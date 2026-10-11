"""
ISBD Pro Loss Engine — Advanced Multi-Objective Loss Functions & Knowledge Distillation
Includes:
- Differentiable 2D SSIM (Structural Similarity)
- Sobel Gradient Edge Loss (for sharp outlines, collars, hair edges)
- Laplacian 2nd-Order Edge Loss (for razor-sharp boundary details & flyaways)
- Cosine Hue & Color Constancy Loss (for authentic skin/fabric chromaticity)
- KL-Divergence Soft Distillation Loss (Hinton-style Teacher-Student knowledge transfer)
- Feature Embedding Cosine Alignment Loss
- Composite Pro Studio & Distillation Loss Suites
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, size, size).contiguous()
    return window


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes Structural Similarity (SSIM) between two images in [0, 1].
    Returns (1 - SSIM) as a loss to minimize.
    """
    channel = img1.size(1)
    window = _gaussian_window(window_size, 1.5, channel).to(img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return torch.clamp(1.0 - ssim_map.mean(), 0.0, 2.0)
    else:
        return torch.clamp(1.0 - ssim_map.mean(dim=[-1, -2, -3]), 0.0, 2.0)


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Extracts spatial image gradients (Sobel filter) and penalizes edge differences.
    Crucial for neck joint, collar borders, clipping paths, and apparel seams.
    """
    device, dtype = pred.device, pred.dtype
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [ 0.0,  0.0,  0.0],
                            [ 1.0,  2.0,  1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)

    pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

    pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)

    target_gx = F.conv2d(target_gray, sobel_x, padding=1)
    target_gy = F.conv2d(target_gray, sobel_y, padding=1)
    target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

    return F.l1_loss(pred_grad, target_grad)


def laplacian_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates 2nd-order spatial derivative (Laplacian kernel).
    Extraordinarily sensitive to fine-grained boundary transitions,
    hair masking flyaways, and razor-sharp clipping path contours.
    """
    device, dtype = pred.device, pred.dtype
    lap_kernel = torch.tensor([[0.0,  1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0,  1.0, 0.0]], device=device, dtype=dtype).view(1, 1, 3, 3)

    pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

    pred_lap = F.conv2d(pred_gray, lap_kernel, padding=1)
    target_lap = F.conv2d(target_gray, lap_kernel, padding=1)

    return F.l1_loss(pred_lap, target_lap)


def color_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalizes angular chromaticity deviation in 3D RGB color space.
    Ensures that color retouching, jewelry gloss, and recoloring
    preserve authentic hue angles without washed-out or discolored tints.
    """
    eps = 1e-7
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + eps)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + eps)
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    return torch.clamp(1.0 - cos_sim.mean(), 0.0, 2.0)


def kl_distillation_loss(student_pred: torch.Tensor, teacher_target: torch.Tensor,
                         temperature: float = 2.0) -> torch.Tensor:
    """
    Hinton-style Knowledge Distillation Soft Loss.
    Scales logits/pixels by temperature T, computes log-softmax of student vs
    softmax of teacher, and weights by T^2 to transfer dark knowledge.
    """
    b, c, h, w = student_pred.shape
    # Flatten spatial features
    s_flat = student_pred.view(b, c, -1) / temperature
    t_flat = teacher_target.view(b, c, -1) / temperature

    s_log_soft = F.log_softmax(s_flat, dim=-1)
    t_soft = F.softmax(t_flat, dim=-1)

    kd = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (temperature ** 2)
    return kd


def feature_cosine_distill_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """
    Cosine alignment between normalized student and teacher feature representations.
    """
    eps = 1e-7
    s_norm = student_feat / (torch.norm(student_feat, dim=1, keepdim=True) + eps)
    t_norm = teacher_feat / (torch.norm(teacher_feat, dim=1, keepdim=True) + eps)
    cos_sim = (s_norm * t_norm).sum(dim=1).mean()
    return torch.clamp(1.0 - cos_sim, 0.0, 2.0)


class ISBDProLoss(nn.Module):
    """
    Composite studio loss function:
    L = L1 + alpha * SSIM + beta * Sobel + delta * Laplacian + eta * Color_Cosine + gamma * MSE
    """
    def __init__(self, ssim_weight: float = 0.20, edge_weight: float = 0.15,
                 laplacian_weight: float = 0.15, color_weight: float = 0.10,
                 mse_weight: float = 0.05):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight
        self.color_weight = color_weight
        self.mse_weight = mse_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        loss = l1

        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * ssim_loss(pred, target)

        if self.edge_weight > 0:
            loss = loss + self.edge_weight * sobel_edge_loss(pred, target)

        if self.laplacian_weight > 0:
            loss = loss + self.laplacian_weight * laplacian_edge_loss(pred, target)

        if self.color_weight > 0:
            loss = loss + self.color_weight * color_cosine_loss(pred, target)

        if self.mse_weight > 0:
            loss = loss + self.mse_weight * F.mse_loss(pred, target)

        return loss


class ISBDDistillationLoss(nn.Module):
    """
    Full Knowledge Distillation Suite:
    Combines Ground Truth Reconstruction (ISBDProLoss) + Teacher Soft-Target KD + Feature Cosine Alignment.
    """
    def __init__(self, pro_loss: Optional[ISBDProLoss] = None, kd_weight: float = 0.25,
                 feat_weight: float = 0.15, temperature: float = 2.0):
        super().__init__()
        self.pro_loss = pro_loss or ISBDProLoss()
        self.kd_weight = kd_weight
        self.feat_weight = feat_weight
        self.temperature = temperature

    def forward(self, student_pred: torch.Tensor, ground_truth: torch.Tensor,
                teacher_target: Optional[torch.Tensor] = None,
                student_feat: Optional[torch.Tensor] = None,
                teacher_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        total = self.pro_loss(student_pred, ground_truth)

        if teacher_target is not None and self.kd_weight > 0:
            kd = kl_distillation_loss(student_pred, teacher_target, self.temperature)
            total = total + self.kd_weight * kd

        if student_feat is not None and teacher_feat is not None and self.feat_weight > 0:
            feat_loss = feature_cosine_distill_loss(student_feat, teacher_feat)
            total = total + self.feat_weight * feat_loss

        return total


class ISBDFrequencySeparationLoss(nn.Module):
    """
    Dual-Branch Frequency Separation Loss:
    - Low-frequency tone matching (macro illumination, skin undertones)
    - High-frequency texture conservation (skin pores, fabric weave, micro edges)
    """
    def __init__(self, kernel_size: int = 7, low_weight: float = 0.5, high_weight: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Low frequency branch (smooth tones & macro light)
        pred_low = self.pool(pred)
        target_low = self.pool(target)
        loss_low = F.l1_loss(pred_low, target_low)

        # High frequency branch (micro texture residual: pores, hair, cloth)
        pred_high = pred - pred_low
        target_high = target - target_low
        loss_high_l1 = F.l1_loss(pred_high, target_high)

        # Cosine alignment of high-pass frequency vectors
        pred_flat = pred_high.view(pred_high.size(0), -1)
        target_flat = target_high.view(target_high.size(0), -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        loss_high_cos = 1.0 - cos_sim

        return (self.low_weight * loss_low) + (self.high_weight * (loss_high_l1 + 0.5 * loss_high_cos))

