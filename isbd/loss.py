"""
ISBD Pro Loss Engine — Advanced Multi-Objective Loss Functions
Includes:
- Differentiable 2D SSIM (Structural Similarity)
- Sobel Gradient Edge Loss (for sharp outlines, collars, hair edges)
- Composite Pro Studio Loss (L1 + SSIM + Sobel + MSE)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # Sobel kernels for horizontal (dx) and vertical (dy) gradients
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [ 0.0,  0.0,  0.0],
                            [ 1.0,  2.0,  1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)

    # Convert RGB to grayscale brightness for edge extraction: 0.299R + 0.587G + 0.114B
    pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

    pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
    pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)

    target_gx = F.conv2d(target_gray, sobel_x, padding=1)
    target_gy = F.conv2d(target_gray, sobel_y, padding=1)
    target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

    return F.l1_loss(pred_grad, target_grad)


class ISBDProLoss(nn.Module):
    """
    Composite studio loss function:
    L = L1 + alpha * SSIM_Loss + beta * Edge_Loss + gamma * MSE
    """
    def __init__(self, ssim_weight: float = 0.25, edge_weight: float = 0.20, mse_weight: float = 0.05):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.mse_weight = mse_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        loss = l1

        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * ssim_loss(pred, target)

        if self.edge_weight > 0:
            loss = loss + self.edge_weight * sobel_edge_loss(pred, target)

        if self.mse_weight > 0:
            loss = loss + self.mse_weight * F.mse_loss(pred, target)

        return loss
