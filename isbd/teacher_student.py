"""
Noisy-Student Teacher-Self-Distillation (EMA Consistency Regularization)
========================================================================
Maintains an Exponential Moving Average (EMA) Teacher model that provides
pseudo-supervision for strongly perturbed Student inputs. This enables
unsupervised self-learning from raw unlabelled or partially edited imagery.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TeacherStudentDistiller:
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.995, consistency_weight: float = 0.35):
        self.student = student_model
        self.ema_decay = float(ema_decay)
        self.consistency_weight = float(consistency_weight)
        self.teacher = copy.deepcopy(student_model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def to(self, device):
        self.student.to(device)
        self.teacher.to(device)
        return self

    def perturb(self, x: torch.Tensor) -> torch.Tensor:
        """Applies stochastic noise, mild blur, and random cutout to student input."""
        device = x.device
        perturbed = x.clone()

        # 1. Mild Gaussian noise
        noise = torch.randn_like(perturbed) * 0.03
        perturbed = torch.clamp(perturbed + noise, 0.0, 1.0)

        # 2. Random Cutout / Patch Masking (15% chance per batch item)
        b, c, h, w = perturbed.shape
        if h > 32 and w > 32:
            mask_h, mask_w = int(h * 0.25), int(w * 0.25)
            for i in range(b):
                if torch.rand(1).item() < 0.35:
                    top = torch.randint(0, h - mask_h, (1,)).item()
                    left = torch.randint(0, w - mask_w, (1,)).item()
                    perturbed[i, :, top:top+mask_h, left:left+mask_w] = 0.5

        return perturbed

    def compute_consistency_loss(self, clean_x: torch.Tensor, perturbed_x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates consistency loss between Student(perturbed_x) and Teacher(clean_x).
        Returns (consistency_loss, teacher_prediction).
        """
        if perturbed_x is None:
            perturbed_x = self.perturb(clean_x)

        with torch.no_grad():
            self.teacher.eval()
            teacher_pred = self.teacher(clean_x)

        student_pred = self.student(perturbed_x)
        consistency_loss = F.smooth_l1_loss(student_pred, teacher_pred)
        return self.consistency_weight * consistency_loss, teacher_pred

    def perturb_and_predict(self, clean_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (perturbed_x, teacher_prediction)."""
        perturbed_x = self.perturb(clean_x)
        with torch.no_grad():
            self.teacher.eval()
            teacher_pred = self.teacher(clean_x)
        return perturbed_x, teacher_pred

    def consistency_loss(self, student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
        """Calculates smooth L1 consistency loss."""
        return F.smooth_l1_loss(student_pred, teacher_pred)

    def update_teacher(self, student_model: Optional[nn.Module] = None):
        """Updates teacher EMA weights."""
        if student_model is not None:
            self.student = student_model
        self.update_teacher_ema()

    def update_teacher_ema(self):
        """Exponential Moving Average update of teacher weights."""
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
                p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=(1.0 - self.ema_decay))
