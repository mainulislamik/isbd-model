"""
Elastic Weight Consolidation (EWC) for ISBD AI
==============================================
Prevents catastrophic forgetting during continuous micro-finetuning on live
Photoshop designer streams and domain-specific harvesting (e.g. jewelry, skin, hair).

Mathematical formulation:
    L_total = L_task(theta) + sum_i (lambda_ewc / 2) * F_i * (theta_i - theta_star_i)^2

where F_i is the diagonal element of the empirical Fisher Information Matrix
computed over historical anchor samples (Golden Ground Truth pairs).
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EWC_ANCHOR_FILE = DATA / "ewc_anchor.pt"


class EWCRegularizer:
    def __init__(self, model: nn.Module, lambda_ewc: float = 80.0):
        self.model = model
        self.lambda_ewc = float(lambda_ewc)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def compute_fisher_matrix(self, dataloader, loss_fn, device="cpu", max_samples: int = 40):
        """
        Computes diagonal empirical Fisher Information Matrix over anchor samples.
        """
        self.model.eval()
        self.model.to(device)
        self.fisher = {n: torch.zeros_like(p, device=device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.optimal_params = {n: p.detach().clone().to(device) for n, p in self.model.named_parameters() if p.requires_grad}

        count = 0
        for batch in dataloader:
            if count >= max_samples:
                break
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                continue

            self.model.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2

            count += x.size(0)

        if count > 0:
            for n in self.fisher:
                self.fisher[n] /= float(count)
                # Apply numerical smoothing clamp
                self.fisher[n] = torch.clamp(self.fisher[n], min=1e-7, max=1e3)

        return count

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Calculates the quadratic EWC penalty loss for current model parameters.
        """
        if not self.fisher or not self.optimal_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.optimal_params and p.requires_grad:
                f = self.fisher[n].to(p.device)
                p_star = self.optimal_params[n].to(p.device)
                penalty += torch.sum(f * (p - p_star) ** 2)

        return 0.5 * self.lambda_ewc * penalty

    def save(self, filepath: Optional[Path] = None):
        target = filepath or EWC_ANCHOR_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        cpu_fisher = {n: f.cpu() for n, f in self.fisher.items()}
        cpu_params = {n: p.cpu() for n, p in self.optimal_params.items()}
        torch.save({
            "lambda_ewc": self.lambda_ewc,
            "fisher": cpu_fisher,
            "optimal_params": cpu_params,
        }, str(target))

    def load(self, filepath: Optional[Path] = None) -> bool:
        target = filepath or EWC_ANCHOR_FILE
        if not target.exists():
            return False
        try:
            checkpoint = torch.load(str(target), map_location="cpu")
            self.lambda_ewc = float(checkpoint.get("lambda_ewc", self.lambda_ewc))
            self.fisher = checkpoint.get("fisher", {})
            self.optimal_params = checkpoint.get("optimal_params", {})
            return len(self.fisher) > 0
        except Exception as e:
            print(f"[EWC] Failed to load anchor: {e}")
            return False

    def compute_fisher(self, dataloader, loss_fn, device="cpu", sample_limit: int = 40):
        return self.compute_fisher_matrix(dataloader, loss_fn, device=device, max_samples=sample_limit)


def get_or_create_ewc_anchor(model: nn.Module, device="cpu") -> Optional[EWCRegularizer]:
    """Helper to return an active EWCRegularizer, loaded from disk if available."""
    ewc = EWCRegularizer(model)
    if ewc.load():
        return ewc
    return None


# Alias for clean importing
EWC = EWCRegularizer

