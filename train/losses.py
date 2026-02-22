import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    # thin wrapper around torch BCEWithLogitsLoss so all loss functions live in one place
    # pos_weight: optional tensor of shape (num_labels,) to up-weight positive samples per label

    def __init__(self, pos_weight=None, reduction: str = 'none'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=self.reduction
        )


class AsymmetricLoss(nn.Module):
    # asl losses

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = 'none',
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # forward function for asl

        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs

        # asl prob margin shift: raise xs_neg by `clip` then clamp at 1.  equiv to zeroing negatives whose p_positive < clip.
        if self.clip is not None and self.clip > 0.0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # binary cross entropy components (unreduced)
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # asl focal modulating weights —> computed without gradient to keep the backward pass clean (matches the official implementation).
        with torch.no_grad():
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = (1.0 - pt) ** gamma

        loss = -weight * (loss_pos + loss_neg)

        if self.reduction == 'mean':
            return loss.mean()
        
        if self.reduction == 'sum':
            return loss.sum()
        
        return loss  # 'none'
