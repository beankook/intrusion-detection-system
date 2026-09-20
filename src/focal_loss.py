"""Focal Loss - proposed contribution (Sec. 3.5.6 / 4.5).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

p_t is the predicted probability of the true class. gamma > 0 down-weights easy samples so
training concentrates on hard (mostly minority) samples; gamma = 0 and alpha = 1 gives plain
cross-entropy. Works for binary (2 logits) and multi-class problems.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_pt = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()
