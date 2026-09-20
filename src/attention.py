"""Feature Attention Mechanism Module - proposed contribution (Sec. 3.5.5 / 4.4).

    alpha = softmax(W x + b)          one importance weight per input feature
    X'    = alpha (.) X               element-wise re-weighting

Because alpha sums to 1, alpha (.) X shrinks the input by ~1/n. With `rescale=True` the
weights are multiplied by n (mean weight = 1) so the classifier sees inputs on the original
scale; the relative feature importances are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    def __init__(self, n_features: int, rescale: bool = True):
        super().__init__()
        self.score = nn.Linear(n_features, n_features)   # W, b  (feature weight calculation)
        self.n, self.rescale = n_features, rescale

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score(x), dim=1)       # alpha (attention layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.weights(x)
        if self.rescale:
            alpha = alpha * self.n
        return alpha * x                                 # X' (feature importance assignment)
