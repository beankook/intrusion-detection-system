"""Focal Loss-Based Deep Learning Classifier Module (Sec. 3.5.6 / 4.5 / 4.6, Algorithm 3 steps 5-9).

`IDSClassifier(use_attention=True)`  + FocalLoss      -> proposed model      ("Focal+Attn")
`IDSClassifier(use_attention=False)` + CrossEntropy   -> baseline            ("CE_NoAttn")
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .attention import FeatureAttention
from .focal_loss import FocalLoss
from .utils import get_logger

log = get_logger()


class IDSClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: list[int], dropout: float = 0.2,
                 use_attention: bool = True, attention_rescale: bool = True):
        super().__init__()
        self.attention = FeatureAttention(n_features, attention_rescale) if use_attention else None
        layers: list[nn.Module] = []
        for a, b in zip([n_features, *hidden][:-1], hidden):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers, nn.Linear(hidden[-1], n_classes))

    def weighted_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x) if self.attention is not None else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.weighted_features(x))


def _alpha(spec, y: np.ndarray, n_classes: int, device) -> torch.Tensor | None:
    if spec is None:
        return None
    if spec == "balanced":
        counts = np.bincount(y, minlength=n_classes).astype(float)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32, device=device)
    w = torch.tensor(list(spec), dtype=torch.float32, device=device)
    if len(w) != n_classes:
        raise ValueError("classifier.focal_alpha must have one entry per class")
    return w


def train_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, cfg, device: torch.device, *,
                     use_attention: bool, loss: str, tag: str) -> tuple[IDSClassifier, list[dict]]:
    """Mini-batch training with theta <- theta - eta * grad(L); best validation-loss weights are kept."""
    c = cfg.classifier
    model = IDSClassifier(X.shape[1], n_classes, list(c.hidden), c.dropout,
                          use_attention, c.attention_rescale).to(device)
    criterion = (FocalLoss(c.focal_gamma, _alpha(c.focal_alpha, y, n_classes, device))
                 if loss == "focal" else nn.CrossEntropyLoss()).to(device)
    opt = (torch.optim.SGD(model.parameters(), lr=c.lr, momentum=0.9) if c.optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=c.lr))

    tr, va = train_test_split(np.arange(len(y)), test_size=c.val_fraction, stratify=y,
                              random_state=cfg.seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    best, best_state, history = float("inf"), None, []
    for epoch in range(1, c.epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen).to(device)]
        total = 0.0
        for i in range(0, len(perm), c.batch_size):
            idx = perm[i:i + c.batch_size]
            loss_v = criterion(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss_v.backward(); opt.step()
            total += loss_v.item() * len(idx)
        model.eval()
        with torch.no_grad():
            out = model(Xt[va_t])
            val_loss = criterion(out, yt[va_t]).item()
            val_acc = (out.argmax(1) == yt[va_t]).float().mean().item()
        history.append(dict(epoch=epoch, train_loss=total / len(perm), val_loss=val_loss, val_acc=val_acc))
        if val_loss < best:
            best, best_state = val_loss, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch in (1, c.epochs):
            log.info("[%s] epoch %3d/%d | train %.4f | val %.4f | val acc %.4f",
                     tag, epoch, c.epochs, history[-1]["train_loss"], val_loss, val_acc)
    model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def predict(model: IDSClassifier, X: np.ndarray, device: torch.device, batch: int = 8192):
    """Returns (predicted labels, class probabilities)."""
    model.eval()
    probs = [torch.softmax(model(torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)), 1).cpu()
             for i in range(0, len(X), batch)]
    p = torch.cat(probs).numpy()
    return p.argmax(1), p


@torch.no_grad()
def attention_features(model: IDSClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    """X' = alpha (.) X for visualisation (Fig. 2)."""
    return model.weighted_features(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


@torch.no_grad()
def mean_attention_weights(model: IDSClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    return model.attention.weights(torch.as_tensor(X, dtype=torch.float32, device=device)).mean(0).cpu().numpy()
