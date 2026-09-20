"""Synthetic attack sample generation + Balanced Dataset Formation (Sec. 3.5.4, Algorithm 2)."""
from __future__ import annotations

import numpy as np
import torch

from .tmg_gan import TMGGAN
from .utils import get_logger

log = get_logger()


def harden_onehot(x: torch.Tensor, slices) -> torch.Tensor:
    """Generators emit soft values for one-hot blocks; snap each block to a valid one-hot vector
    so synthetic records have the same form as real (encoded) records."""
    for a, b in slices or []:
        idx = x[:, a:b].argmax(1, keepdim=True)
        x[:, a:b] = torch.zeros_like(x[:, a:b]).scatter_(1, idx, 1.0)
    return x


@torch.no_grad()
def generate_class_samples(gan: TMGGAN, k: int, n: int, max_attempt_factor: int = 20,
                           chunk: int = 2048, onehot_slices=None) -> np.ndarray:
    """Algorithm 2 for class k: keep a generated sample only if classifier C labels it as class k."""
    gan.generators.eval()
    kept: list[torch.Tensor] = []                       # 1: S = {}
    n_kept, drawn, budget = 0, 0, max(n, 1) * max_attempt_factor
    while n_kept < n and drawn < budget:                # 2: while |S| < n
        x = harden_onehot(gan.generators[k].sample(chunk, gan.device), onehot_slices)  # 3-4: z, x~ = G(z)
        ok = gan.classify(x) == k                       # 5-6: valid attack of this class?
        kept.append(x[ok].cpu()); n_kept += int(ok.sum()); drawn += chunk   # 7: add to S
    if n_kept < n:
        log.warning("class %d: only %d/%d generated samples passed the classifier filter "
                    "(train the GAN longer or raise balance.max_attempt_factor)", k, n_kept, n)
    out = torch.cat(kept)[:n].numpy() if kept else np.empty((0, 0), dtype=np.float32)
    return out                                          # 10: return S


def synthetic_targets(y: np.ndarray, class_names: list[str], bcfg) -> dict[int, int]:
    """How many synthetic samples each class receives."""
    counts = np.bincount(y, minlength=len(class_names))
    if bcfg.mode == "max":
        return {k: int(counts.max() - c) for k, c in enumerate(counts) if c < counts.max()}
    if bcfg.mode == "fixed":
        skip = set(bcfg.majority_classes)
        return {k: int(bcfg.n_synthetic) for k, name in enumerate(class_names) if name not in skip}
    raise ValueError(f"Unknown balance.mode '{bcfg.mode}'")


def build_balanced_dataset(gan: TMGGAN, X: np.ndarray, y: np.ndarray, class_names: list[str], bcfg,
                           onehot_slices=None):
    """D' = D U S.  Returns X', y' and a flag array (0 = real, 1 = synthetic)."""
    Xs, ys = [X], [y]
    for k, n in synthetic_targets(y, class_names, bcfg).items():
        s = generate_class_samples(gan, k, n, bcfg.max_attempt_factor, onehot_slices=onehot_slices)
        if len(s):
            Xs.append(s); ys.append(np.full(len(s), k, dtype=y.dtype))
            log.info("generated %6d synthetic '%s' samples", len(s), class_names[k])
    Xb, yb = np.vstack(Xs).astype(np.float32), np.concatenate(ys)
    synthetic = np.concatenate([np.zeros(len(X), dtype=int), np.ones(len(Xb) - len(X), dtype=int)])
    return Xb, yb, synthetic
