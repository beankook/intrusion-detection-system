"""Figures of the Result Analysis section (Fig. 2 - Fig. 5) plus training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def fig2_feature_embedding(Xw: np.ndarray, y: np.ndarray, synthetic: np.ndarray, path: Path, seed: int):
    """Fig. 2 - 2-D t-SNE embedding of attention-weighted features.
    Legend 'c.s': c = class index, s = 0 real sample / 1 TMG-GAN synthetic sample."""
    perplexity = max(5, min(30, (len(Xw) - 1) // 3))
    emb = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(Xw)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = sorted({(int(c), int(s)) for c, s in zip(y, synthetic)})
    cmap = plt.get_cmap("tab20" if len(groups) > 10 else "tab10")
    for i, (c, s) in enumerate(groups):
        m = (y == c) & (synthetic == s)
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.5, color=cmap(i % cmap.N), label=f"{c}.{s}")
    ax.legend(title="class.synthetic", fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig3_metric_comparison(proposed: dict, baseline: dict, path: Path):
    """Fig. 3 - Focal+Attn vs CE_NoAttn on Precision / Recall / F1 / Accuracy."""
    names, keys = ["Precision", "Recall", "F1", "Accuracy"], ["Precision", "Recall", "F1-score", "Accuracy"]
    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, [proposed[k] for k in keys], w, label="Focal+Attn")
    ax.bar(x + w / 2, [baseline[k] for k in keys], w, label="CE_NoAttn")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.0); ax.set_ylabel("Score"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig4_confusion_matrix(cm_df: pd.DataFrame, path: Path):
    """Fig. 4 - confusion matrix of the proposed system."""
    cm = cm_df.to_numpy()
    n = len(cm)
    fig, ax = plt.subplots(figsize=(1.3 * n + 2.5, 1.0 * n + 2))
    im = ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cm_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(n):
        for j in range(n):
            frac = cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if frac > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig5_traffic_distribution(y_pred: np.ndarray, class_names: list[str], path: Path, max_slices: int = 5):
    """Fig. 5 - distribution of detected traffic categories (small categories -> 'Other Attacks')."""
    counts = pd.Series(np.bincount(y_pred, minlength=len(class_names)), index=class_names)
    counts = counts[counts > 0].sort_values(ascending=False)
    if len(counts) > max_slices:
        counts = pd.concat([counts.iloc[:max_slices - 1],
                            pd.Series({"Other Attacks": counts.iloc[max_slices - 1:].sum()})])
    labels = [f"{n} Traffic" if n == "Normal" else (n if n == "Other Attacks" else f"{n} Attack")
              for n in counts.index]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, *_ = ax.pie(counts, autopct="%1.0f%%", startangle=90, counterclock=False,
                        textprops={"color": "white", "fontsize": 11})
    ax.legend(wedges, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    ax.set_title("Distribution of Detected Traffic Categories –\nEnhanced TMG-GAN IDS")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def training_curves(gan_hist: list[dict], clf_hists: dict[str, list[dict]], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    g = pd.DataFrame(gan_hist).set_index("epoch").rolling(max(1, len(gan_hist) // 100)).mean()
    g.plot(ax=axes[0]); axes[0].set_title("TMG-GAN losses (Algorithm 1)"); axes[0].set_xlabel("epoch")
    for tag, h in clf_hists.items():
        d = pd.DataFrame(h)
        axes[1].plot(d["epoch"], d["val_loss"], label=f"{tag} (val loss)")
    axes[1].set_title("Classifier validation loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def attention_importance(weights: np.ndarray, feature_names: list[str], path: Path, top: int = 20):
    """Mean attention weight per feature - which traffic features the model focuses on."""
    order = np.argsort(weights)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_names[i] for i in order], weights[order])
    ax.set_xlabel("mean attention weight α"); ax.set_title(f"Top-{top} features by attention weight")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
