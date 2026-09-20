"""Performance Evaluation Metrics (Sec. 4.8): Accuracy, Precision, Recall, F1-score, False Alarm Rate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def false_alarm_rate(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """FAR = FP / (FP + TN), computed on Normal (0) vs Attack (1)."""
    tn, fp, _, _ = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro",
                    normal_index: int = 0) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    tb, pb = (y_true != normal_index).astype(int), (y_pred != normal_index).astype(int)
    bp, br, bf, _ = precision_recall_fscore_support(tb, pb, average="binary", zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(p), "Recall": float(r), "F1-score": float(f1),
        "FAR": false_alarm_rate(tb, pb),
        # attack-vs-normal view (TP / TN / FP / FN formulas of Sec. 4.8)
        "Binary Accuracy": float(accuracy_score(tb, pb)),
        "Binary Precision": float(bp), "Binary Recall": float(br), "Binary F1-score": float(bf),
    }


def per_class_report(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    return pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f1, "support": s}).round(4)


def table1(proposed: dict, baseline: dict) -> pd.DataFrame:
    """TABLE 1 of the paper: Focal Loss + Attention vs CrossEntropy."""
    rows = ["Accuracy", "Precision", "Recall", "F1-score", "FAR"]
    return pd.DataFrame({
        "Metric": rows,
        "Focal Loss + Attention": [round(proposed[m], 4) for m in rows],
        "CrossEntropy": [round(baseline[m], 4) for m in rows],
    })


def table2(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    """TABLE 2 / Fig. 4 of the paper: confusion matrix with row totals in the index."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    idx = [f"{n} ({t})" for n, t in zip(class_names, cm.sum(1))]
    df = pd.DataFrame(cm, index=idx, columns=class_names)
    df.index.name = "Actual \\ Predicted"
    return df
