"""Regenerate the reference artefacts in outputs/paper_reference/ from the values PRINTED IN
THE PAPER (Table 1, Table 2, Fig. 3, Fig. 5).

Nothing here is produced by running the model - the numbers are transcribed from the PDF so the
repository carries the target figures next to the ones your own runs produce. Run

    python scripts/paper_reference.py

Transcription notes (all verifiable against the PDF):
  * Table 1 is printed with three numeric columns; the second and third are identical, so only
    the two distinct ones are kept here.
  * Table 2 prints the Worms row twice. The first copy (…, 640, 863) does not sum to its stated
    total of 10130 and repeats the value 863 from the Shellcode row; the second copy
    (…, 640, 8670) sums to 10130 exactly and is the one used here.
  * The Fig. 5 pie is printed with slice labels 60 / 20 / 18 / 12 / 6, which sum to 116 %, and its
    legend lists "Other Attacks" twice. The reference pie is therefore rebuilt from the predicted
    column totals of Table 2, which the paper's own confusion matrix determines exactly.
  * Fig. 2 (t-SNE) cannot be reconstructed from printed numbers and has no reference copy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import visualize as viz  # noqa: E402
from src.evaluate import compute_metrics, per_class_report  # noqa: E402

OUT = Path("outputs/paper_reference")

CLASSES = ["Normal", "DoS", "Reconnaissance", "Shellcode", "Worms"]

# --- Table 1 (paper, Sec. 5.3) ------------------------------------------------------------
TABLE1 = {
    "Focal Loss + Attention": {"Accuracy": 0.8291, "Precision": 0.8675, "Recall": 0.8121, "F1-score": 0.8177},
    "CrossEntropy":           {"Accuracy": 0.8223, "Precision": 0.8675, "Recall": 0.8038, "F1-score": 0.8090},
}

# --- Fig. 5 (paper, Sec. 5.6): slice labels exactly as printed in the pie chart ------------
# These five values sum to 116 %, and the printed legend lists "Other Attacks" twice, so the
# figure cannot be reproduced as printed. Kept here only as a record of what the PDF shows.
FIG5_AS_PRINTED = {"Normal Traffic": 60, "DoS Attack": 20, "Exploit Attack": 18,
                   "Reconnaissance": 12, "Other Attacks": 6}

# --- Table 2 / Fig. 4 (paper, Sec. 5.5): rows = actual, columns = predicted ----------------
CM = np.array([
    [52320,  1680,   980,   580,   440],   # Normal         (56000)
    [ 1620, 18350,  1150,   650,   494],   # DoS            (22264)
    [ 1490,   980, 16840,   620,   561],   # Reconnaissance (20491)
    [  520,   310,   290,  9150,   863],   # Shellcode      (11133)
    [  430,   210,   180,   640,  8670],   # Worms          (10130)
])


def expand(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn the confusion matrix back into (y_true, y_pred) so the repo's own metric code can read it."""
    y_true = np.repeat(np.arange(len(cm)), cm.sum(1))
    y_pred = np.concatenate([np.repeat(np.arange(len(cm)), row) for row in cm])
    return y_true, y_pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = expand(CM)

    t1 = pd.DataFrame({"Metric": list(TABLE1["Focal Loss + Attention"]),
                       "Focal Loss + Attention": list(TABLE1["Focal Loss + Attention"].values()),
                       "CrossEntropy": list(TABLE1["CrossEntropy"].values())})
    t1.to_csv(OUT / "table1_metrics.csv", index=False)

    idx = [f"{n} ({t})" for n, t in zip(CLASSES, CM.sum(1))]
    t2 = pd.DataFrame(CM, index=idx, columns=CLASSES)
    t2.index.name = "Actual \\ Predicted"
    t2.to_csv(OUT / "table2_confusion_matrix.csv")

    per_class_report(y_true, y_pred, CLASSES).to_csv(OUT / "per_class_from_table2.csv", index=False)

    pd.DataFrame({"category": list(FIG5_AS_PRINTED), "percent_as_printed": list(FIG5_AS_PRINTED.values())}
                 ).to_csv(OUT / "fig5_slices_as_printed.csv", index=False)

    # what Table 2 actually implies, next to what Table 1 reports
    implied = compute_metrics(y_true, y_pred, "macro")
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "FAR"],
        "Implied by Table 2": [round(implied[m], 4) for m in ["Accuracy", "Precision", "Recall", "F1-score", "FAR"]],
        "Reported in Table 1": [TABLE1["Focal Loss + Attention"].get(m, "not reported")
                                for m in ["Accuracy", "Precision", "Recall", "F1-score", "FAR"]],
    }).to_csv(OUT / "table1_vs_table2_consistency.csv", index=False)

    viz.fig3_metric_comparison(TABLE1["Focal Loss + Attention"], TABLE1["CrossEntropy"],
                               OUT / "fig3_metric_comparison.png")
    viz.fig4_confusion_matrix(t2, OUT / "fig4_confusion_matrix.png")
    viz.fig5_traffic_distribution(y_pred, CLASSES, OUT / "fig5_traffic_distribution.png")

    # consistency check between the paper's own Table 1 and Table 2
    acc_cm = float(np.trace(CM) / CM.sum())
    (OUT / "consistency_check.json").write_text(json.dumps({
        "table2_total_samples": int(CM.sum()),
        "table2_correct": int(np.trace(CM)),
        "accuracy_implied_by_table2": round(acc_cm, 4),
        "accuracy_reported_in_table1": TABLE1["Focal Loss + Attention"]["Accuracy"],
        "difference": round(acc_cm - TABLE1["Focal Loss + Attention"]["Accuracy"], 4),
        "note": ("Table 2 implies a different accuracy from Table 1. Table 2's row totals equal the "
                 "UNSW-NB15 training-partition counts plus 10 000 synthetic samples per attack class, "
                 "so it was scored on training + synthetic data rather than a held-out test set."),
    }, indent=2))
    print(t1.to_string(index=False))
    print(f"\nTable 2 implies accuracy {acc_cm:.4f} vs {TABLE1['Focal Loss + Attention']['Accuracy']} in Table 1")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
