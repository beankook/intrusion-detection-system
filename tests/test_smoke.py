"""Fast checks of the individual modules and one end-to-end run on placeholder data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import FeatureAttention
from src.config import load_config
from src.evaluate import compute_metrics, false_alarm_rate
from src.focal_loss import FocalLoss


def test_focal_loss_reduces_to_cross_entropy():
    logits, y = torch.randn(32, 5), torch.randint(0, 5, (32,))
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), ce, atol=1e-6)
    assert FocalLoss(gamma=2.0)(logits, y) < ce          # easy samples are down-weighted


def test_attention_weights_sum_to_one():
    att = FeatureAttention(10, rescale=False)
    x = torch.rand(4, 10)
    assert torch.allclose(att.weights(x).sum(1), torch.ones(4), atol=1e-5)
    assert att(x).shape == x.shape


def test_false_alarm_rate():
    y_true, y_pred = np.array([0, 0, 0, 0, 1, 1]), np.array([0, 0, 0, 1, 1, 0])
    assert false_alarm_rate(y_true, y_pred) == 0.25       # FP=1, TN=3
    assert compute_metrics(y_true, y_pred)["FAR"] == 0.25


def test_pipeline_end_to_end(tmp_path):
    from main import run
    from src.inference import IntrusionDetector
    from scripts.make_demo_data import make

    cfg = load_config(overrides={
        "output_dir": str(tmp_path), "device": "cpu",
        "gan": {"epochs": 30, "log_every": 30}, "classifier": {"epochs": 2},
        "balance": {"n_synthetic": 100, "max_attempt_factor": 50},
        "eval": {"tsne_samples_per_group": 30}})
    res = run(cfg, demo=True)
    assert 0.0 <= res["proposed"]["Accuracy"] <= 1.0
    for f in ["table1_metrics.csv", "table2_confusion_matrix.csv", "fig2_feature_embedding.png",
              "fig3_metric_comparison.png", "fig4_confusion_matrix.png", "fig5_traffic_distribution.png"]:
        assert (tmp_path / f).exists(), f
    flows = make(scale=0.002).drop(columns=["attack_cat", "label"])
    out = IntrusionDetector(tmp_path / "model", "cpu").predict(flows)
    assert len(out) == len(flows) and set(out["action"]) <= {"ALLOW", "BLOCK"}
