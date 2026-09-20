"""Enhanced TMG-GAN IDS - full training + evaluation pipeline (Algorithm 3 of the paper).

    python main.py                                  # paper setting (5 classes), real UNSW-NB15 data
    python main.py --config configs/all_classes.yaml
    python main.py --config configs/binary.yaml
    python main.py --demo                           # quick smoke test on generated placeholder data
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from src import visualize as viz
from src.classifier import attention_features, mean_attention_weights, predict, train_classifier
from src.config import load_config, to_dict
from src.evaluate import compute_metrics, per_class_report, table1, table2
from src.generate import build_balanced_dataset
from src.preprocessing import clean, imbalance_report, load_raw, prepare_data
from src.tmg_gan import TMGGAN
from src.utils import get_device, get_logger, set_seed

log = get_logger()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML file overriding configs/default.yaml")
    ap.add_argument("--demo", action="store_true", help="placeholder data + tiny settings (smoke test only)")
    ap.add_argument("--gan-epochs", type=int, default=None)
    ap.add_argument("--clf-epochs", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def build_config(args):
    ov: dict = {}
    if args.demo:
        ov = {"gan": {"epochs": 300, "log_every": 100}, "classifier": {"epochs": 8},
              "balance": {"n_synthetic": 500}, "eval": {"tsne_samples_per_group": 100},
              "output_dir": "outputs/demo"}
    if args.gan_epochs is not None:
        ov.setdefault("gan", {})["epochs"] = args.gan_epochs
    if args.clf_epochs is not None:
        ov.setdefault("classifier", {})["epochs"] = args.clf_epochs
    if args.output_dir:
        ov["output_dir"] = args.output_dir
    if args.seed is not None:
        ov["seed"] = args.seed
    return load_config(args.config, ov)


def run(cfg, demo: bool = False) -> dict:
    t0 = time.time()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir); (out / "model").mkdir(parents=True, exist_ok=True)
    log.info("device: %s | output: %s", device, out)

    # ---- 1. preprocess dataset D --------------------------------------------------------
    if demo:
        from scripts.make_demo_data import make
        log.warning("DEMO MODE: generated placeholder data - results are meaningless, do not report them")
        df = clean(make())
    else:
        df = load_raw(cfg.data.train_csv, cfg.data.test_csv, getattr(cfg.data, "format", "unsw"))
    data = prepare_data(df, cfg)
    names, K = data.class_names, len(data.class_names)
    log.info("features: %d | train: %d | test: %d", data.X_train.shape[1], len(data.y_train), len(data.y_test))
    report = imbalance_report(data.y_train, names)
    log.info("imbalance analysis (training split):\n%s", report.to_string(index=False))
    report.to_csv(out / "imbalance_before.csv", index=False)

    # ---- 2. train TMG-GAN (Algorithm 1) --------------------------------------------------
    gan = TMGGAN(data.X_train.shape[1], K, cfg, device).fit(data.X_train, data.y_train)

    # ---- 3-4. synthetic samples (Algorithm 2) and balanced dataset D' = D U S -----------
    Xb, yb_cat, synthetic = build_balanced_dataset(gan, data.X_train, data.y_train, names, cfg.balance,
                                                  data.preprocessor.onehot_slices())
    imbalance_report(yb_cat, names).to_csv(out / "imbalance_after.csv", index=False)
    log.info("balanced dataset: %d samples (%d synthetic)", len(yb_cat), synthetic.sum())

    # label space of the final classifier
    if cfg.data.task == "binary":
        label_names = ["Normal", "Attack"]
        yb, y_test = (yb_cat != 0).astype(int), (data.y_test != 0).astype(int)
    else:
        label_names, yb, y_test = names, yb_cat, data.y_test
    n_out = len(label_names)

    # ---- 5-9. attention + focal-loss classifier, and the CE / no-attention baseline -----
    proposed, h_prop = train_classifier(Xb, yb, n_out, cfg, device,
                                        use_attention=True, loss="focal", tag="Focal+Attn")
    set_seed(cfg.seed)
    baseline, h_base = train_classifier(Xb, yb, n_out, cfg, device,
                                        use_attention=False, loss="ce", tag="CE_NoAttn")

    # ---- evaluation on the held-out REAL test split --------------------------------------
    pred_p, _ = predict(proposed, data.X_test, device)
    pred_b, _ = predict(baseline, data.X_test, device)
    m_prop = compute_metrics(y_test, pred_p, cfg.eval.average)
    m_base = compute_metrics(y_test, pred_b, cfg.eval.average)
    t1, t2 = table1(m_prop, m_base), table2(y_test, pred_p, label_names)
    t1.to_csv(out / "table1_metrics.csv", index=False)
    t2.to_csv(out / "table2_confusion_matrix.csv")
    per_class_report(y_test, pred_p, label_names).to_csv(out / "per_class_proposed.csv", index=False)
    per_class_report(y_test, pred_b, label_names).to_csv(out / "per_class_baseline.csv", index=False)
    print("\nTABLE 1\n" + t1.to_string(index=False))
    print("\nTABLE 2 (confusion matrix, proposed model)\n" + t2.to_string() + "\n")

    # ---- figures ---------------------------------------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    keep = np.concatenate([
        rng.choice(idx, min(len(idx), cfg.eval.tsne_samples_per_group), replace=False)
        for c in range(K) for s in (0, 1)
        if len(idx := np.flatnonzero((yb_cat == c) & (synthetic == s)))
    ])
    viz.fig2_feature_embedding(attention_features(proposed, Xb[keep], device), yb_cat[keep],
                               synthetic[keep], out / "fig2_feature_embedding.png", cfg.seed)
    viz.fig3_metric_comparison(m_prop, m_base, out / "fig3_metric_comparison.png")
    viz.fig4_confusion_matrix(t2, out / "fig4_confusion_matrix.png")
    viz.fig5_traffic_distribution(pred_p, label_names, out / "fig5_traffic_distribution.png")
    viz.training_curves(gan.history, {"Focal+Attn": h_prop, "CE_NoAttn": h_base}, out / "training_curves.png")
    viz.attention_importance(mean_attention_weights(proposed, data.X_test, device),
                             data.preprocessor.feature_names_, out / "attention_feature_importance.png")

    # ---- 10. save trained model M + everything inference needs ---------------------------
    torch.save(proposed.state_dict(), out / "model" / "proposed_model.pt")
    torch.save(baseline.state_dict(), out / "model" / "baseline_model.pt")
    torch.save(gan.state_dict(), out / "model" / "tmg_gan.pt")
    with open(out / "model" / "preprocessor.pkl", "wb") as f:
        pickle.dump(data.preprocessor, f)
    c = cfg.classifier
    meta = {"label_names": label_names, "normal_index": 0, "gan_class_names": names,
            "model_kwargs": dict(n_features=int(Xb.shape[1]), n_classes=n_out, hidden=list(c.hidden),
                                 dropout=c.dropout, use_attention=True, attention_rescale=c.attention_rescale)}
    (out / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    results = {"demo_placeholder_data": demo, "proposed": m_prop, "baseline": m_base,
               "config": to_dict(cfg), "runtime_s": round(time.time() - t0, 1)}
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info("done in %.1fs - results in %s", time.time() - t0, out)
    return results


if __name__ == "__main__":
    a = parse_args()
    run(build_config(a), demo=a.demo)
