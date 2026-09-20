"""Testing / Deployment phase (Sec. 3.5.7 / 4.7, Algorithm 4).

    x_new -> preprocess -> feature attention -> trained classifier -> decision block
             Normal  -> ALLOW (access allowed)
             Attack  -> BLOCK (access denied)

CLI:  python -m src.inference --model outputs/model --input some_flows.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from .classifier import IDSClassifier, predict
from .preprocessing import clean
from .utils import get_device


class IntrusionDetector:
    """Loads the artefacts written by main.py and classifies raw UNSW-NB15-style flow records."""

    def __init__(self, model_dir: str | Path, device: str = "auto"):
        model_dir = Path(model_dir)
        self.device = get_device(device)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            self.pre = pickle.load(f)
        self.model = IDSClassifier(**self.meta["model_kwargs"]).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "proposed_model.pt", map_location=self.device))
        self.model.eval()
        self.labels = self.meta["label_names"]
        self.normal = self.meta["normal_index"]

    def predict(self, flows: pd.DataFrame) -> pd.DataFrame:
        flows = flows.copy()
        if "attack_cat" not in flows.columns:        # live traffic has no ground truth
            flows["attack_cat"] = "Normal"
        df = clean(flows.assign(_row=range(len(flows))))           # 1: preprocess input sample x
        X = self.pre.transform(df)
        y, p = predict(self.model, X, self.device)                 # 2-4: attention, features, classify
        out = pd.DataFrame({
            "row": df["_row"].to_numpy(),
            "prediction": [self.labels[i] for i in y],             # 5: y <- prediction result
            "confidence": p.max(1).round(4),
            "action": ["ALLOW" if i == self.normal else "BLOCK" for i in y],   # decision block
        })
        return out                                                 # 6: return y


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify network flows with a trained Enhanced TMG-GAN IDS")
    ap.add_argument("--model", default="outputs/model")
    ap.add_argument("--input", required=True, help="CSV with UNSW-NB15 feature columns")
    ap.add_argument("--output", default=None, help="where to write predictions (CSV)")
    args = ap.parse_args()
    res = IntrusionDetector(args.model).predict(pd.read_csv(args.input, low_memory=False))
    print(res.head(20).to_string(index=False))
    print("\n", res["action"].value_counts().to_string())
    if args.output:
        res.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
