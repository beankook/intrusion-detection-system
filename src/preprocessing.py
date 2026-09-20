"""Data Preprocessing Module (Sec. 3.5.2 / 4.2).

  1. data cleaning          - drop id / duplicates, fix missing values, tidy attack names
  2. categorical encoding   - one-hot for proto / service / state
  3. min-max normalisation  - x_norm = (x - x_min) / (x_max - x_min), fitted on the training split only
  4. train / test split     - stratified, D = D_train U D_test
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLS = ["id", "label", "attack_cat"]
_NAME_FIXES = {"Backdoors": "Backdoor", "": "Normal", "nan": "Normal", "-": "Normal"}


# --------------------------------------------------------------------------- #
# loading + cleaning
# --------------------------------------------------------------------------- #
def load_raw(train_csv: str, test_csv: str | None = None, fmt: str = "unsw") -> pd.DataFrame:
    """Read the dataset file(s) and return one cleaned DataFrame with an `attack_cat` column.

    fmt='unsw'   : UNSW-NB15 training/testing-set partition files
    fmt='cicids' : CICIDS2017 MachineLearningCSV flow files (the other dataset used by
                   Ding et al.); its ' Label' column is renamed to `attack_cat` and
                   BENIGN is renamed to Normal.
    """
    paths = [p for p in (train_csv, test_csv) if p and Path(p).exists()]
    if train_csv and not paths and any(ch in str(train_csv) for ch in "*?"):
        paths = sorted(str(p) for p in Path(train_csv).parent.glob(Path(train_csv).name))
    if not paths:
        raise FileNotFoundError(
            f"No dataset found at '{train_csv}'. Run `python scripts/download_data.py`, see "
            "data/README.md, or use `python main.py --demo` for a smoke test on placeholder data."
        )
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    if fmt == "cicids":
        df = _cicids_to_common(df)
    elif fmt != "unsw":
        raise ValueError(f"Unknown data.format '{fmt}' (expected 'unsw' or 'cicids')")
    return clean(df)


def _cicids_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """CICIDS2017 -> the common layout: an `attack_cat` label column plus feature columns."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError("CICIDS2017 files are expected to have a 'Label' column.")
    df = df.rename(columns={label_col: "attack_cat"})
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip().replace({"BENIGN": "Normal"})
    df["label"] = (df["attack_cat"] != "Normal").astype(int)
    # flow identifiers leak the label and are not behavioural features
    return df.drop(columns=[c for c in ["Flow ID", "Source IP", "Destination IP", "Timestamp",
                                        "Source Port", "Destination Port"] if c in df.columns])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning: column names, attack names, missing values, infinities, duplicates."""
    df = df.copy()
    df.columns = [c.strip().lower().lstrip("\ufeff") for c in df.columns]
    if "attack_cat" not in df.columns:
        raise ValueError("Expected an 'attack_cat' column (UNSW-NB15 training/testing-set format).")
    df["attack_cat"] = (
        df["attack_cat"].astype(str).str.strip().replace(_NAME_FIXES)
    )
    if "label" in df.columns:  # rows flagged benign are Normal whatever attack_cat says
        df.loc[df["label"] == 0, "attack_cat"] = "Normal"
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"-": "none", "nan": "none", "": "none"})
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df[num] = df[num].fillna(df[num].median())
    df = df.drop(columns=["id"], errors="ignore").drop_duplicates().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# fitted preprocessor (re-used unchanged at inference time)
# --------------------------------------------------------------------------- #
@dataclass
class Preprocessor:
    categorical_cols: list[str]
    max_categories: int = 10
    log_transform: bool = False
    numeric_cols: list[str] = field(default_factory=list)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.categorical_cols = [c for c in self.categorical_cols if c in df.columns]
        self.numeric_cols = [
            c for c in df.columns
            if c not in self.categorical_cols and c not in DROP_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        for c in self.categorical_cols:
            top = df[c].astype(str).value_counts().index[: self.max_categories].tolist()
            self.categories_[c] = sorted(top) + ["other"]
        num = self._numeric(df)
        self.min_, self.max_ = num.min(axis=0), num.max(axis=0)
        self.feature_names_ = list(self.numeric_cols) + [
            f"{c}={v}" for c in self.categorical_cols for v in self.categories_[c]
        ]
        return self

    def onehot_slices(self) -> list[tuple[int, int]]:
        """Column ranges [start, end) of each one-hot block in the transformed matrix."""
        out, start = [], len(self.numeric_cols)
        for c in self.categorical_cols:
            out.append((start, start + len(self.categories_[c])))
            start = out[-1][1]
        return out

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        num = df.reindex(columns=self.numeric_cols).astype(float).fillna(0.0).to_numpy()
        return np.log1p(np.clip(num, 0, None)) if self.log_transform else num

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # min-max normalisation to [0, 1]; unseen values at inference are clipped
        span = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        num = np.clip((self._numeric(df) - self.min_) / span, 0.0, 1.0)
        blocks = [num]
        # one-hot encoding  x_cat -> [0, 0, ..., 1, ..., 0]
        for c in self.categorical_cols:
            cats = self.categories_[c]
            col = df[c].astype(str) if c in df.columns else pd.Series(["other"] * len(df))
            col = col.where(col.isin(cats), "other")
            idx = col.map({v: i for i, v in enumerate(cats)}).to_numpy()
            oh = np.zeros((len(df), len(cats)), dtype=float)
            oh[np.arange(len(df)), idx] = 1.0
            blocks.append(oh)
        return np.hstack(blocks).astype(np.float32)


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray          # attack-category index (always multi-class; used by the GAN)
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    preprocessor: Preprocessor


def prepare_data(df: pd.DataFrame, cfg) -> DataBundle:
    """Select classes, split (stratified), fit the preprocessor on the training part only."""
    d = cfg.data
    if d.classes != "all":
        missing = set(d.classes) - set(df["attack_cat"].unique())
        if missing:
            raise ValueError(f"Classes {sorted(missing)} not present in the dataset.")
        df = df[df["attack_cat"].isin(d.classes)].reset_index(drop=True)
        class_names = list(d.classes)
    else:
        found = sorted(df["attack_cat"].unique())
        class_names = ["Normal"] + [c for c in found if c != "Normal"]
    y = df["attack_cat"].map({c: i for i, c in enumerate(class_names)}).to_numpy()

    tr, te = train_test_split(
        np.arange(len(df)), test_size=d.test_size, stratify=y, random_state=cfg.seed
    )
    pre = Preprocessor(list(d.categorical_cols), d.max_categories, d.log_transform).fit(df.iloc[tr])
    return DataBundle(
        pre.transform(df.iloc[tr]), y[tr], pre.transform(df.iloc[te]), y[te], class_names, pre
    )


def imbalance_report(y: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Imbalance analysis step of the workflow (N_normal >> N_attack)."""
    counts = np.bincount(y, minlength=len(class_names))
    return pd.DataFrame(
        {"class": class_names, "samples": counts, "share_%": np.round(100 * counts / counts.sum(), 2),
         "imbalance_ratio": np.round(counts.max() / np.maximum(counts, 1), 1)}
    )
