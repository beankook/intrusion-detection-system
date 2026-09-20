"""Download the UNSW-NB15 partition files used by this project.

The official source is https://research.unsw.edu.au/projects/unsw-nb15-dataset. That page is
behind a redirect that is awkward to script, so this helper pulls the identical CSV from a
public GitHub mirror and verifies it. If the mirror disappears, download the file by hand from
the official page and drop it into data/ - nothing else changes.

    python scripts/download_data.py

CICIDS2017 (the second dataset used by Ding et al.) is ~1 GB and is not scripted here:
download "MachineLearningCSV.zip" from https://www.unb.ca/cic/datasets/ids-2017.html,
unzip it into data/cicids2017/ and run `python main.py --config configs/cicids2017.yaml`.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MIRROR = ("https://raw.githubusercontent.com/Ausommet/PacketGuard/main/"
          "Test-Files/UNSW_NB15_training-set.csv")
TARGET = Path("data/UNSW_NB15_training-set.csv")
EXPECTED_ROWS = 175_341          # size of the official training partition
EXPECTED_CLASSES = {"Normal", "Generic", "Exploits", "Fuzzers", "DoS",
                    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"}


def verify(path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    if "attack_cat" not in df.columns:
        raise SystemExit(f"{path} has no 'attack_cat' column - wrong UNSW-NB15 variant.")
    cats = set(df["attack_cat"].astype(str).str.strip().str.replace("Backdoors", "Backdoor"))
    print(f"rows: {len(df)} (expected {EXPECTED_ROWS})  columns: {df.shape[1]}")
    print(df["attack_cat"].value_counts().to_string())
    if len(df) != EXPECTED_ROWS or not EXPECTED_CLASSES <= cats:
        print("\nWARNING: this file does not match the official training partition.", file=sys.stderr)
    else:
        print("\nOK - matches the official UNSW-NB15 training partition.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=MIRROR)
    ap.add_argument("--out", default=str(TARGET))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        print(f"{out} already exists (use --force to re-download)")
    else:
        print(f"downloading {a.url}\n  -> {out}")
        urllib.request.urlretrieve(a.url, out)
    verify(out)


if __name__ == "__main__":
    main()
