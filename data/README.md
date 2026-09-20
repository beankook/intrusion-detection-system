# Datasets

Ding et al., *"TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network
Intrusion Detection"* (IEEE TIFS, vol. 19, pp. 1156–1167, 2024) — the base paper — evaluate on
two datasets: **CICIDS2017** and **UNSW-NB15**. This project uses UNSW-NB15 by default and
supports CICIDS2017 through `configs/cicids2017.yaml`.

## UNSW-NB15 (default)

```bash
python scripts/download_data.py
```

The script downloads `UNSW_NB15_training-set.csv` and checks it has 175,341 records and all ten
attack categories. To do it by hand instead:

1. Download the *Training and Testing Sets* partition from the official page:
   https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Put the CSV file(s) in this folder:

```
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv     # optional
```

Both files are merged, cleaned and re-split (stratified, 80 / 20 by default), so it does not
matter which of the two is called "training" or "testing". If only the first file is present
the pipeline runs on that file alone.

Required columns: the standard 45-column layout (`id, dur, proto, service, state, …, attack_cat, label`).
Some mirrors publish a variant with numeric `xProt / xServ / xState` columns and **no `attack_cat`** —
that one cannot be used here, since the attack category is what the multi-generator GAN conditions on.

### Class counts in the training partition

| Class | Records | | Class | Records |
|---|---|---|---|---|
| Normal | 56,000 | | Reconnaissance | 10,491 |
| Generic | 40,000 | | Analysis | 2,000 |
| Exploits | 33,393 | | Backdoor | 1,746 |
| Fuzzers | 18,184 | | Shellcode | 1,133 |
| DoS | 12,264 | | Worms | 130 |

Worms is outnumbered by Normal 431 : 1 — this is the imbalance the whole method exists to fix.
These counts also explain the row totals in the paper's Table 2: each is the count above plus the
10,000 synthetic samples added per attack class (`balance.n_synthetic` in `configs/default.yaml`).

## CICIDS2017 (the second dataset used by Ding et al.)

Not scripted — it is about 1 GB and needs a form submission.

1. Download `MachineLearningCSV.zip` from https://www.unb.ca/cic/datasets/ids-2017.html
2. Unzip the eight day-files into `data/cicids2017/`
3. Run `python main.py --config configs/cicids2017.yaml`

The loader renames the `Label` column to `attack_cat`, maps `BENIGN` to `Normal`, and drops flow
identifiers (`Flow ID`, IPs, ports, `Timestamp`) that leak the label.

## Licensing

The CSV files are git-ignored — do not commit them. Both datasets have their own terms of use and
require citing their original papers.
