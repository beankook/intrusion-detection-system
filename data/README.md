# Dataset

This project uses the **UNSW-NB15** dataset (Moustafa & Slay, UNSW Canberra).

1. Download the *Training and Testing Sets* partition from the official page:
   https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Put the two CSV files in this folder:

```
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv     # optional
```

Both files are merged, cleaned and re-split (stratified, 80 / 20 by default), so it does not
matter which of the two is called "training" or "testing". If only the first file is present
the pipeline runs on that file alone.

Required columns: the standard 45-column layout (`id, dur, proto, service, state, …, attack_cat, label`).
The CSV files are git-ignored - do not commit them; the dataset has its own terms of use.
