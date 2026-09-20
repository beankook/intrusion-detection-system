# Sample run — real UNSW-NB15 data

Produced by this repository, on the official UNSW-NB15 training partition
(`UNSW_NB15_training-set.csv`, 175,341 records), with the default configuration:

```bash
python scripts/download_data.py
python main.py --output-dir outputs/sample_run
```

Setting: the five classes of the paper's Table 2 (Normal, DoS, Reconnaissance, Shellcode, Worms),
a stratified 80 / 20 split, TMG-GAN trained for 2,000 epochs, 10,000 synthetic samples added to
each attack class, then 30 epochs for each classifier. Seed 42, CPU, ~85 seconds.

## Results (held-out test split, 12,888 real records)

| Metric | Focal Loss + Attention | Cross-Entropy, no attention |
|---|---|---|
| Accuracy | 0.9542 | 0.9431 |
| Precision (macro) | 0.8435 | 0.7977 |
| Recall (macro) | 0.7518 | 0.6455 |
| F1-score (macro) | 0.7829 | 0.6713 |
| FAR | 0.0104 | 0.0120 |

Both models were trained on the same TMG-GAN-balanced data with the same architecture, optimiser,
seed and epoch count; only the attention module and the loss function differ. The proposed
combination wins on every metric, with the largest gain in recall (+0.106) and F1 (+0.112) — the
behaviour the paper predicts, since focal loss is aimed at hard minority samples.

## How this differs from the paper's numbers

These are **not** the paper's numbers and are not meant to be. The paper's Table 2 was scored on
training plus synthetic data (see `../paper_reference/README.md`); this run is scored on a held-out
split of real records only, which is the harder and more honest setting. Accuracy here is higher
because the five-class problem is dominated by Normal traffic, while macro recall is lower because
Worms has only 25 real test records.

Your own numbers will move a little with seed and hardware. Report a mean ± std over several
`--seed` values rather than a single run.

## Files

`table1_metrics.csv`, `table2_confusion_matrix.csv`, `per_class_proposed.csv`,
`per_class_baseline.csv`, `imbalance_before.csv`, `imbalance_after.csv`, `metrics.json`,
`fig2_feature_embedding.png`, `fig3_metric_comparison.png`, `fig4_confusion_matrix.png`,
`fig5_traffic_distribution.png`, `training_curves.png`, `attention_feature_importance.png`,
and `model/` (trained classifier, baseline, TMG-GAN, fitted preprocessor).
