# Reference outputs — values taken from the paper

Everything here is **transcribed from the PDF**, not produced by running this code. It exists so
you can put the target numbers side by side with the numbers your own runs produce
(`outputs/sample_run/`). Regenerate with:

```bash
python scripts/paper_reference.py
```

| File | Source in the paper |
|---|---|
| `table1_metrics.csv` | Table 1 (Sec. 5.3) |
| `table2_confusion_matrix.csv`, `fig4_confusion_matrix.png` | Table 2 / Fig. 4 (Sec. 5.5) |
| `fig3_metric_comparison.png` | Fig. 3 (Sec. 5.4), redrawn from the Table 1 values |
| `fig5_traffic_distribution.png` | Fig. 5 (Sec. 5.6), rebuilt from the Table 2 column totals |
| `fig5_slices_as_printed.csv` | the slice labels exactly as printed in Fig. 5 |
| `per_class_from_table2.csv` | per-class precision / recall / F1 derived from Table 2 |
| `table1_vs_table2_consistency.csv`, `consistency_check.json` | comparison of the two tables |

Fig. 2 (the t-SNE embedding) cannot be reconstructed from printed numbers, so it has no reference copy.

## Issues found while transcribing

1. **Table 1 and Table 2 disagree.** Table 2 contains 120,018 samples of which 105,330 sit on the
   diagonal — an accuracy of **0.8776**, against the **0.8291** printed in Table 1. Macro precision,
   recall and F1 implied by Table 2 (0.8438 / 0.8516 / 0.8471) likewise differ from Table 1
   (0.8675 / 0.8121 / 0.8177). See `table1_vs_table2_consistency.csv`.
2. **Table 2 was scored on training data.** Its row totals are exactly the UNSW-NB15
   *training-partition* counts plus 10,000 synthetic samples per attack class
   (56,000 / 12,264 + 10,000 / 10,491 + 10,000 / 1,133 + 10,000 / 130 + 10,000). No held-out test
   set was used, and synthetic samples were included in the evaluation. Both inflate the result.
3. **Table 1 has a duplicated column.** Three numeric columns are printed; the second and third
   are identical.
4. **Table 2 prints the Worms row twice.** The first copy (`430, 210, 180, 640, 863`) sums to
   2,323 rather than its stated total of 10,130 and repeats `863` from the Shellcode row above it.
   The second copy (`430, 210, 180, 640, 8670`) sums to 10,130 exactly and is the one used here.
5. **Fig. 5 does not sum to 100 %.** The printed slices are 60 / 20 / 18 / 12 / 6 (= 116 %) and the
   legend lists "Other Attacks" twice. The reference pie is therefore rebuilt from the Table 2
   predicted column totals: Normal 47.0 %, DoS 17.9 %, Reconnaissance 16.2 %, Shellcode 9.7 %,
   Worms 9.2 %.
6. **FAR is never reported.** Sec. 4.8 defines it and Sec. 5.3 refers to a "low false alarm rate",
   but no value appears. Table 2 implies a macro FAR of 0.0657.
