# Enhanced TMG-GAN IDS

Implementation of **"Enhanced TMG-GAN: Generative Adversarial Networks Based Imbalanced Learning
with Feature Attention and Focal Loss for Network Intrusion Detection"**
(Srinidhi S, Nikhitaa M, Anika Rathina, Kavya Sri V — Dept. of CSE, College of Engineering Guindy, Anna University).

The system tackles class imbalance in network intrusion detection at three levels:

| Level | Technique | Code |
|---|---|---|
| Data | TMG-GAN — multi-generator GAN with cosine-similarity loss and classifier-filtered sampling (base paper: Ding et al., IEEE TIFS 2024) | `src/tmg_gan.py`, `src/generate.py` |
| Feature | **Feature Attention** — `α = softmax(Wx + b)`, `X' = α ⊙ X` | `src/attention.py` |
| Loss | **Focal Loss** — `FL(p_t) = −α (1 − p_t)^γ log(p_t)` | `src/focal_loss.py` |

## Architecture

```
TRAINING                                                                          
UNSW-NB15 ─► Preprocessing ─► TMG-GAN training ─► Balanced dataset ─► Feature   ─► Focal-loss
             clean /          G1..GK, D, C, F      D' = D ∪ S          Attention     DNN classifier
             min-max /        + cosine sim. loss   (Algorithm 2)       X' = α ⊙ X    (Algorithm 3)
             one-hot          (Algorithm 1)                                               │
                                                                                          ▼
DEPLOYMENT                                                                        trained model M
live flow ─► Preprocessing ─► Feature Attention ─► Classifier ─► Decision ─► Normal → ALLOW
                                                   (Algorithm 4)             Attack → BLOCK
```

## Repository layout

```
├── main.py                    full pipeline: train GAN → balance → train both classifiers → evaluate
├── configs/
│   ├── default.yaml           paper setting: Normal, DoS, Reconnaissance, Shellcode, Worms (+10 000 synthetic each)
│   ├── all_classes.yaml       all 10 UNSW-NB15 classes
│   └── binary.yaml            y ∈ {0 = Normal, 1 = Attack}
├── src/
│   ├── preprocessing.py       Sec. 3.5.2 / 4.2   cleaning, min-max, one-hot, stratified split
│   ├── tmg_gan.py             Sec. 3.5.3 / 4.3   generators, D/C/F network, losses, Algorithm 1
│   ├── generate.py            Sec. 3.5.4         Algorithm 2, balanced dataset formation
│   ├── attention.py           Sec. 3.5.5 / 4.4   feature attention (proposed)
│   ├── focal_loss.py          Sec. 3.5.6 / 4.5   focal loss (proposed)
│   ├── classifier.py          Sec. 4.5 / 4.6     DNN classifier + training loop
│   ├── inference.py           Sec. 3.5.7 / 4.7   Algorithm 4 + allow / block decision
│   ├── evaluate.py            Sec. 4.8           Accuracy, Precision, Recall, F1, FAR, Tables 1–2
│   └── visualize.py           Sec. 5             Figures 2–5
├── scripts/make_demo_data.py  placeholder data for smoke tests (not real traffic)
├── tests/test_smoke.py
└── data/README.md             how to get UNSW-NB15
```

## Setup

```bash
git clone <your-repo-url> && cd enhanced-tmg-gan-ids
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download UNSW-NB15 as described in [`data/README.md`](data/README.md).

## Usage

```bash
python main.py                                   # paper setting
python main.py --config configs/all_classes.yaml # all 10 classes
python main.py --config configs/binary.yaml      # normal vs attack
python main.py --gan-epochs 5000 --clf-epochs 50 --seed 7
python main.py --demo                            # 10-second smoke test on placeholder data
pytest -q                                        # unit + end-to-end tests
```

Classify new flows with a trained model (Algorithm 4):

```bash
python -m src.inference --model outputs/model --input my_flows.csv --output predictions.csv
```

```python
from src.inference import IntrusionDetector
ids = IntrusionDetector("outputs/model")
ids.predict(flows_df)        # → prediction, confidence, action (ALLOW / BLOCK)
```

## Outputs

Everything is written to `outputs/`:

| File | Paper item |
|---|---|
| `table1_metrics.csv` | Table 1 — Focal Loss + Attention vs Cross-Entropy (Accuracy, Precision, Recall, F1, FAR) |
| `table2_confusion_matrix.csv`, `fig4_confusion_matrix.png` | Table 2 / Fig. 4 — confusion matrix of the proposed model |
| `fig2_feature_embedding.png` | Fig. 2 — t-SNE of attention-weighted features; legend `c.s` = class `c`, `s=0` real / `s=1` synthetic |
| `fig3_metric_comparison.png` | Fig. 3 — `Focal+Attn` vs `CE_NoAttn` |
| `fig5_traffic_distribution.png` | Fig. 5 — distribution of detected traffic categories |
| `per_class_*.csv`, `imbalance_before/after.csv`, `training_curves.png`, `attention_feature_importance.png`, `metrics.json` | extra diagnostics |
| `model/` | trained classifier, baseline, TMG-GAN, fitted preprocessor |

## Experimental protocol

- The **baseline** (`CE_NoAttn`) and the **proposed model** (`Focal+Attn`) are trained on the *same*
  TMG-GAN-balanced data with the same network, optimiser, seed and epochs. Only attention and the loss differ,
  so Table 1 / Fig. 3 isolate the contribution of the two proposed components.
- All metrics are computed on a **held-out split of real records only**. Synthetic samples are used for
  training and never for evaluation. The preprocessor is fitted on the training split only.
- Precision / Recall / F1 are macro-averaged (`eval.average`). FAR = FP / (FP + TN) on Normal-vs-Attack.
- Runs are seeded (`seed: 42`). Numbers are reproducible on the same machine and library versions;
  expect small differences across hardware (CPU vs GPU) and PyTorch versions. Report the values from
  your own `outputs/metrics.json`, ideally as mean ± std over several `--seed` values.

## Implementation notes

Choices the paper leaves open, all switchable in `configs/default.yaml`:

- One generator per class; D and C are two heads on the shared feature extractor F.
- `L_cos` has the paper's intra-class term `1 − cos(F(x), F(x̃))` plus an inter-class term that pushes
  generators of different classes apart ("reduce inter-class overlap"); `gan.inter_class_weight: 0` disables it.
  The generator also receives the classification loss used in the original TMG-GAN (`gan.lambda_cls`).
- Generated one-hot blocks are snapped to valid one-hot vectors before the classifier filter of Algorithm 2.
- `α` sums to 1, so `α ⊙ X` shrinks inputs by ≈ 1/n. By default the weights are multiplied by n
  (`classifier.attention_rescale`); relative importances are unchanged.
- Optimiser is Adam by default; `classifier.optimizer: sgd` gives the plain update `θ ← θ − η∇L` of Sec. 4.6.
- `proto` has 130+ values; one-hot keeps the 10 most frequent per column (`data.max_categories`).

## References

1. H. Ding et al., "TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection," *IEEE TIFS*, 2024.
2. T.-Y. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017.
3. N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems," *MilCIS*, 2015.
