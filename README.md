# Jigsaw Toxic Comment Classification

Multi-label text classification of Wikipedia talk-page comments across six toxicity categories — `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Comments can carry zero, one, or several labels at once, and the dataset is strongly imbalanced (the rarest label, `threat`, is <0.3% positive).

This repo is a portfolio artifact built end-to-end from EDA through training, evaluation, and threshold tuning on a single consumer GPU. It includes the negative result of a follow-up experiment — that's deliberate, not an oversight.

## Headline results

| Metric | Value |
|---|---|
| Test micro PR AUC | **0.7575** |
| Test exact-match accuracy (all 6 labels correct) | 0.8337 |
| Macro F1 @ threshold 0.5 | 0.5689 |
| **Macro F1 with per-class threshold tuning** | **0.6118** (+0.043) |
| Best validation micro PR AUC | 0.8765 |
| Total training time | 51.8 min (4 epochs) |

### Per-label test metrics

| Label | Test positives | PR AUC | ROC AUC | F1 (tuned) |
|---|---:|---:|---:|---:|
| toxic         | 6,090 | 0.806 | 0.973 | 0.680 |
| severe_toxic  |   367 | 0.344 | 0.990 | 0.411 |
| obscene       | 3,691 | 0.805 | 0.982 | 0.718 |
| threat        |   211 | 0.603 | 0.997 | 0.556 |
| insult        | 3,427 | 0.792 | 0.981 | 0.704 |
| identity_hate |   712 | 0.678 | 0.992 | 0.601 |

ROC AUC is ≥0.97 on every label — the model **ranks** positives correctly. The PR AUC drop on `severe_toxic` is the precision-recall area being geometrically starved by 0.57% prevalence on test, not a ranking failure. More on this in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## What's in this repo

- **`src/`** — importable modules: `model.py` (shared-trunk DistilBERT), `data.py` (variable-length tokenization + length-bucketed sampler), `losses.py` (focal loss), `evaluate.py` (per-label metrics + threshold tuning with rare-label floor).
- **`notebooks/`** — the canonical training pipeline (`01_training_v7.ipynb`) and the follow-up hierarchical experiment (`02_hierarchical_v8.ipynb`).
- **`docs/`** — methods, experiments narrative, and the v6 postmortem.
- **`artifacts/`** — model card, tuned thresholds, training history.

The trained checkpoint (`best_model_v7.pt`, ~270 MB) is attached to the [v0.1.0 GitHub Release](#) — the link is set once the release is created.

## Methods at a glance

- **Backbone**: `distilbert-base-uncased` (66M params, 6 transformer layers).
- **Head**: a single shared trunk (`Linear(768→384) → ReLU → Dropout`) feeding one 6-way linear classifier. Replaces the per-label MLPs of earlier versions; labels co-occur heavily (>70% of positive rows carry 2+ labels), so the trunk learns "what is toxic" once and the classifier specializes.
- **Loss**: per-class focal loss (γ=2, α derived from positive rates and clamped to [0.25, 0.75]). No `pos_weight` — see [`docs/POSTMORTEM_v6.md`](docs/POSTMORTEM_v6.md) for why.
- **Training**: 4 epochs, AdamW with linear warmup, FP16 via `torch.cuda.amp`, length-bucketed batching with dynamic padding.
- **Threshold tuning**: per-class F1-optimal threshold on validation, floored at 0.5 for any label with fewer than 100 val positives (a rare-label safety net — `threat` triggered it).

Full rationale and design decisions in [`docs/METHODS.md`](docs/METHODS.md).

## Hardware & constraints

This project was trained on a **GTX 1070 (8 GB VRAM)**, which shaped multiple design decisions: FP16 was non-optional, `BATCH_SIZE` capped at 64 at `MAX_LENGTH=192`, freezing the lower 2 DistilBERT layers to stabilize at small batch sizes, and length-bucketed sampling to drop epoch time from a projected 44 min to an actual 12.9 min. Full training run completed in 51.8 min.

These are real engineering constraints; the methods doc explains where each decision came from.

## Reproducing

```bash
git clone https://github.com/berkeragir/jigsaw-toxic-classification.git
cd jigsaw-toxic-classification
pip install -r requirements.txt
bash scripts/download_data.sh         # requires Kaggle CLI + accepted competition rules
jupyter notebook notebooks/01_training_v7.ipynb
```

Or use `src/` directly:

```python
from src import MultiLabelDistilBert, FocalLoss, compute_focal_alpha

model = MultiLabelDistilBert(num_labels=6, freeze_layers=2, head_hidden_size=384)
alpha = compute_focal_alpha(train_labels)  # per-class alpha tensor
criterion = FocalLoss(gamma=2.0, alpha=alpha)
```

## What didn't work

A follow-up experiment (`v8.1`) fine-tuned the last 2 DistilBERT layers end-to-end on the toxic-positive subset to lift `severe_toxic` PR AUC. It **lost** to the frozen-backbone linear probe (0.353 vs 0.362 test PR AUC) — overfitting on 1,446 severe positives. The experiment, its three independent overfitting signals, and the bias-variance interpretation are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). One data point, not a general rule, but worth publishing.

## Future directions

1. **Backbone swap** to `roberta-base` or `deberta-v3-base` — likely the path past the `severe_toxic` ceiling and the next-biggest expected lift on `threat` and `identity_hate`.
2. **Jigsaw 2019 follow-up**: the [Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) challenge shifts the objective from raw toxicity detection to identity-bias mitigation. Same data corpus, different optimization target — a natural extension of this work.
3. **Data augmentation** on `threat` positives, which the hierarchical trick can't help (it isn't structurally nested under another label).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Data from the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Conversation AI / Jigsaw, 2017–2018).
