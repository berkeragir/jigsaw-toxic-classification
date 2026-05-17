# Model card: jigsaw-toxic-classification v7

## Model details

- **Model type**: Fine-tuned multi-label text classifier.
- **Base model**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (66M params, 6 transformer layers).
- **Task head**: shared trunk (`Linear(768 → 384) → ReLU → Dropout`) → single `Linear(384 → 6)` classifier.
- **Total parameters**: 66.7M (52.5M trainable; lower 2 DistilBERT layers frozen).
- **Output**: independent sigmoid probability per label across 6 toxicity categories.
- **Languages**: English only.
- **Version**: v7 (shared trunk + focal-only loss + length bucketing).
- **License**: MIT (model weights and code).

## Intended use

### In scope

- Research and educational use cases: studying multi-label classification, focal loss behavior, threshold calibration on imbalanced data.
- A reference implementation for end-to-end fine-tuning of a small transformer on a Kaggle competition dataset.
- Portfolio / interview discussion artifact.

### Out of scope

- **Real-world content moderation.** This model was trained on Wikipedia talk-page comments from a single 2017–2018 corpus. It has not been audited for fairness, demographic bias, or deployment-readiness. Do not use it to make moderation decisions on user content.
- **Languages other than English.** No multilingual capability.
- **Legal, employment, safety, or other high-stakes decisions about individuals.**

## Training data

- **Source**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Conversation AI / Jigsaw, 2017–2018).
- **Domain**: Wikipedia talk-page comments.
- **Size**: ~159K labeled comments (90/10 train/val split → ~143K train / ~16K val).
- **Labels**: 6 binary labels — `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Multi-label (a comment may carry zero or several).

### Class distribution (train)

| Label | Positive rate |
|---|---:|
| toxic         | 9.59% |
| severe_toxic  | 1.01% |
| obscene       | 5.31% |
| threat        | 0.30% |
| insult        | 4.92% |
| identity_hate | 0.88% |

Strong imbalance — `threat` has roughly 1 positive per 330 comments.

## Evaluation

Evaluated on the official Kaggle test set after filtering rows where labels were marked `-1` (Kaggle's hidden split). Final test set: ~64K rows.

### Aggregate

| Metric | Value |
|---|---:|
| Micro-averaged PR AUC | 0.7575 |
| Exact-match accuracy (all 6 labels correct) | 0.8337 |
| Macro F1 @ threshold 0.5 | 0.5689 |
| Macro F1 with per-class threshold tuning | 0.6118 |

### Per-label

| Label | Test positives | PR AUC | ROC AUC | F1 (tuned) | Threshold |
|---|---:|---:|---:|---:|---:|
| toxic         | 6,090 | 0.806 | 0.973 | 0.680 | 0.628 |
| severe_toxic  |   367 | 0.344 | 0.990 | 0.411 | 0.553 |
| obscene       | 3,691 | 0.805 | 0.982 | 0.718 | 0.687 |
| threat        |   211 | 0.603 | 0.997 | 0.556 | 0.500 (floored) |
| insult        | 3,427 | 0.792 | 0.981 | 0.704 | 0.611 |
| identity_hate |   712 | 0.678 | 0.992 | 0.601 | 0.597 |

`threat` was floored at 0.5 (val positives below 100 → tuning would be noise-driven).

## Limitations and known failure modes

### Rare-label PR AUC is bounded by prevalence, not ranking

`severe_toxic` and `identity_hate` have ROC AUC ≥0.99 but PR AUC of 0.34 and 0.68 respectively. The model **ranks** rare positives near-perfectly, but precision at any practical recall is geometrically limited by the absolute scarcity of positives in the test set. This is a structural property of imbalanced data, not a model deficiency that can be fixed by more training on the same data.

### `severe_toxic` is structurally nested under `toxic`

Almost every `severe_toxic=1` sample is also `toxic=1`. The boundary is "toxic vs very toxic," which is a subjective ordinal cut. See `docs/EXPERIMENTS.md` for an experiment that exploits this nesting for a small lift; the takeaway is that the data, not the architecture, is the bottleneck.

### Bias inherited from the base model

`distilbert-base-uncased` is trained on web text and has documented sociolinguistic biases (gender, race, religion). Those biases are not removed by fine-tuning. The Jigsaw 2018 dataset has its own known biases — the [2019 follow-up competition](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) was created specifically to address them. This model **has not** been audited or trained against unintended-bias metrics.

### Distribution shift

Trained on Wikipedia talk-page comments from a specific time window. Performance will degrade on:
- Other platforms (Twitter, Reddit, YouTube comments) with different conventions.
- Other time periods (slang and adversarial patterns evolve).
- Out-of-distribution languages or code-switched text.

### Train→test gap

Validation micro PR AUC reached 0.877; test micro PR AUC is 0.758 — a ~0.12 gap. This is partly because the Kaggle test set was constructed to remove rows correlated with public leaderboard scores. The gap is informative about how much the val score over-estimates real-world performance.

## Ethical considerations

- Automated toxicity classification is a **support tool**, not a replacement for human review. False positives silence legitimate speech; false negatives let harmful content through. Either failure mode has real cost.
- The dataset's labels were assigned by crowdworkers and reflect their interpretations of toxicity. "Toxic" is not a single objective property; this model has internalized whatever working definition the labelers used.
- Identity-related labels (`identity_hate`) are particularly sensitive — see the 2019 follow-up competition for the harms that arise when toxicity models systematically over-flag mentions of marginalized identities.

If this work were a production system, the next steps would include: bias audit by identity group, demographic-parity metrics, human-in-the-loop review on flagged content, and ongoing distribution monitoring.

## How to use

```python
from src import MultiLabelDistilBert
import torch
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = MultiLabelDistilBert(num_labels=6, freeze_layers=2, head_hidden_size=384, verbose=False)
ckpt = torch.load("best_model_v7.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

inputs = tokenizer("you are an idiot", return_tensors="pt", truncation=True, max_length=192)
with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    probs = torch.sigmoid(logits).squeeze().tolist()

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label, prob in zip(LABELS, probs):
    print(f"  {label:14s} {prob:.3f}")
```

Apply per-class thresholds from `artifacts/tuned_thresholds_v7.json` for binary decisions.

## Contact

Issues and questions: open a GitHub issue on the repository.
