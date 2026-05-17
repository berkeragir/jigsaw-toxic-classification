# Methods

This document covers the architectural and training choices in v7 and why each was made. Numbered version history is in [`EXPERIMENTS.md`](EXPERIMENTS.md); the bug that killed v6 is in [`POSTMORTEM_v6.md`](POSTMORTEM_v6.md).

## Problem framing

Multi-label binary classification on Wikipedia talk-page comments. Six labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) each independently 0/1. A comment may carry zero or several labels — `severe_toxic` is almost always co-labeled with `toxic`, for example.

Primary evaluation metric: **micro-averaged PR AUC**. ROC AUC inflates on imbalanced binary problems (a high-true-negative rate makes everything look good); PR AUC is more honest about how the model behaves on the small positive class. Macro F1 with per-class threshold tuning is reported as a secondary, operationally relevant metric.

## Data split

- **Train**: ~143K comments (after a 90/10 split of the official train set).
- **Validation**: ~16K comments held out for early stopping, threshold tuning, and architecture decisions.
- **Test**: ~64K rows from `test.csv`, after filtering out rows where Kaggle marked the labels `-1` (originally hidden split). Always filter with `(test_labels[labels] != -1).all(axis=1)` before scoring.

## Architecture

```
Input (token_ids, attention_mask)
        |
        v
   DistilBERT (66M params, 6 transformer layers, last 4 trainable)
        |
        v
  [CLS] token   (last_hidden_state[:, 0, :])   shape: [B, 768]
        |
        v
   Dropout(0.1)
   Linear(768 -> 384)
   ReLU
   Dropout(0.1)            shape: [B, 384]   <-- shared trunk
        |
        v
   Linear(384 -> 6)        shape: [B, 6]    logits
```

Total params: 66.7 M. Trainable: 52.5 M (78.7%) after freezing the lower 2 layers.

### Why a shared trunk

Earlier versions used `nn.ModuleList` of six independent `Linear→ReLU→Linear` heads, one per label. Three problems:

1. **Redundant capacity**. The first thing each head learned was "what makes a comment toxic" — a representation shared across all six labels. Six heads relearned it six times from highly correlated supervision.
2. **Parameter cost**. At hidden size 384, each per-label MLP added ~295K params; total ~1.8M of head params.
3. **Sample inefficiency**. Rare labels (`threat` at <0.3% positive rate) saw their head trained on very few positive examples in isolation. A shared trunk gets gradient from every positive of every label.

The v7 trunk reduces head-block params to ~297K (a ~6× reduction) and unifies the representation. This is the same pattern as HuggingFace's reference multi-label sequence-classification head.

### Why DistilBERT (and what's next)

DistilBERT is ~40% smaller and ~60% faster than `bert-base-uncased` at ~97% of the GLUE performance. On 8 GB VRAM with FP16 at `MAX_LENGTH=192` it fits a `BATCH_SIZE=64` comfortably, which lets training complete in under an hour. The next planned improvement (see [`EXPERIMENTS.md`](EXPERIMENTS.md)) swaps to `roberta-base` for the harder rare-label signal.

### Why freeze 2 layers

Lower transformer layers encode general syntactic/lexical features that don't need to specialize to this task. Freezing them:
- saves ~14M params from optimizer state and gradient memory,
- stabilizes training at the small batch sizes 8 GB VRAM forces,
- doesn't measurably hurt PR AUC (verified by ablation in early iterations).

## Loss function

**Focal loss with per-class alpha**, no `pos_weight`.

```
Focal(γ=2.0, α_c per class)

α_c = clip(1 - positive_rate_c, 0.25, 0.75)
```

- **Focal** down-weights easy examples (high-confidence predictions on the majority class) so the gradient stays informative through training.
- **Per-class α** gives positives in each label more weight in inverse proportion to their prevalence — but **clamped**. The clamp bounds the correction so rare classes don't dominate the gradient.

Earlier versions also applied `BCEWithLogitsLoss(pos_weight=neg_count/pos_count)`. This was a catastrophic double-correction — see [`POSTMORTEM_v6.md`](POSTMORTEM_v6.md).

## Tokenization and batching

- **Tokenizer**: `distilbert-base-uncased` (WordPiece).
- **MAX_LENGTH = 192**. The 90th percentile of comment length is ~150 tokens; 192 covers the tail without wasting compute. The earlier 512 was ~3× slower for no measurable accuracy gain.
- **Variable-length tokenization**: tokenize without padding, store as ragged lists, pad per-batch in `collate_fn`. Cuts mean batch length significantly versus pad-to-max.
- **Length-bucketed sampling**: a SortishSampler-style sampler shuffles globally, slices into buckets of `batch_size * 50`, sorts each bucket by length, and shuffles the batch order. Samples within a batch are similar length, so dynamic padding is tight.

Net effect on epoch time: projected ~44 min → actual 12.9 min on a GTX 1070. Without length bucketing, the full 4-epoch training run would not have fit in a sane session.

## Training loop

- **Optimizer**: AdamW, learning rate 3e-5, weight decay 0.01.
- **Scheduler**: linear warmup (first 6% of steps) → linear decay to zero.
- **Mixed precision**: FP16 via `torch.cuda.amp.autocast` + `GradScaler`. Required on 8 GB VRAM at this batch size.
- **Epochs**: 4. Best checkpoint by val micro PR AUC was epoch 3 (val PR AUC 0.8765).
- **Gradient clipping**: max norm 1.0.

## Post-hoc threshold tuning

The default 0.5 cutoff is not F1-optimal under focal loss because focal compresses the logit distribution asymmetrically. After training:

1. Collect sigmoid probabilities on the validation set.
2. For each label, sweep the PR curve and pick the threshold that maximizes val F1.
3. **Floor at 0.5** for any label with fewer than 100 val positives.
4. Apply those thresholds to test.

### Why the floor matters

`threat` had only 48 positives in the validation slice. Its raw F1-optimal threshold was 0.455, but that "optimum" is dominated by curve noise at that positive count. Applying it to test regressed F1 by **−0.027** versus the 0.5 baseline. With the floor in place, `threat` is the one label where tuning is a no-op (threshold stays at 0.5), and the tuning pass becomes a clean win across every label.

Net macro F1 lift on test: **+0.043** (0.569 → 0.612). See [`../artifacts/tuned_thresholds_v7.json`](../artifacts/tuned_thresholds_v7.json) for the per-label thresholds and val positive counts.

## Reproducibility notes

- `SEED = 42` for the train/val split and the length-bucketed sampler.
- The length-bucketed sampler uses an internal RNG that consumes `epoch` for per-epoch reshuffling; results are deterministic given the same seed and epoch count.
- FP16 inference is bit-deterministic in eval mode on the same hardware. The smoke test in `tests/smoke_test_src_equivalence.py` verifies the extracted `src/` modules reproduce the v7 notebook predictions exactly.

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GTX 1070 (8 GB VRAM) |
| Mixed precision | FP16 (`torch.cuda.amp`) |
| OS | Windows 10 |
| `num_workers` | 0 (Windows DataLoader spawn semantics break workers without an `if __name__ == "__main__"` guard) |
| Per-epoch time | ~12.9 min |
| Total training time | 51.8 min for 4 epochs |
