# Experiments

A roughly chronological account of what was tried, what worked, and what didn't. The complete picture matters more than the headline number — negative results stay in.

## Baseline — TF-IDF + Logistic Regression

Before reaching for a transformer, it's worth knowing what a strong classical baseline gets. The baseline is `scripts/baseline_tfidf_logreg.py`: word 1–2 grams with sublinear TF, `min_df=3`, `max_features=200,000`, plus a per-label logistic regression (one-vs-rest, `C=4.0`, liblinear). Same train/val split as v7 (`SEED=42`, `VAL_SPLIT=0.1`), same `-1`-row filtering on the test set, same metrics, same threshold-tuning rule.

| Metric | Baseline | v7 (DistilBERT) | Δ |
|---|---:|---:|---:|
| Test micro PR AUC | 0.722 | **0.758** | +0.036 |
| Macro F1 @ 0.5 | 0.518 | **0.569** | +0.051 |
| Macro F1 tuned | 0.535 | **0.612** | +0.077 |
| Val micro PR AUC (best) | 0.829 | **0.877** | +0.048 |
| Train + predict time | **0.98 min** (CPU) | 51.8 min (GTX 1070) | **53× slower** |

### Per-label test PR AUC

| Label | Baseline | v7 | Δ |
|---|---:|---:|---:|
| toxic | 0.760 | 0.806 | +0.046 |
| obscene | 0.780 | 0.805 | +0.025 |
| insult | 0.707 | 0.792 | **+0.085** |
| identity_hate | 0.490 | 0.678 | **+0.188** |
| threat | 0.465 | 0.603 | **+0.138** |
| severe_toxic | 0.304 | 0.344 | +0.040 |

### Reading the gap

The transformer wins on every label, but the gains are not uniform — they are **concentrated exactly where contextual representation should help**:

- **Surface-profanity labels** (`toxic`, `obscene`): the baseline lands within 0.025–0.046 PR AUC of the transformer. These labels are largely detectable from lexical content, which is what bag-of-ngrams encodes well. The transformer doesn't have much room to differentiate itself here.
- **Contextual labels** (`identity_hate` +0.188, `threat` +0.138, `insult` +0.085): the transformer pulls clearly ahead. A threat or a slur depends on how words combine and who they target, not which words are present. This is the textbook advantage of contextualized embeddings over bag-of-words.
- **`severe_toxic`**: both models struggle similarly (+0.040). Consistent with the structural argument in the v7 section — this label's PR AUC ceiling is set by 1% prevalence and ordinal-cut subjectivity, not by representational power.

### Why publish the baseline at all

Three reasons:

1. **It frames the transformer investment honestly.** Reporting v7 in isolation invites the reader to assume the alternative is 0; reporting against a 0.722 baseline shows what 50 GPU-minutes actually bought (+0.036 micro PR AUC, with most of that landing on the labels where it should).
2. **It identifies which labels deserve more model capacity.** If a backbone swap is going to help anywhere, it's on the labels where the current model already pulls clearly away from bag-of-ngrams — they're the ones whose ceiling is set by representational power, not by data thinness or label structure.
3. **It's a sanity check.** A transformer that does not measurably beat TF-IDF + LogReg on a text-classification task probably has a bug. Future iterations (v9, backbone swaps) re-run this comparison as a guardrail.

The full per-label JSON is in [`../artifacts/baseline_tfidf_logreg.json`](../artifacts/baseline_tfidf_logreg.json).

## v7 — shared trunk + length bucketing (the canonical pipeline)

Configuration in `notebooks/01_training_v7.ipynb`. Architecture and rationale in [`METHODS.md`](METHODS.md). Headline:

| Metric | Value |
|---|---|
| Validation micro PR AUC (best) | 0.8765 |
| Test micro PR AUC | 0.7575 |
| Test exact-match | 0.8337 |
| Macro F1 @ 0.5 | 0.5689 |
| Macro F1 with per-class threshold tuning | 0.6118 (+0.043) |
| Training time | 51.8 min (4 epochs, GTX 1070, FP16) |

The val→test gap of ~0.12 PR AUC is larger than ideal but consistent with the Kaggle test set being drawn from a different distribution than the public leaderboard — the original competition specifically hid rows that correlated most strongly with public scores.

### Per-label test metrics

| Label | Test pos | PR AUC | ROC AUC | F1 @ 0.5 | F1 tuned | Tuned thr |
|---|---:|---:|---:|---:|---:|---:|
| toxic         | 6,090 | 0.806 | 0.973 | 0.614 | **0.680** | 0.628 |
| severe_toxic  |   367 | 0.344 | 0.990 | 0.379 | **0.411** | 0.553 |
| obscene       | 3,691 | 0.805 | 0.982 | 0.639 | **0.718** | 0.687 |
| threat        |   211 | 0.603 | 0.997 | 0.556 | **0.556** | 0.500 (floored) |
| insult        | 3,427 | 0.792 | 0.981 | 0.645 | **0.704** | 0.611 |
| identity_hate |   712 | 0.678 | 0.992 | 0.580 | **0.601** | 0.597 |

Two clusters:

- **Head classes** (`toxic`, `obscene`, `insult`): PR AUC ≈ 0.80, clearly learned.
- **Rare/nested classes** (`severe_toxic`, `threat`, `identity_hate`): PR AUC between 0.34 and 0.68. ROC AUC is ≥0.99 on every one of these — the model **ranks** rare positives near-perfectly; the PR AUC is being geometrically starved by tiny absolute positive counts, not by a ranking failure.

### What drove the v7 result

The biggest single fix was removing a double class-imbalance correction that broke the earlier version (see [`POSTMORTEM_v6.md`](POSTMORTEM_v6.md)). The remaining changes, all landed together in v7:

| Change | Reason | Effect |
|---|---|---|
| Focal-only loss (removed `pos_weight`) | The earlier pipeline used focal γ=2 *and* `pos_weight = neg/pos`. Double-correcting imbalance pushed rare-class logits so far negative their PR AUC collapsed to ~0.03. | Most significant single fix — all rare classes became learnable. |
| Shared trunk (replaced 6 per-label MLPs) | Labels co-occur heavily (>70% of positive rows carry 2+ labels). Per-label heads relearned "what is toxic" six times from correlated signal. | Cleaner representation, ~6× fewer head params, faster train. |
| `FREEZE_LAYERS=2` | Lower DistilBERT layers encode general linguistic features that don't need task-specific tuning. Freezing them stabilizes small-batch training and saves VRAM. | No measurable quality loss, faster per-step. |
| `MAX_LENGTH=192` (was 512) | 90th-percentile comment length is ~150 tokens. 512 was wasted compute on padding. | ~3× faster inference. |
| Length-bucketed sampling + dynamic padding | Sorting similar-length samples into the same batch drops mean padded length from `MAX_LENGTH` to actual content length per batch. | Epoch time 44 min projected → 12.9 min actual. Enabled 4 full epochs in one session. |

### Threshold tuning with a rare-label floor

Per-class F1-optimal thresholds on validation, **floored at 0.5** whenever a label has fewer than 100 val positives. Without the floor, `threat` (48 val positives) drew a val-optimal threshold of 0.455 — and applying it to test regressed F1 by −0.027. The PR curve at that positive count is dominated by noise, and the "optimum" was an artifact.

Rule worth keeping for any imbalanced multi-label problem: always report the val positive count next to a tuned threshold. Floor it (or use CV-stabilized tuning) when positives < 100.

Macro F1 lifted from 0.569 → 0.612 (+0.043) overall. The artifact (per-label thresholds, val positive counts, full per-label test metrics) is in [`../artifacts/tuned_thresholds_v7.json`](../artifacts/tuned_thresholds_v7.json).

## v8 — hierarchical head for `severe_toxic` (small positive lift)

`notebooks/02_hierarchical_v8.ipynb`. ROC AUC ≥0.99 on `severe_toxic` says the model's ranking is near-perfect; the PR AUC of 0.344 is starved by 0.57% test prevalence. The label is also **structurally nested**: nearly every `severe_toxic=1` sample is also `toxic=1`. The boundary is "toxic vs very toxic" — a subjective ordinal cut, not an independent category.

Hypothesis: model `severe_toxic` as `P(severe | toxic=1) · P(toxic)` and train a small head only on toxic-positive rows where the distinction actually has signal.

| Variant | Test PR AUC | Test ROC AUC |
|---|---:|---:|
| v7 baseline (independent head) | 0.3443 | 0.9899 |
| **v8 frozen conditional** (linear probe on v7 trunk, toxic-positive rows only) | **0.3624** | 0.9905 |
| v8 combined: P(sev\|tox) · P(tox) | 0.3615 | 0.9904 |

A logistic regression on the frozen v7 trunk, trained only on the 13.8K toxic-positive rows, lifts severe_toxic test PR AUC by **+0.018** (+5% relative). The combined score barely changes versus the conditional-only score because v7's `P(toxic)` is already near-certain on severe rows; multiplying by it doesn't add ranking information.

## v8.1 — unfrozen fine-tune for the same head (negative result)

Same target as v8: lift `severe_toxic`. But this time unfreeze the last 2 DistilBERT layers and fine-tune end-to-end on the toxic-positive subset (same data as v8).

| Variant | Test PR AUC | Test ROC AUC |
|---|---:|---:|
| v8 frozen conditional (linear probe, 385 params) | **0.3624** | 0.9905 |
| v8.1 unfrozen (last 2 layers + trunk + new head, 14.5M params) | 0.3534 | 0.9891 |
| v8.1 combined: P(sev\|tox)_v8.1 · P(tox) | 0.3528 | 0.9896 |

**v8.1 lost to the frozen probe.** Three independent signals consistent with overfitting:

1. Within-toxic validation PR AUC peaked at epoch 2 (0.548) and **declined** at epoch 3 (0.519) while train loss kept falling — the classic memorization curve.
2. v8.1 trained 14.5M params on 13,765 rows containing only 1,446 severe positives. v8's linear probe has 385 params on the same data — roughly **37,000× fewer**.
3. Even on its own within-toxic validation slice, v8 frozen reached 0.565 PR AUC while v8.1 topped out at 0.548.

### Interpretation

On this particular small rare-class subset, freezing the backbone acted as regularization that more than compensated for the lost representational flexibility. The linear probe couldn't memorize the training data; the unfrozen variant could.

This is a **single experiment**, not a general rule. With more training data — for example, conditional-loss masking over all 143K rows rather than just the 13.8K toxic-positive subset, or text augmentation on severe positives — the unfrozen variant could plausibly win. The right framing here is: on this dataset at this scale, the linear probe is the better choice, and the result is worth reporting because it's not the answer the standard playbook would suggest.

Real further gains on `severe_toxic` likely require a **larger backbone** (`roberta-base`, `deberta-v3-base`) rather than more aggressive fine-tuning of the existing one.

## Open directions

Ranked by expected lift per hour of work:

1. **Backbone swap to `roberta-base` or `deberta-v3-base`.** Expected to help most on the subtler rare labels (`threat`, `identity_hate`) where DistilBERT's distilled representation loses fidelity. Probably the only realistic way past the 0.36 PR AUC ceiling that v8 hit on `severe_toxic`. Cost: ~60 min training plus VRAM pressure on 8 GB (likely `BATCH_SIZE=24–32`).
2. **Apply v7.1 tuned thresholds to the v8 combined score** for a unified per-row submission. Trivial follow-on.
3. **Revisit v8.1 with more data**: either conditional-loss masking on all 143K rows, or text augmentation on severe positives. Lower priority until #1 is in.
4. **Rethink `threat`.** It isn't nested under any single other label, so the hierarchical trick won't help. Likely needs the backbone swap or explicit data augmentation on positives.
5. **Jigsaw 2019 follow-up** ([Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)): same corpus, shifted objective (identity-bias mitigation). A natural extension worth its own repo.
