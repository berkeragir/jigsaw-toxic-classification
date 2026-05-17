# Postmortem: the v6 double-correction bug

An earlier version of this pipeline (v6) shipped a class-imbalance correction that quietly collapsed the rare-label scores to near-random while *appearing* to train normally. This is the story of how it broke, how long it took to notice, and what the fix was.

The broken v6 notebook is not in this repo. It's preserved internally for reference, but the public artifact is the version where this was caught and corrected.

## The bug

v6 stacked **two** independent class-imbalance corrections on top of each other:

1. **Focal loss** with γ=2.0. Focal loss is itself a soft form of imbalance correction — it down-weights easy examples and lets gradient flow through hard / minority cases.
2. **`BCEWithLogitsLoss(pos_weight=neg_count/pos_count)`**. This separately scales the loss on positive examples by the negative-to-positive ratio. For `threat` (~0.3% positive rate), that ratio is roughly **350×**.

Either one of these on its own is a defensible imbalance fix. Both at once is a multiplicative correction far beyond what the optimizer can stably absorb.

## What happened in practice

The combined gradient on rare-class positives became so large that the optimizer pushed those classes' decision boundaries hard in the direction of "everything is positive." This drove logits far from zero on rare labels, the sigmoid saturated, and the next gradient step had nothing to correct from. Training "stabilized" at a bad equilibrium.

The collapse was severe on the rare labels:

| Label | Positive rate | v6 val PR AUC | v7 val PR AUC |
|---|---:|---:|---:|
| toxic         | 9.6%  | ~0.85 | 0.91 |
| obscene       | 5.3%  | ~0.78 | 0.91 |
| insult        | 4.9%  | ~0.70 | 0.82 |
| severe_toxic  | 1.0%  | ~0.05 | 0.45 |
| identity_hate | 0.8%  | ~0.03 | 0.55 |
| threat        | 0.3%  | ~0.03 | 0.51 |

Head classes were merely degraded; the rare classes were essentially random.

## Why it wasn't immediately obvious

This is the part worth dwelling on. The v6 notebook **ran to completion**. Loss curves looked like loss curves — monotonically descending, no spikes, no NaNs. The micro-averaged metrics looked plausible because the head classes dominate the micro-average. Aggregate validation PR AUC was around 0.62, which isn't far enough below a working baseline to scream "broken."

The only signal that something was wrong was in the **per-label breakdown** — and only if you looked at the rare labels and noticed that 0.03 PR AUC on `threat` and `identity_hate` is statistically indistinguishable from random guessing. The training output displayed the per-label metrics but didn't flag them.

The lesson: on imbalanced multi-label problems, **monitor per-class metrics, not just aggregates.** A micro-PR-AUC of 0.62 with two labels at 0.03 is a very different model than the same 0.62 with all labels at 0.62.

## The fix

Removed `pos_weight` entirely. Kept focal loss with a bounded per-class `alpha`:

```python
alpha_c = clip(1 - positive_rate_c, 0.25, 0.75)
```

The clamp is what keeps the rare-class weighting bounded so the optimizer can absorb it. Without the clamp, even alpha alone would have started to recreate the v6 instability for `threat`.

After this single change (with no other architectural difference), the rare-class scores recovered from ~0.03 to the 0.34–0.60 range. The full set of v7 changes is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## What this means for the rest of the work

- The framing "v6 → v7 was a 25× improvement in val PR AUC" is technically true and internally accurate (0.035 → 0.877). It is also misleading shorthand for "we found and removed a bug." The published narrative leads with what v7 *is*, not with the ratio against a broken baseline.
- The rare-label PR AUC numbers in v7 (especially `severe_toxic` at 0.344) sit in a different regime than the v6 numbers. They are not bounded above by "v6 was 0.05 so v7 is 7× better." They are bounded above by the structural difficulty of those labels — see the `severe_toxic` discussion in [`EXPERIMENTS.md`](EXPERIMENTS.md).
- The fix is small. The cost of *not* catching it would have been weeks of trying architectural changes against a baseline that was broken in the loss, not the architecture.
