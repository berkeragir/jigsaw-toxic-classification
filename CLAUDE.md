# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle Jigsaw Toxic Comment Classification — a **multi-label** text classification task with 6 labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Comments can carry zero, one, or several of these simultaneously. The dataset is strongly imbalanced (e.g. `threat` is <0.5% positive), and most of the engineering here is about fighting that imbalance.

The codebase is a series of Jupyter notebooks (`toxic-classification-training.ipynb` → `…-7-shared-trunk.ipynb`) showing the iterative evolution of a single training pipeline. **The canonical current pipeline is `toxic-classification-training-7-shared-trunk.ipynb`** — when asked to "edit the training code" or "add feature X," start there unless the user specifies otherwise. Older numbered notebooks are kept for reference; do not port changes into them. v7's in-notebook markdown documents every architectural change from v6 (shared trunk, MAX_LENGTH=256 with dynamic padding, focal-only loss, FREEZE_LAYERS=2, sliding-window inference).

## Running the code

The notebook is run cell-by-cell in Jupyter. Data files (`train.csv`, `test.csv`, `test_labels.csv`) live in the repo root. No `requirements.txt` exists — the notebook relies on `torch` (with CUDA), `transformers` (DistilBERT), `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`.

Environment specifics that affect code:
- **Windows + GTX 1070 (8GB VRAM).** DataLoaders use `num_workers=0` on purpose — Windows spawn semantics break multiprocessing workers, so don't "fix" this to 2/4 without adding an `if __name__ == "__main__":` guard.
- **FP16 is the default** (`CONFIG['USE_FP16']=True`) via `torch.cuda.amp`. Keep autocast on the forward pass and `GradScaler` around `.backward()` / `.step()` when editing the training loop.
- **Batch size 48** is the known-good value for 8GB VRAM at `MAX_LENGTH=512`. Don't raise above 64 without a VRAM plan.

## The `CONFIG` dict is the control surface

Everything tunable lives in a single `CONFIG` dictionary at the top of the notebook. `MODEL_CONFIGURATION_GUIDE.md` documents the speed/accuracy tradeoffs in depth. The three architecture knobs drive most behavior downstream:

| Flag | Effect |
|---|---|
| `USE_EMBEDDINGS_ONLY` | Bypasses all 6 transformer layers, uses only the embedding layer's `[CLS]` vector. 3–4× faster, ~0.30–0.40 PR AUC. |
| `FREEZE_LAYERS` (0–6) | Freezes first N DistilBERT transformer layers via `param.requires_grad = False`. Frozen params still counted in `total_params` but excluded from the optimizer (`filter(lambda p: p.requires_grad, ...)`). |
| `HEAD_HIDDEN_SIZE` (256/384/768) | Hidden dim of the classification heads. |

When changing these, the notebook's own logging reports trainable vs. frozen params — trust that output over manual math.

## Model architecture

`MultiTaskDistilBert` in the notebook (and conceptually illustrated by `model_architecture_diagram.py` / `_professional.py`, which only render PNGs and are **not** part of training):

- Base: `DistilBertModel.from_pretrained("distilbert-base-uncased")` (768-dim hidden).
- **v7: shared trunk** — one `Dropout → Linear(768, H) → ReLU → Dropout` trunk feeding a single `Linear(H, 6)` classifier. This replaces v6's per-label MLP `ModuleList`; labels co-occur strongly, so they share representation. Don't split it back into per-label heads without a specific reason.
- Pooling: `[CLS]` token (`last_hidden_state[:, 0, :]`), or `embeddings[:, 0, :]` in embeddings-only mode.

Loss (v7): custom `FocalLoss(gamma=2.0, alpha=per_class_tensor)` — focal only, no `pos_weight`. `alpha` is a per-label tensor derived from positive rates and clamped to `[0.25, 0.75]`. v6 used focal + `pos_weight=neg/pos`, which double-corrected rare classes and destabilized training (near-zero PR AUC on `threat`/`identity_hate`); do not reintroduce `pos_weight` without a specific reason.

## Data caching contract

`preprocessed_data/` holds pickled tokenized tensors:
- **v7** (canonical): `train_v7_mlen{N}_full.pkl` / `val_v7_mlen{N}_full.pkl` — variable-length tokenization (no padding); collate_fn pads per batch. Cache key now includes `MAX_LENGTH`, so changing it invalidates properly.
- **v6 (legacy)**: `train_full.pkl` / `val_full.pkl`, `train_tokenized.pkl` / `val_tokenized.pkl` — fixed-512-pad tokenization. Not compatible with v7's dataloader; leave them be or delete.
- Subset runs produce `…_subset{N}.pkl`.

If you change the tokenizer or `SEED`, delete the relevant `.pkl` files — those dimensions are still not in the cache key. Files are ~600MB each; don't commit them.

## Evaluating and serving

- `best_model.pt` is a full checkpoint (`model_state_dict`, optimizer, scheduler, `val_pr_auc`, `pr_aucs`, `roc_aucs`, `config`). Loading it requires reinstantiating `MultiTaskDistilBert` with **the same config** that was saved — always read `checkpoint['config']` first rather than assuming current `CONFIG` matches.
- Primary metric is **micro-averaged PR AUC** (not ROC AUC) because of the class imbalance. The validation loop also reports per-label PR AUC and ROC AUC.
- Test set filtering: `test_labels.csv` uses `-1` to mark excluded rows (Kaggle's original hidden split). Always filter with `(test_labels_df[labels_list] != -1).all(axis=1)` before scoring — scoring on `-1` labels will produce garbage metrics.

## Things that are easy to get wrong

- **`example-classification.ipynb`** is an unrelated pedagogical LSTM toy (custom tokenizer, `nn.LSTM`, single binary label). It is **not** the training code and does not share architecture with the real pipeline. Do not edit it when asked about "the model."
- The `toxic-classification-training-5 copy.ipynb` file is a literal duplicate of `…-5.ipynb` — ignore it.
- When adding a new config flag, update both the `CONFIG` dict AND the printed configuration banner so the notebook's self-report stays accurate (the project relies on that printout for experiment tracking, since there's no MLflow/W&B integration).
