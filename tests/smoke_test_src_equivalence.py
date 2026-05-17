"""Smoke test: verify src/ produces identical predictions to the original v7 notebook.

The strongest available check is replaying the deterministic val_loader
(LengthBucketedBatchSampler with shuffle=False) using the src/ module
pieces, loading best_model_v7.pt, and comparing the first N batches'
predictions against the cached val_inference_v7.npz from the notebook run.

Same architecture + same weights + same input order + eval mode + same
fp16 setting -> predictions should match to ~1e-3 tolerance.

Run from the repo root:
    python tests/smoke_test_src_equivalence.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import (  # noqa: E402
    MultiLabelDistilBert,
    ToxicCommentsDataset,
    LengthBucketedBatchSampler,
    dynamic_pad_collate,
    LABELS,
)

CHECKPOINT_PATH = REPO_ROOT / "best_model_v7.pt"
VAL_CACHE = REPO_ROOT / "preprocessed_data" / "val_v7_mlen192_full.pkl"
VAL_INFERENCE_NPZ = REPO_ROOT / "val_inference_v7.npz"

N_BATCHES_TO_COMPARE = 4  # 4 batches of 128 = 512 samples (~3s on GTX 1070)
TOLERANCE = 1e-3  # fp16 inference has ~1e-4 noise; 1e-3 is generous but safe


def main():
    print("=" * 70)
    print("v7 src/ equivalence smoke test")
    print("=" * 70)

    # ---- Tier 1: checkpoint loads, config matches src/ model ----
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing {CHECKPOINT_PATH}")
    print(f"\n[1/4] Loading checkpoint: {CHECKPOINT_PATH.name}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"      Checkpoint config: FREEZE_LAYERS={cfg.get('FREEZE_LAYERS')}, "
          f"HEAD_HIDDEN_SIZE={cfg.get('HEAD_HIDDEN_SIZE')}, "
          f"MAX_LENGTH={cfg.get('MAX_LENGTH')}, USE_FP16={cfg.get('USE_FP16')}")

    print(f"\n[2/4] Instantiating MultiLabelDistilBert from src/")
    model = MultiLabelDistilBert(
        num_labels=6,
        use_embeddings_only=cfg.get("USE_EMBEDDINGS_ONLY", False),
        freeze_layers=cfg.get("FREEZE_LAYERS", 2),
        head_hidden_size=cfg.get("HEAD_HIDDEN_SIZE", 384),
        verbose=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"State dict mismatch: missing={missing} unexpected={unexpected}"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      State dict loaded cleanly. Total params: {total_params:,}")
    assert total_params == 66_660_486, f"Expected 66,660,486 params, got {total_params}"

    # ---- Tier 2: replay val_loader and run forward ----
    if not VAL_CACHE.exists():
        print(f"\n[SKIP] {VAL_CACHE.name} not found — skipping equivalence check.")
        print("       (Tier 1 architectural verification passed.)")
        return 0
    if not VAL_INFERENCE_NPZ.exists():
        print(f"\n[SKIP] {VAL_INFERENCE_NPZ.name} not found — skipping equivalence check.")
        return 0

    print(f"\n[3/4] Loading val cache + cached predictions")
    with open(VAL_CACHE, "rb") as f:
        val_data = pickle.load(f)
    cached = np.load(VAL_INFERENCE_NPZ)
    val_probs_cached = cached["val_probs"]
    val_true_cached = cached["val_true"]

    val_dataset = ToxicCommentsDataset(
        val_data["input_ids"], val_data["attention_mask"], val_data["labels"]
    )
    val_lengths = [len(x) for x in val_data["input_ids"]]
    val_sampler = LengthBucketedBatchSampler(
        lengths=val_lengths,
        batch_size=cfg.get("BATCH_SIZE", 64) * 2,
        bucket_multiplier=cfg.get("BUCKET_MULTIPLIER", 50),
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dynamic_pad_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    use_fp16 = cfg.get("USE_FP16", True) and torch.cuda.is_available()
    print(f"      Device: {device}  |  fp16: {use_fp16}")

    print(f"\n[4/4] Comparing {N_BATCHES_TO_COMPARE} batches against cached predictions")
    cursor = 0
    max_diff = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= N_BATCHES_TO_COMPARE:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits.float()).cpu().numpy()

            b = probs.shape[0]
            cached_slice = val_probs_cached[cursor:cursor + b]
            diff = np.abs(probs - cached_slice).max()
            max_diff = max(max_diff, diff)
            cursor += b
            print(f"      batch {i}: n={b}  max|diff|={diff:.6f}")

    print()
    if max_diff < TOLERANCE:
        print(f"PASS: max|diff|={max_diff:.6f} < {TOLERANCE} over {cursor} samples")
        print("      src/ module reproduces v7 notebook predictions to fp16 precision.")
        return 0
    print(f"FAIL: max|diff|={max_diff:.6f} >= {TOLERANCE} over {cursor} samples")
    print("      src/ diverges from the notebook. Inspect model.py / data.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
