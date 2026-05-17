"""Variable-length tokenization, dataset, dynamic padding, and length-bucketed sampling.

Pre-v7, the pipeline padded every sample to MAX_LENGTH=512. Most comments
are 50-80 tokens long, so >90% of the compute was on padding tokens. v7:
- Tokenize without padding (returns lists of varying lengths).
- Pad each batch to its own max length in collate_fn.
- Sort similar-length samples into the same batch via a SortishSampler.

Net effect: epoch time projected ~44 min -> actual 12.9 min on 8GB VRAM.
"""

import time
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


def tokenize_texts_varlen(texts, labels, tokenizer, max_length: int) -> Dict:
    """Tokenize without padding. Returns per-sample lists of ids/masks.

    The DataLoader's collate_fn pads each batch independently. Storing as
    object arrays of Python lists keeps memory low and lets workers read
    directly.

    Args:
        texts: 1D array-like of strings.
        labels: shape [N, num_labels].
        tokenizer: HuggingFace tokenizer (DistilBertTokenizer).
        max_length: truncation cap (v7 uses 192; 90th percentile is ~150).
    """
    print(f"  Tokenizing {len(texts):,} texts (variable-length, max {max_length})...")
    t0 = time.time()
    encoded = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} samples/sec)")

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels.astype(np.float32),
    }


class ToxicCommentsDataset(Dataset):
    """Returns raw variable-length ids/masks + labels; padding happens in collate."""

    def __init__(self, input_ids: List[List[int]], attention_masks: List[List[int]], labels: np.ndarray):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def dynamic_pad_collate(batch, pad_token_id: int = 0):
    """Pad each batch to its own max length.

    DistilBERT's pad token id is 0. Attention mask pads with 0 so padded
    positions are masked out.
    """
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LengthBucketedBatchSampler(Sampler):
    """Yields batches of indices grouped by similar sequence length.

    Algorithm ("SortishSampler" pattern):
      1. Shuffle all indices.
      2. Slice into buckets of size `batch_size * bucket_multiplier`.
      3. Sort each bucket by length.
      4. Split each sorted bucket into contiguous batches.
      5. Shuffle the global list of batches.

    Step 1 preserves shuffle randomness at the bucket level; steps 2-4
    ensure each batch pads to approximately its members' lengths.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        bucket_multiplier: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
        seed=None,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """Call between epochs to re-seed shuffling deterministically."""
        self._epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self._rng.integers(0, 2**31) + self._epoch)
            rng.shuffle(indices)

        batches = []
        for start in range(0, n, self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            order = np.argsort(self.lengths[bucket], kind="stable")
            bucket_sorted = bucket[order]
            for bstart in range(0, len(bucket_sorted), self.batch_size):
                batch = bucket_sorted[bstart:bstart + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng = np.random.default_rng(self._rng.integers(0, 2**31) + self._epoch + 1)
            rng.shuffle(batches)

        yield from batches

    def __len__(self):
        n = len(self.lengths)
        if self.drop_last:
            full_buckets, rem = divmod(n, self.bucket_size)
            full_batches = full_buckets * (self.bucket_size // self.batch_size)
            full_batches += rem // self.batch_size
            return full_batches
        return (n + self.batch_size - 1) // self.batch_size
