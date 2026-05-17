"""Jigsaw Toxic Comment Classification — v7 shared-trunk pipeline.

Importable equivalents of the model, data, loss, and evaluation code from
`notebooks/01_training_v7.ipynb`. The notebook is the canonical reference
implementation; this package mirrors it for reuse in scripts/tests/serving.
"""

from src.model import MultiLabelDistilBert
from src.losses import FocalLoss, compute_focal_alpha
from src.data import (
    ToxicCommentsDataset,
    LengthBucketedBatchSampler,
    dynamic_pad_collate,
    tokenize_texts_varlen,
)
from src.evaluate import (
    evaluate_model,
    tune_thresholds,
    apply_thresholds_to_test,
    LABELS,
)

__all__ = [
    "MultiLabelDistilBert",
    "FocalLoss",
    "compute_focal_alpha",
    "ToxicCommentsDataset",
    "LengthBucketedBatchSampler",
    "dynamic_pad_collate",
    "tokenize_texts_varlen",
    "evaluate_model",
    "tune_thresholds",
    "apply_thresholds_to_test",
    "LABELS",
]
