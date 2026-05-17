"""Evaluation: per-label PR AUC / ROC AUC + post-hoc F1-optimal threshold tuning.

Threshold tuning has a floor at 0.5 for any label with fewer than
`min_positives_for_tuning` val positives (default 100). On v7, threat had
48 val positives and an unfloored val-optimal threshold of 0.455 regressed
test F1 by -0.027 — the PR curve at that positive count is dominated by noise.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABELS: List[str] = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, use_fp16: bool = False,
                   labels_list: List[str] = LABELS) -> Tuple[float, Dict, Dict, float]:
    """Evaluate model on a dataloader.

    Returns:
        (avg_loss, pr_aucs_per_label, roc_aucs_per_label, micro_pr_auc)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if use_fp16 and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    pr_aucs: Dict[str, float] = {}
    roc_aucs: Dict[str, float] = {}
    for i, label in enumerate(labels_list):
        try:
            pr_aucs[label] = float(average_precision_score(all_labels[:, i], all_preds[:, i]))
            roc_aucs[label] = float(roc_auc_score(all_labels[:, i], all_preds[:, i]))
        except ValueError:
            pr_aucs[label] = 0.0
            roc_aucs[label] = 0.0

    micro_pr_auc = float(average_precision_score(all_labels, all_preds, average="micro"))
    return avg_loss, pr_aucs, roc_aucs, micro_pr_auc


def tune_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    labels_list: List[str] = LABELS,
    min_positives_for_tuning: int = 100,
) -> Dict:
    """Per-label F1-optimal threshold on validation, floored at 0.5 for rare labels.

    The floor matters: on v7, threat had 48 val positives and the raw F1-optimal
    threshold (0.455) regressed test F1 by -0.027. PR curves at low positive
    counts are dominated by noise, so the val "optimum" is an artifact.

    Returns dict with thresholds, val_positive_counts, and floored_labels.
    """
    thresholds: Dict[str, float] = {}
    val_positive_counts: Dict[str, int] = {}
    floored_labels: List[str] = []

    for i, label in enumerate(labels_list):
        y_true = val_labels[:, i].astype(int)
        y_prob = val_probs[:, i]
        n_pos = int(y_true.sum())
        val_positive_counts[label] = n_pos

        prec, rec, thrs = precision_recall_curve(y_true, y_prob)
        if len(thrs) == 0:
            thresholds[label] = 0.5
            continue
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.argmax(f1s))
        thr_raw = float(thrs[best_idx])

        if n_pos < min_positives_for_tuning:
            thr_use = max(thr_raw, 0.5)
            if thr_use != thr_raw:
                floored_labels.append(label)
        else:
            thr_use = thr_raw

        thresholds[label] = thr_use

    return {
        "thresholds": thresholds,
        "val_positive_counts": val_positive_counts,
        "floored_labels": floored_labels,
        "min_positives_for_tuning": min_positives_for_tuning,
    }


def apply_thresholds_to_test(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    thresholds: Dict[str, float],
    labels_list: List[str] = LABELS,
) -> Dict:
    """Apply per-class thresholds and report per-label + macro F1, vs the 0.5 baseline."""
    per_label = {}
    f1_tuned, f1_05 = [], []

    for i, label in enumerate(labels_list):
        thr = thresholds[label]
        y_true = test_labels[:, i].astype(int)
        y_prob = test_probs[:, i]

        y_pred_t = (y_prob >= thr).astype(int)
        y_pred_5 = (y_prob >= 0.5).astype(int)

        p_t = precision_score(y_true, y_pred_t, zero_division=0)
        r_t = recall_score(y_true, y_pred_t, zero_division=0)
        f_t = f1_score(y_true, y_pred_t, zero_division=0)
        p_5 = precision_score(y_true, y_pred_5, zero_division=0)
        r_5 = recall_score(y_true, y_pred_5, zero_division=0)
        f_5 = f1_score(y_true, y_pred_5, zero_division=0)

        per_label[label] = {
            "threshold": float(thr),
            "test_precision_tuned": float(p_t),
            "test_recall_tuned": float(r_t),
            "test_f1_tuned": float(f_t),
            "test_precision_at_0.5": float(p_5),
            "test_recall_at_0.5": float(r_5),
            "test_f1_at_0.5": float(f_5),
        }
        f1_tuned.append(f_t)
        f1_05.append(f_5)

    macro_tuned = float(np.mean(f1_tuned))
    macro_05 = float(np.mean(f1_05))

    return {
        "per_label_test_metrics": per_label,
        "macro_f1_test_tuned": macro_tuned,
        "macro_f1_test_at_0.5": macro_05,
        "macro_f1_lift": macro_tuned - macro_05,
    }
