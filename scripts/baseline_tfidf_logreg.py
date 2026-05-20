"""TF-IDF + Logistic Regression baseline for Jigsaw multi-label toxicity.

Purpose
-------
Establish a non-transformer baseline so the DistilBERT model in
``notebooks/01_training_v7.ipynb`` can be evaluated against a classical
text classifier on the same data, splits, and metrics. The transformer
work is only justified if it beats this baseline by enough to pay for the
training time and GPU cost.

Methodology
-----------
- Same train/val split as v7: ``train_test_split(test_size=0.1, random_state=42)``.
- Same test filtering as v7: drop rows where any label is ``-1`` in
  ``test_labels.csv`` (Kaggle's excluded hidden-set rows).
- Same metrics as v7: per-label PR AUC / ROC AUC, micro PR AUC, exact-match
  accuracy, and macro F1 with per-class F1-optimal thresholds floored at 0.5
  for labels with fewer than 100 validation positives (via ``src.evaluate``).

Output
------
``artifacts/baseline_tfidf_logreg.json`` — schema mirrors
``artifacts/final_metrics_v7.json`` with extra fields for the
baseline-vs-v7 delta and timing.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import LABELS, apply_thresholds_to_test, tune_thresholds  # noqa: E402

SEED = 42
VAL_SPLIT = 0.1

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=200_000,
    strip_accents="unicode",
)
LOGREG_KWARGS = dict(C=4.0, solver="liblinear", max_iter=200, random_state=SEED)


def main() -> None:
    print("=== TF-IDF + LogReg baseline ===\n")

    # --- Data ---------------------------------------------------------------
    print("[1/4] Loading data...")
    train_df = pd.read_csv(REPO_ROOT / "train.csv")
    test_df = pd.read_csv(REPO_ROOT / "test.csv")
    test_labels_df = pd.read_csv(REPO_ROOT / "test_labels.csv")
    print(f"  train.csv:  {len(train_df):,}")
    print(f"  test.csv:   {len(test_df):,}")

    train_texts, val_texts, y_train, y_val = train_test_split(
        train_df["comment_text"].values,
        train_df[LABELS].values.astype(np.float32),
        test_size=VAL_SPLIT,
        random_state=SEED,
    )
    print(f"  train_split: {len(train_texts):,}")
    print(f"  val_split:   {len(val_texts):,}")

    valid_mask = (test_labels_df[LABELS] != -1).all(axis=1)
    test_subset = test_df.merge(
        test_labels_df.loc[valid_mask, ["id"] + LABELS],
        on="id",
        how="inner",
    )
    test_texts = test_subset["comment_text"].values
    y_test = test_subset[LABELS].values.astype(np.float32)
    print(f"  test_valid:  {len(test_texts):,}\n")

    # --- TF-IDF -------------------------------------------------------------
    print("[2/4] Fitting TF-IDF (1-2 grams, sublinear_tf)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    tfidf_time = time.time() - t0
    print(f"  features: {X_train.shape[1]:,}")
    print(f"  time:     {tfidf_time:.1f}s\n")

    # --- LogReg per label ---------------------------------------------------
    print("[3/4] Training LogReg per label (one-vs-rest)...")
    t0 = time.time()
    val_probs = np.zeros_like(y_val)
    test_probs = np.zeros_like(y_test)
    for i, label in enumerate(LABELS):
        t_label = time.time()
        clf = LogisticRegression(**LOGREG_KWARGS)
        clf.fit(X_train, y_train[:, i].astype(int))
        val_probs[:, i] = clf.predict_proba(X_val)[:, 1]
        test_probs[:, i] = clf.predict_proba(X_test)[:, 1]
        print(f"  {label:<14} {time.time() - t_label:>5.1f}s")
    logreg_time = time.time() - t0
    print(f"  total:    {logreg_time:.1f}s ({logreg_time / 60:.2f} min)\n")

    # --- Metrics ------------------------------------------------------------
    print("[4/4] Computing metrics...")
    val_pr_aucs = {
        label: float(average_precision_score(y_val[:, i], val_probs[:, i]))
        for i, label in enumerate(LABELS)
    }
    test_pr_aucs = {
        label: float(average_precision_score(y_test[:, i], test_probs[:, i]))
        for i, label in enumerate(LABELS)
    }
    test_roc_aucs = {
        label: float(roc_auc_score(y_test[:, i], test_probs[:, i]))
        for i, label in enumerate(LABELS)
    }
    val_micro_pr_auc = float(average_precision_score(y_val, val_probs, average="micro"))
    test_micro_pr_auc = float(average_precision_score(y_test, test_probs, average="micro"))
    test_exact_match = float(
        accuracy_score(y_test, (test_probs >= 0.5).astype(int))
    )

    tuning = tune_thresholds(
        val_probs, y_val.astype(int), LABELS, min_positives_for_tuning=100
    )
    test_results = apply_thresholds_to_test(
        test_probs, y_test, tuning["thresholds"], LABELS
    )

    # --- Compare to v7 ------------------------------------------------------
    with open(REPO_ROOT / "artifacts" / "final_metrics_v7.json") as f:
        v7 = json.load(f)
    v7_train_minutes = v7["total_training_time_hours"] * 60.0
    baseline_train_minutes = (tfidf_time + logreg_time) / 60.0

    # --- Report -------------------------------------------------------------
    print("\n=== Results ===")
    print(f"Val micro PR AUC:   {val_micro_pr_auc:.4f}")
    print(f"Test micro PR AUC:  {test_micro_pr_auc:.4f}")
    print(f"Test exact-match:   {test_exact_match:.4f}")
    print(f"Macro F1 @ 0.5:     {test_results['macro_f1_test_at_0.5']:.4f}")
    print(f"Macro F1 tuned:     {test_results['macro_f1_test_tuned']:.4f}")
    print()
    header = f"  {'label':<14} {'PR AUC':>8} {'ROC AUC':>8} {'F1@0.5':>8} {'F1@tuned':>10}"
    print(header)
    print(f"  {'-' * 12:<14} {'-' * 6:>8} {'-' * 7:>8} {'-' * 6:>8} {'-' * 8:>10}")
    for label in LABELS:
        per = test_results["per_label_test_metrics"][label]
        print(
            f"  {label:<14} "
            f"{test_pr_aucs[label]:>8.3f} "
            f"{test_roc_aucs[label]:>8.3f} "
            f"{per['test_f1_at_0.5']:>8.3f} "
            f"{per['test_f1_tuned']:>10.3f}"
        )

    delta = test_micro_pr_auc - v7["test_micro_pr_auc"]
    speedup = v7_train_minutes / max(baseline_train_minutes, 1e-9)
    print("\n=== vs v7 (DistilBERT, shared trunk) ===")
    print(
        f"  Test micro PR AUC:  baseline {test_micro_pr_auc:.4f}  "
        f"vs v7 {v7['test_micro_pr_auc']:.4f}  (delta {delta:+.4f})"
    )
    print(
        f"  Train+predict time: baseline {baseline_train_minutes:.2f} min  "
        f"vs v7 {v7_train_minutes:.1f} min  ({speedup:.0f}x faster)"
    )

    # --- Save ---------------------------------------------------------------
    out = {
        "method": (
            "TF-IDF (word, 1-2 grams, sublinear_tf, min_df=3, max_features=200k) "
            "+ per-label Logistic Regression (one-vs-rest, C=4.0, liblinear)"
        ),
        "purpose": (
            "Non-transformer baseline. Validates whether the DistilBERT model "
            "in notebooks/01_training_v7.ipynb justifies its training cost."
        ),
        "config": {
            "tfidf": {**TFIDF_KWARGS, "ngram_range": list(TFIDF_KWARGS["ngram_range"])},
            "logreg": LOGREG_KWARGS,
            "split": {"VAL_SPLIT": VAL_SPLIT, "SEED": SEED},
        },
        "n_features_tfidf": int(X_train.shape[1]),
        "n_train_split": int(len(train_texts)),
        "n_val_split": int(len(val_texts)),
        "n_test_valid": int(len(test_texts)),
        "val_micro_pr_auc": val_micro_pr_auc,
        "val_pr_aucs": val_pr_aucs,
        "test_micro_pr_auc": test_micro_pr_auc,
        "test_pr_aucs": test_pr_aucs,
        "test_roc_aucs": test_roc_aucs,
        "test_exact_match": test_exact_match,
        "macro_f1_test_at_0.5": test_results["macro_f1_test_at_0.5"],
        "macro_f1_test_tuned": test_results["macro_f1_test_tuned"],
        "macro_f1_lift": test_results["macro_f1_lift"],
        "per_label_test_metrics": test_results["per_label_test_metrics"],
        "thresholds": tuning["thresholds"],
        "val_positive_counts": tuning["val_positive_counts"],
        "floored_labels": tuning["floored_labels"],
        "timing_seconds": {
            "tfidf_fit_transform": tfidf_time,
            "logreg_train_predict": logreg_time,
            "total": tfidf_time + logreg_time,
        },
        "vs_v7": {
            "test_micro_pr_auc_v7": v7["test_micro_pr_auc"],
            "test_micro_pr_auc_delta": delta,
            "v7_train_time_minutes": v7_train_minutes,
            "baseline_train_time_minutes": baseline_train_minutes,
            "speedup_x": speedup,
        },
    }
    out_path = REPO_ROOT / "artifacts" / "baseline_tfidf_logreg.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
