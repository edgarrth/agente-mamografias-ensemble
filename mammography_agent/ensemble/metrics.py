from __future__ import annotations
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate(y_true, scores, threshold: float) -> dict:
    """Evaluate a binary study-level classifier at one threshold.

    v0.23 keeps ROC-AUC as the threshold-independent ranking metric and adds
    threshold-dependent metrics needed to understand the sensitivity/specificity
    trade-off. None of these values are clinical claims; they are research metrics.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    predicted_positives = int((pred == 1).sum())
    predicted_negatives = int((pred == 0).sum())

    sensitivity = _safe_ratio(int(tp), positives)
    specificity = _safe_ratio(int(tn), negatives)
    precision_ppv = _safe_ratio(int(tp), predicted_positives)
    npv = _safe_ratio(int(tn), predicted_negatives)
    fpr = _safe_ratio(int(fp), negatives)
    accuracy = _safe_ratio(int(tp + tn), int(len(y)))
    balanced_accuracy = (
        float((sensitivity + specificity) / 2.0)
        if sensitivity is not None and specificity is not None
        else None
    )

    sensitivity_reason = None if positives else "Sensitivity requires at least one malignant (ground_truth=1) study."
    specificity_reason = None if negatives else "Specificity/FPR require at least one benign (ground_truth=0) study."
    precision_reason = None if predicted_positives else "PPV requires at least one predicted positive study."
    npv_reason = None if predicted_negatives else "NPV requires at least one predicted negative study."

    auc = float(roc_auc_score(y, s)) if positives and negatives else None
    auc_reason = None if auc is not None else "ROC-AUC requires both benign (0) and malignant (1) ground-truth classes."

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": sensitivity,
        "sensitivity_unavailable_reason": sensitivity_reason,
        "specificity": specificity,
        "specificity_unavailable_reason": specificity_reason,
        "precision_ppv": precision_ppv,
        "precision_ppv_unavailable_reason": precision_reason,
        "npv": npv,
        "npv_unavailable_reason": npv_reason,
        "fpr": fpr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": auc,
        "roc_auc_unavailable_reason": auc_reason,
        "threshold": float(threshold),
    }
