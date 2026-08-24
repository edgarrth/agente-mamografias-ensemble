from __future__ import annotations
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import numpy as np


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate(y_true, scores, threshold: float) -> dict:
    """Evaluate a binary study-level classifier at one threshold.

    v0.30.1 keeps ROC-AUC as the selection-policy ranking metric and adds
    Average Precision (reported as AUPRC) plus F1 so strongly imbalanced
    datasets such as RSNA can be interpreted beyond accuracy alone.
    None of these values are clinical claims; they are research metrics.
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
    f1_denominator = int(2 * tp + fp + fn)
    f1 = _safe_ratio(int(2 * tp), f1_denominator)

    sensitivity_reason = None if positives else "Sensitivity requires at least one malignant (ground_truth=1) study."
    specificity_reason = None if negatives else "Specificity/FPR require at least one benign (ground_truth=0) study."
    precision_reason = None if predicted_positives else "PPV requires at least one predicted positive study."
    npv_reason = None if predicted_negatives else "NPV requires at least one predicted negative study."
    f1_reason = None if f1_denominator else "F1 requires at least one true positive, false positive, or false negative event."

    both_classes = bool(positives and negatives)
    auc = float(roc_auc_score(y, s)) if both_classes else None
    auc_reason = None if auc is not None else "ROC-AUC requires both benign (0) and malignant (1) ground-truth classes."
    average_precision = float(average_precision_score(y, s)) if both_classes else None
    ap_reason = None if average_precision is not None else "Average Precision/AUPRC requires both benign (0) and malignant (1) ground-truth classes."

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
        "f1": f1,
        "f1_unavailable_reason": f1_reason,
        "roc_auc": auc,
        "roc_auc_unavailable_reason": auc_reason,
        "average_precision": average_precision,
        "auprc": average_precision,
        "auprc_method": "average_precision_score",
        "auprc_unavailable_reason": ap_reason,
        "threshold": float(threshold),
    }
