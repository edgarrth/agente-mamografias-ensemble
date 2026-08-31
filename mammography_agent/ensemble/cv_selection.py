from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import load_yaml
from ..score_analysis import derive_threshold_candidates, threshold_strategy_config
from .metrics import evaluate

SCORE_COLUMNS = ["gmic_score", "nyu_score", "glam_score"]
REQUIRED_COLUMNS = {"study_id", "patient_id", "ground_truth", "dataset_source", *SCORE_COLUMNS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _score(df: pd.DataFrame, weights: list[float] | tuple[float, float, float]) -> pd.Series:
    return (
        df["gmic_score"].astype(float) * float(weights[0])
        + df["nyu_score"].astype(float) * float(weights[1])
        + df["glam_score"].astype(float) * float(weights[2])
    )


def build_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        raise ValueError(f"Configuration predictions missing required columns: {missing}")
    if df["study_id"].astype(str).duplicated().any():
        raise ValueError("Configuration predictions contain duplicate study_id values")
    if df["patient_id"].astype(str).duplicated().any():
        raise ValueError("Configuration predictions contain duplicate patient_id values")
    y = df["ground_truth"].astype(int).to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    if classes.tolist() != [0, 1]:
        raise ValueError("Stratified CV requires both ground-truth classes 0 and 1")
    if int(counts.min()) < int(n_splits):
        raise ValueError("Each class must contain at least n_splits samples")

    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    fold = np.empty(len(df), dtype=int)
    dummy = np.zeros(len(df), dtype=np.uint8)
    for fold_index, (_, val_idx) in enumerate(splitter.split(dummy, y), start=1):
        fold[val_idx] = fold_index
    out = df[["study_id", "patient_id", "dataset_source", "ground_truth"]].copy()
    out["fold"] = fold
    return out


def evaluate_cv_grid(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate configured weight/quantile candidates by stratified out-of-fold validation.

    For each fold and weight, threshold values are derived from the other folds' scores only.
    Ground-truth labels are never used to derive thresholds. The held-out fold is used only to
    evaluate the already-derived operating point.
    """
    cfg = load_yaml("experiments.yaml")
    strategy = threshold_strategy_config()
    folds = build_stratified_folds(df, n_splits=n_splits, seed=seed)
    work = df.copy().reset_index(drop=True)
    work["fold"] = folds["fold"].to_numpy()

    rows: list[dict] = []
    for fold_index in range(1, int(n_splits) + 1):
        train = work[work.fold != fold_index].reset_index(drop=True)
        valid = work[work.fold == fold_index].reset_index(drop=True)
        for wid, weights in cfg["weights"].items():
            train_score = _score(train, weights)
            valid_score = _score(valid, weights)
            candidates = derive_threshold_candidates(
                train_score,
                strategy,
                threshold_source="cv_training_fold_score_quantile",
            )
            for candidate in candidates:
                metrics = evaluate(valid["ground_truth"], valid_score, float(candidate["threshold"]))
                rows.append(
                    {
                        "fold": int(fold_index),
                        "weight_id": str(wid),
                        "threshold_id": str(candidate["threshold_id"]),
                        "threshold_quantile": float(candidate["threshold_quantile"]),
                        "threshold": float(candidate["threshold"]),
                        "threshold_source": candidate["threshold_source"],
                        "ground_truth_used_for_threshold_derivation": False,
                        "w_gmic": float(weights[0]),
                        "w_nyu": float(weights[1]),
                        "w_glam": float(weights[2]),
                        "training_studies": int(len(train)),
                        "validation_studies": int(len(valid)),
                        "training_malignant": int((train.ground_truth.astype(int) == 1).sum()),
                        "validation_malignant": int((valid.ground_truth.astype(int) == 1).sum()),
                        **metrics,
                    }
                )

    fold_metrics = pd.DataFrame(rows)
    expected = int(n_splits) * len(cfg["weights"]) * len(strategy["quantiles"])
    if len(fold_metrics) != expected:
        raise AssertionError(f"Expected {expected} fold evaluations, got {len(fold_metrics)}")

    metric_names = [
        "roc_auc", "average_precision", "auprc", "balanced_accuracy", "sensitivity",
        "specificity", "precision_ppv", "npv", "fpr", "accuracy", "f1", "tn", "fp", "fn", "tp", "threshold",
    ]
    agg: dict[str, list[str]] = {name: ["mean", "std"] for name in metric_names}
    summary = fold_metrics.groupby(
        ["weight_id", "threshold_id", "threshold_quantile", "w_gmic", "w_nyu", "w_glam"],
        as_index=False,
    ).agg(agg)
    summary.columns = [
        "_".join([str(part) for part in col if part != ""]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary = summary.rename(columns={
        "weight_id": "weight_id", "threshold_id": "threshold_id",
        "threshold_quantile": "threshold_quantile", "w_gmic": "w_gmic",
        "w_nyu": "w_nyu", "w_glam": "w_glam",
    })

    # Derive the final numeric threshold for every candidate from ALL Configuration scores.
    # This is a refit after CV selection; it still uses scores only, never labels.
    full_thresholds: dict[tuple[str, str], float] = {}
    for wid, weights in cfg["weights"].items():
        full_score = _score(work, weights)
        for candidate in derive_threshold_candidates(
            full_score, strategy, threshold_source="full_configuration_refit_quantile"
        ):
            full_thresholds[(str(wid), str(candidate["threshold_id"]))] = float(candidate["threshold"])
    summary["full_configuration_threshold"] = summary.apply(
        lambda r: full_thresholds[(str(r.weight_id), str(r.threshold_id))], axis=1
    )
    baseline = np.array([0.333333, 0.333333, 0.333334, 0.50], dtype=float)
    summary["baseline_distance"] = summary.apply(
        lambda r: float(np.linalg.norm(np.array([
            float(r.w_gmic), float(r.w_nyu), float(r.w_glam), float(r.full_configuration_threshold)
        ]) - baseline)), axis=1
    )
    return folds, fold_metrics, summary


def rank_cv_candidates(summary: pd.DataFrame, balanced_accuracy_tolerance: float = 0.0) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    # AUC selects weights. Every threshold for a weight shares the same out-of-fold AUC,
    # but grouping makes the policy explicit and prevents threshold metrics from changing
    # which weight wins.
    weight_auc = summary.groupby("weight_id", as_index=False)["roc_auc_mean"].max()
    best_auc = float(weight_auc["roc_auc_mean"].max())
    winning_weights = set(
        weight_auc[np.isclose(weight_auc.roc_auc_mean.astype(float), best_auc, rtol=0.0, atol=1e-12)].weight_id
    )
    out = summary.copy()
    out["is_best_auc_weight"] = out.weight_id.isin(winning_weights)
    # Human-readable global order; authoritative selection is performed separately.
    return out.sort_values(
        ["is_best_auc_weight", "roc_auc_mean", "balanced_accuracy_mean", "sensitivity_mean",
         "specificity_mean", "fp_mean", "baseline_distance", "weight_id", "threshold_id"],
        ascending=[False, False, False, False, False, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_cv_candidate(summary: pd.DataFrame) -> pd.Series:
    if summary.empty:
        raise ValueError("No CV candidates to select")
    selection = load_yaml("experiments.yaml").get("selection", {})
    tol = float(selection.get("balanced_accuracy_tolerance", 0.0))

    weight_auc = summary.groupby("weight_id", as_index=False)["roc_auc_mean"].max()
    if weight_auc.roc_auc_mean.isna().all():
        raise ValueError("CV requires both classes in every validation fold")
    best_auc = float(weight_auc.roc_auc_mean.max())
    winning_weights = set(
        weight_auc[np.isclose(weight_auc.roc_auc_mean.astype(float), best_auc, rtol=0.0, atol=1e-12)].weight_id
    )
    sub = summary[summary.weight_id.isin(winning_weights)].copy()

    best_balanced = float(sub.balanced_accuracy_mean.max())
    sub = sub[sub.balanced_accuracy_mean >= best_balanced - tol].copy()
    best_sens = float(sub.sensitivity_mean.max())
    sub = sub[np.isclose(sub.sensitivity_mean.astype(float), best_sens, rtol=0.0, atol=1e-12)].copy()
    best_spec = float(sub.specificity_mean.max())
    sub = sub[np.isclose(sub.specificity_mean.astype(float), best_spec, rtol=0.0, atol=1e-12)].copy()
    min_fp = float(sub.fp_mean.min())
    sub = sub[np.isclose(sub.fp_mean.astype(float), min_fp, rtol=0.0, atol=1e-12)].copy()
    return sub.sort_values(["baseline_distance", "weight_id", "threshold_id"]).iloc[0]


def run_expanded_cv_selection(
    scores: pd.DataFrame,
    output_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    input_scores_sha256: str | None = None,
    final_manifest_sha256: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, fold_metrics, summary = evaluate_cv_grid(scores, n_splits=n_splits, seed=seed)
    ranking = rank_cv_candidates(summary)
    selected = select_cv_candidate(summary)

    folds.to_csv(output_dir / "fold_assignments.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "candidate_cv_summary.csv", index=False)
    ranking.to_csv(output_dir / "ranking_cv.csv", index=False)

    weights = [float(selected.w_gmic), float(selected.w_nyu), float(selected.w_glam)]
    full_score = _score(scores, weights)
    threshold = float(selected.full_configuration_threshold)
    descriptive = evaluate(scores.ground_truth, full_score, threshold)
    best = {
        "selection_protocol": "stratified_5fold_cv_expanded_grid_v0310",
        "cv_folds": int(n_splits),
        "cv_seed": int(seed),
        "weight_id": str(selected.weight_id),
        "threshold_id": str(selected.threshold_id),
        "threshold_quantile": float(selected.threshold_quantile),
        "threshold": threshold,
        "threshold_source": "full_configuration_refit_quantile_after_cv_selection",
        "ground_truth_used_for_threshold_derivation": False,
        "w_gmic": weights[0], "w_nyu": weights[1], "w_glam": weights[2],
        "cv_roc_auc_mean": float(selected.roc_auc_mean),
        "cv_roc_auc_std": float(selected.roc_auc_std),
        "cv_auprc_mean": float(selected.auprc_mean),
        "cv_auprc_std": float(selected.auprc_std),
        "cv_balanced_accuracy_mean": float(selected.balanced_accuracy_mean),
        "cv_balanced_accuracy_std": float(selected.balanced_accuracy_std),
        "cv_sensitivity_mean": float(selected.sensitivity_mean),
        "cv_sensitivity_std": float(selected.sensitivity_std),
        "cv_specificity_mean": float(selected.specificity_mean),
        "cv_specificity_std": float(selected.specificity_std),
        "configuration_refit_descriptive_metrics": descriptive,
        "input_scores_sha256": input_scores_sha256,
        "final_test_manifest_sha256": final_manifest_sha256,
        "final_test_scores_used": False,
        "models_reexecuted": False,
    }
    (output_dir / "best_configuration.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")

    cfg = load_yaml("experiments.yaml")
    protocol = {
        "protocol_id": "stratified_5fold_cv_expanded_grid_v0310",
        "configuration_studies": int(len(scores)),
        "configuration_class_distribution": {
            "BENIGN": int((scores.ground_truth.astype(int) == 0).sum()),
            "MALIGNANT": int((scores.ground_truth.astype(int) == 1).sum()),
        },
        "weights": int(len(cfg["weights"])),
        "threshold_quantiles": int(len(cfg["threshold_strategy"]["quantiles"])),
        "candidate_configurations": int(len(cfg["weights"]) * len(cfg["threshold_strategy"]["quantiles"])),
        "fold_evaluations": int(n_splits * len(cfg["weights"]) * len(cfg["threshold_strategy"]["quantiles"])),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "shuffle": True,
        "threshold_derivation": "training folds only; score quantiles; labels not used",
        "candidate_evaluation": "held-out fold only",
        "final_numeric_threshold": "selected quantile refit on all Configuration scores after CV selection",
        "selection_order": cfg.get("selection", {}).get("order", []),
        "input_scores_sha256": input_scores_sha256,
        "final_test_manifest_sha256": final_manifest_sha256,
        "final_test_scores_used": False,
        "models_reexecuted": False,
    }
    (output_dir / "selection_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output_dir
