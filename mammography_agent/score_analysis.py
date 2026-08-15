from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .ensemble.metrics import evaluate

from .config import load_yaml, WORKSPACE_ROOT
from .reporting import write_json

MODEL_SCORE_COLUMNS = {
    "gmic": "gmic_score",
    "nyu": "nyu_score",
    "glam": "glam_score",
}


def _finite_numeric(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing/non-finite values")
    return values.astype(float)


def validate_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"study_id", "ground_truth", *MODEL_SCORE_COLUMNS.values()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Score file is missing required columns: {missing}")
    out = df.copy()
    out["study_id"] = out["study_id"].astype(str)
    if out["study_id"].duplicated().any():
        dupes = sorted(out.loc[out.study_id.duplicated(keep=False), "study_id"].unique().tolist())
        raise ValueError(f"Score file contains duplicate study_id values: {dupes[:10]}")
    out["ground_truth"] = pd.to_numeric(out["ground_truth"], errors="raise").astype(int)
    invalid = sorted(set(out["ground_truth"].tolist()) - {0, 1})
    if invalid:
        raise ValueError(f"ground_truth must be binary 0/1; found {invalid}")
    for column in MODEL_SCORE_COLUMNS.values():
        out[column] = _finite_numeric(out[column], column)
    return out


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y.astype(int), score.astype(float)))


def stratified_bootstrap_auc(
    y: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> dict:
    """Stratified patient/study-level bootstrap CI for ROC-AUC.

    The bootstrap preserves the observed benign/malignant counts in every replicate,
    avoiding invalid one-class resamples in very small diagnostic sets.  It is an
    uncertainty description only; it is never used to select weights or thresholds.
    """
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    if len(pos) == 0 or len(neg) == 0:
        return {
            "roc_auc_ci95_low": None,
            "roc_auc_ci95_high": None,
            "bootstrap_replicates": 0,
            "bootstrap_method": "stratified_by_class",
        }
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        draw_pos = rng.choice(pos, size=len(pos), replace=True)
        draw_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([draw_neg, draw_pos])
        values[i] = roc_auc_score(y_arr[idx], s_arr[idx])
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "roc_auc_ci95_low": float(low),
        "roc_auc_ci95_high": float(high),
        "bootstrap_replicates": int(replicates),
        "bootstrap_method": "stratified_by_class",
    }


def threshold_strategy_config() -> dict:
    cfg = load_yaml("experiments.yaml")
    strategy = cfg.get("threshold_strategy", {})
    if strategy.get("mode") != "score_quantiles":
        raise ValueError("config/experiments.yaml threshold_strategy.mode must be score_quantiles")
    quantiles = strategy.get("quantiles", {})
    if not isinstance(quantiles, dict) or len(quantiles) != 5:
        raise ValueError("threshold_strategy.quantiles must contain exactly 5 threshold IDs")
    for key, value in quantiles.items():
        q = float(value)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Invalid threshold quantile {key}={value}")
    return strategy


def derive_threshold_candidates(
    scores: pd.Series | np.ndarray,
    strategy: dict | None = None,
    threshold_source: str = "configuration_score_quantile",
) -> list[dict]:
    """Derive five deterministic, label-independent thresholds from score quantiles.

    Candidate values depend only on the supplied model/ensemble scores. Ground truth is
    deliberately not consulted here; labels may be used later only to evaluate already
    derived candidates. In formal experiments the supplied scores come exclusively from
    the Configuration Set. This keeps the Final Test Set isolated and avoids the
    obsolete fixed 0.40-0.60 grid when a model emits scores on a much lower range.
    """
    strategy = strategy or threshold_strategy_config()
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot derive thresholds from empty/non-finite scores")
    rows = []
    for tid, q in strategy["quantiles"].items():
        rows.append({
            "threshold_id": str(tid),
            "threshold_quantile": float(q),
            "threshold": float(np.quantile(values, float(q))),
            "threshold_source": str(threshold_source),
            "ground_truth_used_for_threshold_derivation": False,
        })
    return rows


def _distribution_row(model: str, label_name: str, values: pd.Series) -> dict:
    arr = values.to_numpy(dtype=float)
    return {
        "model": model,
        "class": label_name,
        "count": int(arr.size),
        "min": float(np.min(arr)) if arr.size else None,
        "q10": float(np.quantile(arr, 0.10)) if arr.size else None,
        "q25": float(np.quantile(arr, 0.25)) if arr.size else None,
        "median": float(np.quantile(arr, 0.50)) if arr.size else None,
        "q75": float(np.quantile(arr, 0.75)) if arr.size else None,
        "q90": float(np.quantile(arr, 0.90)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr, ddof=0)) if arr.size else None,
    }


def analyze_score_frame(df: pd.DataFrame, output_dir: Path, source: str | None = None, include_candidate_thresholds: bool = True) -> Path:
    df = validate_score_frame(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_yaml("ensemble.yaml")["baseline"]
    weights = {k: float(v) for k, v in base["weights"].items()}
    baseline_threshold = float(base["threshold"])
    df = df.copy()
    df["baseline_ensemble_score"] = sum(df[MODEL_SCORE_COLUMNS[m]] * weights[m] for m in MODEL_SCORE_COLUMNS)

    y = df["ground_truth"].astype(int)
    class_distribution = {
        "BENIGN": int((y == 0).sum()),
        "MALIGNANT": int((y == 1).sum()),
    }

    metric_rows = []
    distribution_rows = []
    score_series = {m: df[col] for m, col in MODEL_SCORE_COLUMNS.items()}
    score_series["baseline_ensemble"] = df["baseline_ensemble_score"]
    warnings: list[str] = []

    for model, values in score_series.items():
        auc = _safe_auc(y, values)
        benign = values[y == 0]
        malignant = values[y == 1]
        ci = stratified_bootstrap_auc(y, values)
        metric_rows.append({
            "model": model,
            "roc_auc": auc,
            **ci,
            "mean_benign": float(benign.mean()) if len(benign) else None,
            "mean_malignant": float(malignant.mean()) if len(malignant) else None,
            "median_benign": float(benign.median()) if len(benign) else None,
            "median_malignant": float(malignant.median()) if len(malignant) else None,
            "min_score": float(values.min()),
            "max_score": float(values.max()),
            "directionality_warning": bool(auc is not None and auc < 0.5),
        })
        distribution_rows.append(_distribution_row(model, "ALL", values))
        distribution_rows.append(_distribution_row(model, "BENIGN", benign))
        distribution_rows.append(_distribution_row(model, "MALIGNANT", malignant))
        if auc is not None and auc < 0.5:
            warnings.append(
                f"{model} ROC-AUC is below 0.5 in this analysis set. v0.23 records the warning but does not invert or recalibrate scores."
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(output_dir / "score_distribution.csv", index=False)

    corr_source = pd.DataFrame({m: s for m, s in score_series.items()})
    corr_rows = []
    for i, left in enumerate(corr_source.columns):
        for right in corr_source.columns[i + 1:]:
            corr_rows.append({
                "left": left,
                "right": right,
                "pearson": float(corr_source[left].corr(corr_source[right], method="pearson")),
                "spearman": float(corr_source[left].corr(corr_source[right], method="spearman")),
            })
    pd.DataFrame(corr_rows).to_csv(output_dir / "model_correlations.csv", index=False)

    roc_rows = []
    if y.nunique() >= 2:
        for model, values in score_series.items():
            fpr, tpr, thresholds = roc_curve(y, values)
            for idx, (f, t, threshold) in enumerate(zip(fpr, tpr, thresholds)):
                roc_rows.append({
                    "model": model,
                    "point": idx,
                    "fpr": float(f),
                    "tpr": float(t),
                    "threshold": float(threshold) if math.isfinite(float(threshold)) else None,
                })
    pd.DataFrame(roc_rows, columns=["model", "point", "fpr", "tpr", "threshold"]).to_csv(output_dir / "roc_points.csv", index=False)

    # Preview the adaptive threshold grid only when this is a configuration/diagnostic analysis.
    # Final Test Set analysis must never suggest a post-hoc re-optimization grid.
    exp_cfg = load_yaml("experiments.yaml")
    strategy = threshold_strategy_config()
    threshold_rows = []
    diagnostic_metric_rows = []
    if include_candidate_thresholds:
        for weight_id, w in exp_cfg["weights"].items():
            weighted = df.gmic_score * float(w[0]) + df.nyu_score * float(w[1]) + df.glam_score * float(w[2])
            for row in derive_threshold_candidates(weighted, strategy, threshold_source="analysis_score_quantile"):
                base_row = {
                    "weight_id": weight_id,
                    "w_gmic": float(w[0]),
                    "w_nyu": float(w[1]),
                    "w_glam": float(w[2]),
                    **row,
                }
                threshold_rows.append(base_row)
                diagnostic_metric_rows.append({
                    **base_row,
                    **evaluate(y, weighted, float(row["threshold"])),
                    "diagnostic_only": True,
                    "eligible_for_freeze": False,
                })
        pd.DataFrame(threshold_rows).to_csv(output_dir / "candidate_thresholds.csv", index=False)
        diagnostic_df = pd.DataFrame(diagnostic_metric_rows)
        diagnostic_df.to_csv(output_dir / "diagnostic_configurations.csv", index=False)
        diagnostic_ranking = diagnostic_df.sort_values(
            ["roc_auc","balanced_accuracy","sensitivity","specificity","fn","fp","weight_id","threshold_id"],
            ascending=[False,False,False,False,True,True,True,True],
            na_position="last",
        ).reset_index(drop=True)
        diagnostic_ranking.to_csv(output_dir / "diagnostic_ranking.csv", index=False)

    observed_max = float(df["baseline_ensemble_score"].max())
    if baseline_threshold > observed_max:
        warnings.append(
            f"Baseline threshold {baseline_threshold:.6f} is above the maximum baseline ensemble score observed in this analysis set ({observed_max:.6f})."
        )

    baseline_classification_metrics = evaluate(y, df["baseline_ensemble_score"], baseline_threshold)

    summary = {
        "source": source,
        "studies": int(len(df)),
        "class_distribution": class_distribution,
        "baseline": {
            "weights": weights,
            "threshold": baseline_threshold,
            "ensemble_score_min": float(df["baseline_ensemble_score"].min()),
            "ensemble_score_max": observed_max,
            "ensemble_score_mean": float(df["baseline_ensemble_score"].mean()),
            "roc_auc": _safe_auc(y, df["baseline_ensemble_score"]),
            "roc_auc_uncertainty": stratified_bootstrap_auc(y, df["baseline_ensemble_score"]),
            "classification_metrics": baseline_classification_metrics,
        },
        "statistical_uncertainty": {
            "benign_studies": int((y == 0).sum()),
            "malignant_studies": int((y == 1).sum()),
            "strict_pairwise_auc_step": (float(1.0 / (((y == 0).sum()) * ((y == 1).sum()))) if (y == 0).sum() and (y == 1).sum() else None),
            "bootstrap_method": "stratified_by_class",
            "bootstrap_replicates": 2000,
            "interpretation": "Small diagnostic sets have coarse and uncertain ROC-AUC estimates; confidence intervals are descriptive and are not used for configuration selection.",
        },
        "threshold_strategy": {
            "mode": strategy["mode"],
            "quantiles": {k: float(v) for k, v in strategy["quantiles"].items()},
            "derive_per_weight": bool(strategy.get("derive_per_weight", True)),
            "ground_truth_used_for_threshold_derivation": False,
            "candidate_thresholds_generated": bool(include_candidate_thresholds),
            "diagnostic_configuration_metrics_generated": bool(include_candidate_thresholds),
            "diagnostic_results_eligible_for_freeze": False,
        },
        "warnings": warnings,
        "research_guards": {
            "score_inversion_performed": False,
            "calibration_performed": False,
            "training_performed": False,
            "final_test_scores_required": False,
        },
    }
    write_json(output_dir / "score_summary.json", summary)

    report_lines = [
        "# Score Analysis Report",
        "",
        "> Research-only diagnostic. Scores are not calibrated clinical probabilities.",
        "",
        f"- **source**: {source}",
        f"- **studies**: {len(df)}",
        f"- **class_distribution**: {class_distribution}",
        f"- **baseline_threshold**: {baseline_threshold}",
        f"- **baseline_score_range**: [{float(df.baseline_ensemble_score.min()):.6f}, {observed_max:.6f}]",
        f"- **threshold_strategy**: score quantiles {strategy['quantiles']} derived independently for each weight combination",
        f"- **candidate_thresholds_generated**: {bool(include_candidate_thresholds)}",
        "- **threshold_derivation_uses_ground_truth**: False",
        "- **score_inversion_or_calibration**: None",
        "",
        "## Baseline classification metrics",
        "",
        f"- **TN / FP / FN / TP**: {baseline_classification_metrics['tn']} / {baseline_classification_metrics['fp']} / {baseline_classification_metrics['fn']} / {baseline_classification_metrics['tp']}",
        f"- **Sensitivity**: {baseline_classification_metrics['sensitivity']}",
        f"- **Specificity**: {baseline_classification_metrics['specificity']}",
        f"- **Precision / PPV**: {baseline_classification_metrics['precision_ppv']}",
        f"- **NPV**: {baseline_classification_metrics['npv']}",
        f"- **FPR**: {baseline_classification_metrics['fpr']}",
        f"- **Balanced Accuracy**: {baseline_classification_metrics['balanced_accuracy']}",
        "",
        "## Diagnostic candidate evaluation",
        "",
        "- **diagnostic_only**: True",
        "- **eligible_for_freeze**: False",
        "- When candidate thresholds are enabled, see `diagnostic_configurations.csv` and `diagnostic_ranking.csv` for the 16×5 CPU-only preview on this analysis set.",
        "",
        "## Per-model ROC-AUC",
        "",
    ]
    for row in metric_rows:
        report_lines.append(
            f"- **{row['model']}**: {row['roc_auc']} "
            f"(stratified-bootstrap 95% CI {row['roc_auc_ci95_low']}–{row['roc_auc_ci95_high']}, n=2000)"
        )
    n_neg = int((y == 0).sum()); n_pos = int((y == 1).sum())
    pair_step = (1.0 / (n_neg * n_pos)) if n_neg and n_pos else None
    report_lines += [
        "", "## Statistical uncertainty", "",
        f"- **benign / malignant studies**: {n_neg} / {n_pos}",
        f"- **one strictly ordered positive-negative pair changes AUC by approximately**: {pair_step if pair_step is not None else 'N/A'}",
        "- **CI method**: 2,000-replicate stratified bootstrap preserving class counts.",
        "- These intervals describe uncertainty only. They are not used to choose weights, thresholds, orientation, or aggregation.",
        "- With a 5/5 diagnostic set, modest AUC changes can correspond to only a few pairwise order changes and must not be treated as conclusive.",
        "", "## Warnings", ""
    ]
    report_lines += [f"- {w}" for w in warnings] if warnings else ["- None"]
    (output_dir / "score_analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_dir


def analyze_score_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path)
    if output_path is None:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_path = WORKSPACE_ROOT / "output" / "analyses" / f"score-analysis-{stamp}"
    return analyze_score_frame(df, Path(output_path), source=str(input_path))
