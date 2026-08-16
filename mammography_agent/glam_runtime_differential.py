from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import WORKSPACE_ROOT
from .model_client import ensure_metarepository, run_glam_legacy_cpu_reference, run_model
from .upstream_reference_validation import evaluate_prediction_csv, _compare, REFERENCE_SOURCE


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pairwise_order_disagreements(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    disagreements = 0
    comparable = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa == 0 or sb == 0:
                continue
            comparable += 1
            if sa != sb:
                disagreements += 1
    return disagreements, comparable


def run_glam_runtime_differential(output_dir: str | Path | None = None) -> Path:
    root = WORKSPACE_ROOT
    out = Path(output_dir) if output_dir else root / "output" / "analyses" / f"glam-runtime-differential-{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions"
    pre = out / "preprocessed"
    pred.mkdir(exist_ok=True)
    pre.mkdir(exist_ok=True)

    meta = ensure_metarepository()
    runtime_meta = Path(meta["path"])
    images = runtime_meta / "sample_data" / "images"
    data_pkl = runtime_meta / "sample_data" / "data.pkl"
    if not images.is_dir() or not data_pkl.is_file():
        raise FileNotFoundError(f"Official sample missing under {runtime_meta / 'sample_data'}")

    legacy_csv = pred / "glam_legacy_cpu.csv"
    blackwell_csv = pred / "glam_blackwell_gpu.csv"

    legacy_meta = run_glam_legacy_cpu_reference(
        run_id=f"glam-legacy-cpu-{out.name}",
        image_dir=str(images),
        data_pickle=str(data_pkl),
        output_file=str(legacy_csv),
        preprocessed_dir=str(pre / "legacy_cpu"),
    )
    blackwell_meta = run_model(
        "glam",
        run_id=f"glam-blackwell-gpu-{out.name}",
        image_dir=str(images),
        data_pickle=str(data_pkl),
        output_file=str(blackwell_csv),
        preprocessed_dir=str(pre / "blackwell_gpu"),
        device="gpu",
    )

    legacy_metrics = evaluate_prediction_csv("glam", legacy_csv, data_pkl)
    blackwell_metrics = evaluate_prediction_csv("glam", blackwell_csv, data_pkl)
    legacy_reference = _compare("glam", legacy_metrics)
    blackwell_reference = _compare("glam", blackwell_metrics)

    ldf = pd.read_csv(legacy_csv)
    bdf = pd.read_csv(blackwell_csv)
    cols = ["image_index", "malignant_pred", "malignant_label"]
    merged = ldf[cols].rename(columns={
        "malignant_pred": "legacy_cpu_malignant_pred",
        "malignant_label": "legacy_label",
    }).merge(
        bdf[cols].rename(columns={
            "malignant_pred": "blackwell_gpu_malignant_pred",
            "malignant_label": "blackwell_label",
        }),
        on="image_index",
        how="outer",
        validate="one_to_one",
    )
    if merged[["legacy_cpu_malignant_pred", "blackwell_gpu_malignant_pred"]].isna().any().any():
        raise ValueError("Legacy/Blackwell GLAM image rows do not align one-to-one")
    if not (merged["legacy_label"].astype(int) == merged["blackwell_label"].astype(int)).all():
        raise ValueError("Legacy/Blackwell GLAM labels differ")

    merged["score_delta_blackwell_minus_legacy"] = (
        merged["blackwell_gpu_malignant_pred"].astype(float) - merged["legacy_cpu_malignant_pred"].astype(float)
    )
    merged["abs_score_delta"] = merged["score_delta_blackwell_minus_legacy"].abs()
    merged.to_csv(out / "glam_prediction_differential.csv", index=False)

    a = merged["legacy_cpu_malignant_pred"].astype(float).to_numpy()
    b = merged["blackwell_gpu_malignant_pred"].astype(float).to_numpy()
    pearson = float(pd.Series(a).corr(pd.Series(b), method="pearson"))
    spearman = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    order_disagreements, comparable_pairs = _pairwise_order_disagreements(a, b)

    metric_rows = []
    for runtime_name, m, c in (
        ("legacy_cpu_torch1.1", legacy_metrics, legacy_reference),
        ("blackwell_gpu_torch2.7", blackwell_metrics, blackwell_reference),
    ):
        metric_rows.append({
            "runtime": runtime_name,
            "image_roc_auc": m["image_roc_auc"],
            "image_auprc": m["image_auprc"],
            "breast_roc_auc": m["breast_roc_auc"],
            "breast_auprc": m["breast_auprc"],
            "reference_metrics_match": c["reference_metrics_match"],
        })
    pd.DataFrame(metric_rows).to_csv(out / "glam_runtime_metrics.csv", index=False)

    legacy_match = bool(legacy_reference["reference_metrics_match"])
    blackwell_match = bool(blackwell_reference["reference_metrics_match"])
    if legacy_match and not blackwell_match:
        decision = "BLACKWELL_COMPATIBILITY_MISMATCH"
    elif legacy_match and blackwell_match:
        decision = "BOTH_RUNTIMES_REPRODUCE_REFERENCE"
    elif not legacy_match and not blackwell_match and order_disagreements == 0 and float(np.max(np.abs(a - b))) <= 0.0015:
        decision = "REFERENCE_DRIFT_OR_PUBLISHED_ROUNDING_MISMATCH"
    elif not legacy_match and not blackwell_match:
        decision = "LEGACY_AND_BLACKWELL_BOTH_MISS_REFERENCE"
    else:
        decision = "BLACKWELL_MATCHES_BUT_LEGACY_DOES_NOT"

    summary = {
        "reference_source": REFERENCE_SOURCE,
        "sample_exams": 4,
        "sample_images": int(len(merged)),
        "legacy_runtime": {
            "framework": "PyTorch 1.1.0",
            "device": "CPU",
            "reference_metrics_match": legacy_match,
            "metrics": legacy_metrics,
            "run_metadata": legacy_meta,
        },
        "blackwell_runtime": {
            "framework": "PyTorch 2.7.1 / CUDA 12.8",
            "device": "GPU",
            "reference_metrics_match": blackwell_match,
            "metrics": blackwell_metrics,
            "run_metadata": blackwell_meta,
        },
        "prediction_differential": {
            "pearson": pearson,
            "spearman": spearman,
            "mean_absolute_score_delta": float(merged["abs_score_delta"].mean()),
            "max_absolute_score_delta": float(merged["abs_score_delta"].max()),
            "pairwise_order_disagreements": int(order_disagreements),
            "comparable_pairs": int(comparable_pairs),
            "pairwise_order_disagreement_rate": (float(order_disagreements / comparable_pairs) if comparable_pairs else None),
        },
        "decision": decision,
        "interpretation": {
            "primary_question": "Does the Blackwell compatibility runtime preserve the original GLAM PyTorch 1.1 sample ordering?",
            "legacy_headless_patch": "TkAgg -> Agg only; no model/checkpoint/inference-semantic change intended.",
            "next_if_blackwell_mismatch": "Instrument checkpoint loading and exact forward-boundary/intermediate tensors before using GLAM in formal ensemble freeze.",
        },
        "research_guards": {
            "diagnostic_only": True,
            "eligible_for_freeze": False,
            "cbis_ddsm_used": False,
            "ensemble_weights_changed": False,
            "threshold_changed": False,
            "training_performed": False,
        },
    }
    (out / "glam_runtime_differential_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# GLAM Runtime Differential",
        "",
        "> Diagnostic comparison of the original upstream PyTorch 1.1 CPU runtime against the Blackwell PyTorch 2.7/CUDA 12.8 compatibility runtime on the same official 4-exam sample.",
        "",
        f"- **decision**: {decision}",
        f"- **legacy reference match**: {legacy_match}",
        f"- **Blackwell reference match**: {blackwell_match}",
        f"- **Pearson / Spearman score correlation**: {pearson:.6f} / {spearman:.6f}",
        f"- **mean / max absolute score delta**: {merged['abs_score_delta'].mean():.6f} / {merged['abs_score_delta'].max():.6f}",
        f"- **pairwise ordering disagreements**: {order_disagreements}/{comparable_pairs}",
        "",
        "## Metrics",
        "",
        "| runtime | image AUROC | image AUPRC | breast AUROC | breast AUPRC | upstream match |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in metric_rows:
        report.append(
            f"| {row['runtime']} | {row['image_roc_auc']:.6f} | {row['image_auprc']:.6f} | "
            f"{row['breast_roc_auc']:.6f} | {row['breast_auprc']:.6f} | {row['reference_metrics_match']} |"
        )
    report += [
        "",
        "## Interpretation",
        "",
        "- If legacy PyTorch 1.1 matches the published reference and Blackwell does not, the compatibility runtime is the defect boundary.",
        "- If both runtimes miss the published reference but agree closely with one another, investigate reference/sample drift before changing model code.",
        "- Do not interpret a numerically higher AUROC as better reproduction; this diagnostic requires matching upstream score ordering.",
        "- Results are diagnostic only and cannot freeze ensemble weights, thresholds, or GLAM inclusion.",
    ]
    (out / "glam_runtime_differential_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out
