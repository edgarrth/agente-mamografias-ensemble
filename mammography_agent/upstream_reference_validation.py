from __future__ import annotations

import datetime as dt
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from .model_client import ensure_gpu_model, gpu_probe, run_model
from .config import WORKSPACE_ROOT

REFERENCE_SOURCE = "https://github.com/nyukat/mammography_metarepository#what-results-should-be-expected-on-the-sample-images-with-the-supported-models"
REFERENCE_NOTE = (
    "Upstream states these reproduction metrics are computed on only 4 exams and have high variance; "
    "they are used here only to validate runtime/preprocessing reproduction, not scientific performance."
)

EXPECTED = {
    "gmic": {
        "upstream_name": "nyu_gmic",
        "image_roc_auc": 0.867,
        "image_auprc": 0.851,
        "breast_roc_auc": 0.867,
        "breast_auprc": 0.850,
    },
    "nyu": {
        "upstream_name": "nyu_model",
        "image_roc_auc": None,
        "image_auprc": None,
        "breast_roc_auc": 0.867,
        "breast_auprc": 0.850,
    },
    "glam": {
        "upstream_name": "nyu_glam",
        "image_roc_auc": 0.700,
        "image_auprc": 0.451,
        "breast_roc_auc": 0.733,
        "breast_auprc": 0.461,
    },
}


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _metric(y: list[int], s: list[float]) -> tuple[float, float]:
    roc = float(roc_auc_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    pr = float(auc(recall, precision))
    return roc, pr


def _exam_id_from_image_index(value: object) -> str:
    text = str(value)
    m = re.match(r"^([^_]+)_", text)
    return m.group(1) if m else text


def _laterality_from_image_index(value: object) -> str:
    text = str(value).upper().replace("_", "-")
    if "-L-" in text or text.endswith("-L-CC") or text.endswith("-L-MLO"):
        return "LEFT"
    if "-R-" in text or text.endswith("-R-CC") or text.endswith("-R-MLO"):
        return "RIGHT"
    # canonical examples are e.g. 0_L-CC or 0_L_CC
    parts = re.split(r"[_-]", text)
    if "L" in parts:
        return "LEFT"
    if "R" in parts:
        return "RIGHT"
    raise ValueError(f"Cannot infer laterality from image_index={value!r}")


def _load_labels(data_pkl: Path) -> list[dict]:
    with data_pkl.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list):
        raise ValueError("Upstream sample data.pkl must contain a list of exam dictionaries")
    return data


def _labels_by_exam(data: list[dict]) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    for idx, exam in enumerate(data):
        c = exam.get("cancer_label", {})
        labels[str(idx)] = {
            "LEFT": int(c["left_malignant"]),
            "RIGHT": int(c["right_malignant"]),
        }
    return labels


def evaluate_prediction_csv(model: str, csv_path: Path, data_pkl: Path) -> dict:
    """Reproduce mammography_metarepository/evaluation/score.py semantics exactly."""
    model = model.lower()
    df = pd.read_csv(csv_path)
    data = _load_labels(data_pkl)
    result: dict[str, object] = {"model": model, "prediction_rows": int(len(df))}

    if model in {"gmic", "glam"}:
        required = {"image_index", "malignant_pred", "malignant_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{model} prediction CSV missing columns: {sorted(missing)}")
        # Upstream image-level score.py uses labels copied into the prediction CSV.
        image_auc, image_ap = _metric(
            df["malignant_label"].astype(int).tolist(),
            df["malignant_pred"].astype(float).tolist(),
        )

        # Upstream breast-level conversion from image predictions averages the available
        # L-CC/L-MLO scores and R-CC/R-MLO scores (not max). Preserve that exact contract.
        left_scores: list[float] = []
        right_scores: list[float] = []
        left_labels: list[int] = []
        right_labels: list[int] = []
        for exam in data:
            side_scores = {"L": [], "R": []}
            for view in ("L-CC", "L-MLO", "R-CC", "R-MLO"):
                names = exam.get(view, [])
                if not names:
                    continue
                match = df[df["image_index"].isin(names)]
                if match.empty:
                    raise ValueError(f"{model} output missing sample image {names[0]!r} for view {view}")
                side_scores[view[0]].append(float(match["malignant_pred"].iloc[0]))
            c = exam["cancer_label"]
            if side_scores["L"]:
                left_scores.append(float(np.mean(side_scores["L"])))
                left_labels.append(int(c["left_malignant"]))
            if side_scores["R"]:
                right_scores.append(float(np.mean(side_scores["R"])))
                right_labels.append(int(c["right_malignant"]))
        breast_scores = left_scores + right_scores
        breast_labels = left_labels + right_labels
        breast_auc, breast_ap = _metric(breast_labels, breast_scores)
        image_scores = df["malignant_pred"].astype(float).to_numpy()
        result.update({
            "image_roc_auc": image_auc,
            "image_auprc": image_ap,
            "breast_roc_auc": breast_auc,
            "breast_auprc": breast_ap,
            "breast_rows": int(len(breast_scores)),
            "malignancy_score_min": float(np.min(image_scores)),
            "malignancy_score_max": float(np.max(image_scores)),
            "malignancy_score_mean": float(np.mean(image_scores)),
        })
        return result

    if model == "nyu":
        required = {"left_malignant", "right_malignant"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"nyu prediction CSV missing columns: {sorted(missing)}")
        if len(df) != len(data):
            raise ValueError(f"NYU sample output row count {len(df)} != sample exam count {len(data)}")
        # Exact upstream ordering: all left predictions/labels, then all right.
        scores = df["left_malignant"].astype(float).tolist() + df["right_malignant"].astype(float).tolist()
        labels = [int(exam["cancer_label"]["left_malignant"]) for exam in data] + [
            int(exam["cancer_label"]["right_malignant"]) for exam in data
        ]
        breast_auc, breast_ap = _metric(labels, scores)
        score_arr = np.asarray(scores, dtype=float)
        result.update({
            "image_roc_auc": None,
            "image_auprc": None,
            "breast_roc_auc": breast_auc,
            "breast_auprc": breast_ap,
            "breast_rows": len(scores),
            "malignancy_score_min": float(np.min(score_arr)),
            "malignancy_score_max": float(np.max(score_arr)),
            "malignancy_score_mean": float(np.mean(score_arr)),
        })
        return result

    raise ValueError(f"Unknown model: {model}")


def _compare(model: str, observed: dict) -> dict:
    expected = EXPECTED[model]
    out: dict[str, object] = {
        "model": model,
        "upstream_model": expected["upstream_name"],
    }
    checked: list[bool] = []
    for key in ("image_roc_auc", "image_auprc", "breast_roc_auc", "breast_auprc"):
        exp = expected[key]
        obs = observed.get(key)
        out[f"expected_{key}"] = exp
        out[f"observed_{key}"] = obs
        if exp is None or obs is None:
            out[f"delta_{key}"] = None
            out[f"match_{key}"] = None
        else:
            delta = float(obs) - float(exp)
            # Upstream README reports three decimals. A 0.0015 tolerance accounts only for rounding.
            match = abs(delta) <= 0.0015
            out[f"delta_{key}"] = delta
            out[f"match_{key}"] = match
            checked.append(match)
    out["reference_metrics_match"] = bool(checked) and all(checked)
    return out


def validate_upstream_sample(output_dir: str | Path | None = None) -> Path:
    root = WORKSPACE_ROOT
    out = Path(output_dir) if output_dir else root / "output" / "analyses" / f"upstream-reference-{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = out / "predictions"
    pre_dir = out / "preprocessed"
    pred_dir.mkdir(exist_ok=True)
    pre_dir.mkdir(exist_ok=True)

    # ensure-gpu also materializes the metarepository clone in /workspace/runtime.
    runtime_meta = root / "runtime" / "mammography_metarepository"
    inference_meta: dict[str, object] = {}
    for model in ("gmic", "nyu", "glam"):
        ensure_gpu_model(model)
        probe = gpu_probe(model)
        inference_meta[model] = {"gpu_probe": probe}

    image_dir = runtime_meta / "sample_data" / "images"
    data_pkl = runtime_meta / "sample_data" / "data.pkl"
    if not image_dir.is_dir() or not data_pkl.is_file():
        raise FileNotFoundError(f"Official metarepository sample data missing under {runtime_meta / 'sample_data'}")

    observed_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for model in ("gmic", "nyu", "glam"):
        output_csv = pred_dir / f"{model}.csv"
        pre = pre_dir / model
        run_meta = run_model(
            model=model,
            run_id=f"upstream-reference-{model}-{out.name}",
            image_dir=str(image_dir),
            data_pickle=str(data_pkl),
            output_file=str(output_csv),
            preprocessed_dir=str(pre),
        )
        inference_meta[model]["run"] = run_meta
        observed = evaluate_prediction_csv(model, output_csv, data_pkl)
        observed_rows.append(observed)
        comparison_rows.append(_compare(model, observed))

    observed_df = pd.DataFrame(observed_rows)
    compare_df = pd.DataFrame(comparison_rows)
    observed_df.to_csv(out / "observed_reference_metrics.csv", index=False)
    compare_df.to_csv(out / "reference_metric_comparison.csv", index=False)

    all_match = bool(compare_df["reference_metrics_match"].all())
    registry = root / "models" / "metarepository.json"
    meta_info = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else {}
    summary = {
        "reference_source": REFERENCE_SOURCE,
        "reference_note": REFERENCE_NOTE,
        "sample_exams": 4,
        "sample_images": len(list(image_dir.glob("*.png"))),
        "metarepository": meta_info,
        "expected_metrics": EXPECTED,
        "all_models_reference_metrics_match": all_match,
        "models": comparison_rows,
        "research_guards": {
            "diagnostic_only": True,
            "eligible_for_freeze": False,
            "cbis_ddsm_used": False,
            "ensemble_weights_changed": False,
            "threshold_changed": False,
            "training_performed": False,
        },
    }
    (out / "upstream_reference_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Upstream Reference Validation",
        "",
        "> Runtime-reproduction diagnostic using the official 4-exam sample bundled by the NYU mammography metarepository.",
        "",
        f"- **reference_source**: {REFERENCE_SOURCE}",
        "- **sample_exams**: 4",
        f"- **sample_images**: {summary['sample_images']}",
        f"- **all_models_reference_metrics_match**: {all_match}",
        f"- **metarepository_commit**: {meta_info.get('resolved_commit')}",
        "",
        "## Comparison",
        "",
        "| model | metric | expected | observed | delta | match |",
        "|---|---|---:|---:|---:|---|",
    ]
    metric_labels = {
        "image_roc_auc": "image ROC-AUC", "image_auprc": "image AUPRC",
        "breast_roc_auc": "breast ROC-AUC", "breast_auprc": "breast AUPRC",
    }
    for row in comparison_rows:
        for key, label in metric_labels.items():
            exp = row[f"expected_{key}"]
            if exp is None:
                continue
            obs = row[f"observed_{key}"]
            delta = row[f"delta_{key}"]
            match = row[f"match_{key}"]
            lines.append(f"| {row['model']} | {label} | {exp:.3f} | {obs:.6f} | {delta:+.6f} | {match} |")
    lines += ["", "## Observed malignancy-score ranges", "", "| model | min | max | mean |", "|---|---:|---:|---:|"]
    for row in observed_rows:
        lines.append(f"| {row['model']} | {row['malignancy_score_min']:.6f} | {row['malignancy_score_max']:.6f} | {row['malignancy_score_mean']:.6f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- PASS means the score ordering reproduces the three-decimal AUROC/AUPRC references published for the 4-exam upstream sample.",
        "- A mismatch is evidence to investigate the Blackwell compatibility runtime before interpreting CBIS-DDSM performance as domain shift.",
        "- Matching the 4-exam reference is a runtime reproduction check, not evidence of clinical validity or generalization.",
        "- The upstream project explicitly warns that these sample metrics have high variance because they use only four exams.",
        "",
        "## Research guard",
        "",
        "- Diagnostic only; not eligible for ensemble freeze.",
        "- No CBIS-DDSM labels/scores are used by this validation.",
        "- No ensemble weight, threshold, model weight or training state is changed.",
    ]
    (out / "upstream_reference_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "inference_metadata.json").write_text(json.dumps(inference_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
