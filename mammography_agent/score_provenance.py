from __future__ import annotations

from pathlib import Path
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .prediction_parser import _study_from_image_index
from .config import WORKSPACE_ROOT

_IMAGE_SUFFIX = re.compile(r"[_-](?P<side>L|R)[_-](?P<view>CC|MLO)$", re.IGNORECASE)
_MODELS = ("gmic", "nyu", "glam")


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(y, errors="raise").astype(int)
    score = pd.to_numeric(score, errors="raise").astype(float)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, score))


def _resolve_run_dir(value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    p = p.resolve()
    root = WORKSPACE_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"run_dir must be inside workspace: {p}")
    if not p.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {p}")
    return p


def _parse_image_identity(image_index: str) -> tuple[str, str, str]:
    raw = str(image_index)
    stem = Path(raw).stem
    m = _IMAGE_SUFFIX.search(stem)
    if not m:
        raise ValueError(f"Cannot parse laterality/view from image_index={raw!r}")
    side = "LEFT" if m.group("side").upper() == "L" else "RIGHT"
    view = m.group("view").upper()
    study_key = _study_from_image_index(stem)
    return study_key, side, view


def _require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _reconstruct_study_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate build_batch() identity metadata from an ordered study_id frame."""
    study_ids = frame["study_id"].astype(str).tolist()
    study_keys = [re.sub(r"[^A-Za-z0-9_.-]", "_", sid) for sid in study_ids]
    if len(set(study_keys)) != len(study_keys):
        duplicates = pd.Series(study_keys)[pd.Series(study_keys).duplicated(keep=False)].tolist()
        raise ValueError(f"Cannot reconstruct study_order due to sanitized study_id collision: {duplicates[:10]}")
    return pd.DataFrame({"position": range(len(study_ids)), "study_id": study_ids, "study_key": study_keys})


def _read_context(run_dir: Path):
    selected_path = run_dir / "selected_studies.csv"
    raw_path = run_dir / "raw_model_predictions.csv"
    for p in (selected_path, raw_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required v0.24.2 audit input is missing: {p}")

    selected_all = pd.read_csv(selected_path, dtype={"study_id": str})
    raw = pd.read_csv(raw_path, dtype={"study_id": str})
    _require_columns(selected_all, {"study_id", "patient_id", "ground_truth", "left_ground_truth", "right_ground_truth"}, selected_path)
    _require_columns(raw, {"study_id", "gmic_score", "nyu_score", "glam_score"}, raw_path)

    if selected_all.study_id.duplicated().any() or raw.study_id.duplicated().any():
        raise ValueError("Duplicate study identity detected in selected_studies.csv or raw_model_predictions.csv")
    selected_ids = set(selected_all.study_id.astype(str))
    raw_ids = set(raw.study_id.astype(str))
    unknown = sorted(raw_ids - selected_ids)
    if unknown:
        raise ValueError(f"raw_model_predictions contains studies absent from selected_studies.csv: {unknown[:10]}")

    # A PARTIAL_TIME_LIMIT run may have selected more studies than it completed.
    selected = selected_all[selected_all.study_id.astype(str).isin(raw_ids)].copy()
    selected["ground_truth"] = pd.to_numeric(selected["ground_truth"], errors="raise").astype(int)
    selected["left_ground_truth"] = pd.to_numeric(selected["left_ground_truth"], errors="raise").astype(int)
    selected["right_ground_truth"] = pd.to_numeric(selected["right_ground_truth"], errors="raise").astype(int)

    context = {
        "selected_studies_manifest_count": int(len(selected_all)),
        "processed_studies_count": int(len(raw)),
        "partial_run_detected": bool(len(selected_all) != len(raw)),
    }
    return selected, raw, context


def _read_order_file(path: Path) -> pd.DataFrame:
    order = pd.read_csv(path, dtype={"study_id": str, "study_key": str})
    _require_columns(order, {"study_id", "study_key"}, path)
    if order.study_id.duplicated().any() or order.study_key.duplicated().any():
        raise ValueError(f"Duplicate study identity in {path}")
    return order.reset_index(drop=True)


def _order_for_batch(batch_dir: Path, run_dir: Path, processed_selected: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    order_path = batch_dir / "study_order.csv"
    if order_path.is_file():
        return _read_order_file(order_path), str(order_path.relative_to(run_dir)), False

    # In max_runtime_minutes/chunked mode _infer_three() writes a chunk-level
    # raw_model_predictions.csv. Its row order is the exact NYU batch order.
    local_raw = batch_dir.parent / "raw_model_predictions.csv"
    if local_raw.is_file():
        local = pd.read_csv(local_raw, dtype={"study_id": str})
        _require_columns(local, {"study_id"}, local_raw)
        return _reconstruct_study_order(local[["study_id"]]), str(local_raw.relative_to(run_dir)), True

    # Direct legacy runs without study_order.csv can use the top-level raw file,
    # whose row order is the original build_batch order.
    if batch_dir.parent == run_dir:
        top_raw = run_dir / "raw_model_predictions.csv"
        top = pd.read_csv(top_raw, dtype={"study_id": str})
        return _reconstruct_study_order(top[["study_id"]]), "raw_model_predictions.csv", True

    # Last safe fallback only when the batch unambiguously covers every processed study.
    return _reconstruct_study_order(processed_selected[["study_id"]]), "selected_studies.csv", True


def _batch_has_native_outputs(batch_dir: Path) -> bool:
    return all((batch_dir / f"{model}.csv").is_file() for model in _MODELS)


def _native_inventory(run_dir: Path) -> list[str]:
    found = []
    for p in sorted(run_dir.rglob("*.csv")):
        rel = p.relative_to(run_dir)
        if p.name in {"gmic.csv", "nyu.csv", "glam.csv", "study_order.csv", "raw_model_predictions.csv"}:
            found.append(str(rel))
    return found


def _discover_native_batches(run_dir: Path, selected: pd.DataFrame, raw: pd.DataFrame):
    """Discover native model output layout used by both direct and chunked runs.

    normal_test() writes native outputs in one of two layouts:
      direct:  <run>/model_batch/{gmic,nyu,glam}.csv
      chunked: <run>/chunks/<NNNN>/model_batch/{gmic,nyu,glam}.csv

    v0.24.0/0.24.1 only supported the first layout. v0.24.2 intentionally
    discovers the layout instead of assuming a fixed path.
    """
    direct = run_dir / "model_batch"
    chunk_batches = sorted(p for p in (run_dir / "chunks").glob("*/model_batch") if p.is_dir()) if (run_dir / "chunks").is_dir() else []

    batches: list[Path] = []
    layout: str | None = None
    if _batch_has_native_outputs(direct):
        batches = [direct]
        layout = "direct"
    else:
        complete_chunks = [p for p in chunk_batches if _batch_has_native_outputs(p)]
        if complete_chunks:
            batches = complete_chunks
            layout = "chunked"

    if not batches:
        inventory = _native_inventory(run_dir)
        inventory_text = ", ".join(inventory[:30]) if inventory else "<none>"
        raise FileNotFoundError(
            "No complete native model batch was found. Expected either "
            f"{direct}/{{gmic,nyu,glam}}.csv or chunks/*/model_batch/{{gmic,nyu,glam}}.csv. "
            f"Relevant CSV files discovered under run_dir: {inventory_text}"
        )

    raw_ids = set(raw.study_id.astype(str))
    seen: set[str] = set()
    contexts = []
    for batch_dir in batches:
        order, order_source, reconstructed = _order_for_batch(batch_dir, run_dir, selected)
        order_ids = set(order.study_id.astype(str))
        overlap = sorted(seen & order_ids)
        if overlap:
            raise ValueError(f"Native batches overlap on study IDs: {overlap[:10]}")
        unknown = sorted(order_ids - raw_ids)
        if unknown:
            raise ValueError(f"Native batch {batch_dir} contains studies absent from top-level raw_model_predictions.csv: {unknown[:10]}")
        seen.update(order_ids)
        contexts.append({
            "batch_dir": batch_dir,
            "order": order,
            "order_source": order_source,
            "order_reconstructed": reconstructed,
            "studies": int(len(order)),
        })

    missing = sorted(raw_ids - seen)
    if missing:
        raise ValueError(f"Native output batches do not cover all processed studies. Missing: {missing[:10]}")

    provenance = {
        "native_output_layout": layout,
        "native_batch_count": int(len(contexts)),
        "native_batches": [
            {
                "path": str(c["batch_dir"].relative_to(run_dir)),
                "studies": c["studies"],
                "study_order_source": c["order_source"],
                "study_order_reconstructed": c["order_reconstructed"],
            }
            for c in contexts
        ],
    }
    return contexts, provenance


def _load_image_model(model: str, batch_dir: Path, order: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    path = batch_dir / f"{model}.csv"
    df = pd.read_csv(path)
    _require_columns(df, {"image_index", "malignant_pred"}, path)
    records = []
    key_to_sid = dict(zip(order.study_key.astype(str), order.study_id.astype(str)))
    for row in df.itertuples(index=False):
        key, side, view = _parse_image_identity(getattr(row, "image_index"))
        if key not in key_to_sid:
            raise ValueError(f"{model} native output {path} contains unknown study_key={key!r}")
        records.append({
            "study_id": key_to_sid[key],
            "model": model,
            "native_level": "VIEW",
            "laterality": side,
            "view": view,
            "native_score": float(getattr(row, "malignant_pred")),
            "native_score_column": "malignant_pred",
            "pipeline_score_semantics": "malignancy score",
            "native_source": str(path.relative_to(run_dir)),
        })
    out = pd.DataFrame(records)
    expected = len(order) * 4
    if len(out) != expected:
        raise ValueError(f"{model} expected {expected} view-level scores in {path}, found {len(out)}")
    if out.duplicated(["study_id", "laterality", "view"]).any():
        raise ValueError(f"{model} contains duplicate study/laterality/view native outputs in {path}")
    return out


def _load_nyu(batch_dir: Path, order: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    path = batch_dir / "nyu.csv"
    df = pd.read_csv(path)
    _require_columns(df, {"left_malignant", "right_malignant"}, path)
    if len(df) != len(order):
        raise ValueError(f"NYU native output rows {len(df)} != study_order rows {len(order)} in {path}")
    records=[]
    for i, row in df.reset_index(drop=True).iterrows():
        sid = str(order.iloc[i].study_id)
        for side, col in (("LEFT", "left_malignant"), ("RIGHT", "right_malignant")):
            records.append({
                "study_id": sid,
                "model": "nyu",
                "native_level": "BREAST",
                "laterality": side,
                "view": "",
                "native_score": float(row[col]),
                "native_score_column": col,
                "pipeline_score_semantics": "malignancy score",
                "native_source": str(path.relative_to(run_dir)),
            })
    return pd.DataFrame(records)


def _load_all_native(run_dir: Path, batch_contexts: list[dict]) -> pd.DataFrame:
    pieces = []
    for ctx in batch_contexts:
        batch_dir, order = ctx["batch_dir"], ctx["order"]
        pieces.extend([
            _load_image_model("gmic", batch_dir, order, run_dir),
            _load_nyu(batch_dir, order, run_dir),
            _load_image_model("glam", batch_dir, order, run_dir),
        ])
    native = pd.concat(pieces, ignore_index=True)
    if native.duplicated(["study_id", "model", "laterality", "view"]).any():
        dup = native[native.duplicated(["study_id", "model", "laterality", "view"], keep=False)]
        raise ValueError(f"Duplicate native outputs after combining batches: {dup[['study_id','model','laterality','view']].head(10).to_dict('records')}")
    return native


def _build_breast_scores(native: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    context = selected[["study_id", "patient_id", "ground_truth", "left_ground_truth", "right_ground_truth"]].copy()
    records=[]
    for (sid, model, side), grp in native.groupby(["study_id", "model", "laterality"], sort=False):
        max_idx = grp["native_score"].astype(float).idxmax()
        peak = grp.loc[max_idx]
        score = float(grp["native_score"].max())
        records.append({
            "study_id": str(sid),
            "model": model,
            "laterality": side,
            "breast_score": score,
            "breast_aggregation": "max_native_score",
            "peak_view": str(peak["view"] or "BREAST_NATIVE"),
            "view_count": int(len(grp)),
        })
    out=pd.DataFrame(records).merge(context,on="study_id",how="left",validate="many_to_one")
    if out.patient_id.isna().any():
        raise ValueError("Breast-level audit could not map one or more study IDs to selected_studies.csv")
    out["breast_ground_truth"] = np.where(out.laterality.eq("LEFT"), out.left_ground_truth, out.right_ground_truth).astype(int)
    return out


def _build_study_reconstruction(breast: pd.DataFrame, raw: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    records=[]
    raw_idx=raw.set_index("study_id")
    selected_idx=selected.set_index("study_id")
    for (sid, model), grp in breast.groupby(["study_id","model"],sort=False):
        idx=grp.breast_score.astype(float).idxmax(); peak=grp.loc[idx]
        reconstructed=float(grp.breast_score.max())
        stored=float(raw_idx.loc[str(sid), f"{model}_score"])
        left_rows=grp.loc[grp.laterality.eq("LEFT"),"breast_score"]
        right_rows=grp.loc[grp.laterality.eq("RIGHT"),"breast_score"]
        if len(left_rows) != 1 or len(right_rows) != 1:
            raise ValueError(f"{model}/{sid}: expected exactly one LEFT and one RIGHT breast score")
        left=float(left_rows.iloc[0]); right=float(right_rows.iloc[0])
        sel=selected_idx.loc[str(sid)]
        malignant_sides=[]
        if int(sel.left_ground_truth)==1: malignant_sides.append("LEFT")
        if int(sel.right_ground_truth)==1: malignant_sides.append("RIGHT")
        records.append({
            "study_id": str(sid),
            "patient_id": str(sel.patient_id),
            "ground_truth": int(sel.ground_truth),
            "left_ground_truth": int(sel.left_ground_truth),
            "right_ground_truth": int(sel.right_ground_truth),
            "model": model,
            "left_breast_score": left,
            "right_breast_score": right,
            "reconstructed_study_score": reconstructed,
            "stored_study_score": stored,
            "absolute_difference": abs(reconstructed-stored),
            "reconstruction_match": bool(np.isclose(reconstructed,stored,rtol=1e-9,atol=1e-12)),
            "study_aggregation": "max(left_breast_score,right_breast_score)",
            "peak_laterality": str(peak.laterality),
            "peak_view": str(peak.peak_view),
            "malignant_lateralities": "+".join(malignant_sides) if malignant_sides else "NONE",
            "peak_side_matches_malignancy": (str(peak.laterality) in malignant_sides) if malignant_sides else None,
        })
    return pd.DataFrame(records)


def _model_summary(breast: pd.DataFrame, recon: pd.DataFrame) -> list[dict]:
    rows=[]
    for model in _MODELS:
        b=breast[breast.model.eq(model)].copy()
        r=recon[recon.model.eq(model)].copy()
        malignant=r[r.ground_truth.eq(1)]
        side_matches=malignant.peak_side_matches_malignancy.dropna()
        rows.append({
            "model": model,
            "breast_level_roc_auc": _safe_auc(b.breast_ground_truth, b.breast_score),
            "study_level_roc_auc": _safe_auc(r.ground_truth, r.stored_study_score),
            "study_reconstruction_all_match": bool(r.reconstruction_match.all()),
            "max_reconstruction_absolute_difference": float(r.absolute_difference.max()),
            "malignant_studies": int(len(malignant)),
            "malignant_peak_side_matches": int(side_matches.astype(bool).sum()) if len(side_matches) else 0,
            "malignant_peak_side_match_rate": float(side_matches.astype(bool).mean()) if len(side_matches) else None,
        })
    return rows


def audit_score_provenance(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    run_dir=_resolve_run_dir(run_dir)
    selected, raw, context_info=_read_context(run_dir)
    batch_contexts, batch_provenance=_discover_native_batches(run_dir, selected, raw)
    context_info.update(batch_provenance)

    native=_load_all_native(run_dir,batch_contexts)
    native=native.merge(selected[["study_id","patient_id","ground_truth","left_ground_truth","right_ground_truth"]],on="study_id",how="left",validate="many_to_one")
    if native.patient_id.isna().any():
        raise ValueError("Native audit contains study IDs that cannot be mapped to selected_studies.csv")
    breast=_build_breast_scores(native,selected)
    recon=_build_study_reconstruction(breast,raw,selected)
    summary_rows=_model_summary(breast,recon)

    if output_dir is None:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output=WORKSPACE_ROOT/"output"/"analyses"/f"score-provenance-{stamp}"
    else:
        output=Path(output_dir)
        if not output.is_absolute(): output=WORKSPACE_ROOT/output
    output.mkdir(parents=True,exist_ok=True)

    native.to_csv(output/"native_model_scores.csv",index=False)
    breast.to_csv(output/"breast_level_scores.csv",index=False)
    recon.to_csv(output/"study_score_reconstruction.csv",index=False)
    pd.DataFrame(summary_rows).to_csv(output/"model_provenance_metrics.csv",index=False)

    warnings=[]
    for row in summary_rows:
        if not row["study_reconstruction_all_match"]:
            warnings.append(f"{row['model']}: reconstructed study scores do not match raw_model_predictions.csv")
        auc=row["breast_level_roc_auc"]
        if auc is not None and auc < 0.5:
            warnings.append(f"{row['model']}: breast-level ROC-AUC is below 0.5 in this diagnostic set")
        rate=row["malignant_peak_side_match_rate"]
        if rate is not None and rate < 1.0:
            warnings.append(f"{row['model']}: highest-scoring breast is not the malignant breast in all malignant studies ({rate:.3f} match rate)")

    payload={
        "source_run": str(run_dir),
        "studies": int(len(selected)),
        "breasts": int(len(selected)*2),
        "native_scores": int(len(native)),
        "ground_truth_contract": {"BENIGN":0,"MALIGNANT":1},
        "audit_input_provenance": context_info,
        "pipeline_aggregation_contract": {
            "gmic": "view malignant_pred -> max per breast -> max across breasts",
            "nyu": "left_malignant/right_malignant -> max across breasts",
            "glam": "view malignant_pred -> max per breast -> max across breasts",
        },
        "score_semantics_contract": {
            "gmic_native_column": "malignant_pred",
            "nyu_native_columns": ["left_malignant","right_malignant"],
            "glam_native_column": "malignant_pred",
            "calibrated_clinical_probability_claimed": False,
            "score_inversion_performed": False,
        },
        "models": summary_rows,
        "warnings": warnings,
        "research_guards": {
            "diagnostic_only": True,
            "eligible_for_freeze": False,
            "model_inference_performed": False,
            "weights_changed": False,
            "threshold_changed": False,
            "aggregation_changed": False,
        },
    }
    (output/"score_provenance_summary.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")

    batch_lines=[]
    for b in context_info["native_batches"]:
        batch_lines.append(
            f"- `{b['path']}`: {b['studies']} studies; order source `{b['study_order_source']}`; reconstructed={b['study_order_reconstructed']}"
        )
    lines=[
        "# Score Provenance Audit",
        "",
        "> Diagnostic-only audit. No GPU inference, model weights, ensemble weights, threshold, calibration, score inversion or production aggregation were changed.",
        "",
        f"- **source_run**: {run_dir}",
        f"- **studies audited**: {len(selected)}",
        f"- **ground_truth**: 0=BENIGN, 1=MALIGNANT",
        f"- **native_output_layout**: {context_info['native_output_layout']}",
        f"- **native_batch_count**: {context_info['native_batch_count']}",
        f"- **partial_run_detected**: {context_info['partial_run_detected']}",
        "- **eligible_for_freeze**: False",
        "",
        "## Native batch provenance",
        "",
        *batch_lines,
        "",
        "## Current aggregation contract",
        "",
        "- GMIC: `malignant_pred` per view -> maximum within each breast -> maximum across breasts.",
        "- NYU: `left_malignant` and `right_malignant` -> maximum across breasts.",
        "- GLAM: `malignant_pred` per view -> maximum within each breast -> maximum across breasts.",
        "- Therefore the current pipeline does **not** average left/right breasts when creating the stored study score.",
        "",
        "## Model provenance metrics",
        "",
        "| model | breast ROC-AUC | study ROC-AUC | reconstruction | malignant peak-side match |",
        "|---|---:|---:|---|---:|",
    ]
    for r in summary_rows:
        b="NA" if r["breast_level_roc_auc"] is None else f"{r['breast_level_roc_auc']:.4f}"
        s="NA" if r["study_level_roc_auc"] is None else f"{r['study_level_roc_auc']:.4f}"
        m="NA" if r["malignant_peak_side_match_rate"] is None else f"{r['malignant_peak_side_match_rate']:.4f}"
        lines.append(f"| {r['model']} | {b} | {s} | {'MATCH' if r['study_reconstruction_all_match'] else 'MISMATCH'} | {m} |")
    lines += ["", "## Warnings", ""]
    if warnings:
        lines += [f"- {w}" for w in warnings]
    else:
        lines.append("- None")
    lines += [
        "",
        "## Files",
        "",
        "- `native_model_scores.csv`: original view/breast outputs with study/laterality mapping and exact source CSV path.",
        "- `breast_level_scores.csv`: reconstructed breast-level scores and breast ground truth.",
        "- `study_score_reconstruction.csv`: proof of the exact breast-to-study aggregation used by the current pipeline.",
        "- `model_provenance_metrics.csv`: breast-level vs study-level ROC-AUC and malignant-side alignment.",
        "- `score_provenance_summary.json`: machine-readable audit summary including direct/chunked source provenance.",
    ]
    (output/"score_provenance_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return output
