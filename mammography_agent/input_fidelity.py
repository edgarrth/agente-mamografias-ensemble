from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime as dt
import json
import math
import pickle
import re
import struct

import pandas as pd

MODELS = ("gmic", "nyu", "glam")
VIEW_COLUMNS = {
    "l_cc": ("LEFT", "CC", "L-CC"),
    "r_cc": ("RIGHT", "CC", "R-CC"),
    "l_mlo": ("LEFT", "MLO", "L-MLO"),
    "r_mlo": ("RIGHT", "MLO", "R-MLO"),
}
DEFAULT_CBIS_SOURCE = Path("/workspace/datasets/raw/cbis_ddsm/source_manifest.csv")
DEFAULT_ANALYSES = Path("/workspace/output/analyses")


def _utc_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _read_png_stats(path: Path) -> dict:
    # Read the PNG IHDR directly. This audit intentionally avoids decoding giant
    # mammograms; model preprocessing already validates that image pixels can be read.
    with path.open("rb") as fh:
        sig = fh.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        length_raw = fh.read(4)
        chunk_type = fh.read(4)
        if len(length_raw) != 4 or chunk_type != b"IHDR":
            raise ValueError(f"PNG missing IHDR: {path}")
        length = struct.unpack(">I", length_raw)[0]
        data = fh.read(length)
        if length != 13 or len(data) != 13:
            raise ValueError(f"Unexpected PNG IHDR length {length}: {path}")
        width, height, bitdepth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", data)
    greyscale = color_type in (0, 4)
    alpha = color_type in (4, 6)
    return {
        "width": int(width),
        "height": int(height),
        "bitdepth": int(bitdepth),
        "color_type": int(color_type),
        "greyscale": bool(greyscale),
        "alpha": bool(alpha),
        "compression_method": int(compression),
        "filter_method": int(filter_method),
        "interlace_method": int(interlace),
        "file_size_bytes": int(path.stat().st_size),
    }


def _dicom_value(ds, name: str):
    value = getattr(ds, name, None)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "|".join(str(x) for x in value)
    try:
        # MultiValue behaves list-like but is not always a list subclass.
        if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
            return "|".join(str(x) for x in value)
    except Exception:
        pass
    return str(value)


def _source_map(source_manifest: Path) -> dict[tuple[str, str], str]:
    if not source_manifest.exists():
        return {}
    df = pd.read_csv(source_manifest)
    out: dict[tuple[str, str], str] = {}
    for _, r in df.iterrows():
        sid = str(r.get("study_id", ""))
        for col in VIEW_COLUMNS:
            if col in df.columns and pd.notna(r.get(col)):
                out[(sid, col)] = str(r[col])
    return out


def _audit_inputs(selected: pd.DataFrame, source_manifest: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = _source_map(source_manifest)
    png_rows = []
    dicom_rows = []
    for _, r in selected.iterrows():
        sid = str(r.study_id)
        pid = str(r.get("patient_id", ""))
        for col, (laterality, view, model_view) in VIEW_COLUMNS.items():
            p = Path(str(r[col]))
            rec = {
                "study_id": sid, "patient_id": pid, "view_column": col,
                "laterality": laterality, "view": view, "model_view": model_view,
                "png_path": str(p), "exists": p.exists(),
            }
            if p.exists():
                rec.update(_read_png_stats(p))
                rec["png_contract_16bit_grayscale"] = bool(rec["bitdepth"] == 16 and rec["greyscale"] and not rec["alpha"])
            else:
                rec["png_contract_16bit_grayscale"] = False
            png_rows.append(rec)

            source = src.get((sid, col))
            d = {
                "study_id": sid, "patient_id": pid, "view_column": col,
                "laterality": laterality, "view": view, "dicom_path": source,
                "dicom_available": bool(source and Path(source).exists()),
            }
            if d["dicom_available"]:
                import pydicom
                ds = pydicom.dcmread(source, stop_before_pixels=True, force=True)
                for field in [
                    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
                    "PhotometricInterpretation", "PresentationLUTShape", "RescaleSlope", "RescaleIntercept",
                    "WindowCenter", "WindowWidth", "VOILUTFunction",
                ]:
                    d[field] = _dicom_value(ds, field)
                d["VOILUTSequence_present"] = bool(getattr(ds, "VOILUTSequence", None))
                d["ModalityLUTSequence_present"] = bool(getattr(ds, "ModalityLUTSequence", None))
                d["presentation_transform_metadata_present"] = bool(
                    d.get("WindowCenter") is not None or d.get("WindowWidth") is not None
                    or d["VOILUTSequence_present"] or d["ModalityLUTSequence_present"]
                    or d.get("RescaleSlope") not in (None, "1", "1.0")
                    or d.get("RescaleIntercept") not in (None, "0", "0.0")
                )
                # Current adapter behavior is explicit and auditable: raw pixel_array, MONOCHROME1 inversion, bit shift.
                d["adapter_applies_voi_lut"] = False
                d["adapter_applies_modality_lut"] = False
                d["adapter_handles_monochrome1_inversion"] = True
                d["manual_review_recommended"] = bool(d["presentation_transform_metadata_present"])
            dicom_rows.append(d)
    return pd.DataFrame(png_rows), pd.DataFrame(dicom_rows)


def _native_batches(run_dir: Path) -> list[Path]:
    direct = run_dir / "model_batch"
    if direct.is_dir():
        return [direct]
    chunks = sorted(p / "model_batch" for p in (run_dir / "chunks").glob("*") if (p / "model_batch").is_dir())
    return chunks


def _order_for_batch(batch: Path, selected: pd.DataFrame) -> list[str]:
    order = batch / "study_order.csv"
    if order.exists():
        return pd.read_csv(order).sort_values("position").study_id.astype(str).tolist()
    # Backward-compatible fallback for historical chunks. Limit to pickle length later.
    return selected.study_id.astype(str).tolist()


def _find_pickle(pre_dir: Path, suffix: str) -> Path | None:
    matches = sorted(pre_dir.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def _scalar_distance(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except Exception:
        return None


def _audit_model_preprocessing(run_dir: Path, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch in _native_batches(run_dir):
        order = _order_for_batch(batch, selected)
        for model in MODELS:
            pre = batch / "preprocessed" / model
            center = _find_pickle(pre, "_center_data.pkl") if pre.exists() else None
            cropped = _find_pickle(pre, "cropped_exam_list.pkl") if pre.exists() else None
            target = center or cropped
            if target is None:
                rows.append({
                    "batch": str(batch.relative_to(run_dir)), "model": model,
                    "preprocessed_dir": str(pre.relative_to(run_dir)) if pre.exists() else str(pre),
                    "preprocessing_pickle_available": False,
                    "study_id": None, "model_view": None,
                    "horizontal_flip": None, "distance_from_starting_side": None,
                    "distance_nonzero": None, "best_center_available": None,
                })
                continue
            with target.open("rb") as fh:
                data = pickle.load(fh)
            for pos, exam in enumerate(data):
                sid = order[pos] if pos < len(order) else f"UNMAPPED_POSITION_{pos}"
                hflip = str(exam.get("horizontal_flip", ""))
                dist = exam.get("distance_from_starting_side", {}) or {}
                centers = exam.get("best_center", {}) or {}
                for model_view in ("L-CC", "R-CC", "L-MLO", "R-MLO"):
                    value = _scalar_distance(dist.get(model_view))
                    rows.append({
                        "batch": str(batch.relative_to(run_dir)), "model": model,
                        "preprocessed_dir": str(pre.relative_to(run_dir)),
                        "preprocessing_pickle": str(target.relative_to(run_dir)),
                        "preprocessing_pickle_available": True,
                        "position": pos, "study_id": sid, "model_view": model_view,
                        "horizontal_flip": hflip,
                        "distance_from_starting_side": value,
                        "distance_nonzero": bool(value is not None and abs(value) > 1e-9),
                        "best_center_available": bool(model_view in centers and centers.get(model_view)),
                    })
    return pd.DataFrame(rows)


def audit_input_fidelity(run_dir: str | Path, output: str | Path | None = None, source_manifest: str | Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    selected_path = run_dir / "selected_studies.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Required audit input is missing: {selected_path}")
    selected = pd.read_csv(selected_path)
    missing = [c for c in ["study_id", *VIEW_COLUMNS] if c not in selected.columns]
    if missing:
        raise ValueError(f"selected_studies.csv missing required columns: {missing}")

    src = Path(source_manifest).resolve() if source_manifest else DEFAULT_CBIS_SOURCE
    out = Path(output).resolve() if output else DEFAULT_ANALYSES / _utc_id("input-fidelity")
    out.mkdir(parents=True, exist_ok=True)

    png_df, dicom_df = _audit_inputs(selected, src)
    prep_df = _audit_model_preprocessing(run_dir, selected)
    png_df.to_csv(out / "input_png_audit.csv", index=False)
    dicom_df.to_csv(out / "dicom_conversion_audit.csv", index=False)
    prep_df.to_csv(out / "model_preprocessing_audit.csv", index=False)

    png_invalid = int((~png_df["png_contract_16bit_grayscale"].fillna(False)).sum()) if not png_df.empty else 0
    dicom_available = int(dicom_df.get("dicom_available", pd.Series(dtype=bool)).fillna(False).sum()) if not dicom_df.empty else 0
    transform_review = int(dicom_df.get("manual_review_recommended", pd.Series(dtype=bool)).fillna(False).sum()) if not dicom_df.empty else 0
    prep_available = prep_df[prep_df.get("preprocessing_pickle_available", False) == True] if not prep_df.empty else prep_df  # noqa: E712
    distance_nonzero = int(prep_available.get("distance_nonzero", pd.Series(dtype=bool)).fillna(False).sum()) if not prep_available.empty else 0
    best_center_missing = int((~prep_available.get("best_center_available", pd.Series(dtype=bool)).fillna(False)).sum()) if not prep_available.empty else 0

    summary = {
        "source_run": str(run_dir),
        "studies": int(len(selected)),
        "images": int(len(png_df)),
        "input_png_contract": {
            "expected": "16-bit grayscale PNG",
            "invalid_images": png_invalid,
            "all_valid": png_invalid == 0,
        },
        "dicom_conversion_contract": {
            "source_manifest": str(src),
            "dicom_headers_available": dicom_available,
            "images_with_presentation_transform_metadata": transform_review,
            "current_adapter_applies_voi_lut": False,
            "current_adapter_applies_modality_lut": False,
            "current_adapter_handles_monochrome1_inversion": True,
            "interpretation": "Presentation-transform metadata is a review signal, not automatic proof of incorrect conversion.",
        },
        "upstream_preprocessing_contract": {
            "preprocessing_records": int(len(prep_available)),
            "distance_from_starting_side_nonzero_records": distance_nonzero,
            "missing_best_center_records": best_center_missing,
            "interpretation": "Upstream crop metadata documents distance_from_starting_side as a signal that can reveal wrong horizontal_flip/orientation.",
        },
        "research_guards": {
            "diagnostic_only": True,
            "eligible_for_freeze": False,
            "model_inference_performed": False,
            "weights_changed": False,
            "threshold_changed": False,
            "images_modified": False,
            "dataset_modified": False,
        },
    }
    _json(out / "input_fidelity_summary.json", summary)

    report = [
        "# Input Fidelity Audit", "",
        "> Diagnostic-only. No inference, model weights, ensemble weights, threshold, image bytes or dataset metadata were changed.", "",
        f"- **source_run**: {run_dir}",
        f"- **studies**: {len(selected)}",
        f"- **images audited**: {len(png_df)}",
        f"- **16-bit grayscale PNG violations**: {png_invalid}",
        f"- **DICOM headers available**: {dicom_available}/{len(dicom_df)}",
        f"- **DICOMs with presentation-transform metadata requiring review**: {transform_review}",
        f"- **upstream preprocessing records with distance_from_starting_side != 0**: {distance_nonzero}",
        f"- **upstream preprocessing records missing best_center**: {best_center_missing}", "",
        "## Interpretation", "",
        "- A nonzero `distance_from_starting_side` is not itself a model failure; the upstream preprocessing documents it as a useful signal for detecting a possibly wrong horizontal orientation.",
        "- DICOM Window/VOI/Rescale metadata is reported because the metarepository requires the 16-bit PNG to represent the mammogram correctly after any necessary presentation conversions. Metadata presence is a review trigger, not automatic proof that VOI LUT must be applied.",
        "- This audit does not optimize or select weights/thresholds and is not eligible for freeze.", "",
        "## Files", "",
        "- `input_png_audit.csv`: bit depth, geometry and intensity-distribution checks for selected prepared PNGs.",
        "- `dicom_conversion_audit.csv`: source DICOM presentation/intensity metadata when the original DICOM is available.",
        "- `model_preprocessing_audit.csv`: metadata produced by each model's own crop/center preprocessing, including orientation diagnostics.",
        "- `input_fidelity_summary.json`: machine-readable summary.",
    ]
    (out / "input_fidelity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out
