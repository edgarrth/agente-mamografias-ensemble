from __future__ import annotations

from pathlib import Path
import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from .workspace import safe_workspace_path

VIEW_COLUMNS = {
    "l_cc": "L_CC",
    "r_cc": "R_CC",
    "l_mlo": "L_MLO",
    "r_mlo": "R_MLO",
}
WORKSPACE_ROOT = Path("/workspace")


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _first_number(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        # pydicom MultiValue and ordinary list/tuple both support indexing.
        if not isinstance(value, (str, bytes)) and hasattr(value, "__len__") and len(value) > 0:
            value = value[0]
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _detect_dataset_source(selected: pd.DataFrame) -> str:
    if "dataset_source" in selected.columns:
        values = sorted({str(v).strip().lower() for v in selected["dataset_source"].dropna() if str(v).strip()})
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            raise ValueError(f"DICOM presentation counterfactual requires one dataset per run; found {values}")
    ids = [str(v).strip().upper() for v in selected.get("study_id", pd.Series(dtype=str)).dropna()]
    if ids and all(v.startswith("CMMD_") for v in ids):
        return "cmmd"
    if ids and all(v.startswith("CBIS-DDSM_") or v.startswith("CBIS_") for v in ids):
        return "cbis_ddsm"
    return "selected_dataset"


def _resolve_source_manifest(dataset_source: str, source_manifest: str | Path | None) -> Path:
    if source_manifest:
        path = Path(source_manifest).resolve()
    else:
        path = WORKSPACE_ROOT / "datasets" / "raw" / dataset_source / "source_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Source manifest not found: {path}. Pass --source-manifest explicitly if this dataset stores it elsewhere."
        )
    return path


def _selected_dicom_records(selected: pd.DataFrame, source_manifest: Path) -> list[dict]:
    source = pd.read_csv(source_manifest)
    required = {"study_id", *VIEW_COLUMNS.keys()}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"source_manifest.csv missing required columns: {missing}")

    source = source.copy()
    source["study_id"] = source["study_id"].astype(str)
    lookup = source.set_index("study_id", drop=False)
    records: list[dict] = []
    for _, row in selected.iterrows():
        sid = str(row["study_id"])
        if sid not in lookup.index:
            raise KeyError(f"Selected study {sid} is absent from {source_manifest}")
        src = lookup.loc[sid]
        if isinstance(src, pd.DataFrame):
            raise ValueError(f"Duplicate study_id {sid} in {source_manifest}")
        for col, view in VIEW_COLUMNS.items():
            raw_path = safe_workspace_path(str(src[col]))
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            records.append({"study_id": sid, "view_column": col, "view": view, "dicom_path": raw_path})
    return records


def _native_range(ds) -> tuple[float, float]:
    bits = int(getattr(ds, "BitsStored", 16) or 16)
    signed = int(getattr(ds, "PixelRepresentation", 0) or 0) == 1
    if signed:
        return float(-(2 ** (bits - 1))), float(2 ** (bits - 1) - 1)
    return 0.0, float(2**bits - 1)


def _modality_range(ds) -> tuple[float, float]:
    lo, hi = _native_range(ds)
    slope = _first_number(getattr(ds, "RescaleSlope", None), 1.0) or 1.0
    intercept = _first_number(getattr(ds, "RescaleIntercept", None), 0.0) or 0.0
    a, b = lo * slope + intercept, hi * slope + intercept
    return (min(a, b), max(a, b))


def _scale_to_u16(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite values produced by DICOM presentation transform")
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(data.min()), float(data.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint16)
    scaled = (np.clip(data, lo, hi) - lo) / (hi - lo)
    return np.rint(scaled * 65535.0).astype(np.uint16)


def _presentation_invert_if_needed(arr: np.ndarray, ds) -> np.ndarray:
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        return np.uint16(65535) - arr
    return arr


def _current_adapter_u16(raw: np.ndarray, ds) -> np.ndarray:
    """Mirror ManifestDatasetAdapter._convert_to_png semantics exactly."""
    arr = np.asarray(raw)
    if arr.min() < 0:
        raise ValueError("Current adapter rejects signed DICOM pixels")
    bits = int(getattr(ds, "BitsStored", 16) or 16)
    max_native = (1 << bits) - 1
    arr = np.clip(arr, 0, max_native).astype(np.uint16)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = np.uint16(max_native) - arr
    if bits < 16:
        arr = np.left_shift(arr, 16 - bits).astype(np.uint16)
    return arr


def _apply_modality(raw: np.ndarray, ds) -> np.ndarray:
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # pydicom < 3 compatibility
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    return np.asarray(apply_modality_lut(raw, ds))


def _apply_voi(modality: np.ndarray, ds) -> np.ndarray:
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom < 3 compatibility
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    # prefer_lut=True follows an explicit VOI LUT sequence when present and otherwise
    # falls back to WindowCenter/WindowWidth, including VOILUTFunction semantics.
    return np.asarray(apply_voi_lut(modality, ds, prefer_lut=True))


def _voi_output_range(ds, arr: np.ndarray) -> tuple[float, float]:
    seq = getattr(ds, "VOILUTSequence", None)
    if seq:
        try:
            descriptor = seq[0].LUTDescriptor
            bits = int(descriptor[2])
            return 0.0, float(2**bits - 1)
        except Exception:
            return float(np.min(arr)), float(np.max(arr))
    # pydicom windowing maps into the modality output range.
    return _modality_range(ds)


def _array_stats(arr: np.ndarray) -> dict:
    x = np.asarray(arr, dtype=np.float64) / 65535.0
    q = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "normalized_min": float(x.min()),
        "normalized_max": float(x.max()),
        "normalized_mean": float(x.mean()),
        "normalized_std": float(x.std()),
        "normalized_q01": float(q[0]),
        "normalized_q05": float(q[1]),
        "normalized_median": float(q[2]),
        "normalized_q95": float(q[3]),
        "normalized_q99": float(q[4]),
        "zero_fraction": float(np.mean(x == 0.0)),
        "dynamic_range": float(x.max() - x.min()),
    }


def _pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    af = np.asarray(a, dtype=np.float64) / 65535.0
    bf = np.asarray(b, dtype=np.float64) / 65535.0
    delta = bf - af
    exact = bool(np.array_equal(a, b))
    # Avoid expensive corrcoef copies for exact arrays and degenerate images.
    if exact:
        corr = 1.0
    else:
        av = af.ravel()
        bv = bf.ravel()
        astd, bstd = float(av.std()), float(bv.std())
        corr = float(np.corrcoef(av, bv)[0, 1]) if astd > 0 and bstd > 0 else None
    return {
        "exact_equal": exact,
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "root_mean_squared_difference": float(np.sqrt(np.mean(delta * delta))),
        "max_absolute_difference": float(np.max(np.abs(delta))),
        "pearson": corr,
    }


def _metadata_record(ds) -> dict:
    def text(name: str):
        value = getattr(ds, name, None)
        if value is None:
            return None
        try:
            if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
                return "|".join(str(v) for v in value)
        except Exception:
            pass
        return str(value)

    return {
        "BitsAllocated": text("BitsAllocated"),
        "BitsStored": text("BitsStored"),
        "PixelRepresentation": text("PixelRepresentation"),
        "PhotometricInterpretation": text("PhotometricInterpretation"),
        "RescaleSlope": text("RescaleSlope"),
        "RescaleIntercept": text("RescaleIntercept"),
        "WindowCenter": text("WindowCenter"),
        "WindowWidth": text("WindowWidth"),
        "VOILUTFunction": text("VOILUTFunction"),
        "VOILUTSequence_present": bool(getattr(ds, "VOILUTSequence", None)),
        "ModalityLUTSequence_present": bool(getattr(ds, "ModalityLUTSequence", None)),
    }


def _write_u16_png(path: Path, arr: np.ndarray) -> None:
    import png

    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr, dtype=np.uint16)
    with path.open("wb") as fh:
        png.Writer(width=a.shape[1], height=a.shape[0], greyscale=True, bitdepth=16).write(
            fh, (row.tolist() for row in a)
        )


def run_dicom_presentation_counterfactual(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    source_manifest: str | Path | None = None,
    write_images: bool = False,
) -> Path:
    """Compare current DICOM conversion with Modality and VOI presentation branches.

    This diagnostic is deliberately classifier-free and label-blind. It reads only the
    study/view identity from the selected run and the original DICOM paths from the raw
    source manifest. It never mutates raw/prepared dataset files.
    """
    import pydicom

    run_dir = Path(run_dir).resolve()
    selected_path = run_dir / "selected_studies.csv"
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    selected = pd.read_csv(selected_path)
    required = {"study_id", *VIEW_COLUMNS.keys()}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"selected_studies.csv missing required columns: {missing}")

    dataset_source = _detect_dataset_source(selected)
    manifest = _resolve_source_manifest(dataset_source, source_manifest)
    records = _selected_dicom_records(selected, manifest)

    out = Path(output_dir).resolve() if output_dir else WORKSPACE_ROOT / "output" / "analyses" / f"dicom-presentation-{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)

    branch_rows: list[dict] = []
    comparison_rows: list[dict] = []
    metadata_rows: list[dict] = []

    for rec in records:
        ds = pydicom.dcmread(rec["dicom_path"], force=True)
        raw = np.asarray(ds.pixel_array)
        current = _current_adapter_u16(raw, ds)

        modality_native = _apply_modality(raw, ds)
        slope = _first_number(getattr(ds, "RescaleSlope", None), 1.0) or 1.0
        intercept = _first_number(getattr(ds, "RescaleIntercept", None), 0.0) or 0.0
        if not getattr(ds, "ModalityLUTSequence", None) and abs(slope - 1.0) <= 1e-12 and abs(intercept) <= 1e-12:
            # Identity Modality stage: preserve the production adapter byte-for-byte,
            # including its historical left-shift convention for <16-bit pixels.
            modality = current.copy()
        else:
            mod_lo, mod_hi = _modality_range(ds)
            modality = _presentation_invert_if_needed(_scale_to_u16(modality_native, mod_lo, mod_hi), ds)

        has_voi = bool(
            getattr(ds, "VOILUTSequence", None)
            or getattr(ds, "WindowCenter", None) is not None
            or getattr(ds, "WindowWidth", None) is not None
        )
        if has_voi:
            voi_native = _apply_voi(modality_native, ds)
            voi_lo, voi_hi = _voi_output_range(ds, voi_native)
            voi = _presentation_invert_if_needed(_scale_to_u16(voi_native, voi_lo, voi_hi), ds)
        else:
            voi = modality.copy()

        arrays = {"current_adapter": current, "modality_lut": modality, "voi_presentation": voi}
        for branch, arr in arrays.items():
            row = {
                "study_id": rec["study_id"],
                "view": rec["view"],
                "dicom_path": str(rec["dicom_path"]),
                "branch": branch,
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                **_array_stats(arr),
            }
            branch_rows.append(row)
            if write_images:
                _write_u16_png(out / "transformed_images" / branch / f"{rec['study_id']}_{rec['view']}.png", arr)

        meta = {"study_id": rec["study_id"], "view": rec["view"], "dicom_path": str(rec["dicom_path"]), **_metadata_record(ds)}
        metadata_rows.append(meta)

        for candidate_name, candidate in (("modality_lut", modality), ("voi_presentation", voi)):
            comparison_rows.append({
                "study_id": rec["study_id"],
                "view": rec["view"],
                "candidate": candidate_name,
                **_pair_stats(current, candidate),
            })

    branches = pd.DataFrame(branch_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metadata = pd.DataFrame(metadata_rows)
    branches.to_csv(out / "presentation_branch_stats.csv", index=False)
    comparisons.to_csv(out / "presentation_pairwise_comparison.csv", index=False)
    metadata.to_csv(out / "presentation_dicom_metadata.csv", index=False)

    metric_cols = [
        "normalized_mean", "normalized_std", "normalized_q01", "normalized_median",
        "normalized_q99", "zero_fraction", "dynamic_range",
    ]
    group_rows = []
    for branch, group in branches.groupby("branch", sort=True):
        row = {"branch": branch, "images": int(len(group))}
        for col in metric_cols:
            row[f"median_{col}"] = float(group[col].median())
            row[f"mean_{col}"] = float(group[col].mean())
        group_rows.append(row)
    groups = pd.DataFrame(group_rows)
    groups.to_csv(out / "presentation_group_summary.csv", index=False)

    pair_summary = {}
    for candidate, group in comparisons.groupby("candidate", sort=True):
        pair_summary[candidate] = {
            "images": int(len(group)),
            "exact_equal_images": int(group.exact_equal.astype(bool).sum()),
            "all_exact_equal": bool(group.exact_equal.astype(bool).all()),
            "median_mean_absolute_difference": float(group.mean_absolute_difference.median()),
            "max_mean_absolute_difference": float(group.mean_absolute_difference.max()),
            "median_root_mean_squared_difference": float(group.root_mean_squared_difference.median()),
            "median_pearson": float(pd.to_numeric(group.pearson, errors="coerce").median()),
        }

    wc = int(metadata.WindowCenter.notna().sum()) if "WindowCenter" in metadata else 0
    ww = int(metadata.WindowWidth.notna().sum()) if "WindowWidth" in metadata else 0
    voi_seq = int(metadata.VOILUTSequence_present.astype(bool).sum()) if "VOILUTSequence_present" in metadata else 0
    modality_seq = int(metadata.ModalityLUTSequence_present.astype(bool).sum()) if "ModalityLUTSequence_present" in metadata else 0

    summary = {
        "source_run": str(run_dir),
        "dataset_source": dataset_source,
        "source_manifest": str(manifest),
        "studies": int(selected.study_id.astype(str).nunique()),
        "images": int(len(records)),
        "branches": ["current_adapter", "modality_lut", "voi_presentation"],
        "presentation_metadata": {
            "window_center_images": wc,
            "window_width_images": ww,
            "voi_lut_sequence_images": voi_seq,
            "modality_lut_sequence_images": modality_seq,
            "voi_function_counts": {str(k): int(v) for k, v in metadata.VOILUTFunction.fillna("<none>").value_counts().to_dict().items()},
        },
        "pairwise_vs_current": pair_summary,
        "interpretation": {
            "purpose": "Determine whether explicit DICOM Modality/VOI presentation transforms materially differ from the adapter's current raw-pixel conversion on already-inspected diagnostic images.",
            "decision_guard": "A visible/statistical difference is not by itself permission to change the adapter. No branch is selected by AUC or model score in this diagnostic.",
            "next_step": "If Modality is identical and VOI differs, document the presentation metadata and decide from DICOM/source semantics whether VOI belongs in the input contract; do not retune on labels.",
        },
        "research_guards": {
            "diagnostic_only": True,
            "eligible_for_freeze": False,
            "ground_truth_used": False,
            "model_scores_used": False,
            "classifier_inference_performed": False,
            "model_weights_changed": False,
            "ensemble_weights_changed": False,
            "threshold_changed": False,
            "raw_dataset_modified": False,
            "prepared_dataset_modified": False,
            "transformed_images_written": bool(write_images),
        },
    }
    _json(out / "dicom_presentation_summary.json", summary)

    group_by_name = groups.set_index("branch") if not groups.empty else pd.DataFrame()
    lines = [
        "# DICOM Presentation Counterfactual",
        "",
        "> Classifier-free, label-blind diagnostic. Raw/prepared dataset bytes, model weights, ensemble weights and thresholds are not changed.",
        "",
        f"- **source_run**: {run_dir}",
        f"- **dataset_source**: {dataset_source}",
        f"- **studies / images**: {summary['studies']} / {summary['images']}",
        f"- **WindowCenter / WindowWidth present**: {wc}/{len(metadata)} / {ww}/{len(metadata)}",
        f"- **VOI LUT sequence present**: {voi_seq}/{len(metadata)}",
        f"- **Modality LUT sequence present**: {modality_seq}/{len(metadata)}",
        "",
        "## Branch summaries",
        "",
        "| branch | images | median mean | median std | median q01 | median q50 | median q99 | median dynamic range | median zero fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in ["current_adapter", "modality_lut", "voi_presentation"]:
        if branch not in group_by_name.index:
            continue
        r = group_by_name.loc[branch]
        lines.append(
            f"| {branch} | {int(r['images'])} | {r['median_normalized_mean']:.6f} | {r['median_normalized_std']:.6f} | "
            f"{r['median_normalized_q01']:.6f} | {r['median_normalized_median']:.6f} | {r['median_normalized_q99']:.6f} | "
            f"{r['median_dynamic_range']:.6f} | {r['median_zero_fraction']:.6f} |"
        )
    lines += ["", "## Pairwise difference vs current adapter", ""]
    for candidate in ["modality_lut", "voi_presentation"]:
        p = pair_summary.get(candidate, {})
        lines += [
            f"### {candidate}",
            "",
            f"- exact equal images: **{p.get('exact_equal_images', 0)}/{p.get('images', 0)}**",
            f"- median normalized MAE: **{p.get('median_mean_absolute_difference', float('nan')):.8f}**",
            f"- max image normalized MAE: **{p.get('max_mean_absolute_difference', float('nan')):.8f}**",
            f"- median Pearson: **{p.get('median_pearson', float('nan')):.8f}**",
            "",
        ]
    lines += [
        "## Interpretation guard",
        "",
        "- The comparison does not use ground truth, model scores, AUC or classifier inference.",
        "- `current_adapter` mirrors the production DICOM-to-16-bit-PNG conversion.",
        "- `modality_lut` applies the DICOM Modality LUT/rescale stage before presentation inversion.",
        "- `voi_presentation` additionally applies the explicit VOI LUT or WindowCenter/WindowWidth + VOILUTFunction stage when present.",
        "- A difference does not automatically mean the current adapter is wrong and does not authorize selecting a branch by downstream AUC.",
        "",
        "## Files",
        "",
        "- `dicom_presentation_summary.json`",
        "- `presentation_dicom_metadata.csv`",
        "- `presentation_branch_stats.csv`",
        "- `presentation_group_summary.csv`",
        "- `presentation_pairwise_comparison.csv`",
    ]
    if write_images:
        lines.append("- `transformed_images/` (diagnostic copies only)")
    (out / "dicom_presentation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
