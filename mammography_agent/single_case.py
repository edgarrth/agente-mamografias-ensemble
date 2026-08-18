from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WORKSPACE_ROOT, load_yaml
from .datasets.adapters import ManifestDatasetAdapter
from .datasets.rsna import deterministic_selection_key
from .ensemble.soft_voting import vote
from .logging_utils import audit
from .object_storage import persist_result_json, persist_single_case
from .orientation_policy import POLICY_ID as ORIENTATION_POLICY_ID, resolve_orientation
from .pipeline import _infer_three
from .reporting import write_json
from .storage import save_run, save_web_inference

REQUIRED_VIEWS = ("L_CC", "R_CC", "L_MLO", "R_MLO")
REQUIRED_VIEW_SET = set(REQUIRED_VIEWS)
VIEW_TO_COLUMN = {"L_CC": "l_cc", "R_CC": "r_cc", "L_MLO": "l_mlo", "R_MLO": "r_mlo"}
VIEW_CODE_MAP = {
    "R-10242": "CC",          # DICOM/SNOMED legacy code: cranio-caudal
    "399162004": "CC",        # SNOMED CT: cranio-caudal
    "R-10226": "MLO",         # DICOM/SNOMED legacy code: medio-lateral oblique
    "399368009": "MLO",       # SNOMED CT: medio-lateral oblique
}


def _id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"web-{stamp}-{uuid.uuid4().hex[:8]}"


def _workspace_path(value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    p = p.resolve()
    root = WORKSPACE_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path outside workspace is forbidden: {p}")
    return p


def _safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())[:120]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return ""


def _norm_view_text(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", " ").replace("_", " ")
    collapsed = " ".join(text.split())
    if collapsed in {"CC", "CRANIO CAUDAL", "CRANIOCAUDAL"}:
        return "CC"
    if collapsed in {
        "MLO", "MEDIO LATERAL OBLIQUE", "MEDIOLATERAL OBLIQUE", "MEDIOLATERALOBLIQUE",
    }:
        return "MLO"
    if re.search(r"\bCC\b", collapsed):
        return "CC"
    if re.search(r"\bMLO\b", collapsed):
        return "MLO"
    if "CRANIO" in collapsed and "CAUD" in collapsed:
        return "CC"
    if "OBLIQUE" in collapsed and ("MEDIO" in collapsed or "LATERAL" in collapsed):
        return "MLO"
    return ""


def _dicom_text_candidates(ds: Any) -> list[tuple[str, object]]:
    """Conservative presentation metadata candidates; no clinical labels are inspected."""
    candidates: list[tuple[str, object]] = [
        ("ViewPosition", getattr(ds, "ViewPosition", "")),
        ("SeriesDescription", getattr(ds, "SeriesDescription", "")),
        ("ProtocolName", getattr(ds, "ProtocolName", "")),
        ("RequestedProcedureDescription", getattr(ds, "RequestedProcedureDescription", "")),
        ("StudyDescription", getattr(ds, "StudyDescription", "")),
        ("ImageComments", getattr(ds, "ImageComments", "")),
    ]
    image_type = getattr(ds, "ImageType", None)
    if image_type:
        try:
            candidates.append(("ImageType", " ".join(str(x) for x in image_type)))
        except Exception:
            candidates.append(("ImageType", str(image_type)))
    return candidates


def _view_from_dataset(ds: Any) -> tuple[str, str]:
    """Resolve CC/MLO from standard mammography tags and conservative descriptive fallbacks."""
    sequence = getattr(ds, "ViewCodeSequence", None)
    if sequence:
        try:
            item = sequence[0]
            code_value = str(getattr(item, "CodeValue", "") or "").strip()
            if code_value in VIEW_CODE_MAP:
                return VIEW_CODE_MAP[code_value], f"ViewCodeSequence:{code_value}"
            meaning = _norm_view_text(getattr(item, "CodeMeaning", ""))
            if meaning:
                return meaning, "ViewCodeSequence:CodeMeaning"
        except Exception:
            pass

    for source, value in _dicom_text_candidates(ds):
        resolved = _norm_view_text(value)
        if resolved:
            return resolved, source

    for sequence_name in ("PerformedProtocolCodeSequence", "ProcedureCodeSequence"):
        sequence = getattr(ds, sequence_name, None)
        if not sequence:
            continue
        try:
            for item in sequence:
                resolved = _norm_view_text(getattr(item, "CodeMeaning", ""))
                if resolved:
                    return resolved, f"{sequence_name}:CodeMeaning"
        except Exception:
            pass
    return "", "unresolved"


def _read_dicom_metadata(path: Path) -> dict[str, Any]:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required to inspect Web DICOM uploads") from exc

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as exc:
        raise ValueError(f"{path.name} is not a readable DICOM: {type(exc).__name__}: {exc}") from exc

    side = _norm_side(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "")))
    view, view_source = _view_from_dataset(ds)
    detected = f"{side}_{view}" if side and view else ""
    transfer_uid = ""
    try:
        transfer_uid = str(ds.file_meta.TransferSyntaxUID)
    except Exception:
        pass
    return {
        "path": str(path),
        "name": path.name,
        "sha256": _sha256(path),
        "patient_id": str(getattr(ds, "PatientID", "") or "").strip(),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "") or "").strip(),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "") or "").strip(),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "") or "").strip(),
        "modality": str(getattr(ds, "Modality", "") or "").strip(),
        "laterality": side,
        "view": view,
        "view_source": view_source,
        "detected_view": detected if detected in REQUIRED_VIEW_SET else None,
        "transfer_syntax_uid": transfer_uid,
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "") or ""),
    }


def _write_dicom_preview(source: Path, destination: Path, *, max_dimension: int = 900) -> None:
    """Create an 8-bit display preview. This representation is never used for model inference."""
    try:
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise RuntimeError("numpy and pydicom are required for DICOM previews") from exc

    ds = pydicom.dcmread(source)
    arr = np.asarray(ds.pixel_array)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{source.name}: expected a 2D mammography image; got shape {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"{source.name}: pixel data contains no finite values")
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        display = np.zeros(arr.shape, dtype=np.uint8)
    else:
        display = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            display = 1.0 - display
        display = np.rint(display * 255.0).astype(np.uint8)

    if max(display.shape) > max_dimension:
        stride = int(np.ceil(max(display.shape) / max_dimension))
        display = display[::stride, ::stride]

    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    height, width = int(display.shape[0]), int(display.shape[1])
    raw = b"".join(b"\x00" + row.tobytes() for row in display)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def create_dicom_previews(dicom_paths: list[str]) -> dict[str, Any]:
    """Prepare cached presentation-only previews for Web review of unresolved projections."""
    if not dicom_paths:
        return {"status": "READY", "previews": []}
    paths = [_workspace_path(value) for value in dicom_paths]
    preview_root = WORKSPACE_ROOT / "input" / "web_previews"
    previews: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        destination = preview_root / f"{digest[:24]}.png"
        error = None
        if not destination.exists():
            try:
                _write_dicom_preview(path, destination)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        previews.append({
            "name": path.name,
            "preview_path": str(destination) if destination.exists() else None,
            "sha256": digest,
            "error": error,
            "inference_input": False,
        })
    return {"status": "READY", "previews": previews}


def _normalized_override(value: object) -> str | None:
    text = str(value or "AUTO").strip().upper().replace("-", "_")
    if text in {"", "AUTO"}:
        return None
    if text == "IGNORE":
        return "IGNORE"
    if text not in REQUIRED_VIEW_SET:
        raise ValueError(f"Invalid view assignment {value!r}; expected AUTO/IGNORE or {sorted(REQUIRED_VIEW_SET)}")
    return text


def inspect_dicom_case(
    dicom_paths: list[str],
    view_assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect one Web mammography exam without reading cancer labels or executing models."""
    if len(dicom_paths) < 4:
        raise ValueError("At least four DICOM files are required for the four-view ensemble")
    paths = [_workspace_path(value) for value in dicom_paths]
    if len({str(p) for p in paths}) != len(paths):
        raise ValueError("Duplicate DICOM paths were submitted")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".dcm", ".dicom", ""}:
            raise ValueError(f"Unsupported Web input {path.name}; use DICOM")

    assignments = {str(k): _normalized_override(v) for k, v in (view_assignments or {}).items()}
    rows: list[dict[str, Any]] = []
    for path in paths:
        rec = _read_dicom_metadata(path)
        override = assignments.get(path.name)
        assigned = None if override == "IGNORE" else (override or rec.get("detected_view"))
        rec["override"] = override
        rec["assigned_view"] = assigned
        rec["selected"] = False
        rec["selected_as"] = None
        rows.append(rec)

    patient_ids = sorted({r["patient_id"] for r in rows if r["patient_id"]})
    study_uids = sorted({r["study_instance_uid"] for r in rows if r["study_instance_uid"]})
    errors: list[str] = []
    warnings: list[str] = []
    if len(patient_ids) > 1:
        errors.append(f"The upload contains multiple PatientID values ({len(patient_ids)}); submit one patient/exam")
    if len(study_uids) > 1:
        errors.append(f"The upload contains multiple StudyInstanceUID values ({len(study_uids)}); submit one exam")
    if not patient_ids:
        warnings.append("PatientID is absent; an internal pseudonymous case id will be generated")
    if not study_uids:
        warnings.append("StudyInstanceUID is absent; file hashes will be used for internal exam identity")
    non_mg = sorted({r["modality"] for r in rows if r["modality"] and r["modality"].upper() not in {"MG", "DX"}})
    if non_mg:
        warnings.append(f"Unexpected DICOM modality values detected: {non_mg}")

    candidates: dict[str, list[dict[str, Any]]] = {view: [] for view in REQUIRED_VIEWS}
    unresolved = []
    for row in rows:
        assigned = row.get("assigned_view")
        if assigned in REQUIRED_VIEW_SET:
            candidates[str(assigned)].append(row)
        elif row.get("override") != "IGNORE":
            unresolved.append(row["name"])

    identity_seed = patient_ids[0] if patient_ids else (study_uids[0] if study_uids else "WEB")
    selected: dict[str, dict[str, Any]] = {}
    for view in REQUIRED_VIEWS:
        group = candidates[view]
        if not group:
            continue
        group.sort(
            key=lambda r: deterministic_selection_key(
                identity_seed,
                view,
                r.get("sop_instance_uid") or r["sha256"],
            )
        )
        chosen = group[0]
        chosen["selected"] = True
        chosen["selected_as"] = view
        selected[view] = chosen
        if len(group) > 1:
            warnings.append(f"{view}: {len(group)} candidates; one was selected deterministically without labels/scores")

    missing = [view for view in REQUIRED_VIEWS if view not in selected]
    if missing:
        errors.append(f"Missing required views after DICOM metadata/manual assignments: {', '.join(missing)}")
    if unresolved:
        warnings.append("Some DICOMs have unresolved view metadata; assign them manually in Streamlit if needed")

    identity_payload = "|".join(
        [patient_ids[0] if patient_ids else "", study_uids[0] if study_uids else ""]
        + sorted(r["sha256"] for r in rows)
    )
    digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    patient_internal = f"WEBP_{digest[:12]}"
    study_internal = f"WEBS_{digest[:16]}"

    # Do not expose raw PatientID in the result. Keep only whether identity metadata was present.
    public_rows = []
    for row in rows:
        public_rows.append({
            key: row.get(key)
            for key in (
                "path", "name", "sha256", "modality", "laterality", "view", "view_source",
                "detected_view", "override", "assigned_view", "selected", "selected_as",
                "transfer_syntax_uid", "rows", "columns", "photometric",
            )
        })

    return {
        "status": "READY" if not errors else "NEEDS_ATTENTION",
        "ready": not errors,
        "required_views": list(REQUIRED_VIEWS),
        "selected_views": {
            view: {
                "name": rec["name"],
                "path": rec["path"],
                "sha256": rec["sha256"],
                "detected_view": rec.get("detected_view"),
                "view_source": rec.get("view_source"),
                "candidate_count": len(candidates[view]),
            }
            for view, rec in selected.items()
        },
        "files": public_rows,
        "unresolved_files": unresolved,
        "missing_views": missing,
        "warnings": warnings,
        "errors": errors,
        "identity": {
            "patient_id_present": bool(patient_ids),
            "study_instance_uid_present": bool(study_uids),
            "internal_patient_id": patient_internal,
            "internal_study_id": study_internal,
        },
        "ground_truth_received": False,
        "labels_used": False,
        "selection_policy": "DICOM_METADATA_THEN_DETERMINISTIC_LABEL_BLIND_SHA256_V1",
    }


def _convert_input_to_png(source: Path, destination: Path) -> None:
    """Reuse the canonical DICOM->16-bit PNG converter used by dataset adapters."""
    source = _workspace_path(source)
    destination = _workspace_path(destination)
    converter = ManifestDatasetAdapter("single_case_upload", {})
    converter._convert_to_png(source, destination)


def _build_uploaded_case(
    *,
    run_dir: Path,
    dicom_paths: list[str],
    view_assignments: dict[str, str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inspection = inspect_dicom_case(dicom_paths, view_assignments)
    if not inspection["ready"]:
        raise ValueError("; ".join(inspection["errors"]))

    canonical_dir = run_dir / "input" / "canonical"
    canonical: dict[str, str] = {}
    for view in REQUIRED_VIEWS:
        selected = inspection["selected_views"][view]
        destination = canonical_dir / f"{view}.png"
        _convert_input_to_png(Path(selected["path"]), destination)
        canonical[VIEW_TO_COLUMN[view]] = str(destination)
        selected["canonical_png"] = str(destination)

    identity = inspection["identity"]
    frame = pd.DataFrame([{
        "study_id": identity["internal_study_id"],
        "patient_id": identity["internal_patient_id"],
        "dataset_source": "rsna_web_dicom",
        "ground_truth": float("nan"),
        "left_ground_truth": float("nan"),
        "right_ground_truth": float("nan"),
        "horizontal_flip": "NO",
        **canonical,
    }])
    preparation = {
        "mode": "web_dicom_upload",
        "input_format": "DICOM only; no train.csv/ground truth",
        "uploaded_dicom_count": len(dicom_paths),
        "selected_view_count": 4,
        "required_views": list(REQUIRED_VIEWS),
        "selected_views": inspection["selected_views"],
        "files": inspection["files"],
        "warnings": inspection["warnings"],
        "selection_policy": inspection["selection_policy"],
        "ground_truth_received": False,
        "labels_used": False,
    }
    return frame, preparation


def _baseline_config() -> tuple[dict[str, float], float, float]:
    ensemble = load_yaml("ensemble.yaml")
    baseline = ensemble["baseline"]
    weights = {k: float(v) for k, v in baseline["weights"].items()}
    threshold = float(baseline["threshold"])
    discordance = float(ensemble.get("discordance", {}).get("range_threshold", 0.30))
    return weights, threshold, discordance


def web_ensemble_config() -> dict[str, Any]:
    """Read-only Web ensemble defaults. This never mutates batch configuration files."""
    weights, threshold, discordance = _baseline_config()
    return {
        "weights": weights,
        "threshold": threshold,
        "discordance_threshold": discordance,
        "source": "config/ensemble.yaml:baseline",
        "editable_fields": ["weights"],
        "batch_configuration_mutated": False,
    }


def _resolve_web_weights(override: dict[str, float] | None) -> tuple[dict[str, float], str]:
    baseline, _, _ = _baseline_config()
    if override is None:
        return baseline, "BASELINE"
    required = {"gmic", "nyu", "glam"}
    supplied = {str(k).strip().lower() for k in override}
    if supplied != required:
        raise ValueError("Web ensemble weights must contain exactly gmic, nyu and glam")
    weights = {str(k).strip().lower(): float(v) for k, v in override.items()}
    if any(v < 0.0 or v > 1.0 for v in weights.values()):
        raise ValueError("Web ensemble weights must be between 0 and 1")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Web ensemble weights must sum to 1")
    return weights, "WEB_OVERRIDE"




def _resolve_web_device(value: str | None) -> str:
    device = str(value or "cpu").strip().lower()
    if device not in {"cpu", "gpu"}:
        raise ValueError("Web inference device must be cpu or gpu")
    return device

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_web_progress(run_dir: Path, run_id: str, *, stage: str, state: str, message: str, models: dict[str, Any] | None = None, started_at: str | None = None, error: str | None = None) -> None:
    payload = {
        "run_id": run_id,
        "stage": stage,
        "state": state,
        "message": message,
        "models": models or {},
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if started_at:
        payload["started_at_utc"] = started_at
    if error:
        payload["error"] = error
    write_json(run_dir / "web_progress.json", payload)


def get_single_case_progress(run_id: str) -> dict[str, Any]:
    safe = _safe_token(run_id)
    if safe != run_id or not run_id.startswith("web-"):
        raise ValueError("Invalid Web run_id")
    path = WORKSPACE_ROOT / "output" / "single_cases" / run_id / "web_progress.json"
    if not path.exists():
        return {"run_id": run_id, "stage": "QUEUED", "state": "PENDING", "message": "La evaluación está iniciándose.", "models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def run_single_case(
    *,
    dicom_paths: list[str],
    view_assignments: dict[str, str] | None = None,
    ensemble_weights: dict[str, float] | None = None,
    inference_device: str = "cpu",
    request_run_id: str | None = None,
) -> dict[str, Any]:
    """Infer exactly one four-view mammography exam; never train and never consume labels."""
    run_id = str(request_run_id or _id())
    if _safe_token(run_id) != run_id or not run_id.startswith("web-"):
        raise ValueError("Invalid Web run_id")
    run_dir = WORKSPACE_ROOT / "output" / "single_cases" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    web_device = _resolve_web_device(inference_device)
    audit("WEB_SINGLE_CASE_STARTED", run_id=run_id, uploaded_dicoms=len(dicom_paths), inference_device=web_device)
    model_progress = {model: {"state": "PENDING"} for model in ("gmic", "nyu", "glam")}
    started_iso = started.isoformat()
    _write_web_progress(
        run_dir, run_id, stage="PREPARATION", state="RUNNING",
        message="Preparando las cuatro proyecciones mamográficas.", models=model_progress, started_at=started_iso,
    )

    try:
        df, preparation = _build_uploaded_case(
            run_dir=run_dir,
            dicom_paths=dicom_paths,
            view_assignments=view_assignments,
        )
        df.to_csv(run_dir / "input_study.csv", index=False)
        _write_web_progress(
            run_dir, run_id, stage="ORIENTATION", state="RUNNING",
            message="Aplicando la política de orientación del estudio.", models=model_progress, started_at=started_iso,
        )

        resolved = resolve_orientation(df, run_dir / "orientation_resolution", run_id)
        resolved.to_csv(run_dir / "resolved_study.csv", index=False)

        # Mechanical label blindness: the Web frame contains no clinical labels; keep
        # all label-shaped columns NaN immediately before the common model pipeline.
        inference_input = resolved.copy()
        for label_column in ("ground_truth", "left_ground_truth", "right_ground_truth"):
            inference_input[label_column] = float("nan")
        def _model_progress_callback(*, model: str, state: str, elapsed_seconds: float | None = None) -> None:
            item = dict(model_progress.get(model, {}))
            item["state"] = state
            if elapsed_seconds is not None:
                item["elapsed_seconds"] = float(elapsed_seconds)
            model_progress[model] = item
            label = {"gmic": "GMIC", "nyu": "NYU / DMV-CNN", "glam": "GLAM"}.get(model, model.upper())
            verb = "Ejecutando" if state == "RUNNING" else "Completado"
            _write_web_progress(
                run_dir, run_id, stage="MODELS", state="RUNNING",
                message=f"{verb}: {label}.", models=model_progress, started_at=started_iso,
            )

        scores = _infer_three(
            inference_input, run_dir, run_id, device=web_device,
            web_label_blind_compat=True, progress_callback=_model_progress_callback,
        )
        if len(scores) != 1:
            raise RuntimeError(f"Expected one prediction row; got {len(scores)}")

        _, threshold, discordance_threshold = _baseline_config()
        weights, weights_source = _resolve_web_weights(ensemble_weights)
        row = scores.iloc[0]
        model_scores = {
            "gmic": float(row.gmic_score),
            "nyu": float(row.nyu_score),
            "glam": float(row.glam_score),
        }
        _write_web_progress(
            run_dir, run_id, stage="ENSEMBLE", state="RUNNING",
            message="Integrando las probabilidades de los tres modelos.", models=model_progress, started_at=started_iso,
        )
        ensemble = vote(model_scores, weights, threshold, discordance_threshold)

        orientation_summary_path = run_dir / "orientation_resolution" / "orientation_policy_summary.json"
        orientation_summary = (
            json.loads(orientation_summary_path.read_text(encoding="utf-8"))
            if orientation_summary_path.exists()
            else {"policy_id": ORIENTATION_POLICY_ID}
        )
        resolution_path = run_dir / "orientation_resolution" / "orientation_resolution.csv"
        orientation_resolution = {}
        if resolution_path.exists():
            orientation_resolution = _jsonable(pd.read_csv(resolution_path).iloc[0].to_dict())
        xai_path = run_dir / "xai_artifacts.json"
        xai = json.loads(xai_path.read_text(encoding="utf-8")) if xai_path.exists() else {}
        resource_path = run_dir / "resource_metrics.csv"
        resources = pd.read_csv(resource_path).to_dict("records") if resource_path.exists() else []
        inference_elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()

        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": "SUCCESS",
            "research_only": True,
            "inference_only": True,
            "training_performed": False,
            "ground_truth_received": False,
            "ground_truth_used": False,
            "input_preparation": preparation,
            "study": {
                "study_id": str(df.iloc[0].study_id),
                "patient_id": str(df.iloc[0].patient_id),
                "dataset_source": str(df.iloc[0].dataset_source),
            },
            "classification": ensemble.classification,
            "ensemble_malignancy_score": float(ensemble.ensemble_malignancy_score),
            "threshold": float(ensemble.threshold),
            "weights": {k: float(v) for k, v in ensemble.weights.items()},
            "weights_source": weights_source,
            "inference_device": web_device,
            "batch_configuration_mutated": False,
            "model_scores": model_scores,
            "model_range": float(ensemble.model_range),
            "model_std": float(ensemble.model_std),
            "discordance": bool(ensemble.discordance),
            "discordance_threshold": discordance_threshold,
            "orientation": {"summary": orientation_summary, "resolution": orientation_resolution},
            "xai_artifacts": xai,
            "resource_metrics": resources,
            "inference_elapsed_seconds": float(inference_elapsed),
            "overall_elapsed_seconds": float(inference_elapsed),
            "output_dir": str(run_dir),
            "persistence": {"postgresql": {"status": "PENDING"}, "minio": {"status": "PENDING"}},
        }
        result_path = run_dir / "single_case_result.json"
        write_json(result_path, _jsonable(payload))

        _write_web_progress(
            run_dir, run_id, stage="PERSISTENCE", state="RUNNING",
            message="Registrando resultados y evidencias de la evaluación.", models=model_progress, started_at=started_iso,
        )
        # MinIO is audit persistence only. A MinIO failure never changes the model result.
        try:
            canonical_views = {
                view: str(preparation["selected_views"][view]["canonical_png"])
                for view in REQUIRED_VIEWS
            }
            minio_result = persist_single_case(
                run_id=run_id,
                original_dicoms=preparation["files"],
                canonical_views=canonical_views,
                run_dir=run_dir,
                result_path=result_path,
            )
        except Exception as exc:
            minio_result = {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "non_blocking": True,
            }
            audit("WEB_MINIO_PERSISTENCE_FAILED", run_id=run_id, error=str(exc))

        payload["persistence"]["minio"] = minio_result
        payload["persistence"]["postgresql"] = {"status": "SUCCESS"}
        payload["overall_elapsed_seconds"] = float((dt.datetime.now(dt.timezone.utc) - started).total_seconds())
        write_json(result_path, _jsonable(payload))

        if minio_result.get("status") == "SUCCESS":
            try:
                persist_result_json(run_id=run_id, result_path=result_path)
            except Exception as exc:
                payload["persistence"]["minio"]["result_refresh_error"] = f"{type(exc).__name__}: {exc}"
                write_json(result_path, _jsonable(payload))

        save_run(run_id, "single_case_web", "SUCCESS", str(run_dir))
        save_web_inference(
            run_id=run_id,
            status="SUCCESS",
            classification=payload["classification"],
            ensemble_score=payload["ensemble_malignancy_score"],
            threshold=payload["threshold"],
            ensemble_weights=payload["weights"],
            weights_source=payload["weights_source"],
            inference_device=payload["inference_device"],
            overall_elapsed_seconds=payload["overall_elapsed_seconds"],
            gmic_score=model_scores["gmic"],
            nyu_score=model_scores["nyu"],
            glam_score=model_scores["glam"],
            minio_bucket=minio_result.get("bucket"),
            minio_prefix=minio_result.get("prefix"),
            minio_status=minio_result.get("status"),
            artifact_path=str(run_dir),
        )
        audit(
            "WEB_SINGLE_CASE_COMPLETED",
            run_id=run_id,
            classification=payload["classification"],
            ensemble_malignancy_score=payload["ensemble_malignancy_score"],
            minio_status=minio_result.get("status"),
            overall_elapsed_seconds=payload["overall_elapsed_seconds"],
            inference_device=web_device,
        )
        _write_web_progress(
            run_dir, run_id, stage="COMPLETED", state="SUCCESS",
            message="Evaluación completada.", models=model_progress, started_at=started_iso,
        )
        return _jsonable(payload)
    except Exception as exc:
        failed_elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        write_json(run_dir / "error.json", {
            "type": type(exc).__name__, "message": str(exc), "run_id": run_id,
            "training_performed": False, "ground_truth_received": False,
            "overall_elapsed_seconds": float(failed_elapsed),
        })
        _write_web_progress(
            run_dir, run_id, stage="FAILED", state="FAILED",
            message="La evaluación se interrumpió.", models=model_progress, started_at=started_iso, error=str(exc),
        )
        audit("WEB_SINGLE_CASE_FAILED", run_id=run_id, error=str(exc))
        try:
            save_run(run_id, "single_case_web", "FAILED", str(run_dir))
        except Exception:
            pass
        raise
