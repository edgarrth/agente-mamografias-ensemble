from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WORKSPACE_ROOT, WEB_SCRATCH_ROOT, load_yaml
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
    """Validate a Web input/runtime path without changing the batch workspace contract.

    v0.33.0 stages Web uploads/runs under WEB_SCRATCH_ROOT. Existing absolute
    WORKSPACE_ROOT inputs remain readable for backward compatibility, but Web output
    is never persisted there.
    """
    p = Path(value)
    if not p.is_absolute():
        p = WEB_SCRATCH_ROOT / p
    p = p.resolve()
    roots = (WEB_SCRATCH_ROOT.resolve(), WORKSPACE_ROOT.resolve())
    if not any(p == root or root in p.parents for root in roots):
        raise ValueError(f"Path outside approved Web runtime roots is forbidden: {p}")
    return p


def _is_under(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _cleanup_web_scratch(run_dir: Path, dicom_paths: list[str]) -> None:
    """Remove Web-only scratch data. Batch workspace paths are never deleted."""
    scratch = WEB_SCRATCH_ROOT.resolve()
    if _is_under(run_dir, scratch):
        shutil.rmtree(run_dir, ignore_errors=True)

    upload_parents: set[Path] = set()
    preview_root = WEB_SCRATCH_ROOT / "previews"
    for raw in dicom_paths:
        path = Path(raw)
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if not _is_under(resolved, scratch):
            continue
        try:
            if resolved.is_file():
                digest = _sha256(resolved)
                preview = preview_root / f"{digest[:24]}.png"
                preview.unlink(missing_ok=True)
                resolved.unlink(missing_ok=True)
            if _is_under(resolved.parent, WEB_SCRATCH_ROOT / "uploads"):
                upload_parents.add(resolved.parent)
        except Exception:
            pass
    for parent in sorted(upload_parents, key=lambda x: len(x.parts), reverse=True):
        shutil.rmtree(parent, ignore_errors=True)


_WEB_PROGRESS: dict[str, dict[str, Any]] = {}
_WEB_PROGRESS_LOCK = threading.Lock()


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
    preview_root = WEB_SCRATCH_ROOT / "previews"
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
        "editable_fields": ["weights", "threshold"],
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




def _resolve_web_threshold(override: float | None) -> tuple[float, str]:
    """Resolve a Web-only decision threshold without mutating batch YAML/configuration."""
    _, baseline_threshold, _ = _baseline_config()
    if override is None:
        return float(baseline_threshold), "BASELINE"
    threshold = float(override)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Web decision threshold must be between 0 and 1")
    return threshold, "WEB_OVERRIDE"


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


def _write_web_progress(
    run_dir: Path,
    run_id: str,
    *,
    stage: str,
    state: str,
    message: str,
    models: dict[str, Any] | None = None,
    stages: dict[str, Any] | None = None,
    started_at: str | None = None,
    error: str | None = None,
) -> None:
    # Progress is transient process state, not a persisted case artifact.
    payload = {
        "run_id": run_id,
        "stage": stage,
        "state": state,
        "message": message,
        "models": _jsonable(models or {}),
        "stages": _jsonable(stages or {}),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if started_at:
        payload["started_at_utc"] = started_at
    if error:
        payload["error"] = error
    with _WEB_PROGRESS_LOCK:
        _WEB_PROGRESS[run_id] = payload
        # Keep only a bounded number of compact progress records.
        if len(_WEB_PROGRESS) > 256:
            oldest = next(iter(_WEB_PROGRESS))
            _WEB_PROGRESS.pop(oldest, None)


def get_single_case_progress(run_id: str) -> dict[str, Any]:
    safe = _safe_token(run_id)
    if safe != run_id or not run_id.startswith("web-"):
        raise ValueError("Invalid Web run_id")
    with _WEB_PROGRESS_LOCK:
        payload = _WEB_PROGRESS.get(run_id)
        if payload is not None:
            return json.loads(json.dumps(payload))
    return {
        "run_id": run_id,
        "stage": "QUEUED",
        "state": "PENDING",
        "message": "La evaluación está iniciándose.",
        "models": {},
        "stages": {},
    }

def run_single_case(
    *,
    dicom_paths: list[str],
    view_assignments: dict[str, str] | None = None,
    ensemble_weights: dict[str, float] | None = None,
    decision_threshold: float | None = None,
    inference_device: str = "cpu",
    request_run_id: str | None = None,
) -> dict[str, Any]:
    """Infer exactly one four-view mammography exam; never train and never consume labels."""
    run_id = str(request_run_id or _id())
    if _safe_token(run_id) != run_id or not run_id.startswith("web-"):
        raise ValueError("Invalid Web run_id")
    run_dir = WEB_SCRATCH_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    wall_started = time.monotonic()
    web_device = _resolve_web_device(inference_device)
    weights, weights_source = _resolve_web_weights(ensemble_weights)
    threshold, threshold_source = _resolve_web_threshold(decision_threshold)
    _, _, discordance_threshold = _baseline_config()
    audit("WEB_SINGLE_CASE_STARTED", run_id=run_id, uploaded_dicoms=len(dicom_paths), inference_device=web_device)
    audit(
        "WEB_CONFIGURATION_RESOLVED",
        run_id=run_id,
        inference_device=web_device,
        weights=weights,
        weights_source=weights_source,
        threshold=threshold,
        threshold_source=threshold_source,
        discordance_threshold=discordance_threshold,
        batch_configuration_mutated=False,
    )

    model_progress = {model: {"state": "PENDING"} for model in ("gmic", "nyu", "glam")}
    stage_progress = {
        stage: {"state": "PENDING"}
        for stage in ("PREPARATION", "ORIENTATION", "MODEL_INPUT_PREPARATION", "ENSEMBLE", "PERSISTENCE")
    }
    stage_started_clock: dict[str, float] = {}
    started_iso = started.isoformat()

    def _publish(stage: str, message: str, *, state: str = "RUNNING", error: str | None = None) -> None:
        _write_web_progress(
            run_dir,
            run_id,
            stage=stage,
            state=state,
            message=message,
            models=model_progress,
            stages=stage_progress,
            started_at=started_iso,
            error=error,
        )

    def _begin_stage(key: str, message: str) -> None:
        stage_started_clock[key] = time.monotonic()
        stage_progress[key] = {"state": "RUNNING"}
        audit("WEB_STAGE_STARTED", run_id=run_id, stage=key, message=message)
        _publish(key, message)

    def _finish_stage(key: str) -> float:
        elapsed = time.monotonic() - stage_started_clock[key]
        stage_progress[key] = {"state": "SUCCESS", "elapsed_seconds": float(elapsed)}
        audit("WEB_STAGE_COMPLETED", run_id=run_id, stage=key, elapsed_seconds=float(elapsed))
        return float(elapsed)

    _begin_stage("PREPARATION", "Preparando el estudio mamográfico.")

    try:
        df, preparation = _build_uploaded_case(
            run_dir=run_dir,
            dicom_paths=dicom_paths,
            view_assignments=view_assignments,
        )
        df.to_csv(run_dir / "input_study.csv", index=False)
        _finish_stage("PREPARATION")

        _begin_stage("ORIENTATION", "Aplicando la política de orientación del estudio.")
        resolved = resolve_orientation(df, run_dir / "orientation_resolution", run_id, source_path_resolver=_workspace_path)
        resolved.to_csv(run_dir / "resolved_study.csv", index=False)
        _finish_stage("ORIENTATION")

        # Mechanical label blindness: the Web frame contains no clinical labels; keep
        # all label-shaped columns NaN immediately before the common model pipeline.
        inference_input = resolved.copy()
        for label_column in ("ground_truth", "left_ground_truth", "right_ground_truth"):
            inference_input[label_column] = float("nan")

        def _stage_progress_callback(*, stage: str, state: str, elapsed_seconds: float | None = None) -> None:
            item: dict[str, Any] = {"state": state}
            if elapsed_seconds is not None:
                item["elapsed_seconds"] = float(elapsed_seconds)
            stage_progress[stage] = item
            audit("WEB_STAGE_PROGRESS", run_id=run_id, stage=stage, state=state, elapsed_seconds=elapsed_seconds)
            message = (
                "Preparando las entradas canónicas para los modelos."
                if state == "RUNNING"
                else "Entradas de modelos preparadas."
            )
            _publish(stage, message, state="FAILED" if state == "FAILED" else "RUNNING")

        def _model_progress_callback(*, model: str, state: str, elapsed_seconds: float | None = None) -> None:
            item = dict(model_progress.get(model, {}))
            item["state"] = state
            if elapsed_seconds is not None:
                item["elapsed_seconds"] = float(elapsed_seconds)
            model_progress[model] = item
            audit(
                "WEB_MODEL_PROGRESS", run_id=run_id, model=model, state=state,
                elapsed_seconds=elapsed_seconds, inference_device=web_device,
            )
            label = {"gmic": "GMIC", "nyu": "NYU / DMV-CNN", "glam": "GLAM"}.get(model, model.upper())
            if state == "RUNNING":
                message = f"Ejecutando: {label}."
            elif state == "FAILED":
                message = f"Error durante la ejecución de {label}."
            else:
                message = f"Completado: {label}."
            _publish("MODELS", message, state="FAILED" if state == "FAILED" else "RUNNING")

        scores = _infer_three(
            inference_input,
            run_dir,
            run_id,
            device=web_device,
            web_label_blind_compat=True,
            progress_callback=_model_progress_callback,
            stage_progress_callback=_stage_progress_callback,
        )
        if len(scores) != 1:
            raise RuntimeError(f"Expected one prediction row; got {len(scores)}")

        row = scores.iloc[0]
        model_scores = {
            "gmic": float(row.gmic_score),
            "nyu": float(row.nyu_score),
            "glam": float(row.glam_score),
        }
        audit("WEB_MODEL_SCORES_COLLECTED", run_id=run_id, **model_scores)

        _begin_stage("ENSEMBLE", "Integrando las probabilidades de los tres modelos.")
        ensemble = vote(model_scores, weights, threshold, discordance_threshold)
        _finish_stage("ENSEMBLE")
        audit(
            "WEB_ENSEMBLE_COMPUTED", run_id=run_id,
            ensemble_malignancy_score=float(ensemble.ensemble_malignancy_score),
            classification=ensemble.classification, threshold=float(ensemble.threshold),
            threshold_source=threshold_source, weights=weights, weights_source=weights_source,
            model_range=float(ensemble.model_range), discordance=bool(ensemble.discordance),
        )

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
        inference_elapsed = time.monotonic() - wall_started

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
            "threshold_source": threshold_source,
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
            # Resource metrics are retained as model-runtime diagnostics. Web-visible
            # execution times are reported separately from monotonic wall-clock timing.
            "resource_metrics": resources,
            "execution_timings": {
                "stages": _jsonable(stage_progress),
                "models": _jsonable(model_progress),
            },
            "inference_elapsed_seconds": float(inference_elapsed),
            "overall_elapsed_seconds": float(inference_elapsed),
            "local_persistence": False,
            "persistence": {
                "postgresql": {"status": "PENDING"},
                "minio": {"status": "PENDING"},
                "local_scratch": {"status": "TRANSIENT", "retained": False},
            },
        }
        result_path = run_dir / "single_case_result.json"
        write_json(result_path, _jsonable(payload))

        _begin_stage("PERSISTENCE", "Registrando el resultado de la evaluación.")
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
        audit(
            "WEB_MINIO_PERSISTENCE_COMPLETED", run_id=run_id, status=minio_result.get("status"),
            bucket=minio_result.get("bucket"), prefix=minio_result.get("prefix"),
            object_count=minio_result.get("object_count"),
        )

        durable_ref = (
            f"minio://{minio_result.get('bucket')}/{minio_result.get('prefix')}"
            if minio_result.get("status") == "SUCCESS"
            else f"postgresql://web_inference_runs/{run_id}"
        )
        save_run(run_id, "single_case_web", "SUCCESS", durable_ref)
        overall_before_db = time.monotonic() - wall_started
        save_web_inference(
            run_id=run_id,
            status="SUCCESS",
            classification=payload["classification"],
            ensemble_score=payload["ensemble_malignancy_score"],
            threshold=payload["threshold"],
            threshold_source=payload["threshold_source"],
            ensemble_weights=payload["weights"],
            weights_source=payload["weights_source"],
            inference_device=payload["inference_device"],
            overall_elapsed_seconds=float(overall_before_db),
            gmic_score=model_scores["gmic"],
            nyu_score=model_scores["nyu"],
            glam_score=model_scores["glam"],
            minio_bucket=minio_result.get("bucket"),
            minio_prefix=minio_result.get("prefix"),
            minio_status=minio_result.get("status"),
            artifact_path=durable_ref,
        )
        payload["persistence"]["postgresql"] = {"status": "SUCCESS"}
        audit("WEB_POSTGRESQL_PERSISTENCE_COMPLETED", run_id=run_id, status="SUCCESS")
        _finish_stage("PERSISTENCE")
        payload["execution_timings"] = {
            "stages": _jsonable(stage_progress),
            "models": _jsonable(model_progress),
        }
        payload["overall_elapsed_seconds"] = float(time.monotonic() - wall_started)
        write_json(result_path, _jsonable(payload))

        if minio_result.get("status") == "SUCCESS":
            try:
                persist_result_json(run_id=run_id, result_path=result_path)
            except Exception as exc:
                payload["persistence"]["minio"]["result_refresh_error"] = f"{type(exc).__name__}: {exc}"
                write_json(result_path, _jsonable(payload))

        # Do not return transient filesystem locations that will be removed below.
        for item in (payload.get("input_preparation", {}).get("files") or []):
            item.pop("path", None)
        for item in (payload.get("input_preparation", {}).get("selected_views") or {}).values():
            item.pop("path", None)
            item.pop("canonical_png", None)

        audit(
            "WEB_SINGLE_CASE_COMPLETED",
            run_id=run_id,
            classification=payload["classification"],
            ensemble_malignancy_score=payload["ensemble_malignancy_score"],
            minio_status=minio_result.get("status"),
            overall_elapsed_seconds=payload["overall_elapsed_seconds"],
            inference_device=web_device,
            threshold=payload["threshold"],
            threshold_source=payload["threshold_source"],
            weights_source=payload["weights_source"],
        )
        _publish("COMPLETED", "Evaluación completada.", state="SUCCESS")
        return _jsonable(payload)
    except Exception as exc:
        failed_elapsed = time.monotonic() - wall_started
        for stage_name, item in list(stage_progress.items()):
            if str(item.get("state")) == "RUNNING":
                elapsed = time.monotonic() - stage_started_clock.get(stage_name, time.monotonic())
                stage_progress[stage_name] = {"state": "FAILED", "elapsed_seconds": float(max(0.0, elapsed))}
        write_json(run_dir / "error.json", {
            "type": type(exc).__name__, "message": str(exc), "run_id": run_id,
            "training_performed": False, "ground_truth_received": False,
            "overall_elapsed_seconds": float(failed_elapsed),
            "execution_timings": {"stages": _jsonable(stage_progress), "models": _jsonable(model_progress)},
        })
        _publish("FAILED", "La evaluación se interrumpió.", state="FAILED", error=str(exc))
        audit(
            "WEB_SINGLE_CASE_FAILED", run_id=run_id, error=str(exc),
            error_type=type(exc).__name__, overall_elapsed_seconds=float(failed_elapsed),
            inference_device=web_device, threshold=threshold, threshold_source=threshold_source,
        )
        try:
            save_run(run_id, "single_case_web", "FAILED", f"postgresql://research_runs/{run_id}")
        except Exception:
            pass
        raise
    finally:
        _cleanup_web_scratch(run_dir, dicom_paths)
        audit("WEB_SCRATCH_CLEANUP_COMPLETED", run_id=run_id, scratch_retained=False)

