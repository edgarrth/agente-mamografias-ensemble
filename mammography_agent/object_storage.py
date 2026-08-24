from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable

from .logging_utils import audit

DEFAULT_BUCKET = "mammography-web"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def settings() -> dict[str, Any]:
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", "minio:9000").strip(),
        "access_key": os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "mammography")).strip(),
        "secret_key": os.getenv(
            "MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "mammography_research")
        ).strip(),
        "secure": _env_bool("MINIO_SECURE", False),
        "bucket": os.getenv("MINIO_WEB_BUCKET", DEFAULT_BUCKET).strip() or DEFAULT_BUCKET,
        "console_public_url": os.getenv("MINIO_CONSOLE_PUBLIC_URL", "http://localhost:9001").strip(),
        "enabled": _env_bool("MINIO_WEB_ENABLED", True),
    }


def _client():
    cfg = settings()
    if not cfg["enabled"]:
        raise RuntimeError("MinIO Web persistence is disabled by MINIO_WEB_ENABLED")
    if not cfg["endpoint"] or not cfg["access_key"] or not cfg["secret_key"]:
        raise RuntimeError("MinIO configuration is incomplete")
    try:
        from minio import Minio
    except ImportError as exc:
        raise RuntimeError("Python package 'minio' is required for Web object persistence") from exc
    return Minio(
        cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        secure=bool(cfg["secure"]),
    )


def _ensure_bucket(client, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_type(path: Path) -> str:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return "application/dicom"
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or "application/octet-stream"


def _upload(client, bucket: str, object_name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    client.fput_object(bucket, object_name, str(path), content_type=_content_type(path))
    return {
        "object": object_name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "content_type": _content_type(path),
    }


def persist_single_case(
    *,
    run_id: str,
    original_dicoms: Iterable[dict[str, Any]],
    canonical_views: dict[str, str],
    run_dir: str | Path,
    result_path: str | Path,
) -> dict[str, Any]:
    """Persist Web evidence to MinIO without making MinIO part of model inference.

    Only compact audit artifacts are uploaded. Large model_batch/preprocessed directories
    remain in the Web scratch volume only for the lifetime of the evaluation and are deleted
    after durable PostgreSQL/MinIO persistence completes.
    """
    cfg = settings()
    if not cfg["enabled"]:
        return {"status": "DISABLED", "bucket": cfg["bucket"], "prefix": None, "objects": []}

    run_dir = Path(run_dir)
    result_path = Path(result_path)
    prefix = f"runs/{run_id}"
    client = _client()
    _ensure_bucket(client, cfg["bucket"])

    uploaded: list[dict[str, Any]] = []
    originals = list(original_dicoms)
    for idx, item in enumerate(originals, start=1):
        path = Path(str(item["path"]))
        suffix = path.suffix.lower() if path.suffix else ".dcm"
        canonical = str(item.get("selected_as") or item.get("detected_view") or "UNASSIGNED")
        object_name = f"{prefix}/input/{idx:02d}_{canonical}{suffix}"
        record = _upload(client, cfg["bucket"], object_name, path)
        record.update({"kind": "input_dicom", "source_name": path.name, "selected_as": item.get("selected_as")})
        uploaded.append(record)

    for view, value in sorted(canonical_views.items()):
        path = Path(value)
        object_name = f"{prefix}/canonical/{view}.png"
        record = _upload(client, cfg["bucket"], object_name, path)
        record.update({"kind": "canonical_png", "view": view})
        uploaded.append(record)

    audit_candidates = [
        run_dir / "raw_model_predictions.csv",
        run_dir / "resource_metrics.csv",
        run_dir / "xai_artifacts.json",
        run_dir / "orientation_resolution" / "orientation_policy_summary.json",
        run_dir / "orientation_resolution" / "orientation_resolution.csv",
    ]
    for path in audit_candidates:
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        object_name = f"{prefix}/audit/{rel}"
        record = _upload(client, cfg["bucket"], object_name, path)
        record.update({"kind": "audit_artifact"})
        uploaded.append(record)

    result_record = _upload(client, cfg["bucket"], f"{prefix}/result/single_case_result.json", result_path)
    result_record.update({"kind": "result"})
    uploaded.append(result_record)

    manifest = {
        "run_id": run_id,
        "bucket": cfg["bucket"],
        "prefix": prefix,
        "artifact_object_count": len(uploaded),
        "objects": uploaded,
        "console_public_url": cfg.get("console_public_url"),
    }
    manifest_path = run_dir / "minio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_object = f"{prefix}/manifest/minio_manifest.json"
    manifest_record = _upload(client, cfg["bucket"], manifest_object, manifest_path)
    total_objects = len(uploaded) + 1

    audit("WEB_MINIO_PERSISTED", run_id=run_id, bucket=cfg["bucket"], prefix=prefix, objects=total_objects)
    return {
        "status": "SUCCESS",
        "bucket": cfg["bucket"],
        "prefix": prefix,
        "object_count": total_objects,
        "artifact_object_count": len(uploaded),
        "manifest_object": manifest_object,
        "manifest_sha256": manifest_record["sha256"],
        "objects": uploaded,
        "console_public_url": cfg.get("console_public_url"),
    }


def status() -> dict[str, Any]:
    cfg = settings()
    base = {
        "enabled": bool(cfg["enabled"]),
        "endpoint": cfg["endpoint"],
        "bucket": cfg["bucket"],
        "secure": bool(cfg["secure"]),
        "console_public_url": cfg.get("console_public_url"),
    }
    if not cfg["enabled"]:
        return {**base, "reachable": False, "status": "DISABLED"}
    try:
        client = _client()
        exists = client.bucket_exists(cfg["bucket"])
        return {**base, "reachable": True, "status": "READY", "bucket_exists": bool(exists)}
    except Exception as exc:
        return {**base, "reachable": False, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}


def persist_result_json(*, run_id: str, result_path: str | Path) -> dict[str, Any]:
    """Overwrite the compact result object after persistence metadata has been attached."""
    cfg = settings()
    if not cfg["enabled"]:
        return {"status": "DISABLED"}
    client = _client()
    _ensure_bucket(client, cfg["bucket"])
    object_name = f"runs/{run_id}/result/single_case_result.json"
    record = _upload(client, cfg["bucket"], object_name, Path(result_path))
    return {"status": "SUCCESS", "bucket": cfg["bucket"], **record}
