from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from . import __version__
from .datasets.manager import statuses, request_download, prepare, inspect
from .graph import run_graph
from .health_logging import install_healthcheck_access_filter
from .logging_utils import log_configuration_additions, audit
from .model_client import status as model_status
from .object_storage import status as minio_status
from .single_case import create_dicom_previews, get_single_case_progress, inspect_dicom_case, run_single_case, web_ensemble_config
from .storage import init_db, load_latest_web_inference_settings, load_web_evaluation_settings, save_web_evaluation_settings
from .workspace import ensure_workspace


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_healthcheck_access_filter()
    ensure_workspace()
    log_configuration_additions()
    init_db()
    audit("APPLICATION_READY", version=__version__)
    yield


app = FastAPI(title="Mammography AI Agent", version=__version__, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "research_only": True, "version": __version__}


@app.get("/workspace/status")
def workspace_status():
    return {"datasets": statuses(), "models": model_status()}


@app.get("/datasets")
def datasets():
    return statuses()


class DatasetSelection(BaseModel):
    datasets: list[str]


@app.post("/datasets/download")
def download(req: DatasetSelection):
    try:
        return request_download(req.datasets)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/datasets/prepare")
def prep(req: DatasetSelection):
    try:
        return prepare(req.datasets)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/datasets/inspect")
def inspect_dataset(req: DatasetSelection):
    try:
        return inspect(req.datasets)
    except Exception as e:
        raise HTTPException(400, str(e))


class WebDicomCaseRequest(BaseModel):
    dicom_paths: list[str]
    view_assignments: dict[str, Literal["AUTO", "IGNORE", "L_CC", "R_CC", "L_MLO", "R_MLO"]] = {}
    ensemble_weights: dict[Literal["gmic", "nyu", "glam"], float] | None = None
    decision_threshold: float | None = None
    inference_device: Literal["cpu", "gpu"] = "cpu"
    run_id: str | None = None

    @field_validator("dicom_paths")
    @classmethod
    def validate_dicom_count(cls, value: list[str]):
        if len(value) < 4:
            raise ValueError("At least four DICOM files are required")
        if len(value) > 20:
            raise ValueError("A Web unit case accepts at most 20 DICOM files")
        return value

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value):
        if value is None:
            return value
        import re
        if not re.fullmatch(r"web-[A-Za-z0-9_.-]{8,120}", value):
            raise ValueError("run_id must be a Web evaluation identifier")
        return value


    @field_validator("decision_threshold")
    @classmethod
    def validate_decision_threshold(cls, value):
        if value is None:
            return value
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")
        return value

    @field_validator("ensemble_weights")
    @classmethod
    def validate_ensemble_weights(cls, value):
        if value is None:
            return value
        if set(value) != {"gmic", "nyu", "glam"}:
            raise ValueError("ensemble_weights must contain exactly gmic, nyu and glam")
        weights = {k: float(v) for k, v in value.items()}
        if any(v < 0.0 or v > 1.0 for v in weights.values()):
            raise ValueError("ensemble weights must be between 0 and 1")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("ensemble weights must sum to 1")
        return weights


@app.post("/single-cases/inspect")
def inspect_single_case(req: WebDicomCaseRequest):
    try:
        return inspect_dicom_case(req.dicom_paths, req.view_assignments)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/single-cases/previews")
def preview_single_case(req: WebDicomCaseRequest):
    try:
        return create_dicom_previews(req.dicom_paths)
    except Exception as exc:
        raise HTTPException(400, str(exc))


def _execute_single_case_request(payload: dict):
    return run_single_case(
        dicom_paths=list(payload.get("dicom_paths") or []),
        view_assignments=dict(payload.get("view_assignments") or {}),
        ensemble_weights=(dict(payload["ensemble_weights"]) if payload.get("ensemble_weights") is not None else None),
        decision_threshold=(float(payload["decision_threshold"]) if payload.get("decision_threshold") is not None else None),
        inference_device=str(payload.get("inference_device") or "cpu"),
        request_run_id=(str(payload["run_id"]) if payload.get("run_id") else None),
    )


@app.post("/single-cases/run")
def run_web_single_case(req: WebDicomCaseRequest):
    try:
        return run_graph(req.model_dump(), _execute_single_case_request)
    except Exception as exc:
        raise HTTPException(400, str(exc))




@app.get("/single-cases/progress/{run_id}")
def single_case_progress(run_id: str):
    try:
        return get_single_case_progress(run_id)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/single-cases/ensemble-config")
def single_case_ensemble_config():
    return web_ensemble_config()



class WebEvaluationSettingsRequest(BaseModel):
    inference_device: Literal["cpu", "gpu"] = "cpu"
    weight_mode: Literal["BASELINE", "CUSTOM"] = "BASELINE"
    ensemble_weights: dict[Literal["gmic", "nyu", "glam"], float]
    threshold_mode: Literal["BASELINE", "CUSTOM"] = "BASELINE"
    decision_threshold: float

    @field_validator("ensemble_weights")
    @classmethod
    def validate_settings_weights(cls, value):
        if set(value) != {"gmic", "nyu", "glam"}:
            raise ValueError("ensemble_weights must contain exactly gmic, nyu and glam")
        weights = {k: float(v) for k, v in value.items()}
        if any(v < 0.0 or v > 1.0 for v in weights.values()):
            raise ValueError("ensemble weights must be between 0 and 1")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("ensemble weights must sum to 1")
        return weights

    @field_validator("decision_threshold")
    @classmethod
    def validate_settings_threshold(cls, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")
        return value


@app.get("/single-cases/web-settings")
def get_web_evaluation_settings():
    persisted = load_web_evaluation_settings()
    if persisted is not None:
        return {"persisted": True, "migrated_from_run": False, **persisted}
    migrated = load_latest_web_inference_settings()
    if migrated is not None:
        saved = save_web_evaluation_settings(
            inference_device=migrated["inference_device"],
            weight_mode=migrated["weight_mode"],
            weights=migrated["weights"],
            threshold_mode=migrated["threshold_mode"],
            decision_threshold=migrated["decision_threshold"],
        )
        if saved is not None:
            return {"persisted": True, "migrated_from_run": True, **saved}
    baseline = web_ensemble_config()
    import os
    device = str(os.getenv("WEB_INFERENCE_DEVICE", "cpu")).strip().lower()
    if device not in {"cpu", "gpu"}:
        device = "cpu"
    return {
        "persisted": False,
        "settings_key": "default",
        "inference_device": device,
        "weight_mode": "BASELINE",
        "weights": baseline["weights"],
        "threshold_mode": "BASELINE",
        "decision_threshold": baseline["threshold"],
        "updated_at": None,
    }


@app.put("/single-cases/web-settings")
def put_web_evaluation_settings(req: WebEvaluationSettingsRequest):
    try:
        saved = save_web_evaluation_settings(
            inference_device=req.inference_device,
            weight_mode=req.weight_mode,
            weights={k: float(v) for k, v in req.ensemble_weights.items()},
            threshold_mode=req.threshold_mode,
            decision_threshold=float(req.decision_threshold),
        )
        if saved is None:
            raise RuntimeError("PostgreSQL is not configured; Web settings cannot be persisted")
        audit(
            "WEB_SETTINGS_UPDATED",
            inference_device=saved["inference_device"],
            weight_mode=saved["weight_mode"],
            weights=saved["weights"],
            threshold_mode=saved["threshold_mode"],
            decision_threshold=saved["decision_threshold"],
        )
        return {"persisted": True, **saved}
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.get("/single-cases/storage-status")
def single_case_storage_status():
    return {
        "postgresql": {"configured": True, "role": "metadata + inference scores"},
        "minio": minio_status(),
        "workspace": {"role": "batch-only persistent workspace; Web case artifacts are not stored here"},
        "web_scratch": {"role": "transient shared runtime; removed after Web evaluation", "persistent": False},
    }
