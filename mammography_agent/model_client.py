from __future__ import annotations
import os
import requests

RUNNER_URL = os.getenv("MODEL_RUNNER_URL", "http://model-runner:8010").rstrip("/")


def _model_path(model: str) -> str:
    model = model.strip().lower()
    if model not in {"gmic", "nyu", "glam"}:
        raise ValueError(f"Unknown model: {model}")
    return f"{RUNNER_URL}/models/{model}"


def status():
    try:
        r = requests.get(f"{RUNNER_URL}/models", timeout=20)
        r.raise_for_status()
        items = r.json()
        for item in items:
            item["reachable"] = True
            item["runner_url"] = RUNNER_URL
        return items
    except Exception as exc:
        return [
            {"model": model, "runner_url": RUNNER_URL, "reachable": False, "error": str(exc)}
            for model in ("gmic", "nyu", "glam")
        ]




def _raise_runner_error(r, operation: str, model: str):
    if r.ok:
        return
    try:
        payload = r.json()
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    except Exception:
        detail = r.text
    raise RuntimeError(
        f"{model} model-runner {operation} failed ({r.status_code}): {detail}"
    )

def ensure_model(model: str):
    r = requests.post(f"{_model_path(model)}/ensure", timeout=7200)
    _raise_runner_error(r, "ensure", model)
    return r.json()



def ensure_gpu_model(model: str, force_rebuild: bool = False):
    r = requests.post(
        f"{_model_path(model)}/ensure-gpu",
        params={"force_rebuild": str(bool(force_rebuild)).lower()},
        timeout=7200,
    )
    _raise_runner_error(r, "ensure-gpu", model)
    return r.json()


def gpu_probe(model: str):
    r = requests.post(f"{_model_path(model)}/gpu-probe", timeout=7200)
    _raise_runner_error(r, "gpu-probe", model)
    return r.json()


def smoke_test(model: str):
    r = requests.post(f"{_model_path(model)}/smoke-test", timeout=7200)
    _raise_runner_error(r, "smoke-test", model)
    return r.json()


def run_model(model: str, run_id: str, image_dir: str, data_pickle: str, output_file: str, preprocessed_dir: str):
    payload = {
        "run_id": run_id,
        "image_dir": image_dir,
        "data_pickle": data_pickle,
        "output_file": output_file,
        "preprocessed_dir": preprocessed_dir,
    }
    r = requests.post(f"{_model_path(model)}/run", json=payload, timeout=60 * 60 * 24)
    if not r.ok:
        raise RuntimeError(f"{model} model-runner call failed ({r.status_code}): {r.text}")
    return r.json()
