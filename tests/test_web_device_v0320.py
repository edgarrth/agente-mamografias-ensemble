from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mammography_agent import model_client, single_case
from mammography_agent.api import WebDicomCaseRequest


def test_web_request_defaults_to_cpu_and_rejects_unknown_device():
    req = WebDicomCaseRequest(
        dicom_paths=["/workspace/a.dcm", "/workspace/b.dcm", "/workspace/c.dcm", "/workspace/d.dcm"]
    )
    assert req.inference_device == "cpu"
    with pytest.raises(Exception):
        WebDicomCaseRequest(
            dicom_paths=["a.dcm", "b.dcm", "c.dcm", "d.dcm"],
            inference_device="tpu",
        )


def test_web_device_resolution_is_local_and_strict():
    assert single_case._resolve_web_device(None) == "cpu"
    assert single_case._resolve_web_device("CPU") == "cpu"
    assert single_case._resolve_web_device("gpu") == "gpu"
    with pytest.raises(ValueError, match="cpu or gpu"):
        single_case._resolve_web_device("auto")


def test_model_client_does_not_serialize_device_when_batch_uses_default(monkeypatch):
    captured = {}

    class Response:
        ok = True
        def json(self):
            return {"status": "SUCCESS"}

    def fake_post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr(model_client.requests, "post", fake_post)
    model_client.run_model("gmic", "batch-run", "/workspace/images", "/workspace/data.pkl", "/workspace/out.csv", "/workspace/pre", device=None)
    assert "device" not in captured


def test_model_client_serializes_explicit_web_device(monkeypatch):
    captured = {}

    class Response:
        ok = True
        def json(self):
            return {"status": "SUCCESS"}

    def fake_post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr(model_client.requests, "post", fake_post)
    model_client.run_model("gmic", "web-run", "/workspace/images", "/workspace/data.pkl", "/workspace/out.csv", "/workspace/pre", device="cpu")
    assert captured["device"] == "cpu"


def test_pipeline_batch_calls_keep_default_device_contract():
    text = Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert "device: str | None = None" in text
    assert "web_label_blind_compat: bool = False" in text
    assert 'if device is None:' in text
    assert 'result=run_model(model,f"{run_id}-{model}",str(images),str(pkl),str(out),str(pre))' in text
    assert 'result=run_model(model,f"{run_id}-{model}",str(images),str(pkl),str(out),str(pre),device=device)' in text
    # Formal/batch call sites continue invoking _infer_three without a device override.
    assert 'pred=_infer_three(sub,cdir,f"{run_id}-c{chunk_index:04d}")' in text


def test_streamlit_cpu_mode_does_not_require_gpu_probe():
    # Importing Streamlit UI would execute the app; verify the helper implementation contract textually.
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert 'requested_device: str | None = None' in text
    assert 'if device == "gpu":' in text
    assert 'st.session_state.setdefault("web_inference_device", DEFAULT_WEB_DEVICE)' in text
    assert 'WEB_INFERENCE_DEVICE' in text
    assert 'main_tab, config_tab, method_tab = st.tabs(["Evaluación del estudio", "Configuración y estado", "Metodología y trazabilidad"])' in text
    assert 'st.subheader("3. Configuración del ensemble")' not in text


def test_web_configuration_files_are_not_mutated_by_device_selector():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert '"inference_device": web_device' in text
    assert 'os.environ[' not in text
    assert '.write_text(' not in text
    assert 'yaml.safe_dump' not in text
    assert 'config/ensemble.yaml' in text
    assert 'config/experiments.yaml' in text


def test_api_executor_propagates_web_device_without_global_mutation(monkeypatch):
    import mammography_agent.api as api
    captured = {}

    def fake_run_single_case(**kwargs):
        captured.update(kwargs)
        return {"status": "SUCCESS"}

    monkeypatch.setattr(api, "run_single_case", fake_run_single_case)
    result = api._execute_single_case_request({
        "dicom_paths": ["a", "b", "c", "d"],
        "view_assignments": {},
        "ensemble_weights": None,
        "inference_device": "cpu",
    })
    assert result["status"] == "SUCCESS"
    assert captured["inference_device"] == "cpu"
    assert captured["ensemble_weights"] is None
