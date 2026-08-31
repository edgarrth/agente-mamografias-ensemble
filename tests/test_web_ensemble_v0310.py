from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mammography_agent import single_case
from mammography_agent.api import WebDicomCaseRequest


def test_web_weights_accept_valid_override_and_reject_invalid():
    weights, source = single_case._resolve_web_weights({"gmic": 0.5, "nyu": 0.3, "glam": 0.2})
    assert source == "WEB_OVERRIDE"
    assert weights == {"gmic": 0.5, "nyu": 0.3, "glam": 0.2}

    with pytest.raises(ValueError, match="sum to 1"):
        single_case._resolve_web_weights({"gmic": 0.5, "nyu": 0.4, "glam": 0.4})
    with pytest.raises(ValueError, match="exactly gmic, nyu and glam"):
        single_case._resolve_web_weights({"gmic": 0.5, "nyu": 0.5})


def test_api_weights_are_web_only_and_validated():
    req = WebDicomCaseRequest(
        dicom_paths=["/workspace/a.dcm", "/workspace/b.dcm", "/workspace/c.dcm", "/workspace/d.dcm"],
        ensemble_weights={"gmic": 0.5, "nyu": 0.3, "glam": 0.2},
    )
    assert req.ensemble_weights == {"gmic": 0.5, "nyu": 0.3, "glam": 0.2}
    dumped = req.model_dump()
    assert "ground_truth" not in dumped
    assert dumped["decision_threshold"] is None

    with pytest.raises(Exception):
        WebDicomCaseRequest(
            dicom_paths=["a.dcm", "b.dcm", "c.dcm", "d.dcm"],
            ensemble_weights={"gmic": 0.6, "nyu": 0.3, "glam": 0.3},
        )


def test_web_config_endpoint_payload_does_not_mutate_batch(monkeypatch):
    monkeypatch.setattr(single_case, "_baseline_config", lambda: ({"gmic": 0.4, "nyu": 0.3, "glam": 0.3}, 0.45, 0.25))
    cfg = single_case.web_ensemble_config()
    assert cfg["weights"] == {"gmic": 0.4, "nyu": 0.3, "glam": 0.3}
    assert cfg["threshold"] == 0.45
    assert cfg["editable_fields"] == ["weights", "threshold"]
    assert cfg["batch_configuration_mutated"] is False


def test_streamlit_exposes_web_weights_device_config_and_elapsed_time():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert '["Evaluación del estudio", "Configuración y estado", "Metodología y trazabilidad"]' in text
    assert '"ensemble_weights": (web_weights if config_mode == "Configuración personalizada" else None)' in text
    assert 'key="web_inference_device"' in text
    assert '"inference_device": web_device' in text
    assert "model_tools.validate_gpu --models all" in text
    assert "Tiempo transcurrido hasta la interrupción" in text
    assert "CC · cráneo-caudal" in text
    assert "MLO · medio-lateral oblicua" in text


def test_batch_weight_files_are_not_written_by_web_code():
    text = Path("mammography_agent/single_case.py").read_text(encoding="utf-8")
    assert 'load_yaml("ensemble.yaml")' in text
    assert "experiments.yaml" not in text
    assert ".write_text(" not in text.split("def web_ensemble_config", 1)[1].split("def _jsonable", 1)[0]
