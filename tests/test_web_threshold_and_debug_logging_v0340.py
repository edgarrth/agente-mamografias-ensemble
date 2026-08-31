from __future__ import annotations

from pathlib import Path

import pytest

from mammography_agent import single_case
from mammography_agent.api import WebDicomCaseRequest


def test_web_threshold_override_is_validated_and_does_not_mutate_batch(monkeypatch):
    monkeypatch.setattr(single_case, "_baseline_config", lambda: ({"gmic": 0.4, "nyu": 0.3, "glam": 0.3}, 0.5, 0.3))
    value, source = single_case._resolve_web_threshold(None)
    assert value == 0.5
    assert source == "BASELINE"
    value, source = single_case._resolve_web_threshold(0.025)
    assert value == 0.025
    assert source == "WEB_OVERRIDE"
    with pytest.raises(ValueError, match="between 0 and 1"):
        single_case._resolve_web_threshold(1.1)


def test_api_accepts_web_only_decision_threshold():
    req = WebDicomCaseRequest(
        dicom_paths=["a.dcm", "b.dcm", "c.dcm", "d.dcm"],
        decision_threshold=0.025,
    )
    assert req.decision_threshold == 0.025
    dumped = req.model_dump()
    assert "ground_truth" not in dumped
    with pytest.raises(Exception):
        WebDicomCaseRequest(
            dicom_paths=["a.dcm", "b.dcm", "c.dcm", "d.dcm"],
            decision_threshold=-0.01,
        )


def test_api_executor_propagates_threshold_without_global_mutation(monkeypatch):
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
        "decision_threshold": 0.031,
        "inference_device": "cpu",
    })
    assert result["status"] == "SUCCESS"
    assert captured["decision_threshold"] == 0.031


def test_streamlit_threshold_is_web_only_and_configurable():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert 'key="web_threshold_mode"' in text
    assert 'key="web_decision_threshold"' in text
    assert '"decision_threshold": (threshold_value if threshold_mode == "Configuración personalizada" else None)' in text
    assert "no modifica config/ensemble.yaml ni config/experiments.yaml" in text


def test_web_debug_logging_events_and_helper_script_exist():
    single = Path("mammography_agent/single_case.py").read_text(encoding="utf-8")
    runner = Path("model_runner/api.py").read_text(encoding="utf-8")
    script = Path("scripts/web-debug-logs.sh").read_text(encoding="utf-8")
    for event in (
        "WEB_CONFIGURATION_RESOLVED",
        "WEB_STAGE_STARTED",
        "WEB_STAGE_COMPLETED",
        "WEB_MODEL_PROGRESS",
        "WEB_MODEL_SCORES_COLLECTED",
        "WEB_ENSEMBLE_COMPUTED",
        "WEB_MINIO_PERSISTENCE_COMPLETED",
        "WEB_POSTGRESQL_PERSISTENCE_COMPLETED",
    ):
        assert event in single
    assert "MODEL_RUNTIME_READY" in runner
    assert "runtime_prepare_elapsed_seconds" in runner
    assert "model_runner_total_elapsed_seconds" in runner
    assert "docker compose logs --no-color fastapi model-runner" in script
    assert 'grep -F -- "$run_id"' in script


def test_batch_configuration_files_are_not_written_by_web_threshold_code():
    text = Path("mammography_agent/single_case.py").read_text(encoding="utf-8")
    resolver = text.split("def _resolve_web_threshold", 1)[1].split("def _resolve_web_device", 1)[0]
    assert 'load_yaml("ensemble.yaml")' not in resolver  # baseline is read only via helper
    assert ".write_text(" not in resolver
    assert "experiments.yaml" not in resolver
