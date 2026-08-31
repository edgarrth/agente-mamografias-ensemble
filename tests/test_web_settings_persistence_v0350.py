from __future__ import annotations

from pathlib import Path

import pytest

import mammography_agent.api as api
from mammography_agent.api import WebEvaluationSettingsRequest


def _settings():
    return {
        "settings_key": "default",
        "inference_device": "cpu",
        "weight_mode": "CUSTOM",
        "weights": {"gmic": 0.30, "nyu": 0.20, "glam": 0.50},
        "threshold_mode": "CUSTOM",
        "decision_threshold": 0.02,
        "updated_at": "2026-08-19T00:00:00+00:00",
    }


def test_web_settings_request_validates_weights_and_threshold():
    req = WebEvaluationSettingsRequest(
        inference_device="cpu",
        weight_mode="CUSTOM",
        ensemble_weights={"gmic": 0.30, "nyu": 0.20, "glam": 0.50},
        threshold_mode="CUSTOM",
        decision_threshold=0.02,
    )
    assert sum(req.ensemble_weights.values()) == pytest.approx(1.0)
    with pytest.raises(Exception):
        WebEvaluationSettingsRequest(
            ensemble_weights={"gmic": 0.4, "nyu": 0.4, "glam": 0.4},
            decision_threshold=0.02,
        )
    with pytest.raises(Exception):
        WebEvaluationSettingsRequest(
            ensemble_weights={"gmic": 0.3, "nyu": 0.2, "glam": 0.5},
            decision_threshold=1.1,
        )


def test_web_settings_get_returns_persisted_postgres_values(monkeypatch):
    monkeypatch.setattr(api, "load_web_evaluation_settings", lambda: _settings())
    result = api.get_web_evaluation_settings()
    assert result["persisted"] is True
    assert result["weights"]["glam"] == pytest.approx(0.50)
    assert result["decision_threshold"] == pytest.approx(0.02)



def test_web_settings_get_migrates_latest_successful_run_when_settings_table_empty(monkeypatch):
    monkeypatch.setattr(api, "load_web_evaluation_settings", lambda: None)
    monkeypatch.setattr(api, "load_latest_web_inference_settings", lambda: {
        "inference_device": "cpu",
        "weight_mode": "CUSTOM",
        "weights": {"gmic": 0.30, "nyu": 0.20, "glam": 0.50},
        "threshold_mode": "CUSTOM",
        "decision_threshold": 0.02,
        "migrated_from_run": True,
        "source_created_at": "2026-08-19T00:00:00+00:00",
    })
    monkeypatch.setattr(api, "save_web_evaluation_settings", lambda **kwargs: {
        "settings_key": "default",
        **kwargs,
        "updated_at": "2026-08-19T00:01:00+00:00",
    })
    result = api.get_web_evaluation_settings()
    assert result["persisted"] is True
    assert result["migrated_from_run"] is True
    assert result["decision_threshold"] == pytest.approx(0.02)
    assert result["weight_mode"] == "CUSTOM"

def test_web_settings_put_uses_web_only_postgres_storage(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "audit", lambda *a, **k: None)

    def fake_save(**kwargs):
        captured.update(kwargs)
        return _settings()

    monkeypatch.setattr(api, "save_web_evaluation_settings", fake_save)
    req = WebEvaluationSettingsRequest(
        inference_device="cpu",
        weight_mode="CUSTOM",
        ensemble_weights={"gmic": 0.30, "nyu": 0.20, "glam": 0.50},
        threshold_mode="CUSTOM",
        decision_threshold=0.02,
    )
    result = api.put_web_evaluation_settings(req)
    assert result["persisted"] is True
    assert captured["threshold_mode"] == "CUSTOM"
    assert captured["weights"] == {"gmic": 0.30, "nyu": 0.20, "glam": 0.50}
    assert captured["decision_threshold"] == pytest.approx(0.02)


def test_streamlit_hydrates_and_explicitly_updates_web_settings():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert '"GET", "/single-cases/web-settings"' in text
    assert '"PUT", "/single-cases/web-settings"' in text
    assert '_hydrate_web_settings(persisted_web_settings, ensemble_config)' in text
    assert 'Actualizar configuración' in text
    assert 'Hay cambios pendientes' not in text
    assert 'guardan automáticamente' not in text


def test_persistence_does_not_write_batch_configuration_files():
    storage = Path("mammography_agent/storage.py").read_text(encoding="utf-8")
    api_text = Path("mammography_agent/api.py").read_text(encoding="utf-8")
    assert "web_evaluation_settings" in storage
    for forbidden in ("config/ensemble.yaml", "config/experiments.yaml", "config/models.yaml"):
        assert forbidden not in storage
    settings_section = api_text.split('class WebEvaluationSettingsRequest', 1)[1]
    assert '.write_text(' not in settings_section
    assert 'open(' not in settings_section
