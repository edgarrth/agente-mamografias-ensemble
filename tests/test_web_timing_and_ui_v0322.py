from __future__ import annotations

from pathlib import Path

from mammography_agent import object_storage, single_case


def test_progress_payload_preserves_stage_elapsed_times(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    run_id = "web-20260818T230000Z-timing01"
    run_dir = tmp_path / "output" / "single_cases" / run_id
    run_dir.mkdir(parents=True)
    single_case._write_web_progress(
        run_dir,
        run_id,
        stage="MODELS",
        state="RUNNING",
        message="Ejecutando: GLAM.",
        models={
            "gmic": {"state": "SUCCESS", "elapsed_seconds": 51.2},
            "nyu": {"state": "SUCCESS", "elapsed_seconds": 22.4},
            "glam": {"state": "RUNNING"},
        },
        stages={
            "PREPARATION": {"state": "SUCCESS", "elapsed_seconds": 2.5},
            "ORIENTATION": {"state": "SUCCESS", "elapsed_seconds": 1.2},
            "MODEL_INPUT_PREPARATION": {"state": "SUCCESS", "elapsed_seconds": 183.7},
            "ENSEMBLE": {"state": "PENDING"},
            "PERSISTENCE": {"state": "PENDING"},
        },
    )
    payload = single_case.get_single_case_progress(run_id)
    assert payload["stages"]["MODEL_INPUT_PREPARATION"]["elapsed_seconds"] == 183.7
    assert payload["models"]["gmic"]["elapsed_seconds"] == 51.2


def test_web_ui_blocks_duplicate_run_and_collapses_technical_sections():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.session_state["web_eval_running"] = True' in text
    assert 'disabled=(not ready) or running' in text
    assert '"Evaluación en curso…" if running else "Ejecutar evaluación"' in text
    assert 'with st.expander("Resultados por modelo", expanded=False):' in text
    assert 'with st.expander("Tiempos de ejecución", expanded=False):' in text
    assert 'with st.expander("Preparación del estudio", expanded=False):' in text
    assert 'with st.expander("Normalización de orientación", expanded=False):' in text
    assert '"Preparación de entradas para modelos"' in text
    assert 'Trazabilidad de evidencias habilitada' not in text


def test_web_visible_model_timing_uses_wall_clock_not_runtime_metric():
    text = Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert "model_started = time.monotonic()" in text
    assert "wall_elapsed = time.monotonic() - model_started" in text
    assert 'progress_callback(model=model, state="SUCCESS", elapsed_seconds=wall_elapsed)' in text
    # Runtime metrics remain available for diagnostics, but no longer drive Web progress timing.
    assert 'resources.append({"model":model,**(result.get("resource_metrics") or {})})' in text


def test_model_input_preparation_has_independent_timing_callback():
    text = Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert "stage_progress_callback=None" in text
    assert 'stage="MODEL_INPUT_PREPARATION", state="RUNNING"' in text
    assert 'stage="MODEL_INPUT_PREPARATION", state="SUCCESS"' in text


def test_minio_status_exposes_configurable_console_url(monkeypatch):
    monkeypatch.setenv("MINIO_CONSOLE_PUBLIC_URL", "http://example.test:9001")
    cfg = object_storage.settings()
    assert cfg["console_public_url"] == "http://example.test:9001"
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "MINIO_CONSOLE_PUBLIC_URL" in compose
    assert "MINIO_CONSOLE_PUBLIC_URL=http://localhost:9001" in env
