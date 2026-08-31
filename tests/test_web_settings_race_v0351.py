from __future__ import annotations

from pathlib import Path


def _streamlit_text() -> str:
    return Path("ui/streamlit_app.py").read_text(encoding="utf-8")


def test_failed_initial_get_never_triggers_automatic_settings_write():
    text = _streamlit_text()
    # The configuration API is read during hydration, but PUT exists only behind explicit buttons.
    assert '"GET", "/single-cases/web-settings"' in text
    assert '"PUT", "/single-cases/web-settings"' in text
    assert "autosave_allowed(" not in text
    assert "persistencia automática" not in text.lower()
    assert "Actualizar configuración" in text
    assert "update_clicked = action_col.button(" in text
    assert "if update_clicked:" in text
    assert "saved_settings = _persist_web_settings(" in text


def test_new_session_blocks_evaluation_until_postgres_settings_are_hydrated():
    text = _streamlit_text()
    assert "return None, last_error" in text
    assert "if settings is None:" in text
    assert "return False" in text
    assert "settings_hydrated = bool(st.session_state.get(\"_web_settings_hydrated\"))" in text
    assert "ready = study_ready and weights_valid and runtimes_ready and settings_hydrated" in text
    assert "Reintentar configuración" in text


def test_successful_hydration_sets_persisted_fingerprint_without_writing():
    text = _streamlit_text()
    assert '"_web_settings_last_saved_fingerprint" not in st.session_state' in text
    assert 'st.session_state["_web_settings_last_saved_fingerprint"] = _web_settings_fingerprint(ensemble_config)' in text
    hydration_segment = text.split('if just_hydrated and settings_hydrated', 1)[1].split('with st.sidebar:', 1)[0]
    assert "_persist_web_settings(" not in hydration_segment


def test_unsaved_configuration_changes_can_run_but_are_not_persisted_implicitly():
    text = _streamlit_text()
    assert "ready = study_ready and weights_valid and runtimes_ready and settings_hydrated" in text
    assert "Hay cambios de configuración pendientes" not in text
    assert "if update_clicked:" in text
    hydration_segment = text.split('if just_hydrated and settings_hydrated', 1)[1].split('with st.sidebar:', 1)[0]
    assert "_persist_web_settings(" not in hydration_segment


def test_restore_baseline_is_an_explicit_separate_action():
    text = _streamlit_text()
    assert "Restaurar configuración base" in text
    assert "if reset_clicked:" in text
    assert 'weight_mode="Configuración base"' in text
    assert 'threshold_mode="Configuración base"' in text
    assert 'st.session_state["_web_settings_hydrated"] = False' in text
