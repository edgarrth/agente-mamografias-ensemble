from __future__ import annotations

from pathlib import Path


def test_durable_docker_volumes_have_stable_explicit_names():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "name: ${POSTGRES_VOLUME_NAME:-mammography-postgres-data}" in compose
    assert "name: ${MINIO_VOLUME_NAME:-mammography-minio-data}" in compose
    assert "name: ${WEB_SCRATCH_VOLUME_NAME:-mammography-web-scratch}" in compose


def test_env_example_documents_stable_volume_names():
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "POSTGRES_VOLUME_NAME=mammography-postgres-data" in env
    assert "MINIO_VOLUME_NAME=mammography-minio-data" in env
    assert "WEB_SCRATCH_VOLUME_NAME=mammography-web-scratch" in env


def test_requested_routine_streamlit_messages_are_removed():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    forbidden = (
        "Hay cambios de configuración pendientes",
        "Hay cambios pendientes. Pulse «Actualizar configuración»",
        "La acción permanece bloqueada hasta que finalice la evaluación actual",
        "Modo CPU: la evaluación Web no requiere validación GPU",
        "Los parámetros de esta sección se aplican únicamente a las evaluaciones iniciadas desde Streamlit",
    )
    for message in forbidden:
        assert message not in text


def test_unsaved_web_configuration_no_longer_blocks_current_inference():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert "ready = study_ready and weights_valid and runtimes_ready and settings_hydrated" in text
    assert "and not settings_dirty" not in text.split("ready = study_ready and weights_valid and runtimes_ready and settings_hydrated", 1)[0][-100:]
    assert "Actualizar configuración" in text
    assert "if update_clicked:" in text


def test_legacy_volume_migration_helper_is_guarded():
    script = Path("scripts/migrate-legacy-durable-volumes.sh").read_text(encoding="utf-8")
    assert "volume_in_use" in script
    assert "volume_nonempty" in script
    assert "No se sobrescribirá" in script
    assert "LEGACY_POSTGRES_VOLUME_NAME" in script
    assert "LEGACY_MINIO_VOLUME_NAME" in script
