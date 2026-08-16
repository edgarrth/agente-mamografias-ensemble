from pathlib import Path


def test_app_image_copies_version_file():
    assert "COPY VERSION /app/VERSION" in Path("docker/app.Dockerfile").read_text(encoding="utf-8")


def test_runner_image_copies_version_file():
    assert "COPY VERSION /runner/VERSION" in Path("docker/model-runner.Dockerfile").read_text(encoding="utf-8")


def test_no_stale_runner_010_health_version():
    text=Path("model_runner/api.py").read_text(encoding="utf-8")
    assert '"version": "0.10.0"' not in text
    assert 'APP_VERSION = "0.29.0"' in text
