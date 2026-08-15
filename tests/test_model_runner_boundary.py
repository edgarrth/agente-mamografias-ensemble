from pathlib import Path
import yaml


def test_model_runner_dockerfile_has_no_ml_framework_stack():
    text = Path("docker/model-runner.Dockerfile").read_text(encoding="utf-8").lower()
    install_lines = "\n".join(line for line in text.splitlines() if "pip install" in line)
    for forbidden in ("torch", "tensorflow", "cuda", "cudnn"):
        assert forbidden not in install_lines


def test_model_runner_uses_official_docker_cli_not_debian_docker_io():
    text = Path("docker/model-runner.Dockerfile").read_text(encoding="utf-8").lower()
    assert "arg docker_cli_image=docker:29-cli" in text
    assert "from ${docker_cli_image}" in text
    assert "docker.io" not in "\n".join(
        line for line in text.splitlines() if line.strip().startswith("run ")
    )


def test_runner_exposes_docker_doctor_endpoint():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert '@app.get("/doctor")' in text
    assert "direct_socket_ping" in text
    assert '"docker", "version"' in text
    assert '"docker", "info"' in text


def test_models_have_distinct_local_images():
    cfg = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    tags = [cfg["models"][m]["local_image"] for m in ("gmic", "nyu", "glam")]
    assert len(set(tags)) == 3
    assert all(tag.startswith("mammography-model-") for tag in tags)
