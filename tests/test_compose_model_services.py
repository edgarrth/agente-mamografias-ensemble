from pathlib import Path
import yaml


def _has_docker_socket_mount(service: dict) -> bool:
    for v in service.get("volumes", []):
        if isinstance(v, str) and v == "/var/run/docker.sock:/var/run/docker.sock":
            return True
        if isinstance(v, dict) and v.get("source") == "/var/run/docker.sock" and v.get("target") == "/var/run/docker.sock":
            return True
    return False


def test_compose_has_one_model_runner_and_no_per_model_controllers():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert "model-runner" in services
    assert "gmic-runtime" not in services
    assert "nyu-runtime" not in services
    assert "glam-runtime" not in services
    runner = services["model-runner"]
    assert runner["container_name"] == "mammography-model-runner"
    assert _has_docker_socket_mount(runner)
    assert runner["environment"]["DEFAULT_MODEL_DEVICE"] == "${DEFAULT_MODEL_DEVICE:-cpu}"
    assert runner["environment"]["GMIC_DEVICE"] == "${GMIC_DEVICE:-cpu}"
    assert runner["environment"]["NYU_DEVICE"] == "${NYU_DEVICE:-cpu}"
    assert runner["environment"]["GLAM_DEVICE"] == "${GLAM_DEVICE:-cpu}"
    assert runner["environment"]["DOCKER_HOST"] == "unix:///var/run/docker.sock"
    assert runner["build"]["args"]["DOCKER_CLI_IMAGE"] == "${DOCKER_CLI_IMAGE:-docker:29-cli}"
    assert all(not _has_docker_socket_mount(services[s]) for s in ["fastapi", "streamlit", "bootstrap"])


def test_fastapi_routes_only_to_single_model_runner():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["fastapi"]["environment"]
    assert "http://model-runner:8010" in env["MODEL_RUNNER_URL"]
    assert "GMIC_RUNTIME_URL" not in env
    assert "NYU_RUNTIME_URL" not in env
    assert "GLAM_RUNTIME_URL" not in env
