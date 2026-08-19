from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "deployment" / "production"


def test_production_compose_uses_prebuilt_images_only():
    text = (PROD / "docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "build:" not in text
    assert "${APP_IMAGE" in text
    assert "${MODEL_RUNNER_IMAGE" in text
    assert "${EDGE_IMAGE" in text
    assert "${RUNTIME_ASSETS_IMAGE" in text


def test_production_profile_does_not_bind_host_workspace_or_config():
    text = (PROD / "docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "HOST_WORKSPACE" not in text
    assert "./config:" not in text
    assert "runtime_workspace:/workspace" in text
    assert "/var/run/docker.sock" in text


def test_only_edge_publishes_public_ports():
    data = yaml.safe_load((PROD / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    services = data["services"]
    publishers = {name: svc.get("ports") for name, svc in services.items() if svc.get("ports")}
    assert set(publishers) == {"edge"}
    assert "80:80" in publishers["edge"]
    assert "443:443" in publishers["edge"]


def test_production_runner_is_forced_to_cpu_at_deployment_level():
    data = yaml.safe_load((PROD / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    env = data["services"]["model-runner"]["environment"]
    assert env["DEFAULT_MODEL_DEVICE"] == "cpu"
    assert env["GMIC_DEVICE"] == "cpu"
    assert env["NYU_DEVICE"] == "cpu"
    assert env["GLAM_DEVICE"] == "cpu"
    assert env["ALLOW_GPU"] == "false"


def test_runtime_assets_seed_only_runtime_and_models():
    script = (ROOT / "scripts" / "production" / "publish-platform-images.sh").read_text(encoding="utf-8")
    assert '"$TMP_DIR/seed/runtime"' in script
    assert '"$TMP_DIR/seed/models"' in script
    assert 'cp -a "$HOST_WORKSPACE/datasets' not in script
    assert 'cp -a "$HOST_WORKSPACE/output' not in script


def test_pull_script_preserves_validated_local_model_names():
    text = (ROOT / "scripts" / "production" / "pull-production-images.sh").read_text(encoding="utf-8")
    for ref in (
        "mammography-model-gmic:research",
        "mammography-model-nyu:research",
        "mammography-model-glam:research",
        "mammography-model-gmic:blackwell-cu128",
        "mammography-model-nyu:blackwell-cu128",
        "mammography-model-glam:blackwell-cu128",
    ):
        assert ref in text


def test_production_edge_has_auth_and_reverse_proxy():
    text = (PROD / "Caddyfile").read_text(encoding="utf-8")
    assert "basic_auth" in text
    assert "reverse_proxy streamlit:8501" in text
