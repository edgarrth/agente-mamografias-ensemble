from pathlib import Path
import yaml


def _models():
    return yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))["models"]


def test_glam_has_model_owned_blackwell_gpu_profile():
    glam = _models()["glam"]
    gpu = glam["gpu_compatibility"]
    assert gpu["enabled"] is True
    assert gpu["profile"] == "blackwell-cu128"
    assert gpu["image"] == "mammography-model-glam:blackwell-cu128"
    assert gpu["torch"] == "2.7.1"
    assert gpu["torchvision"] == "0.22.1"
    assert gpu["cuda_wheel"] == "12.8"
    assert len(gpu["compatibility_code_patches"]) >= 5


def test_glam_blackwell_dockerfile_preserves_commit_and_runtime_contract():
    text = Path("docker/model-compat/glam-blackwell.Dockerfile").read_text(encoding="utf-8")
    assert "17a0019860441e2ea8d7b7c7e0aaeada735e871f" in text
    assert "torch==2.7.1" in text
    assert "torchvision==0.22.1" in text
    assert "https://download.pytorch.org/whl/cu128" in text
    assert 'matplotlib.use(\\"Agg\\")' in text or 'TkAgg' in text  # sed transformation is explicit in the Dockerfile
    assert "torch.backends.cudnn.is_available()" in text
    assert "device=camlocal.device" in text
    assert "device=x_original_pytorch.device" in text
    assert 'rounding_mode="floor"' in text
    assert "align_corners=True" in text


def test_glam_gpu_profile_is_model_metadata_not_environment_profile():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    assert "GPU_RUNTIME_PROFILE" not in compose
    assert "GPU_RUNTIME_PROFILE" not in env
    assert "GLAM_DEVICE" in compose
