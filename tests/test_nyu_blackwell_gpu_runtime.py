from pathlib import Path
import yaml


def _models():
    return yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))["models"]


def test_nyu_has_model_owned_blackwell_gpu_profile():
    nyu = _models()["nyu"]
    gpu = nyu["gpu_compatibility"]
    assert gpu["enabled"] is True
    assert gpu["profile"] == "blackwell-cu128"
    assert gpu["image"] == "mammography-model-nyu:blackwell-cu128"
    assert gpu["torch"] == "2.7.1"
    assert gpu["torchvision"] == "0.22.1"
    assert gpu["cuda_wheel"] == "12.8"


def test_nyu_blackwell_dockerfile_preserves_commit_and_modernizes_runtime_only():
    text = Path("docker/model-compat/nyu-blackwell.Dockerfile").read_text(encoding="utf-8")
    assert "de2b0855d02984df0f516008bb4513ff71460e21" in text
    assert "torch==2.7.1" in text
    assert "torchvision==0.22.1" in text
    assert "https://download.pytorch.org/whl/cu128" in text
    assert "src/modeling/run_model.py" in text
    assert "src/heatmaps/run_producer.py" in text
    assert "torch.backends.cudnn.is_available()" in text
    assert "sample_image_model.p" not in text  # weights come unchanged from the pinned upstream repo


def test_gpu_audit_metadata_is_model_owned_not_gmic_hardcoded():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert 'compat.get("compatibility_code_patches", [])' in text
    assert '"compatibility_code_patch": "torch.has_cudnn' not in text
