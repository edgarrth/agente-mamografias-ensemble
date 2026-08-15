from pathlib import Path
import yaml


def test_gmic_has_separate_blackwell_gpu_image():
    cfg = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    g = cfg["models"]["gmic"]
    assert g["local_image"] == "mammography-model-gmic:research"
    gpu = g["gpu_compatibility"]
    assert gpu["image"] == "mammography-model-gmic:blackwell-cu128"
    assert gpu["torch"] == "2.7.1"
    assert gpu["cuda_wheel"] == "12.8"
    assert gpu["compatibility_code_patches"]


def test_blackwell_dockerfile_preserves_gmic_commit_and_uses_cu128():
    text = Path("docker/model-compat/gmic-blackwell.Dockerfile").read_text(encoding="utf-8")
    assert "3bf4ce81dfa40553f108c8bfaf03bf006e082761" in text
    assert "torch==2.7.1" in text
    assert "torchvision==0.22.1" in text
    assert "https://download.pytorch.org/whl/cu128" in text


def test_runner_gpu_probe_is_fail_safe_and_gpu_does_not_use_legacy_image():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert '@app.post("/models/{model}/gpu-probe")' in text
    assert '@app.post("/models/{model}/ensure-gpu")' in text
    assert 'subprocess.run(["docker", "rm", "-f", container]' in text
    assert "GPU_PROBE_REQUIRED" in text
    assert "selected_image = gpu_image_tag(model)" in text
    assert "selected_image = image_tag(model)" in text
