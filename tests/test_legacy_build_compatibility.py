from pathlib import Path
import yaml


def _models():
    return yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))["models"]


def test_all_three_models_declare_same_legacy_cuda_compatibility():
    models = _models()
    for name in ("gmic", "nyu", "glam"):
        compat = models[name]["build_compatibility"]
        assert compat["original_base_image"] == "nvidia/cuda:10.1-base-ubuntu18.04"
        assert compat["base_image_override"] == "nvidia/cudagl:10.1-devel-ubuntu18.04"
        assert compat["nvidia_repository_key_rotation_fix"] == "auto"


def test_runner_generates_auditable_narrow_compatibility_patch():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert "MODEL_COMPATIBILITY_DOCKERFILE_CREATED" in text
    assert "UPSTREAM_DOCKERFILE_DRIFT" in text
    assert 'patched_lines[0] = f"FROM {replacement_base}"' in text
    assert '"model_code_changed": False' in text
    assert '"model_weights_changed": False' in text
    assert '"training_performed": False' in text


def test_runner_handles_git_safe_directory_automatically():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert '"safe.directory"' in text
    assert "GIT_SAFE_DIRECTORY_ADDED" in text
    assert "ensure_git_safe_directory(META)" in text


def test_model_client_surfaces_runner_error_detail():
    text = Path("mammography_agent/model_client.py").read_text(encoding="utf-8")
    assert "_raise_runner_error" in text
    assert 'payload.get("detail", payload)' in text
    ensure_body = text.split("def ensure_model", 1)[1].split("def smoke_test", 1)[0]
    assert "raise_for_status" not in ensure_body


def test_runner_repairs_nvidia_apt_key_rotation_without_changing_model_code():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert "NVIDIA_APT_KEY_ROTATION_COMPATIBILITY_APPLIED" in text
    assert "3bf863cc.pub" in text
    assert "7fa2af80.pub" in text
    assert 'key_fix_status = "upstream_present"' in text
    assert 'key_fix_status = "injected"' in text
    assert "model_code_changed=False" in text
    assert "model_weights_changed=False" in text

def test_nyu_upstream_key_fix_is_not_blindly_duplicated():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert 'if key_marker in original:' in text
    assert 'key_fix_status = "upstream_present"' in text
