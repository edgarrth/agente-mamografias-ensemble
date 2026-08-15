from pathlib import Path
import yaml


def test_gmic_blackwell_preserves_legacy_integer_index_semantics():
    text=Path("docker/model-compat/gmic-blackwell.Dockerfile").read_text(encoding="utf-8")
    assert 'max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode="floor")' in text
    assert 'max_idx_x = max_linear_idx / W_map|max_idx_x = torch.div' in text


def test_gmic_patch_is_declared_in_model_metadata():
    cfg=yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    patches=cfg["models"]["gmic"]["gpu_compatibility"]["compatibility_code_patches"]
    assert any("integer quotient/remainder" in p for p in patches)


def test_gmic_build_revision_forces_one_time_rebuild_without_touching_other_profiles():
    cfg=yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    assert cfg["models"]["gmic"]["gpu_compatibility"]["build_revision"] == 3
    assert cfg["models"]["nyu"]["gpu_compatibility"].get("build_revision", 1) == 1
    assert cfg["models"]["glam"]["gpu_compatibility"].get("build_revision", 1) == 1
    runner=Path("model_runner/api.py").read_text(encoding="utf-8")
    assert "previous_revision != build_revision" in runner
    assert "probe.unlink()" in runner
