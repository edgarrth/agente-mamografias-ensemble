from pathlib import Path
import yaml


def test_glam_blackwell_tolerates_metarepo_malignant_only_label_contract():
    text = Path("docker/model-compat/glam-blackwell.Dockerfile").read_text(encoding="utf-8")
    assert 'cancer_label.get("left_benign", np.nan)' in text
    assert 'cancer_label.get("right_benign", np.nan)' in text
    assert 'cancer_label["left_malignant"]' in text
    assert 'cancer_label["right_malignant"]' in text
    assert "Do not fabricate benign ground truth" in text


def test_glam_build_revision_is_two_for_label_contract_fix():
    cfg = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    glam = cfg["models"]["glam"]["gpu_compatibility"]
    assert glam["build_revision"] == 2
    assert any("malignant-only cancer_label" in p for p in glam["compatibility_code_patches"])
    assert cfg["models"]["gmic"]["gpu_compatibility"]["build_revision"] == 3
    assert cfg["models"]["nyu"]["gpu_compatibility"].get("build_revision", 1) == 1
