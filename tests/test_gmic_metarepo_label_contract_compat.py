from pathlib import Path
import yaml


def test_gmic_blackwell_tolerates_metarepo_malignant_only_label_contract():
    text = Path("docker/model-compat/gmic-blackwell.Dockerfile").read_text(encoding="utf-8")
    assert 'cancer_label.get("left_benign", np.nan)' in text
    assert 'cancer_label.get("right_benign", np.nan)' in text
    assert 'cancer_label["left_malignant"]' in text
    assert 'cancer_label["right_malignant"]' in text
    assert "Do not invent benign ground truth" in text


def test_gmic_build_revision_is_three_for_label_contract_fix():
    cfg = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    g = cfg["models"]["gmic"]["gpu_compatibility"]
    assert g["build_revision"] == 3
    assert any("malignant-only cancer_label" in p for p in g["compatibility_code_patches"])
    assert cfg["models"]["nyu"]["gpu_compatibility"].get("build_revision", 1) == 1
    assert cfg["models"]["glam"]["gpu_compatibility"].get("build_revision", 1) == 1


def test_generated_metarepository_batch_does_not_invent_benign_labels():
    # The metarepository contract used by this project needs only breast-level
    # malignant labels. The GMIC compatibility layer must tolerate that contract.
    source = Path("mammography_agent/metarepo_format.py").read_text(encoding="utf-8")
    assert '"left_malignant":left' in source
    assert '"right_malignant":right' in source
    assert '"left_benign"' not in source
    assert '"right_benign"' not in source
