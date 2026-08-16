from pathlib import Path

def test_orientation_preflight_cli_is_preprocess_only():
    text=Path("experiments/orientation_preflight.py").read_text(encoding="utf-8")
    assert "audit_existing_run" in text
    policy=Path("mammography_agent/orientation_policy.py").read_text(encoding="utf-8")
    assert "classifier_inference_performed" in policy
    assert "ground_truth_used" in policy
    assert "model_scores_used" in policy
