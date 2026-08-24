from pathlib import Path


def test_final_evaluation_persists_individual_and_ensemble_comparison():
    text=Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert 'final_model_comparison.csv' in text
    assert 'individual_models_reference_threshold_0_5' in text
    assert 'final_test_manifest.csv changed after experiment planning' in text
    assert 'Final Test contains formally excluded diagnostic patients' in text
