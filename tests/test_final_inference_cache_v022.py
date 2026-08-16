from pathlib import Path


def test_final_evaluation_reuses_existing_inference_cache():
    text=Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert 'cached_scores=final_inference_dir/"raw_model_predictions.csv"' in text
    assert 'audit("FINAL_TEST_SCORES_REUSED"' in text
    assert 'refusing silent re-inference' in text


def test_experiment_plan_enforces_final_isolation_before_freeze():
    text=Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert '"final_inference_before_freeze":False' in text
    assert '"configuration_inference_before_freeze":True' in text
