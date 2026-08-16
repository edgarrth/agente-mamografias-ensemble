import json
from pathlib import Path
import pandas as pd
from mammography_agent.score_analysis import analyze_score_frame


def test_score_analysis_writes_research_evidence(tmp_path: Path):
    df=pd.DataFrame({
        "study_id":["A","B","C","D","E","F"],
        "ground_truth":[0,0,0,1,1,1],
        "gmic_score":[.10,.12,.08,.03,.04,.05],
        "nyu_score":[.03,.04,.05,.06,.07,.08],
        "glam_score":[.09,.10,.08,.02,.03,.04],
    })
    out=analyze_score_frame(df,tmp_path,"unit-test")
    assert out==tmp_path
    for name in [
        "score_summary.json","model_metrics.csv","score_distribution.csv",
        "model_correlations.csv","roc_points.csv","candidate_thresholds.csv",
        "diagnostic_configurations.csv","diagnostic_ranking.csv","score_analysis_report.md",
    ]:
        assert (tmp_path/name).exists(), name
    summary=json.loads((tmp_path/"score_summary.json").read_text())
    assert summary["research_guards"]["score_inversion_performed"] is False
    assert summary["research_guards"]["calibration_performed"] is False
    assert summary["threshold_strategy"]["ground_truth_used_for_threshold_derivation"] is False
    candidates=pd.read_csv(tmp_path/"candidate_thresholds.csv")
    assert len(candidates)==80
    assert set(candidates.threshold_source)=={"analysis_score_quantile"}
    assert summary["baseline"]["classification_metrics"]["specificity"] is not None
    assert "balanced_accuracy" in summary["baseline"]["classification_metrics"]
    assert summary["threshold_strategy"]["diagnostic_results_eligible_for_freeze"] is False
    diagnostic=pd.read_csv(tmp_path/"diagnostic_configurations.csv")
    assert len(diagnostic)==80
    assert set(diagnostic.diagnostic_only)=={True}
    assert set(diagnostic.eligible_for_freeze)=={False}
    assert {"specificity","precision_ppv","npv","fpr","balanced_accuracy"}.issubset(diagnostic.columns)
    metrics=pd.read_csv(tmp_path/"model_metrics.csv")
    assert set(metrics.model)=={"gmic","nyu","glam","baseline_ensemble"}
