import json
import pandas as pd
from mammography_agent import pipeline


def test_normal_test_completes_voting_metrics_and_reports_after_three_model_scores(monkeypatch, tmp_path):
    df = pd.DataFrame({
        "study_id": ["S1", "S2"],
        "patient_id": ["P1", "P2"],
        "ground_truth": [0, 1],
        "dataset_source": ["cbis_ddsm", "cbis_ddsm"],
    })
    scores = df.copy()
    scores["gmic_score"] = [0.10, 0.90]
    scores["nyu_score"] = [0.20, 0.80]
    scores["glam_score"] = [0.30, 0.70]

    monkeypatch.setattr(pipeline, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "load_datasets", lambda datasets, samples=None: df.copy())
    monkeypatch.setattr(pipeline, "_infer_three", lambda frame, run_dir, run_id: scores.copy())
    monkeypatch.setattr(pipeline, "save_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "audit", lambda *args, **kwargs: None)

    run_dir = pipeline.normal_test(["cbis_ddsm"], samples=2, max_runtime_minutes=None)

    pred = pd.read_csv(run_dir / "predictions.csv")
    assert len(pred) == 2
    assert pred.loc[0, "classification"] == "NO_CANCER"
    assert pred.loc[1, "classification"] == "CANCER"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["tp"] == 1 and metrics["tn"] == 1
    assert (run_dir / "configuration_used.yaml").exists()
    assert (run_dir / "normal_test_report.md").exists()
