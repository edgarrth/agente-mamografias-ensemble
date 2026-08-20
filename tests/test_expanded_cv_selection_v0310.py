import json
from pathlib import Path

import numpy as np
import pandas as pd

from mammography_agent.ensemble.cv_selection import (
    build_stratified_folds,
    evaluate_cv_grid,
    run_expanded_cv_selection,
    select_cv_candidate,
)


def _scores(n=100):
    # 20% positives: enough for five stratified folds in a fast unit test.
    y=np.array(([0]*80)+([1]*20),dtype=int)
    rng=np.random.default_rng(42)
    g=np.clip(.10 + y*.25 + rng.normal(0,.10,n),0,1)
    nscore=np.clip(.08 + y*.20 + rng.normal(0,.12,n),0,1)
    glam=np.clip(.05 + y*.30 + rng.normal(0,.09,n),0,1)
    return pd.DataFrame({
        "study_id":[f"S{i:03d}" for i in range(n)],
        "patient_id":[f"P{i:03d}" for i in range(n)],
        "dataset_source":"unit",
        "ground_truth":y,
        "gmic_score":g,
        "nyu_score":nscore,
        "glam_score":glam,
    })


def test_stratified_folds_cover_each_study_once_and_preserve_positive_counts():
    df=_scores()
    folds=build_stratified_folds(df,5,42)
    assert len(folds)==len(df)
    assert sorted(folds.fold.unique().tolist())==[1,2,3,4,5]
    assert folds.study_id.nunique()==len(df)
    positives=folds.groupby("fold").ground_truth.sum().tolist()
    assert positives==[4,4,4,4,4]


def test_expanded_cv_grid_has_680_candidates_and_3400_fold_evaluations():
    df=_scores()
    folds, fold_metrics, summary=evaluate_cv_grid(df,5,42)
    assert len(folds)==100
    assert len(fold_metrics)==3400
    assert len(summary)==680
    assert summary.weight_id.nunique()==40
    assert summary.threshold_id.nunique()==17
    assert set(fold_metrics.threshold_source)=={"cv_training_fold_score_quantile"}
    assert set(fold_metrics.ground_truth_used_for_threshold_derivation)=={False}


def test_cv_selection_writes_additive_artifacts_without_final_scores(tmp_path: Path):
    df=_scores()
    out=run_expanded_cv_selection(
        df,tmp_path,n_splits=5,seed=42,
        input_scores_sha256="abc",final_manifest_sha256="def",
    )
    for name in [
        "fold_assignments.csv","fold_metrics.csv","candidate_cv_summary.csv",
        "ranking_cv.csv","best_configuration.json","selection_protocol.json",
    ]:
        assert (out/name).exists(), name
    best=json.loads((out/"best_configuration.json").read_text())
    assert best["selection_protocol"]=="stratified_5fold_cv_expanded_grid_v0310"
    assert best["final_test_scores_used"] is False
    assert best["models_reexecuted"] is False
    assert best["ground_truth_used_for_threshold_derivation"] is False
    protocol=json.loads((out/"selection_protocol.json").read_text())
    assert protocol["candidate_configurations"]==680
    assert protocol["fold_evaluations"]==3400


def test_cv_selector_returns_one_candidate():
    _,_,summary=evaluate_cv_grid(_scores(),5,42)
    selected=select_cv_candidate(summary)
    assert str(selected.weight_id).startswith("W")
    assert str(selected.threshold_id).startswith("T")


def test_freeze_prefers_expanded_cv_selection_when_present(tmp_path: Path, monkeypatch):
    import yaml
    import mammography_agent.pipeline as pipeline
    run=tmp_path/"output"/"experiments"/"experiment-unit"
    expanded=run/"configuration_selection_v0310"
    expanded.mkdir(parents=True)
    (run/"best_configuration.json").write_text(json.dumps({
        "w_gmic":.2,"w_nyu":.2,"w_glam":.6,"threshold":.04942
    }))
    (expanded/"best_configuration.json").write_text(json.dumps({
        "w_gmic":.3,"w_nyu":.2,"w_glam":.5,"threshold":.08
    }))
    monkeypatch.setattr(pipeline,"WORKSPACE_ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"audit",lambda *a,**k:None)
    path=pipeline.freeze_experiment("experiment-unit")
    frozen=yaml.safe_load(path.read_text())
    assert frozen["selection_source"]=="expanded_cv_v0310"
    assert frozen["weights"]=={"gmic":.3,"nyu":.2,"glam":.5}
    assert frozen["threshold"]==.08
