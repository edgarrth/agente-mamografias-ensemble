from pathlib import Path
import json
import pandas as pd
import mammography_agent.pipeline as pipeline


def _dataset(n=20):
    rows=[]
    for i in range(n):
        gt=i%2
        rows.append({
            "study_id":f"S{i:03d}","patient_id":f"P{i:03d}","ground_truth":gt,"dataset_source":"unit",
            "l_cc":"a","r_cc":"b","l_mlo":"c","r_mlo":"d",
        })
    return pd.DataFrame(rows)


def _fake_infer_factory(calls):
    def fake(df, run_dir: Path, run_id: str):
        calls.append((run_id,len(df)))
        run_dir.mkdir(parents=True,exist_ok=True)
        out=df[["study_id","patient_id","ground_truth","dataset_source"]].copy()
        # Deterministic discrimination sufficient to exercise ranking/metrics.
        out["gmic_score"]=out.ground_truth.map({0:.03,1:.08}).astype(float)
        out["nyu_score"]=out.ground_truth.map({0:.04,1:.09}).astype(float)
        out["glam_score"]=out.ground_truth.map({0:.02,1:.07}).astype(float)
        out.to_csv(run_dir/"raw_model_predictions.csv",index=False)
        return out
    return fake


def test_experimental_flow_keeps_final_isolated_and_reuses_final_cache(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(pipeline,"WORKSPACE_ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"_id",lambda prefix:"experiment-v022-unit" if prefix=="experiment" else f"{prefix}-unit")
    monkeypatch.setattr(pipeline,"load_datasets",lambda datasets,samples=None:_dataset().head(samples) if samples else _dataset())
    monkeypatch.setattr(pipeline,"_infer_three",_fake_infer_factory(calls))
    monkeypatch.setattr(pipeline,"audit",lambda *a,**k:None)
    def fake_resolve(frame, output_dir, *args, **kwargs):
        out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out/"orientation_policy_summary.json").write_text(json.dumps({"policy_id":pipeline.ORIENTATION_POLICY_ID}), encoding="utf-8")
        return frame.copy()
    monkeypatch.setattr(pipeline,"resolve_orientation",fake_resolve)
    monkeypatch.setattr(pipeline,"save_run",lambda *a,**k:None)

    run_dir=pipeline.experimental_test(["unit"],configuration_ratio=.30,seed=42)
    assert run_dir.name=="experiment-v022-unit"
    assert calls==[("experiment-v022-unit",6)]
    assert (run_dir/"configuration_set_manifest.csv").exists()
    assert len(pd.read_csv(run_dir/"final_test_manifest.csv"))==14
    assert not (run_dir/"final_inference").exists()
    assert len(pd.read_csv(run_dir/"all_configurations.csv"))==680
    assert (run_dir/"configuration_score_analysis"/"score_summary.json").exists()
    assert (run_dir/"configuration_score_analysis"/"diagnostic_configurations.csv").exists()

    pipeline.freeze_experiment("experiment-v022-unit")
    pipeline.final_evaluation("experiment-v022-unit")
    assert calls[-1]==("experiment-v022-unit-final",14)
    call_count=len(calls)
    assert (run_dir/"final_score_analysis"/"score_summary.json").exists()
    assert not (run_dir/"final_score_analysis"/"candidate_thresholds.csv").exists()
    assert not (run_dir/"final_score_analysis"/"diagnostic_configurations.csv").exists()
    assert not (run_dir/"final_score_analysis"/"diagnostic_ranking.csv").exists()

    # Second final evaluation must reuse cached raw scores, not invoke models again.
    pipeline.final_evaluation("experiment-v022-unit")
    assert len(calls)==call_count
