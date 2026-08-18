from pathlib import Path
import json
import pandas as pd
import pytest
import mammography_agent.pipeline as pipeline


def _frame(n=65):
    return pd.DataFrame([
        {"study_id":f"S{i:03d}","patient_id":f"P{i:03d}","ground_truth":i%2,"dataset_source":"unit",
         "l_cc":"a","r_cc":"b","l_mlo":"c","r_mlo":"d","horizontal_flip":"NO"}
        for i in range(n)
    ])


def _fake_infer(calls, fail_chunk=None):
    def fake(df, run_dir: Path, run_id: str):
        calls.append(run_id)
        if fail_chunk is not None and run_id.endswith(f"c{fail_chunk:04d}"):
            raise RuntimeError("synthetic interruption")
        run_dir.mkdir(parents=True,exist_ok=True)
        out=df[["study_id","patient_id","ground_truth","dataset_source"]].copy()
        out["gmic_score"]=0.1+out.ground_truth*0.7
        out["nyu_score"]=0.2+out.ground_truth*0.6
        out["glam_score"]=0.15+out.ground_truth*0.65
        out.to_csv(run_dir/"raw_model_predictions.csv",index=False)
        (run_dir/"xai_artifacts.json").write_text(json.dumps({"gmic":[],"nyu":[],"glam":[]}),encoding="utf-8")
        pd.DataFrame([{"model":"gmic"},{"model":"nyu"},{"model":"glam"}]).to_csv(run_dir/"resource_metrics.csv",index=False)
        return out
    return fake


def test_chunked_inference_checkpoint_and_resume(tmp_path, monkeypatch):
    df=_frame(65); calls=[]
    monkeypatch.setattr(pipeline,"_infer_three",_fake_infer(calls,fail_chunk=1))
    monkeypatch.setattr(pipeline,"audit",lambda *a,**k:None)
    with pytest.raises(RuntimeError,match="synthetic interruption"):
        pipeline._infer_three_chunked(df,tmp_path/"infer","run",chunk_size=25)
    assert (tmp_path/"infer/chunks/0000/chunk_status.json").exists()
    assert json.loads((tmp_path/"infer/chunks/0000/chunk_status.json").read_text())["status"]=="SUCCESS"
    assert json.loads((tmp_path/"infer/chunks/0001/chunk_status.json").read_text())["status"]=="FAILED"

    calls2=[]
    monkeypatch.setattr(pipeline,"_infer_three",_fake_infer(calls2))
    out=pipeline._infer_three_chunked(df,tmp_path/"infer","run",chunk_size=25)
    assert len(out)==65
    assert calls2==["run-c0001","run-c0002"]  # chunk 0000 reused
    progress=json.loads((tmp_path/"infer/chunk_progress.json").read_text())
    assert progress["status"]=="SUCCESS"
    assert progress["reused_chunks"]==1
    assert progress["completed_studies"]==65


def test_success_chunk_hash_mismatch_is_not_silently_reused(tmp_path, monkeypatch):
    df=_frame(30); calls=[]
    monkeypatch.setattr(pipeline,"_infer_three",_fake_infer(calls))
    monkeypatch.setattr(pipeline,"audit",lambda *a,**k:None)
    pipeline._infer_three_chunked(df,tmp_path/"infer","run",chunk_size=25)
    changed=df.copy(); changed.loc[0,"ground_truth"]=1-changed.loc[0,"ground_truth"]
    with pytest.raises(RuntimeError,match="input hash differs"):
        pipeline._infer_three_chunked(changed,tmp_path/"infer","run",chunk_size=25)


def test_orientation_chunked_resume(tmp_path, monkeypatch):
    df=_frame(55); calls=[]
    def fake_resolve(sub, out, run_id):
        calls.append(run_id); out=Path(out); out.mkdir(parents=True,exist_ok=True)
        resolved=sub.copy(); resolved.to_csv(out/"resolved_manifest.csv",index=False)
        pd.DataFrame({"study_id":resolved.study_id,"triggered":False,"orientation_changed":False}).to_csv(out/"orientation_resolution.csv",index=False)
        pd.DataFrame({"study_id":resolved.study_id.repeat(4),"model_view":["L-CC","R-CC","L-MLO","R-MLO"]*len(resolved)}).to_csv(out/"orientation_view_evidence.csv",index=False)
        pd.DataFrame({"study_id":resolved.study_id.repeat(4),"model_view":["L-CC","R-CC","L-MLO","R-MLO"]*len(resolved)}).to_csv(out/"orientation_original_views.csv",index=False)
        return resolved
    monkeypatch.setattr(pipeline,"resolve_orientation",fake_resolve)
    monkeypatch.setattr(pipeline,"audit",lambda *a,**k:None)
    out=pipeline._resolve_orientation_chunked(df,tmp_path/"orientation","ori",chunk_size=25)
    assert len(out)==55 and len(calls)==3
    calls.clear()
    out2=pipeline._resolve_orientation_chunked(df,tmp_path/"orientation","ori",chunk_size=25)
    assert len(out2)==55 and calls==[]
    summary=json.loads((tmp_path/"orientation/orientation_policy_summary.json").read_text())
    assert summary["execution_mode"]=="chunked_resumable"


def test_default_formal_chunk_size_is_25(monkeypatch):
    monkeypatch.setattr(pipeline,"load_yaml",lambda name:{"formal_inference":{"chunk_size":25,"resume_enabled":True,"cache_policy":"SUCCESS_MARKER_AND_HASHES"}} if name=="experiments.yaml" else {})
    assert pipeline._formal_inference_config()["chunk_size"]==25
