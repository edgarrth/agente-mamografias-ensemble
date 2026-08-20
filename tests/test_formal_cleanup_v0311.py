from pathlib import Path
import json
import pandas as pd

from mammography_agent import pipeline


def test_inference_cleanup_preserves_resume_contract_and_retains_xai(tmp_path):
    cdir=tmp_path/"0000"; cdir.mkdir()
    sub=pd.DataFrame([{"study_id":"S1","patient_id":"P1","ground_truth":0,"dataset_source":"rsna","l_cc":"a","r_cc":"b","l_mlo":"c","r_mlo":"d"}])
    pred=pd.DataFrame([{"study_id":"S1","patient_id":"P1","ground_truth":0,"dataset_source":"rsna","gmic_score":0.1,"nyu_score":0.2,"glam_score":0.3}])
    pred.to_csv(cdir/"raw_model_predictions.csv",index=False)
    batch=cdir/"model_batch"; batch.mkdir()
    (batch/"images").mkdir(); (batch/"images"/"source.png").write_bytes(b"big")
    (batch/"preprocessed").mkdir()
    (batch/"data.pkl").write_bytes(b"pickle")
    for model in pipeline.MODELS:
        (batch/f"{model}.csv").write_text("score\n0.1\n",encoding="utf-8")
    (batch/"study_order.csv").write_text("study_id,study_key\nS1,S1\n",encoding="utf-8")
    xai=batch/"preprocessed"/"a.png"; xai.write_bytes(b"png")
    (cdir/"xai_artifacts.json").write_text(json.dumps({"gmic":[str(xai)],"nyu":[],"glam":[]}),encoding="utf-8")
    status={"status":"SUCCESS","studies":1,"input_manifest_sha256":pipeline._frame_sha256(sub),"predictions_sha256":pipeline._sha256_file(cdir/"raw_model_predictions.csv")}
    (cdir/"chunk_status.json").write_text(json.dumps(status),encoding="utf-8")

    pipeline._retain_xai_and_cleanup_model_batch(cdir,4)
    assert batch.exists()
    assert not (batch/"images").exists() and not (batch/"preprocessed").exists() and not (batch/"data.pkl").exists()
    assert all((batch/f"{m}.csv").exists() for m in pipeline.MODELS)
    assert (batch/"study_order.csv").exists()
    payload=json.loads((cdir/"xai_artifacts.json").read_text())
    assert len(payload["gmic"]) == 1 and Path(payload["gmic"][0]).exists()
    valid,cached,reason=pipeline._chunk_cache_status(sub,cdir,0)
    assert valid and reason is None and len(cached)==1


def test_orientation_cleanup_preserves_resume_contract(tmp_path):
    cdir=tmp_path/"0000"; cdir.mkdir()
    sub=pd.DataFrame([{"study_id":"S1","patient_id":"P1","ground_truth":0,"dataset_source":"rsna"}])
    resolved=sub.copy(); resolved["horizontal_flip"]="NO"
    resolved.to_csv(cdir/"resolved_manifest.csv",index=False)
    for name in ("original","counterfactual"):
        p=cdir/name/"model_batch"; p.mkdir(parents=True); (p/"large.bin").write_bytes(b"123")
    status={"status":"SUCCESS","studies":1,"input_manifest_sha256":pipeline._frame_sha256(sub),"resolved_manifest_sha256":pipeline._sha256_file(cdir/"resolved_manifest.csv")}
    (cdir/"orientation_chunk_status.json").write_text(json.dumps(status),encoding="utf-8")

    pipeline._cleanup_orientation_temporaries(cdir)
    assert not (cdir/"original").exists() and not (cdir/"counterfactual").exists()
    valid,cached,reason=pipeline._orientation_chunk_cache_status(sub,cdir,0)
    assert valid and reason is None and len(cached)==1
