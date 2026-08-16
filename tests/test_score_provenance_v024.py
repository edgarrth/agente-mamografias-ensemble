from pathlib import Path
import pandas as pd
import numpy as np
import mammography_agent.score_provenance as sp


def test_parse_image_identity_supports_current_batch_names():
    assert sp._parse_image_identity("P_00001_L_CC") == ("P_00001","LEFT","CC")
    assert sp._parse_image_identity("P_00001_R_MLO.png") == ("P_00001","RIGHT","MLO")


def test_score_provenance_reconstructs_current_max_aggregation(tmp_path, monkeypatch):
    root=tmp_path/"workspace"; run=root/"output"/"normal_tests"/"normal-x"; batch=run/"model_batch"
    batch.mkdir(parents=True)
    selected=pd.DataFrame([
        {"study_id":"S1","patient_id":"P1","ground_truth":1,"left_ground_truth":1,"right_ground_truth":0},
        {"study_id":"S2","patient_id":"P2","ground_truth":0,"left_ground_truth":0,"right_ground_truth":0},
    ])
    selected.to_csv(run/"selected_studies.csv",index=False)
    order=pd.DataFrame({"position":[0,1],"study_id":["S1","S2"],"study_key":["S1","S2"]})
    order.to_csv(batch/"study_order.csv",index=False)
    for model,vals in {"gmic":[.8,.7,.2,.1,.1,.2,.3,.25],"glam":[.75,.65,.15,.1,.2,.1,.25,.2]}.items():
        names=["S1_L_CC","S1_L_MLO","S1_R_CC","S1_R_MLO","S2_L_CC","S2_L_MLO","S2_R_CC","S2_R_MLO"]
        pd.DataFrame({"image_index":names,"malignant_pred":vals}).to_csv(batch/f"{model}.csv",index=False)
    pd.DataFrame({"left_malignant":[.9,.15],"right_malignant":[.2,.25]}).to_csv(batch/"nyu.csv",index=False)
    pd.DataFrame({
        "study_id":["S1","S2"],"patient_id":["P1","P2"],"ground_truth":[1,0],"dataset_source":["x","x"],
        "gmic_score":[.8,.3],"nyu_score":[.9,.25],"glam_score":[.75,.25],
    }).to_csv(run/"raw_model_predictions.csv",index=False)
    monkeypatch.setattr(sp,"WORKSPACE_ROOT",root)
    out=sp.audit_score_provenance(run)
    recon=pd.read_csv(out/"study_score_reconstruction.csv")
    assert recon.reconstruction_match.all()
    assert set(recon.study_aggregation)=={"max(left_breast_score,right_breast_score)"}
    metrics=pd.read_csv(out/"model_provenance_metrics.csv")
    assert metrics.study_reconstruction_all_match.all()
    summary=(out/"score_provenance_summary.json").read_text()
    assert '"aggregation_changed": false' in summary
    assert '"model_inference_performed": false' in summary


def test_score_provenance_reconstructs_missing_study_order_for_legacy_run(tmp_path, monkeypatch):
    root=tmp_path/"workspace"; run=root/"output"/"normal_tests"/"normal-legacy"; batch=run/"model_batch"
    batch.mkdir(parents=True)
    selected=pd.DataFrame([
        {"study_id":"S 1","patient_id":"P1","ground_truth":1,"left_ground_truth":1,"right_ground_truth":0},
        {"study_id":"S 2","patient_id":"P2","ground_truth":0,"left_ground_truth":0,"right_ground_truth":0},
    ])
    selected.to_csv(run/"selected_studies.csv",index=False)
    # Intentionally no model_batch/study_order.csv: this is the legacy-run contract.
    names=["S_1_L_CC","S_1_L_MLO","S_1_R_CC","S_1_R_MLO","S_2_L_CC","S_2_L_MLO","S_2_R_CC","S_2_R_MLO"]
    for model,vals in {"gmic":[.8,.7,.2,.1,.1,.2,.3,.25],"glam":[.75,.65,.15,.1,.2,.1,.25,.2]}.items():
        pd.DataFrame({"image_index":names,"malignant_pred":vals}).to_csv(batch/f"{model}.csv",index=False)
    pd.DataFrame({"left_malignant":[.9,.15],"right_malignant":[.2,.25]}).to_csv(batch/"nyu.csv",index=False)
    pd.DataFrame({
        "study_id":["S 1","S 2"],"patient_id":["P1","P2"],"ground_truth":[1,0],"dataset_source":["x","x"],
        "gmic_score":[.8,.3],"nyu_score":[.9,.25],"glam_score":[.75,.25],
    }).to_csv(run/"raw_model_predictions.csv",index=False)
    monkeypatch.setattr(sp,"WORKSPACE_ROOT",root)
    out=sp.audit_score_provenance(run)
    summary=(out/"score_provenance_summary.json").read_text()
    assert '"study_order_reconstructed": true' in summary
    assert '"study_order_source": "raw_model_predictions.csv"' in summary
    recon=pd.read_csv(out/"study_score_reconstruction.csv")
    assert recon.reconstruction_match.all()


def _write_native_batch(batch, study_ids, gmic_study_scores, nyu_study_scores, glam_study_scores, *, write_order=True):
    batch.mkdir(parents=True, exist_ok=True)
    keys=[sid.replace(" ","_") for sid in study_ids]
    if write_order:
        pd.DataFrame({"position":range(len(study_ids)),"study_id":study_ids,"study_key":keys}).to_csv(batch/"study_order.csv",index=False)
    names=[]; gvals=[]; lvals=[]
    for key,gs,ls in zip(keys,gmic_study_scores,glam_study_scores):
        names += [f"{key}_L_CC",f"{key}_L_MLO",f"{key}_R_CC",f"{key}_R_MLO"]
        gvals += [gs, max(gs-.01,0), max(gs-.02,0), max(gs-.03,0)]
        lvals += [ls, max(ls-.01,0), max(ls-.02,0), max(ls-.03,0)]
    pd.DataFrame({"image_index":names,"malignant_pred":gvals}).to_csv(batch/"gmic.csv",index=False)
    pd.DataFrame({"image_index":names,"malignant_pred":lvals}).to_csv(batch/"glam.csv",index=False)
    # Put the study score on LEFT and a lower value on RIGHT so max() reproduces it.
    pd.DataFrame({
        "left_malignant":nyu_study_scores,
        "right_malignant":[max(v-.05,0) for v in nyu_study_scores],
    }).to_csv(batch/"nyu.csv",index=False)


def test_score_provenance_discovers_exact_chunked_normal_test_layout(tmp_path, monkeypatch):
    root=tmp_path/"workspace"; run=root/"output"/"normal_tests"/"normal-chunked"
    run.mkdir(parents=True)
    selected=pd.DataFrame([
        {"study_id":"S1","patient_id":"P1","ground_truth":1,"left_ground_truth":1,"right_ground_truth":0},
        {"study_id":"S2","patient_id":"P2","ground_truth":0,"left_ground_truth":0,"right_ground_truth":0},
    ])
    selected.to_csv(run/"selected_studies.csv",index=False)
    top_raw=pd.DataFrame({
        "study_id":["S1","S2"],"patient_id":["P1","P2"],"ground_truth":[1,0],"dataset_source":["x","x"],
        "gmic_score":[.8,.3],"nyu_score":[.9,.25],"glam_score":[.75,.25],
    })
    top_raw.to_csv(run/"raw_model_predictions.csv",index=False)
    chunk=run/"chunks"/"0000"
    chunk.mkdir(parents=True)
    top_raw.to_csv(chunk/"raw_model_predictions.csv",index=False)
    _write_native_batch(chunk/"model_batch",["S1","S2"],[.8,.3],[.9,.25],[.75,.25],write_order=True)

    monkeypatch.setattr(sp,"WORKSPACE_ROOT",root)
    out=sp.audit_score_provenance(run)
    payload=__import__("json").loads((out/"score_provenance_summary.json").read_text())
    assert payload["audit_input_provenance"]["native_output_layout"] == "chunked"
    assert payload["audit_input_provenance"]["native_batch_count"] == 1
    assert payload["audit_input_provenance"]["native_batches"][0]["path"] == "chunks/0000/model_batch"
    recon=pd.read_csv(out/"study_score_reconstruction.csv")
    assert recon.reconstruction_match.all()
    native=pd.read_csv(out/"native_model_scores.csv")
    assert native.native_source.str.startswith("chunks/0000/model_batch/").all()


def test_score_provenance_combines_multiple_chunks_and_reconstructs_chunk_order(tmp_path, monkeypatch):
    root=tmp_path/"workspace"; run=root/"output"/"normal_tests"/"normal-multichunk"
    run.mkdir(parents=True)
    selected=pd.DataFrame([
        {"study_id":"S 1","patient_id":"P1","ground_truth":1,"left_ground_truth":1,"right_ground_truth":0},
        {"study_id":"S 2","patient_id":"P2","ground_truth":0,"left_ground_truth":0,"right_ground_truth":0},
        {"study_id":"S 3","patient_id":"P3","ground_truth":1,"left_ground_truth":1,"right_ground_truth":0},
        {"study_id":"S 4","patient_id":"P4","ground_truth":0,"left_ground_truth":0,"right_ground_truth":0},
    ])
    selected.to_csv(run/"selected_studies.csv",index=False)
    all_raw=pd.DataFrame({
        "study_id":["S 1","S 2","S 3","S 4"],"patient_id":["P1","P2","P3","P4"],"ground_truth":[1,0,1,0],"dataset_source":["x"]*4,
        "gmic_score":[.8,.3,.7,.2],"nyu_score":[.9,.25,.8,.2],"glam_score":[.75,.25,.65,.15],
    })
    all_raw.to_csv(run/"raw_model_predictions.csv",index=False)
    for idx, rows in enumerate(([0,1],[2,3])):
        chunk=run/"chunks"/f"{idx:04d}"; chunk.mkdir(parents=True)
        c_raw=all_raw.iloc[rows].reset_index(drop=True)
        c_raw.to_csv(chunk/"raw_model_predictions.csv",index=False)
        _write_native_batch(
            chunk/"model_batch",
            c_raw.study_id.tolist(),
            c_raw.gmic_score.tolist(),
            c_raw.nyu_score.tolist(),
            c_raw.glam_score.tolist(),
            write_order=False,
        )

    monkeypatch.setattr(sp,"WORKSPACE_ROOT",root)
    out=sp.audit_score_provenance(run)
    payload=__import__("json").loads((out/"score_provenance_summary.json").read_text())
    prov=payload["audit_input_provenance"]
    assert prov["native_output_layout"] == "chunked"
    assert prov["native_batch_count"] == 2
    assert all(b["study_order_reconstructed"] for b in prov["native_batches"])
    assert {b["study_order_source"] for b in prov["native_batches"]} == {
        "chunks/0000/raw_model_predictions.csv","chunks/0001/raw_model_predictions.csv"
    }
    recon=pd.read_csv(out/"study_score_reconstruction.csv")
    assert len(recon) == 12
    assert recon.reconstruction_match.all()
