import json
from pathlib import Path
import pandas as pd
import yaml

from mammography_agent import pipeline
from mammography_agent.ensemble.metrics import evaluate


def _dataset(n_benign=72, n_malignant=33):
    rows=[]
    for i in range(n_benign):
        rows.append({"study_id":f"B{i:03d}","patient_id":f"PB{i:03d}","ground_truth":0,"dataset_source":"cbis_ddsm"})
    for i in range(n_malignant):
        rows.append({"study_id":f"M{i:03d}","patient_id":f"PM{i:03d}","ground_truth":1,"dataset_source":"cbis_ddsm"})
    return pd.DataFrame(rows)


def test_stratified_sampling_is_reproducible_and_proportional():
    df=_dataset()
    a,ainfo=pipeline._sample_normal_dataset(df,10,"stratified",42)
    b,binfo=pipeline._sample_normal_dataset(df,10,"stratified",42)
    assert a.study_id.tolist()==b.study_id.tolist()
    assert ainfo["requested_class_distribution"]=={"BENIGN":7,"MALIGNANT":3}
    assert ainfo["actual_class_distribution"]=={"BENIGN":7,"MALIGNANT":3}
    assert binfo==ainfo


def test_balanced_sampling_selects_equal_classes():
    df=_dataset()
    out,info=pipeline._sample_normal_dataset(df,10,"balanced",42)
    assert len(out)==10
    assert info["requested_class_distribution"]=={"BENIGN":5,"MALIGNANT":5}
    assert info["actual_class_distribution"]=={"BENIGN":5,"MALIGNANT":5}


def test_single_class_metrics_explain_why_auc_and_sensitivity_are_unavailable():
    m=evaluate([0,0,0],[0.1,0.2,0.3],0.5)
    assert m["tn"]==3 and m["tp"]==0
    assert m["sensitivity"] is None
    assert "malignant" in m["sensitivity_unavailable_reason"].lower()
    assert m["roc_auc"] is None
    assert "both benign" in m["roc_auc_unavailable_reason"].lower()


def test_normal_test_persists_sampling_and_run_summary(monkeypatch,tmp_path):
    df=_dataset(6,4)
    monkeypatch.setattr(pipeline,"WORKSPACE_ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"load_datasets",lambda datasets,samples=None:df.copy())
    monkeypatch.setattr(pipeline,"save_run",lambda *a,**k:None)
    monkeypatch.setattr(pipeline,"audit",lambda *a,**k:None)
    monkeypatch.setattr(pipeline,"resolve_orientation",lambda frame,*a,**k: frame.copy())

    def fake_infer(frame,run_dir,run_id):
        out=frame[["study_id","patient_id","ground_truth","dataset_source"]].copy()
        out["gmic_score"]=out.ground_truth.map({0:0.1,1:0.9})
        out["nyu_score"]=out.ground_truth.map({0:0.2,1:0.8})
        out["glam_score"]=out.ground_truth.map({0:0.15,1:0.85})
        return out
    monkeypatch.setattr(pipeline,"_infer_three",fake_infer)

    run=pipeline.normal_test(["cbis_ddsm"],samples=4,sampling="balanced",seed=42)
    summary=json.loads((run/"run_summary.json").read_text())
    assert summary["processed_studies"]==4
    assert summary["processed_images"]==16
    assert summary["sampling_strategy"]=="balanced"
    assert summary["sampling_seed"]==42
    assert summary["actual_class_distribution"]=={"BENIGN":2,"MALIGNANT":2}
    assert summary["overall_elapsed_seconds"]>=0
    cfg=yaml.safe_load((run/"configuration_used.yaml").read_text())
    assert cfg["sampling"]["requested_class_distribution"]=={"BENIGN":2,"MALIGNANT":2}
    assert (run/"selected_studies.csv").exists()
