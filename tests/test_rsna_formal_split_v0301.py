from pathlib import Path
import pandas as pd
import pytest

import mammography_agent.pipeline as pipeline


def _rsna_frame():
    rows=[]
    for i in range(11427):
        rows.append({"study_id":f"RSNA_B{i:05d}","patient_id":f"B{i:05d}","ground_truth":0,"dataset_source":"rsna"})
    for i in range(486):
        rows.append({"study_id":f"RSNA_M{i:05d}","patient_id":f"M{i:05d}","ground_truth":1,"dataset_source":"rsna"})
    return pd.DataFrame(rows)


def test_rsna_diagnostic_set_is_excluded_before_full_formal_split(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline,"WORKSPACE_ROOT",tmp_path)
    manifest=tmp_path/"datasets"/"manifests"/"rsna_diagnostic_exclusion_v1.csv"
    manifest.parent.mkdir(parents=True)
    diag=pd.DataFrame(
        [{"study_id":f"RSNA_B{i:05d}","patient_id":f"B{i:05d}","ground_truth":0} for i in range(5)] +
        [{"study_id":f"RSNA_M{i:05d}","patient_id":f"M{i:05d}","ground_truth":1} for i in range(5)]
    )
    diag.to_csv(manifest,index=False)

    prepared=_rsna_frame()
    formal,excluded,summary=pipeline._apply_formal_exclusions(prepared,["rsna"])
    assert len(prepared)==11913
    assert len(excluded)==10
    assert len(formal)==11903
    assert summary["remaining_formal_pool_class_distribution"]=={"BENIGN":11422,"MALIGNANT":481}

    config,final,split=pipeline._split_formal_pool(formal,.30,42)
    assert len(config)==3570
    assert len(final)==8333
    assert split["formal_pool_coverage_fraction"]==pytest.approx(1.0)
    assert split["configuration_class_distribution"]=={"BENIGN":3426,"MALIGNANT":144}
    assert split["final_test_class_distribution"]=={"BENIGN":7996,"MALIGNANT":337}
    assert set(config.study_id).isdisjoint(set(final.study_id))
    assert set(config.study_id) | set(final.study_id) == set(formal.study_id)
    assert set(zip(config.dataset_source,config.patient_id)).isdisjoint(set(zip(final.dataset_source,final.patient_id)))
    assert set(excluded.patient_id).isdisjoint(set(config.patient_id))
    assert set(excluded.patient_id).isdisjoint(set(final.patient_id))


def test_rsna_formal_exclusion_manifest_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline,"WORKSPACE_ROOT",tmp_path)
    with pytest.raises(FileNotFoundError,match="Formal exclusion manifest required for rsna"):
        pipeline._apply_formal_exclusions(_rsna_frame().head(20),["rsna"])


def test_split_rejects_non_exhaustive_or_invalid_ratio():
    df=pd.DataFrame([
        {"study_id":f"S{i}","patient_id":f"P{i}","ground_truth":i%2,"dataset_source":"unit"}
        for i in range(20)
    ])
    with pytest.raises(ValueError,match="configuration_ratio"):
        pipeline._split_formal_pool(df,1.0,42)
