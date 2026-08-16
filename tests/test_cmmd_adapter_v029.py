from pathlib import Path

import pandas as pd

from mammography_agent.datasets.cmmd import CMMDDatasetAdapter, VIEW_CODE_MAP


def _cfg(root: Path):
    return {
        "name": "CMMD",
        "official_information": "https://www.cancerimagingarchive.net/collection/cmmd/",
        "raw_dir": str(root / "datasets/raw/cmmd"),
        "processed_dir": str(root / "datasets/processed/cmmd"),
        "source_manifest": str(root / "datasets/raw/cmmd/source_manifest.csv"),
        "canonical_manifest": str(root / "datasets/manifests/cmmd.csv"),
        "clinical_catalog": str(root / "datasets/manifests/cmmd_clinical_rows.csv"),
        "all_four_view_manifest": str(root / "datasets/manifests/cmmd_all_four_view.csv"),
        "incomplete_manifest": str(root / "datasets/rejected/cmmd_incomplete_studies.csv"),
        "excluded_manifest": str(root / "datasets/rejected/cmmd_nonbenchmark_four_view.csv"),
        "conflicts_manifest": str(root / "datasets/rejected/cmmd_clinical_conflicts.csv"),
        "dicom_index_cache": str(root / "runtime/dataset_cache/cmmd_dicom_index.csv"),
        "reuse_dicom_index_cache": True,
    }


def _patch_workspace(monkeypatch, root: Path):
    import mammography_agent.workspace as ws
    import mammography_agent.logging_utils as lu
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", root)
    monkeypatch.setattr(lu, "WORKSPACE_ROOT", root)


def _touch_patient(raw: Path, patient: str, four_view: bool = True):
    count = 4 if four_view else 2
    for i in range(count):
        p = raw / "cmmd" / patient / f"{i + 1}.dcm"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"DICOM_PLACEHOLDER")


def _write_clinical(raw: Path):
    md = raw / "metadata"
    md.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"ID1": "D1-0001", "LeftRight": "L", "classification": "Benign", "Age": 40, "abnormality": "mass", "subtype": None},
        {"ID1": "D1-0001", "LeftRight": "R", "classification": "Benign", "Age": 40, "abnormality": "mass", "subtype": None},
        {"ID1": "D1-0002", "LeftRight": "L", "classification": "Benign", "Age": 50, "abnormality": "mass", "subtype": None},
        {"ID1": "D1-0002", "LeftRight": "R", "classification": "Malignant", "Age": 50, "abnormality": "mass", "subtype": None},
        {"ID1": "D2-0001", "LeftRight": "L", "classification": "Malignant", "Age": 55, "abnormality": "mass", "subtype": "Luminal B"},
        {"ID1": "D1-0003", "LeftRight": "L", "classification": "Benign", "Age": 45, "abnormality": "mass", "subtype": None},
    ]).to_excel(md / "CMMD_clinicaldata_revision.xlsx", index=False)


def _fake_index(raw: Path) -> pd.DataFrame:
    rows = []
    patients = {
        "D1-0001": [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")],
        "D1-0002": [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")],
        "D2-0001": [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")],
        "D1-0003": [("L", "CC"), ("L", "MLO")],
    }
    for patient, views in patients.items():
        for i, (side, view) in enumerate(views, 1):
            rows.append({
                "path": str((raw / "cmmd" / patient / f"{i}.dcm").resolve()),
                "is_dicom": True,
                "patient_id": patient,
                "study_uid": f"study-{patient}",
                "series_uid": f"series-{patient}-{i}",
                "sop_uid": f"sop-{patient}-{i}",
                "laterality": side,
                "view": view,
                "view_code": "399162004" if view == "CC" else "399368009",
                "canonical_view": f"{side}_{view}",
                "bits_stored": "8",
                "bits_allocated": "8",
                "rows": "2294",
                "columns": "1914",
                "photometric": "MONOCHROME2",
                "read_error": "",
            })
    return pd.DataFrame(rows)


def test_cmmd_view_code_contract_is_explicit():
    assert VIEW_CODE_MAP["399162004"] == "CC"
    assert VIEW_CODE_MAP["399368009"] == "MLO"


def test_cmmd_missing_manual_metadata_is_actionable(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    _touch_patient(raw, "D1-0001", four_view=True)
    adapter = CMMDDatasetAdapter("cmmd", cfg)
    result = adapter.inspect(force_dicom_index=True)
    assert result["status"] == "METADATA_REQUIRED"
    assert result["metadata_auto_download"] is False


def test_cmmd_inspect_exposes_clean_d1_binary_four_view_subset(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    _write_clinical(raw)
    _touch_patient(raw, "D1-0001", four_view=True)
    _touch_patient(raw, "D1-0002", four_view=True)
    _touch_patient(raw, "D2-0001", four_view=True)
    _touch_patient(raw, "D1-0003", four_view=False)

    adapter = CMMDDatasetAdapter("cmmd", cfg)
    monkeypatch.setattr(adapter, "_build_dicom_index", lambda force=False: _fake_index(raw))
    result = adapter.inspect(force_dicom_index=True)

    assert result["status"] == "INSPECTED"
    assert result["dicom_patients"] == 4
    assert result["four_view_patients_all_cmmd"] == 3
    assert result["two_or_incomplete_patients"] == 1
    assert result["benchmark_studies"] == 2
    assert result["benchmark_ground_truth_counts"] == {"BENIGN": 1, "MALIGNANT": 1}

    source = pd.read_csv(cfg["source_manifest"])
    assert set(source.patient_id) == {"D1-0001", "D1-0002"}
    mixed = source[source.patient_id.eq("D1-0002")].iloc[0]
    assert int(mixed.ground_truth) == 1
    assert int(mixed.left_ground_truth) == 0
    assert int(mixed.right_ground_truth) == 1
    assert set(source.cmmd_cohort) == {"D1"}

    excluded = pd.read_csv(cfg["excluded_manifest"])
    assert set(excluded.patient_id) == {"D2-0001"}
