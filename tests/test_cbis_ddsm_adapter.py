from pathlib import Path
import pandas as pd
import pytest

from mammography_agent.datasets.cbis_ddsm import (
    CBISDDSMDatasetAdapter,
    OFFICIAL_METADATA_FILES,
    normalize_laterality,
    normalize_pathology,
    normalize_view,
)


def _write_dicom(path: Path, patient: str, laterality: str, view: str, study_uid: str, series_uid: str):
    # The adapter first resolves official metadata paths without pixel decoding.
    # Unit tests use a placeholder file; real DICOM decoding is provided by pydicom in the app image.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"DICOM_PLACEHOLDER")


def _cfg(root: Path):
    return {
        "name": "CBIS-DDSM",
        "official_information": "https://www.cancerimagingarchive.net/collection/cbis-ddsm/",
        "raw_dir": str(root / "datasets/raw/cbis_ddsm"),
        "processed_dir": str(root / "datasets/processed/cbis_ddsm"),
        "source_manifest": str(root / "datasets/raw/cbis_ddsm/source_manifest.csv"),
        "canonical_manifest": str(root / "datasets/manifests/cbis_ddsm.csv"),
        "metadata_catalog": str(root / "datasets/manifests/cbis_ddsm_metadata_rows.csv"),
        "view_catalog": str(root / "datasets/manifests/cbis_ddsm_view_catalog.csv"),
        "incomplete_manifest": str(root / "datasets/rejected/cbis_ddsm_incomplete_studies.csv"),
        "unresolved_manifest": str(root / "datasets/rejected/cbis_ddsm_unresolved_metadata_rows.csv"),
        "dicom_index_cache": str(root / "runtime/dataset_cache/cbis_ddsm_dicom_index.csv"),
        "minimum_full_image_pixels": 1,
    }


def _patch_workspace(monkeypatch, root: Path):
    import mammography_agent.workspace as ws
    import mammography_agent.logging_utils as lu
    monkeypatch.setattr(ws, "WORKSPACE_ROOT", root)
    monkeypatch.setattr(lu, "WORKSPACE_ROOT", root)


def _patch_dicom_runtime(monkeypatch, adapter):
    def fake_index(files, force=False):
        rows=[]
        for path in files:
            text=Path(path).as_posix().upper()
            patient=(__import__("re").search(r"P_\d{5}", text) or [""])[0]
            lat="LEFT" if "_LEFT_" in text else "RIGHT" if "_RIGHT_" in text else ""
            view="MLO" if "_MLO" in text else "CC" if "_CC" in text else ""
            rows.append({"path":str(path),"is_dicom":True,"patient_id":patient,"study_uid":"","series_uid":"","sop_uid":"","laterality":lat,"view":view,"series_description":"","rows":4000,"columns":2000,"pixels":8000000,"bits_stored":"12","photometric":"MONOCHROME2","read_error":""})
        return pd.DataFrame(rows)
    def fake_convert(source, dest):
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(b"PNG_PLACEHOLDER")
    monkeypatch.setattr(adapter, "_build_dicom_index", fake_index)
    monkeypatch.setattr(adapter, "_convert_to_png", fake_convert)


def _metadata_row(patient, laterality, view, pathology, relative_path):
    return {
        "patient_id": patient,
        "breast density": 2,
        "left or right breast": laterality,
        "image view": view,
        "abnormality id": 1,
        "abnormality type": "mass",
        "assessment": 4,
        "pathology": pathology,
        "subtlety": 3,
        "image file path": relative_path,
        "cropped image file path": "unused",
        "ROI mask file path": "unused",
    }


def _make_official_files(raw: Path, rows: list[dict]):
    raw.mkdir(parents=True, exist_ok=True)
    columns = list(_metadata_row("P_00000", "LEFT", "CC", "BENIGN", "x").keys())
    pd.DataFrame(rows, columns=columns).to_csv(raw / "mass_case_description_train_set.csv", index=False)
    for name in OFFICIAL_METADATA_FILES[1:]:
        pd.DataFrame(columns=columns).to_csv(raw / name, index=False)


def test_cbis_normalizers_do_not_invent_unknown_pathology():
    assert normalize_pathology("MALIGNANT") == 1
    assert normalize_pathology("BENIGN") == 0
    assert normalize_pathology("BENIGN_WITHOUT_CALLBACK") == 0
    assert normalize_pathology("BI-RADS 5") is None
    assert normalize_laterality("L") == "LEFT"
    assert normalize_laterality("right") == "RIGHT"
    assert normalize_view("CC") == "CC"
    assert normalize_view("MLO") == "MLO"


def test_cbis_official_adapter_generates_four_view_manifest_and_pngs(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    rows = []
    specs = [
        ("LEFT", "CC", "MALIGNANT"),
        ("RIGHT", "CC", "BENIGN"),
        ("LEFT", "MLO", "MALIGNANT"),
        ("RIGHT", "MLO", "BENIGN_WITHOUT_CALLBACK"),
    ]
    for i, (lat, view, pathology) in enumerate(specs, start=1):
        study_uid = f"1.2.826.0.1.3680043.10.543.{i}"
        series_uid = f"1.2.826.0.1.3680043.10.543.{100+i}"
        case = f"Mass-Training_P_00001_{lat}_{view}"
        rel = f"{case}/{study_uid}/{series_uid}/1-1.dcm"
        _write_dicom(raw / "CBIS-DDSM" / rel, "P_00001", lat, view, study_uid, series_uid)
        rows.append(_metadata_row("P_00001", lat, view, pathology, rel))
    _make_official_files(raw, rows)

    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)
    _patch_dicom_runtime(monkeypatch, adapter)
    inspected = adapter.inspect()
    assert inspected["complete_four_view_studies"] == 1
    assert inspected["unresolved_metadata_rows"] == 0
    result = adapter.prepare()
    assert result["status"] == "AVAILABLE"
    manifest = pd.read_csv(cfg["canonical_manifest"])
    assert len(manifest) == 1
    assert int(manifest.iloc[0].ground_truth) == 1
    assert int(manifest.iloc[0].left_ground_truth) == 1
    assert int(manifest.iloc[0].right_ground_truth) == 0
    for col in ["l_cc", "r_cc", "l_mlo", "r_mlo"]:
        assert Path(manifest.iloc[0][col]).exists()
        assert Path(manifest.iloc[0][col]).suffix.lower() == ".png"


def test_cbis_incomplete_views_are_rejected_not_synthesized(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    rows = []
    for i, view in enumerate(["CC", "MLO"], start=1):
        study_uid = f"1.2.826.0.1.3680043.10.999.{i}"
        series_uid = f"1.2.826.0.1.3680043.10.999.{100+i}"
        rel = f"Mass-Training_P_00002_LEFT_{view}/{study_uid}/{series_uid}/1-1.dcm"
        _write_dicom(raw / "CBIS-DDSM" / rel, "P_00002", "LEFT", view, study_uid, series_uid)
        rows.append(_metadata_row("P_00002", "LEFT", view, "MALIGNANT", rel))
    _make_official_files(raw, rows)

    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)
    _patch_dicom_runtime(monkeypatch, adapter)
    result = adapter.prepare()
    assert result["status"] == "INSUFFICIENT_FOUR_VIEW_STUDIES"
    assert result["converted_studies"] == 0
    source = pd.read_csv(cfg["source_manifest"])
    assert source.empty
    rejected = pd.read_csv(cfg["incomplete_manifest"])
    assert len(rejected) == 1
    assert "RIGHT_CC" in rejected.iloc[0].missing_views
    assert "RIGHT_MLO" in rejected.iloc[0].missing_views


def test_cbis_supplement_excludes_known_roi_and_crop_series(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", _cfg(tmp_path))
    resolved = pd.DataFrame([
        {
            "patient_id": "P_00003", "laterality": "LEFT", "view": "CC", "ground_truth": 1,
            "resolved_image": str(tmp_path / "left_cc.dcm"), "resolution_method": "path_suffix_4",
            "image file path": "Mass-Training_P_00003_LEFT_CC/1.2.3/1.2.4/1-1.dcm",
            "cropped image file path": "Mass-Training_P_00003_LEFT_CC/1.2.3/9.9.1/1-1.dcm",
            "roi mask file path": "Mass-Training_P_00003_LEFT_CC/1.2.3/9.9.2/1-1.dcm",
            "official_split": "train", "metadata_file": "x", "lesion_type": "mass",
        }
    ])
    index = pd.DataFrame([
        {"path": str(tmp_path / "right_cc_full.dcm"), "is_dicom": True, "patient_id": "P_00003", "laterality": "RIGHT", "view": "CC", "pixels": 8_000_000, "series_uid": "8.8.8", "series_description": ""},
        {"path": str(tmp_path / "right_cc_mask.dcm"), "is_dicom": True, "patient_id": "P_00003", "laterality": "RIGHT", "view": "CC", "pixels": 8_000_000, "series_uid": "9.9.2", "series_description": ""},
    ])
    out = adapter._supplement_unreferenced_views(resolved, index)
    supplemental = out[out.resolution_method == "dicom_header_supplement"]
    assert len(supplemental) == 1
    assert supplemental.iloc[0].resolved_image.endswith("right_cc_full.dcm")


def test_cbis_inspect_missing_metadata_returns_actionable_status(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    _write_dicom(raw / "CBIS-DDSM" / "placeholder.dcm", "P_00001", "LEFT", "CC", "1.2.3", "1.2.4")
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)

    result = adapter.inspect()

    assert result["status"] == "METADATA_REQUIRED"
    assert result["dicom_present"] is True
    assert result["dicom_index_started"] is False
    assert set(result["official_metadata_missing"]) == set(OFFICIAL_METADATA_FILES)
    assert Path(result["metadata_instructions"]).exists()
    prepared = adapter.prepare()
    assert prepared["status"] == "METADATA_REQUIRED"
    assert prepared["converted_studies"] == 0


def test_cbis_accepts_tcia_description_filename_aliases(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    columns = list(_metadata_row("P_00000", "LEFT", "CC", "BENIGN", "x").keys())
    aliases = {
        "Mass-Training-Description.csv": [_metadata_row("P_00001", "LEFT", "CC", "MALIGNANT", "x")],
        "Mass-Test-Description.csv": [],
        "Calc-Training-Description.csv": [],
        "Calc-Test-Description.csv": [],
    }
    for name, rows in aliases.items():
        pd.DataFrame(rows, columns=columns).to_csv(raw / name, index=False)
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)
    files = adapter._metadata_files(strict=True)
    assert set(files) == set(OFFICIAL_METADATA_FILES)
    assert files["mass_case_description_train_set.csv"].name == "Mass-Training-Description.csv"


def test_cbis_v029_download_is_manual_only_and_reuses_existing_files(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    _write_dicom(raw / "CBIS-DDSM" / "already_here.dcm", "P_00001", "LEFT", "CC", "1.2.3", "1.2.4")
    _make_official_files(raw, [_metadata_row("P_00001", "LEFT", "CC", "MALIGNANT", "x")])
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)

    result = adapter.download()
    assert result["status"] == "READY_FOR_INSPECT"
    assert result["download_performed"] is False
    assert result["dicom_download_performed"] is False
    assert result["metadata_download_performed"] is False
    assert result["metadata_auto_download"] is False
    assert (raw / "CBIS-DDSM" / "already_here.dcm").exists()


def test_cbis_v029_download_never_creates_missing_metadata(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    _write_dicom(raw / "CBIS-DDSM" / "already_here.dcm", "P_00001", "LEFT", "CC", "1.2.3", "1.2.4")
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)

    result = adapter.download()
    assert result["status"] == "METADATA_REQUIRED"
    assert result["download_performed"] is False
    assert result["metadata_download_performed"] is False
    assert result["metadata_auto_download"] is False
    for name in OFFICIAL_METADATA_FILES:
        assert not (raw / "metadata" / name).exists()


def test_cbis_v015_auxiliary_metadata_enriches_cached_series_identity(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    metadata_dir = raw / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "PatientID": "Calc-Test_P_00038_RIGHT_MLO_1",
            "StudyInstanceUID": "1.2.3",
            "SeriesInstanceUID": "1.2.3.4",
            "Collection": "cbis_ddsm",
        }
    ]).to_csv(metadata_dir / "metadata.csv", index=False)
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)
    index = pd.DataFrame([
        {
            "path": str(raw / "CBIS-DDSM" / "opaque.dcm"), "is_dicom": True,
            "patient_id": "", "study_uid": "", "series_uid": "1.2.3.4", "sop_uid": "1.2.3.4.5",
            "laterality": "", "view": "", "series_description": "", "rows": 4000,
            "columns": 2000, "pixels": 8_000_000, "bits_stored": "12", "photometric": "MONOCHROME2", "read_error": "",
        }
    ])
    enriched, info = adapter._enrich_dicom_index_with_auxiliary(index)
    assert info["used"] is True
    assert info["matched_dicom_objects"] == 1
    assert enriched.iloc[0].patient_id == "P_00038"
    assert enriched.iloc[0].laterality == "RIGHT"
    assert enriched.iloc[0]["view"] == "MLO"
    assert enriched.iloc[0].study_uid == "1.2.3"


def test_cbis_v015_inventory_reports_distinct_object_types(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    raw = Path(cfg["raw_dir"])
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)
    full = raw / "case" / "study" / "full" / "1.dcm"
    crop = raw / "case" / "study" / "crop" / "2.dcm"
    roi = raw / "case" / "study" / "roi" / "3.dcm"
    other = raw / "case" / "study" / "other" / "4.dcm"
    for x in (full, crop, roi, other):
        x.parent.mkdir(parents=True, exist_ok=True); x.write_bytes(b"x")
    metadata = pd.DataFrame([{
        "image file path": "case/study/full/1.dcm",
        "cropped image file path": "case/study/crop/2.dcm",
        "ROI mask file path": "case/study/roi/3.dcm",
    }])
    index = pd.DataFrame([
        {"path": str(x.resolve()), "is_dicom": True, "series_uid": ""} for x in (full, crop, roi, other)
    ])
    inv = adapter._dicom_object_inventory(metadata, index)
    assert inv["dicom_objects"] == 4
    assert inv["full_mammogram_images"] == 1
    assert inv["cropped_images"] == 1
    assert inv["roi_masks"] == 1
    assert inv["other_dicom_images"] == 1


def test_cbis_v015_cached_dicom_index_avoids_raw_tree_rescan(tmp_path, monkeypatch):
    _patch_workspace(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    cfg["reuse_dicom_index_cache"] = True
    raw = Path(cfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    adapter = CBISDDSMDatasetAdapter("cbis_ddsm", cfg)
    cached_path = raw / "CBIS-DDSM" / "opaque" / "1.dcm"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"placeholder")
    cache = Path(cfg["dicom_index_cache"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "path": str(cached_path), "file_size": 11, "file_mtime_ns": 1,
        "is_dicom": True, "patient_id": "P_00001", "study_uid": "1.2.3",
        "series_uid": "1.2.3.4", "sop_uid": "1.2.3.4.5", "laterality": "LEFT",
        "view": "CC", "series_description": "", "rows": 4000, "columns": 2000,
        "pixels": 8_000_000, "bits_stored": "12", "photometric": "MONOCHROME2", "read_error": "",
    }]).to_csv(cache, index=False)
    monkeypatch.setattr(adapter, "_all_candidate_files", lambda: (_ for _ in ()).throw(AssertionError("raw tree must not be rescanned")))
    metadata = pd.DataFrame([_metadata_row("P_00001", "LEFT", "CC", "MALIGNANT", "case/1.2.3/1.2.3.4/1.dcm")])
    metadata = metadata.rename(columns={"left or right breast": "left or right breast", "image view": "image view"})
    metadata["patient_id"] = metadata["patient_id"].map(lambda x: x)
    metadata["laterality"] = "LEFT"
    metadata["view"] = "CC"
    metadata["ground_truth"] = 1
    metadata["official_split"] = "train"
    metadata["metadata_file"] = "fixture"
    metadata["lesion_type"] = "mass"
    metadata["metadata_row"] = 0
    resolved, unresolved, index = adapter._resolve_metadata_images(metadata, force_dicom_index=False)
    assert len(unresolved) == 0
    assert index is not None
    assert resolved.iloc[0].resolution_method in {"path_suffix_1", "dicom_uid"}
