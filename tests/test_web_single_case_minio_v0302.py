from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mammography_agent import object_storage, single_case


def _make_paths(root: Path):
    names = ["a.dcm", "b.dcm", "c.dcm", "d.dcm"]
    paths = []
    for name in names:
        p = root / name
        p.write_bytes((name * 10).encode())
        paths.append(str(p))
    return paths


def _fake_meta(path: Path, view: str):
    side, projection = view.split("_")
    return {
        "path": str(path), "name": path.name, "sha256": single_case._sha256(path),
        "patient_id": "PATIENT-SECRET", "study_instance_uid": "1.2.3",
        "series_instance_uid": f"1.2.3.{path.stem}", "sop_instance_uid": f"9.9.{path.stem}",
        "modality": "MG", "laterality": side, "view": projection,
        "view_source": "ViewCodeSequence:R-10242" if projection == "CC" else "ViewCodeSequence:R-10226",
        "detected_view": view, "transfer_syntax_uid": "1.2.840.10008.1.2.1",
        "rows": 100, "columns": 80, "photometric": "MONOCHROME2",
    }


def test_dicom_only_inspection_resolves_four_views_without_ground_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    paths = _make_paths(tmp_path)
    mapping = dict(zip([Path(p).name for p in paths], single_case.REQUIRED_VIEWS))
    monkeypatch.setattr(single_case, "_read_dicom_metadata", lambda p: _fake_meta(p, mapping[p.name]))

    result = single_case.inspect_dicom_case(paths)

    assert result["ready"] is True
    assert result["ground_truth_received"] is False
    assert result["labels_used"] is False
    assert set(result["selected_views"]) == set(single_case.REQUIRED_VIEWS)
    # Raw DICOM PatientID is intentionally not returned to the UI/API payload.
    assert "patient_id" not in result["files"][0]
    assert result["identity"]["internal_patient_id"].startswith("WEBP_")


def test_manual_view_assignment_recovers_missing_dicom_view_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    paths = _make_paths(tmp_path)
    views = ["L_CC", "R_CC", "L_MLO", "R_MLO"]
    mapping = dict(zip([Path(p).name for p in paths], views))

    def meta(path):
        rec = _fake_meta(path, mapping[path.name])
        if path.name == "d.dcm":
            rec.update({"view": "", "view_source": "unresolved", "detected_view": None})
        return rec

    monkeypatch.setattr(single_case, "_read_dicom_metadata", meta)
    first = single_case.inspect_dicom_case(paths)
    assert first["ready"] is False
    assert "R_MLO" in first["missing_views"]

    second = single_case.inspect_dicom_case(paths, {"d.dcm": "R_MLO"})
    assert second["ready"] is True
    assert second["selected_views"]["R_MLO"]["name"] == "d.dcm"


def test_web_inference_is_label_blind_and_reuses_common_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    paths = _make_paths(tmp_path)
    mapping = dict(zip([Path(p).name for p in paths], single_case.REQUIRED_VIEWS))
    monkeypatch.setattr(single_case, "_read_dicom_metadata", lambda p: _fake_meta(p, mapping[p.name]))

    def fake_convert(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"PNG16")

    observed = {}
    monkeypatch.setattr(single_case, "_convert_input_to_png", fake_convert)
    monkeypatch.setattr(single_case, "resolve_orientation", lambda df, *a, **k: df.copy())

    def fake_infer(df, run_dir, run_id, device=None, **kwargs):
        observed["ground_truth"] = df.iloc[0]["ground_truth"]
        observed["left"] = df.iloc[0]["left_ground_truth"]
        observed["right"] = df.iloc[0]["right_ground_truth"]
        out = df[["study_id", "patient_id", "ground_truth", "dataset_source"]].copy()
        out["gmic_score"] = [0.80]
        out["nyu_score"] = [0.70]
        out["glam_score"] = [0.90]
        return out

    monkeypatch.setattr(single_case, "_infer_three", fake_infer)
    monkeypatch.setattr(single_case, "_baseline_config", lambda: ({"gmic": 1/3, "nyu": 1/3, "glam": 1/3}, 0.5, 0.3))
    monkeypatch.setattr(single_case, "persist_single_case", lambda **kwargs: {
        "status": "SUCCESS", "bucket": "mammography-web", "prefix": f"runs/{kwargs['run_id']}", "object_count": 10
    })
    monkeypatch.setattr(single_case, "persist_result_json", lambda **kwargs: {"status": "SUCCESS"})
    monkeypatch.setattr(single_case, "save_run", lambda *a, **k: None)
    monkeypatch.setattr(single_case, "save_web_inference", lambda **k: None)
    monkeypatch.setattr(single_case, "audit", lambda *a, **k: None)

    result = single_case.run_single_case(dicom_paths=paths)

    assert pd.isna(observed["ground_truth"])
    assert pd.isna(observed["left"])
    assert pd.isna(observed["right"])
    assert result["classification"] == "CANCER"
    assert result["training_performed"] is False
    assert result["ground_truth_received"] is False
    assert result["ground_truth_used"] is False
    assert result["persistence"]["minio"]["status"] == "SUCCESS"
    assert result["local_persistence"] is False
    assert not (tmp_path / "output" / "experiments").exists()
    assert not (tmp_path / "output" / "normal_tests").exists()


def test_minio_failure_is_non_blocking_for_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    paths = _make_paths(tmp_path)
    mapping = dict(zip([Path(p).name for p in paths], single_case.REQUIRED_VIEWS))
    monkeypatch.setattr(single_case, "_read_dicom_metadata", lambda p: _fake_meta(p, mapping[p.name]))
    monkeypatch.setattr(single_case, "_convert_input_to_png", lambda s, d: (d.parent.mkdir(parents=True, exist_ok=True), d.write_bytes(b"PNG")))
    monkeypatch.setattr(single_case, "resolve_orientation", lambda df, *a, **k: df.copy())

    def fake_infer(df, run_dir, run_id, device=None, **kwargs):
        out = df[["study_id", "patient_id", "ground_truth", "dataset_source"]].copy()
        out["gmic_score"], out["nyu_score"], out["glam_score"] = 0.2, 0.3, 0.4
        return out

    monkeypatch.setattr(single_case, "_infer_three", fake_infer)
    monkeypatch.setattr(single_case, "_baseline_config", lambda: ({"gmic": 1/3, "nyu": 1/3, "glam": 1/3}, 0.5, 0.3))
    monkeypatch.setattr(single_case, "persist_single_case", lambda **kwargs: (_ for _ in ()).throw(ConnectionError("minio down")))
    monkeypatch.setattr(single_case, "save_run", lambda *a, **k: None)
    monkeypatch.setattr(single_case, "save_web_inference", lambda **k: None)
    monkeypatch.setattr(single_case, "audit", lambda *a, **k: None)

    result = single_case.run_single_case(dicom_paths=paths)
    assert result["status"] == "SUCCESS"
    assert result["classification"] == "NO_CANCER"
    assert result["persistence"]["minio"]["status"] == "FAILED"
    assert result["persistence"]["minio"]["non_blocking"] is True


class _FakeMinio:
    def __init__(self):
        self.bucket = False
        self.uploads = []

    def bucket_exists(self, bucket):
        return self.bucket

    def make_bucket(self, bucket):
        self.bucket = True

    def fput_object(self, bucket, object_name, file_path, content_type=None):
        self.uploads.append((bucket, object_name, Path(file_path).name, content_type))


def test_minio_persists_inputs_canonical_result_and_manifest(tmp_path, monkeypatch):
    fake = _FakeMinio()
    monkeypatch.setattr(object_storage, "_client", lambda: fake)
    monkeypatch.setattr(object_storage, "settings", lambda: {
        "endpoint": "minio:9000", "access_key": "a", "secret_key": "b", "secure": False,
        "bucket": "mammography-web", "enabled": True,
    })
    monkeypatch.setattr(object_storage, "audit", lambda *a, **k: None)

    originals = []
    canonical = {}
    for idx, view in enumerate(single_case.REQUIRED_VIEWS, start=1):
        dcm = tmp_path / f"{idx}.dcm"
        dcm.write_bytes(f"dcm-{view}".encode())
        originals.append({"path": str(dcm), "selected_as": view, "detected_view": view})
        png = tmp_path / f"{view}.png"
        png.write_bytes(f"png-{view}".encode())
        canonical[view] = str(png)
    result_path = tmp_path / "single_case_result.json"
    result_path.write_text(json.dumps({"status": "SUCCESS"}))

    result = object_storage.persist_single_case(
        run_id="web-test", original_dicoms=originals, canonical_views=canonical,
        run_dir=tmp_path, result_path=result_path,
    )
    names = [x[1] for x in fake.uploads]
    assert result["status"] == "SUCCESS"
    assert len([x for x in names if "/input/" in x]) == 4
    assert len([x for x in names if "/canonical/" in x]) == 4
    assert "runs/web-test/result/single_case_result.json" in names
    assert "runs/web-test/manifest/minio_manifest.json" in names


def test_api_contract_has_no_ground_truth_or_training_fields():
    from mammography_agent.api import WebDicomCaseRequest

    req = WebDicomCaseRequest(dicom_paths=["/workspace/a.dcm", "/workspace/b.dcm", "/workspace/c.dcm", "/workspace/d.dcm"])
    assert set(req.model_dump()) == {"dicom_paths", "view_assignments", "ensemble_weights", "decision_threshold", "inference_device", "run_id"}
    with pytest.raises(Exception):
        WebDicomCaseRequest(dicom_paths=["a.dcm", "b.dcm", "c.dcm"])


def test_view_detection_uses_conservative_descriptive_metadata_fallbacks():
    from types import SimpleNamespace

    ds_mlo = SimpleNamespace(ViewCodeSequence=None, ViewPosition="", SeriesDescription="RSNA left MLO")
    view, source = single_case._view_from_dataset(ds_mlo)
    assert view == "MLO"
    assert source == "SeriesDescription"

    ds_cc = SimpleNamespace(ViewCodeSequence=None, ViewPosition="", SeriesDescription="", ProtocolName="Acquisition CC")
    view, source = single_case._view_from_dataset(ds_cc)
    assert view == "CC"
    assert source == "ProtocolName"


def test_preview_generation_is_presentation_only_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    paths = _make_paths(tmp_path)
    calls = []

    def fake_preview(source, destination, max_dimension=900):
        calls.append((Path(source).name, Path(destination).name, max_dimension))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"PNG")

    monkeypatch.setattr(single_case, "_write_dicom_preview", fake_preview)

    first = single_case.create_dicom_previews(paths)
    second = single_case.create_dicom_previews(paths)

    assert first["status"] == "READY"
    assert len(first["previews"]) == 4
    assert all(item["inference_input"] is False for item in first["previews"])
    assert all(Path(item["preview_path"]).is_file() for item in first["previews"])
    assert len(calls) == 4
    assert second["previews"] == first["previews"]
    assert len(calls) == 4  # cached previews are not regenerated


def test_dicom_preview_renderer_produces_display_png(tmp_path, monkeypatch):
    import struct
    import sys
    from types import SimpleNamespace

    import numpy as np

    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", tmp_path)
    path = tmp_path / "preview.dcm"
    path.write_bytes(b"synthetic-dicom-placeholder")
    pixels = np.arange(32 * 24, dtype=np.uint16).reshape(32, 24)
    fake_pydicom = SimpleNamespace(
        dcmread=lambda source: SimpleNamespace(
            pixel_array=pixels,
            PhotometricInterpretation="MONOCHROME2",
        )
    )
    monkeypatch.setitem(sys.modules, "pydicom", fake_pydicom)

    destination = tmp_path / "preview.png"
    single_case._write_dicom_preview(path, destination, max_dimension=16)

    assert destination.is_file()
    payload = destination.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, bitdepth, color_type = struct.unpack(">IIBB", payload[16:26])
    assert max(width, height) <= 16
    assert bitdepth == 8
    assert color_type == 0  # grayscale
