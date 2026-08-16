from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mammography_agent import dicom_presentation_counterfactual as mod


def test_counterfactual_report_contract_is_label_blind_and_preserves_identity_modality(tmp_path, monkeypatch):
    # The release test does not depend on a host pydicom installation. It exercises
    # the full report/orchestration contract with a DICOM-like object; the production
    # pydicom 3.x API calls are validated against the pinned project dependency and
    # official pydicom API contract.
    monkeypatch.setattr(mod, "safe_workspace_path", lambda value: Path(value).resolve())

    pixels = np.array([[0, 32, 64, 96], [128, 160, 192, 255]], dtype=np.uint8)

    class FakeDS:
        pixel_array = pixels
        BitsAllocated = 8
        BitsStored = 8
        HighBit = 7
        PixelRepresentation = 0
        PhotometricInterpretation = "MONOCHROME2"
        RescaleSlope = "1"
        RescaleIntercept = "0"
        WindowCenter = "128"
        WindowWidth = "256"
        VOILUTFunction = "SIGMOID"
        VOILUTSequence = None
        ModalityLUTSequence = None

    fake_pydicom = SimpleNamespace(dcmread=lambda *args, **kwargs: FakeDS())
    monkeypatch.setitem(sys.modules, "pydicom", fake_pydicom)
    monkeypatch.setattr(mod, "_apply_modality", lambda raw, ds: np.asarray(raw))
    monkeypatch.setattr(
        mod,
        "_apply_voi",
        lambda arr, ds: np.sqrt(np.asarray(arr, dtype=np.float64) / 255.0) * 255.0,
    )

    run = tmp_path / "run"
    run.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = {}
    for col in mod.VIEW_COLUMNS:
        path = raw / f"{col}.dcm"
        path.write_bytes(b"synthetic")
        paths[col] = str(path)

    # Deliberately include an absurd label: the diagnostic must not read it.
    pd.DataFrame([{
        "study_id": "CMMD_D1-TEST",
        "dataset_source": "cmmd",
        "ground_truth": 12345,
        **{col: f"/prepared/{col}.png" for col in mod.VIEW_COLUMNS},
    }]).to_csv(run / "selected_studies.csv", index=False)

    manifest = tmp_path / "source_manifest.csv"
    pd.DataFrame([{"study_id": "CMMD_D1-TEST", **paths}]).to_csv(manifest, index=False)

    out = mod.run_dicom_presentation_counterfactual(
        run,
        output_dir=tmp_path / "out",
        source_manifest=manifest,
        write_images=False,
    )

    summary = json.loads((out / "dicom_presentation_summary.json").read_text())
    assert summary["dataset_source"] == "cmmd"
    assert summary["studies"] == 1
    assert summary["images"] == 4
    assert summary["presentation_metadata"]["window_center_images"] == 4
    assert summary["presentation_metadata"]["window_width_images"] == 4
    assert summary["pairwise_vs_current"]["modality_lut"]["all_exact_equal"] is True
    assert summary["pairwise_vs_current"]["modality_lut"]["exact_equal_images"] == 4
    assert summary["pairwise_vs_current"]["voi_presentation"]["all_exact_equal"] is False
    assert summary["research_guards"]["ground_truth_used"] is False
    assert summary["research_guards"]["model_scores_used"] is False
    assert summary["research_guards"]["classifier_inference_performed"] is False
    assert summary["research_guards"]["raw_dataset_modified"] is False
    assert summary["research_guards"]["prepared_dataset_modified"] is False

    pair = pd.read_csv(out / "presentation_pairwise_comparison.csv")
    assert len(pair) == 8
    assert pair.loc[pair.candidate.eq("modality_lut"), "exact_equal"].astype(bool).all()
    assert not pair.loc[pair.candidate.eq("voi_presentation"), "exact_equal"].astype(bool).all()

    report = (out / "dicom_presentation_report.md").read_text()
    assert "DICOM Presentation Counterfactual" in report
    assert "modality_lut" in report
    assert "voi_presentation" in report
