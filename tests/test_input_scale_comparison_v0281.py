from __future__ import annotations

from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import struct
import zlib

import mammography_agent.input_scale_comparison as mod
import mammography_agent.workspace as workspace_mod


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png(path: Path, value: int, w: int = 12, h: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)
    pixel = struct.pack(">H", value)
    raw = b"".join(b"\x00" + pixel * w for _ in range(h))
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")
    path.write_bytes(data)


def test_input_scale_comparison_raw_and_cropped_classifier_free(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    run = workspace / "output" / "normal_tests" / "normal-x"
    run.mkdir(parents=True)
    cbis_img = workspace / "datasets" / "cbis"
    cbis_img.mkdir(parents=True)
    rows = []
    for study in ("S1", "S2"):
        rec = {"study_id": study, "patient_id": study, "ground_truth": 0, "left_ground_truth": 0, "right_ground_truth": 0, "horizontal_flip": "NO"}
        for col, view in mod.VIEW_COLUMNS.items():
            p = cbis_img / f"{study}_{view}.png"
            _write_png(p, 1000 if study == "S1" else 2000)
            rec[col] = str(p)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(run / "selected_studies.csv", index=False)

    sample = workspace / "runtime" / "mammography_metarepository" / "sample_data"
    images = sample / "images"; images.mkdir(parents=True)
    data = []
    for i in range(4):
        exam = {"horizontal_flip": "NO", "cancer_label": {"left_malignant": 0, "right_malignant": 0}}
        for view in ("L-CC", "R-CC", "L-MLO", "R-MLO"):
            stem = f"{i}_{view}"
            _write_png(images / f"{stem}.png", 4000)
            exam[view] = [stem]
        data.append(exam)
    with (sample / "data.pkl").open("wb") as fh:
        pickle.dump(data, fh, protocol=4)

    monkeypatch.setattr(mod, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(workspace_mod, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(mod, "ensure_metarepository", lambda: {"status": "READY", "path": str(workspace / "runtime" / "mammography_metarepository"), "resolved_commit": "abc"})

    calls = []
    def fake_preprocess(model, run_id, image_dir, data_pickle, preprocessed_dir):
        calls.append((model, run_id))
        cropped = Path(preprocessed_dir) / "cropped"
        cropped.mkdir(parents=True, exist_ok=True)
        for p in Path(image_dir).glob("*.png"):
            shutil.copy2(p, cropped / p.name)
        return {"status": "SUCCESS", "operation": "PREPROCESS_ONLY", "classifier_inference_performed": False, "cropped_images": str(cropped)}
    monkeypatch.setattr(mod, "preprocess_model", fake_preprocess)

    out = mod.compare_input_scale(run, workspace / "output" / "analyses" / "scale-test", include_nyu_crop=True)
    image_stats = pd.read_csv(out / "input_scale_image_stats.csv")
    assert set(image_stats["source"]) == {"cbis_ddsm", "official_sample"}
    assert set(image_stats["stage"]) == {"raw_prepared_png", "nyu_upstream_cropped"}
    assert len(calls) == 2
    summary = (out / "input_scale_summary.json").read_text()
    assert '"classifier_inference_performed": false' in summary
    assert '"ground_truth_used": false' in summary
    comparison = pd.read_csv(out / "input_scale_comparison.csv")
    raw_mean = comparison[(comparison.stage == "raw_prepared_png") & (comparison.metric == "median_normalized_mean")].iloc[0]
    assert raw_mean["ratio_cbis_to_official"] < 1.0
