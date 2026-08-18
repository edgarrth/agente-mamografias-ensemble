from __future__ import annotations

import math
import pickle
from pathlib import Path

from mammography_agent import pipeline, single_case


def test_web_cpu_label_contract_adds_only_optional_benign_metadata(tmp_path):
    pkl = tmp_path / "data.pkl"
    original = [{
        "L-CC": ["a"], "R-CC": ["b"], "L-MLO": ["c"], "R-MLO": ["d"],
        "cancer_label": {"left_malignant": 0, "right_malignant": 0},
        "horizontal_flip": "NO",
    }]
    with pkl.open("wb") as fh:
        pickle.dump(original, fh, protocol=4)

    pipeline._apply_web_label_blind_compat(pkl)

    with pkl.open("rb") as fh:
        patched = pickle.load(fh)
    labels = patched[0]["cancer_label"]
    assert labels["left_malignant"] == 0
    assert labels["right_malignant"] == 0
    assert math.isnan(labels["left_benign"])
    assert math.isnan(labels["right_benign"])


def test_canonical_batch_builder_still_omits_optional_benign_keys():
    source = Path("mammography_agent/metarepo_format.py").read_text(encoding="utf-8")
    assert '"left_benign"' not in source
    assert '"right_benign"' not in source


def test_web_progress_is_written_and_retrievable(tmp_path, monkeypatch):
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", tmp_path)
    run_id = "web-20260818T220503Z-test0001"
    run_dir = tmp_path / "output" / "single_cases" / run_id
    run_dir.mkdir(parents=True)
    single_case._write_web_progress(
        run_dir, run_id,
        stage="MODELS", state="RUNNING", message="Ejecutando: GMIC.",
        models={"gmic": {"state": "RUNNING"}}, started_at="2026-08-18T22:05:03+00:00",
    )
    payload = single_case.get_single_case_progress(run_id)
    assert payload["stage"] == "MODELS"
    assert payload["models"]["gmic"]["state"] == "RUNNING"
    assert "GMIC" in payload["message"]


def test_streamlit_ui_hides_deploy_and_uses_live_progress_and_value_oriented_storage():
    text = Path("ui/streamlit_app.py").read_text(encoding="utf-8")
    assert '[data-testid="stToolbar"] {display:none !important;}' in text
    assert "Trazabilidad de evidencias habilitada" in text
    assert "Evidencias reproducibles" in text
    assert "MinIO disponible" not in text
    assert 'f"/single-cases/progress/{run_id}"' in text
    assert "ThreadPoolExecutor" in text
    assert "scrollIntoView" in text
    assert "Preparación y normalización de las cuatro proyecciones." not in text
    assert "Cálculo del resultado combinado del ensemble." not in text


def test_pipeline_progress_callback_is_opt_in_and_batch_call_sites_remain_default():
    text = Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert "progress_callback=None" in text
    assert "web_label_blind_compat: bool = False" in text
    assert 'pred=_infer_three(sub,cdir,f"{run_id}-c{chunk_index:04d}")' in text
    assert "_apply_web_label_blind_compat(pkl)" in text
