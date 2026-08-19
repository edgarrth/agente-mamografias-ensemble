from __future__ import annotations

from pathlib import Path

from mammography_agent import single_case


def _meta(path: Path, view: str):
    side, projection = view.split("_")
    return {
        "path": str(path), "name": path.name, "sha256": single_case._sha256(path),
        "patient_id": "P500", "study_instance_uid": "1.2.3", "series_instance_uid": "1.2.3.4",
        "sop_instance_uid": f"1.2.3.{path.stem}", "modality": "MG", "laterality": side,
        "view": projection, "view_source": "test", "detected_view": view,
        "transfer_syntax_uid": "1.2.840.10008.1.2.1", "rows": 10, "columns": 10,
        "photometric": "MONOCHROME2",
    }


def test_web_case_uses_scratch_and_leaves_batch_workspace_untouched(tmp_path, monkeypatch):
    workspace = tmp_path / "batch-workspace"
    scratch = tmp_path / "web-scratch"
    workspace.mkdir()
    scratch.mkdir()
    sentinel = workspace / "output" / "experiments" / "batch-sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("batch-stays")

    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", scratch)
    uploads = scratch / "uploads" / "session" / "case"
    uploads.mkdir(parents=True)
    paths = []
    mapping = {}
    for idx, view in enumerate(single_case.REQUIRED_VIEWS, start=1):
        p = uploads / f"{idx}.dcm"
        p.write_bytes(view.encode())
        paths.append(str(p))
        mapping[p.name] = view

    monkeypatch.setattr(single_case, "_read_dicom_metadata", lambda p: _meta(p, mapping[p.name]))
    monkeypatch.setattr(single_case, "_convert_input_to_png", lambda s, d: (d.parent.mkdir(parents=True, exist_ok=True), d.write_bytes(b"png")))
    monkeypatch.setattr(single_case, "resolve_orientation", lambda df, *a, **k: df.copy())

    def fake_infer(df, run_dir, run_id, **kwargs):
        out = df[["study_id", "patient_id", "ground_truth", "dataset_source"]].copy()
        out["gmic_score"] = 0.1
        out["nyu_score"] = 0.2
        out["glam_score"] = 0.3
        return out

    monkeypatch.setattr(single_case, "_infer_three", fake_infer)
    monkeypatch.setattr(single_case, "_baseline_config", lambda: ({"gmic": 1/3, "nyu": 1/3, "glam": 1/3}, 0.5, 0.3))
    monkeypatch.setattr(single_case, "persist_single_case", lambda **kw: {
        "status": "SUCCESS", "bucket": "mammography-web", "prefix": f"runs/{kw['run_id']}", "object_count": 10
    })
    monkeypatch.setattr(single_case, "persist_result_json", lambda **kw: {"status": "SUCCESS"})
    captured = {}
    monkeypatch.setattr(single_case, "save_run", lambda *a, **kw: captured.setdefault("save_run", a))
    monkeypatch.setattr(single_case, "save_web_inference", lambda **kw: captured.setdefault("web", kw))
    monkeypatch.setattr(single_case, "audit", lambda *a, **kw: None)

    run_id = "web-20260818T230000Z-storage01"
    result = single_case.run_single_case(dicom_paths=paths, request_run_id=run_id)

    assert result["status"] == "SUCCESS"
    assert result["local_persistence"] is False
    assert result["persistence"]["minio"]["status"] == "SUCCESS"
    assert captured["web"]["artifact_path"] == f"minio://mammography-web/runs/{run_id}"
    assert not (scratch / "runs" / run_id).exists()
    assert not uploads.exists()
    assert sentinel.read_text() == "batch-stays"
    assert not (workspace / "output" / "single_cases").exists()


def test_compose_uses_dedicated_web_scratch_volume_not_project_folder():
    text = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "web_scratch:/web-scratch" in text
    assert "WEB_SCRATCH_ROOT: /web-scratch" in text
    assert "WEB_PERSIST_LOCAL: ${WEB_PERSIST_LOCAL:-false}" in text
    assert "web_scratch: {}" in text


def test_batch_build_batch_call_stays_on_historical_default_resolver():
    pipeline = Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    metarepo = Path("mammography_agent/metarepo_format.py").read_text(encoding="utf-8")
    assert "images,pkl=build_batch(df,batch)" in pipeline
    assert "source_path_resolver or safe_workspace_path" in metarepo
    assert 'pred=_infer_three(sub,cdir,f"{run_id}-c{chunk_index:04d}")' in pipeline


def test_web_progress_is_memory_only_not_written_to_project(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(single_case, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(single_case, "WEB_SCRATCH_ROOT", scratch)
    run_id = "web-20260818T230000Z-progress01"
    single_case._write_web_progress(
        scratch / "runs" / run_id,
        run_id,
        stage="MODELS", state="RUNNING", message="Ejecutando GMIC",
        models={"gmic": {"state": "RUNNING"}}, stages={},
    )
    assert single_case.get_single_case_progress(run_id)["stage"] == "MODELS"
    assert not workspace.exists()
    assert not (scratch / "runs" / run_id / "web_progress.json").exists()
