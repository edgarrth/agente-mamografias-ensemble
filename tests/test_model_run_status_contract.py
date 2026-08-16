from model_runner.api import _run_success_payload


def test_completed_model_run_overrides_image_ready_status_with_success():
    payload = _run_success_payload(
        info={"status": "READY", "model": "gmic", "image": "example"},
        output_file="/workspace/output/gmic.csv",
        xai=["/workspace/output/xai.png"],
        resource_metrics={"elapsed_seconds": 1.0},
        stdout="ok",
    )

    assert payload["status"] == "SUCCESS"
    assert payload["model"] == "gmic"
    assert payload["image"] == "example"
    assert payload["output_file"] == "/workspace/output/gmic.csv"


def test_pipeline_requires_completed_run_status_not_image_readiness():
    from pathlib import Path
    text = Path("mammography_agent/pipeline.py").read_text(encoding="utf-8")
    assert 'result.get("status")!="SUCCESS"' in text
