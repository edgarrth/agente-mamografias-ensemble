from pathlib import Path


def test_gpu_probe_success_log_does_not_pass_model_twice():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert 'log("GPU_RUNTIME_PROBE_PASSED", model=model, **result)' not in text
    assert 'log("GPU_RUNTIME_PROBE_PASSED", **result)' in text


def test_gpu_probe_result_contains_model_before_logging():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert '"status": "GPU_READY", "model": model, "image": tag' in text
