from pathlib import Path


def test_model_runner_uses_monitoring_samples_not_ambiguous_samples():
    text=Path("model_runner/api.py").read_text(encoding="utf-8")
    assert '"monitoring_samples": len(metric_samples)' in text
    assert '"samples": len(' not in text
