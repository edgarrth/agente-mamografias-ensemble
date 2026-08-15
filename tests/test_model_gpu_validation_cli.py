from model_tools import validate_gpu as vg


def test_expand_models_supports_all_subset_and_deduplication():
    assert vg.expand_models(["all"]) == ["gmic", "nyu", "glam"]
    assert vg.expand_models(["gmic", "nyu", "gmic"]) == ["gmic", "nyu"]


def test_validate_gpu_models_runs_ensure_then_probe_then_smoke(monkeypatch):
    calls = []

    def ensure(model, force_rebuild=False):
        calls.append(("ensure", model, force_rebuild))
        return {"status": "READY", "model": model, "rebuild_performed": force_rebuild}

    def probe(model):
        calls.append(("probe", model))
        return {"status": "GPU_READY", "model": model}

    def smoke(model):
        calls.append(("smoke", model))
        return {"status": "READY", "model": model}

    monkeypatch.setattr(vg, "ensure_gpu_model", ensure)
    monkeypatch.setattr(vg, "gpu_probe", probe)
    monkeypatch.setattr(vg, "smoke_test", smoke)
    monkeypatch.setattr(vg, "status", lambda: [
        {"model": "gmic", "device": "gpu"},
        {"model": "nyu", "device": "gpu"},
        {"model": "glam", "device": "gpu"},
    ])

    result = vg.validate_gpu_models(
        ["gmic", "nyu"],
        force_rebuild=True,
        persist_report=False,
    )

    assert result["overall_status"] == "READY"
    assert calls == [
        ("ensure", "gmic", True),
        ("ensure", "nyu", True),
        ("probe", "gmic"),
        ("probe", "nyu"),
        ("smoke", "gmic"),
        ("smoke", "nyu"),
    ]
    assert result["models"]["gmic"]["configured_device"] == "gpu"
    assert result["models"]["nyu"]["smoke_test"]["status"] == "PASS"


def test_validate_gpu_models_skips_gpu_smoke_if_device_is_cpu(monkeypatch):
    monkeypatch.setattr(vg, "ensure_gpu_model", lambda model, force_rebuild=False: {"status": "READY"})
    monkeypatch.setattr(vg, "gpu_probe", lambda model: {"status": "GPU_READY"})
    monkeypatch.setattr(vg, "smoke_test", lambda model: (_ for _ in ()).throw(AssertionError("must not run")))
    monkeypatch.setattr(vg, "status", lambda: [{"model": "gmic", "device": "cpu"}])

    result = vg.validate_gpu_models(["gmic"], persist_report=False)

    assert result["overall_status"] == "FAILED"
    assert result["models"]["gmic"]["smoke_test"]["status"] == "SKIPPED"
    assert "GMIC_DEVICE=gpu" in result["models"]["gmic"]["smoke_test"]["reason"]
