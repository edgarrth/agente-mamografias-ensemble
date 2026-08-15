from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from mammography_agent.model_client import ensure_gpu_model, gpu_probe, smoke_test, status

SUPPORTED_MODELS = ("gmic", "nyu", "glam")


def expand_models(values: Iterable[str]) -> list[str]:
    normalized = [str(v).strip().lower() for v in values if str(v).strip()]
    if not normalized:
        raise ValueError("At least one model is required")
    if "all" in normalized:
        return list(SUPPORTED_MODELS)
    unknown = [m for m in normalized if m not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(normalized))


def _record_step(
    summary: dict,
    phase: str,
    model: str,
    fn: Callable[[], dict],
    *,
    fail_fast: bool,
) -> bool:
    started = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - started
        summary["models"][model][phase] = {
            "status": "PASS",
            "elapsed_seconds": elapsed,
            "result": result,
        }
        return True
    except Exception as exc:  # CLI must preserve evidence for other selected models.
        elapsed = time.monotonic() - started
        summary["models"][model][phase] = {
            "status": "FAIL",
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }
        summary["overall_status"] = "FAILED"
        if fail_fast:
            raise
        return False


def _configured_devices() -> dict[str, str]:
    return {
        str(item.get("model", "")).lower(): str(item.get("device", "")).lower()
        for item in status()
        if item.get("model")
    }


def validate_gpu_models(
    models: Iterable[str],
    *,
    force_rebuild: bool = False,
    fail_fast: bool = False,
    require_gpu_device: bool = True,
    persist_report: bool = True,
) -> dict:
    selected = expand_models(models)
    run_id = datetime.now(timezone.utc).strftime("gpu-validation-%Y%m%dT%H%M%SZ")
    started = time.monotonic()
    summary: dict = {
        "run_id": run_id,
        "operation": "ensure_gpu_then_probe_then_smoke_test",
        "selected_models": selected,
        "force_rebuild": bool(force_rebuild),
        "require_gpu_device_for_smoke": bool(require_gpu_device),
        "overall_status": "READY",
        "models": {m: {} for m in selected},
    }

    # Phase 1: ensure every selected model first. This mirrors release validation:
    # all current configured image revisions are present before probing/smoking any one model.
    ensure_ok: dict[str, bool] = {}
    for model in selected:
        ensure_ok[model] = _record_step(
            summary,
            "ensure_gpu",
            model,
            lambda m=model: ensure_gpu_model(m, force_rebuild=force_rebuild),
            fail_fast=fail_fast,
        )

    # Phase 2: CUDA allocation/kernel probe for every successfully ensured runtime.
    probe_ok: dict[str, bool] = {}
    for model in selected:
        if not ensure_ok.get(model):
            summary["models"][model]["gpu_probe"] = {
                "status": "SKIPPED",
                "reason": "ensure_gpu failed",
            }
            probe_ok[model] = False
            continue
        probe_ok[model] = _record_step(
            summary,
            "gpu_probe",
            model,
            lambda m=model: gpu_probe(m),
            fail_fast=fail_fast,
        )

    devices = _configured_devices()
    for model in selected:
        summary["models"][model]["configured_device"] = devices.get(model, "unknown")

    # Phase 3: end-to-end upstream smoke test. By default it must actually route to GPU.
    for model in selected:
        if not probe_ok.get(model):
            summary["models"][model]["smoke_test"] = {
                "status": "SKIPPED",
                "reason": "gpu_probe failed or was skipped",
            }
            continue
        device = devices.get(model, "unknown")
        if require_gpu_device and device != "gpu":
            summary["models"][model]["smoke_test"] = {
                "status": "SKIPPED",
                "reason": f"configured device is {device!r}; set {model.upper()}_DEVICE=gpu to validate the GPU image",
            }
            summary["overall_status"] = "FAILED"
            if fail_fast:
                raise RuntimeError(summary["models"][model]["smoke_test"]["reason"])
            continue
        _record_step(
            summary,
            "smoke_test",
            model,
            lambda m=model: smoke_test(m),
            fail_fast=fail_fast,
        )

    summary["elapsed_seconds"] = time.monotonic() - started
    if any(
        step.get("status") == "FAIL"
        for model_data in summary["models"].values()
        for step in model_data.values()
        if isinstance(step, dict)
    ):
        summary["overall_status"] = "FAILED"
    if any(
        isinstance(model_data.get("smoke_test"), dict)
        and model_data["smoke_test"].get("status") == "SKIPPED"
        for model_data in summary["models"].values()
    ):
        summary["overall_status"] = "FAILED"

    if persist_report:
        workspace = Path(os.getenv("WORKSPACE_ROOT", "/workspace"))
        report_dir = workspace / "output" / "model_validation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{run_id}.json"
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["report"] = str(report_path)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure current GPU runtime revisions, run CUDA probes, then execute upstream smoke tests "
            "for one or more models. Use --models all for GMIC + NYU + GLAM."
        )
    )
    parser.add_argument("--models", nargs="+", required=True, help="gmic nyu glam, any subset, or all")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild selected GPU images even when the configured build_revision is already present.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed phase instead of collecting results for the remaining models.",
    )
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Allow smoke tests when a selected model is configured for CPU. Default requires *_DEVICE=gpu.",
    )
    args = parser.parse_args()

    try:
        result = validate_gpu_models(
            args.models,
            force_rebuild=args.force_rebuild,
            fail_fast=args.fail_fast,
            require_gpu_device=not args.allow_cpu_smoke,
        )
    except Exception as exc:
        print(json.dumps({"overall_status": "FAILED", "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result.get("overall_status") == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
