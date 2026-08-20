from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mammography_agent.config import WORKSPACE_ROOT
from mammography_agent.ensemble.cv_selection import run_expanded_cv_selection, sha256_file


def main(experiment_id: str, folds: int = 5, seed: int = 42) -> Path:
    run_dir = WORKSPACE_ROOT / "output" / "experiments" / experiment_id
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if (run_dir / "frozen_configuration.yaml").exists():
        raise RuntimeError("Experiment is already frozen; expanded Configuration selection must occur before freeze")
    forbidden = [run_dir / "final_predictions.csv", run_dir / "final_metrics.json", run_dir / "final_model_comparison.csv"]
    if any(p.exists() for p in forbidden) or (run_dir / "final_inference" / "raw_model_predictions.csv").exists():
        raise RuntimeError("Final Test scores/artifacts already exist; refusing post-hoc Configuration reselection")

    scores_path = run_dir / "configuration_inference" / "raw_model_predictions.csv"
    config_manifest = run_dir / "configuration_set_manifest.csv"
    final_manifest = run_dir / "final_test_manifest.csv"
    plan_path = run_dir / "experiment_plan.json"
    for required in (scores_path, config_manifest, final_manifest, plan_path):
        if not required.exists():
            raise FileNotFoundError(required)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_final_hash = plan.get("final_test_manifest_sha256")
    observed_final_hash = sha256_file(final_manifest)
    if expected_final_hash and observed_final_hash != expected_final_hash:
        raise RuntimeError("Final Test manifest hash differs from the original experiment plan")

    scores = pd.read_csv(scores_path, dtype={"study_id": str, "patient_id": str})
    manifest = pd.read_csv(config_manifest, dtype={"study_id": str, "patient_id": str})
    if len(scores) != len(manifest):
        raise RuntimeError(f"Configuration score count mismatch: scores={len(scores)} manifest={len(manifest)}")
    if scores.study_id.astype(str).tolist() != manifest.study_id.astype(str).tolist():
        raise RuntimeError("Configuration scores and manifest differ in study identity/order")
    if int(plan.get("configuration_studies", len(scores))) != len(scores):
        raise RuntimeError("Configuration score count differs from the original experiment plan")

    audit_copy = run_dir / "v0302_audit" / "raw_model_predictions.csv"
    if audit_copy.exists() and sha256_file(audit_copy) != sha256_file(scores_path):
        raise RuntimeError("Live Configuration predictions differ from the preserved v0.30.2 audit copy")

    output_dir = run_dir / "configuration_selection_v0310"
    result = run_expanded_cv_selection(
        scores,
        output_dir,
        n_splits=int(folds),
        seed=int(seed),
        input_scores_sha256=sha256_file(scores_path),
        final_manifest_sha256=observed_final_hash,
    )
    print(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.experiment, a.folds, a.seed)
