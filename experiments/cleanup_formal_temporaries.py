from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from mammography_agent.config import WORKSPACE_ROOT
from mammography_agent.pipeline import (
    _chunk_cache_status,
    _orientation_chunk_cache_status,
    _retain_xai_and_cleanup_model_batch,
    _cleanup_orientation_temporaries,
    _formal_inference_config,
)


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def cleanup_configuration(experiment_id: str, apply: bool=False) -> dict:
    run_dir=WORKSPACE_ROOT/"output"/"experiments"/experiment_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Experiment not found: {run_dir}")

    top_scores=run_dir/"configuration_inference"/"raw_model_predictions.csv"
    final_manifest=run_dir/"final_test_manifest.csv"
    if not top_scores.is_file() or not final_manifest.is_file():
        raise RuntimeError("Configuration scores and final_test_manifest.csv must exist before cleanup")

    audit_scores=run_dir/"v0302_audit"/"raw_model_predictions.csv"
    if audit_scores.exists() and _sha256(audit_scores) != _sha256(top_scores):
        raise RuntimeError("Top-level Configuration scores differ from the audited copy; refusing cleanup")

    cfg=_formal_inference_config()
    orientation_root=run_dir/"configuration_orientation"/"chunks"
    inference_root=run_dir/"configuration_inference"/"chunks"
    summary={
        "experiment_id":experiment_id,
        "mode":"apply" if apply else "dry_run",
        "configuration_scores_sha256":_sha256(top_scores),
        "final_test_manifest_sha256":_sha256(final_manifest),
        "orientation_chunks_validated":0,
        "orientation_workdirs_targeted":0,
        "inference_chunks_validated":0,
        "model_batches_targeted":0,
        "xai_retention_per_model_per_chunk":cfg["xai_retention_per_model_per_chunk"],
        "predictions_modified":False,
        "final_test_scores_used":False,
    }

    if orientation_root.is_dir():
        for cdir in sorted(p for p in orientation_root.iterdir() if p.is_dir()):
            idx=int(cdir.name)
            manifest=cdir/"input_manifest.csv"
            if not manifest.is_file():
                raise RuntimeError(f"Missing orientation input manifest: {manifest}")
            sub=pd.read_csv(manifest,dtype={"study_id":str,"patient_id":str,"dataset_source":str})
            valid,_,reason=_orientation_chunk_cache_status(sub,cdir,idx)
            if not valid:
                raise RuntimeError(f"Orientation chunk {idx:04d} not safely reusable: {reason}")
            summary["orientation_chunks_validated"] += 1
            if (cdir/"original").exists() or (cdir/"counterfactual").exists():
                summary["orientation_workdirs_targeted"] += 1
                if apply:
                    _cleanup_orientation_temporaries(cdir)

    if inference_root.is_dir():
        for cdir in sorted(p for p in inference_root.iterdir() if p.is_dir()):
            idx=int(cdir.name)
            manifest=cdir/"chunk_manifest.csv"
            if not manifest.is_file():
                raise RuntimeError(f"Missing inference chunk manifest: {manifest}")
            sub=pd.read_csv(manifest,dtype={"study_id":str,"patient_id":str,"dataset_source":str})
            valid,_,reason=_chunk_cache_status(sub,cdir,idx)
            if not valid:
                raise RuntimeError(f"Inference chunk {idx:04d} not safely reusable: {reason}")
            summary["inference_chunks_validated"] += 1
            batch=cdir/"model_batch"
            heavy=any((batch/name).exists() for name in ("images","preprocessed","data.pkl")) if batch.exists() else False
            if heavy:
                summary["model_batches_targeted"] += 1
                if apply:
                    _retain_xai_and_cleanup_model_batch(cdir,cfg["xai_retention_per_model_per_chunk"])

    out=run_dir/"configuration_cleanup_v0311.json"
    if apply:
        out.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description="Safely validate and prune heavyweight Configuration chunk temporaries without changing scores or Final Test data.")
    p.add_argument("--experiment",required=True)
    p.add_argument("--apply",action="store_true",help="Actually remove validated temporaries. Without this flag, perform a dry run only.")
    args=p.parse_args()
    print(json.dumps(cleanup_configuration(args.experiment,apply=args.apply),indent=2))


if __name__ == "__main__":
    main()
