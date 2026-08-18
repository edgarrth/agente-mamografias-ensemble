from __future__ import annotations
import os, json
from sqlalchemy import create_engine, text

URL=os.getenv("DATABASE_URL")

def init_db():
    if not URL: return
    e=create_engine(URL)
    with e.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS research_runs(
          run_id TEXT PRIMARY KEY, run_type TEXT NOT NULL, status TEXT NOT NULL,
          artifact_path TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW())"""))

def save_run(run_id, run_type, status, artifact_path):
    if not URL: return
    e=create_engine(URL)
    with e.begin() as c:
        c.execute(text("""INSERT INTO research_runs(run_id,run_type,status,artifact_path)
          VALUES(:a,:b,:c,:d) ON CONFLICT(run_id) DO UPDATE SET status=EXCLUDED.status, artifact_path=EXCLUDED.artifact_path"""),
          {"a":run_id,"b":run_type,"c":status,"d":artifact_path})


def save_web_inference(
    *,
    run_id: str,
    status: str,
    classification: str | None,
    ensemble_score: float | None,
    threshold: float | None,
    ensemble_weights: dict[str, float] | None,
    weights_source: str | None,
    inference_device: str | None,
    overall_elapsed_seconds: float | None,
    gmic_score: float | None,
    nyu_score: float | None,
    glam_score: float | None,
    minio_bucket: str | None,
    minio_prefix: str | None,
    minio_status: str | None,
    artifact_path: str,
):
    """Additive Web-only persistence; the existing research_runs contract stays unchanged."""
    if not URL:
        return
    e = create_engine(URL)
    with e.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS web_inference_runs(
          run_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          classification TEXT,
          ensemble_score DOUBLE PRECISION,
          threshold DOUBLE PRECISION,
          ensemble_weights_json TEXT,
          weights_source TEXT,
          inference_device TEXT,
          overall_elapsed_seconds DOUBLE PRECISION,
          gmic_score DOUBLE PRECISION,
          nyu_score DOUBLE PRECISION,
          glam_score DOUBLE PRECISION,
          minio_bucket TEXT,
          minio_prefix TEXT,
          minio_status TEXT,
          artifact_path TEXT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT NOW())"""))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS ensemble_weights_json TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS weights_source TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS inference_device TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS overall_elapsed_seconds DOUBLE PRECISION"))
        c.execute(text("""INSERT INTO web_inference_runs(
          run_id,status,classification,ensemble_score,threshold,ensemble_weights_json,weights_source,inference_device,overall_elapsed_seconds,
          gmic_score,nyu_score,glam_score,minio_bucket,minio_prefix,minio_status,artifact_path)
          VALUES(:run_id,:status,:classification,:ensemble_score,:threshold,:ensemble_weights_json,:weights_source,:inference_device,:overall_elapsed_seconds,
                 :gmic_score,:nyu_score,:glam_score,:minio_bucket,:minio_prefix,:minio_status,:artifact_path)
          ON CONFLICT(run_id) DO UPDATE SET
            status=EXCLUDED.status,
            classification=EXCLUDED.classification,
            ensemble_score=EXCLUDED.ensemble_score,
            threshold=EXCLUDED.threshold,
            ensemble_weights_json=EXCLUDED.ensemble_weights_json,
            weights_source=EXCLUDED.weights_source,
            inference_device=EXCLUDED.inference_device,
            overall_elapsed_seconds=EXCLUDED.overall_elapsed_seconds,
            gmic_score=EXCLUDED.gmic_score,
            nyu_score=EXCLUDED.nyu_score,
            glam_score=EXCLUDED.glam_score,
            minio_bucket=EXCLUDED.minio_bucket,
            minio_prefix=EXCLUDED.minio_prefix,
            minio_status=EXCLUDED.minio_status,
            artifact_path=EXCLUDED.artifact_path"""), {
            "run_id": run_id,
            "status": status,
            "classification": classification,
            "ensemble_score": ensemble_score,
            "threshold": threshold,
            "ensemble_weights_json": json.dumps(ensemble_weights or {}, sort_keys=True),
            "weights_source": weights_source,
            "inference_device": inference_device,
            "overall_elapsed_seconds": overall_elapsed_seconds,
            "gmic_score": gmic_score,
            "nyu_score": nyu_score,
            "glam_score": glam_score,
            "minio_bucket": minio_bucket,
            "minio_prefix": minio_prefix,
            "minio_status": minio_status,
            "artifact_path": artifact_path,
        })
