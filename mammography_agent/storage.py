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
        c.execute(text("""CREATE TABLE IF NOT EXISTS web_evaluation_settings(
          settings_key TEXT PRIMARY KEY,
          inference_device TEXT NOT NULL,
          weight_mode TEXT NOT NULL,
          gmic_weight DOUBLE PRECISION NOT NULL,
          nyu_weight DOUBLE PRECISION NOT NULL,
          glam_weight DOUBLE PRECISION NOT NULL,
          threshold_mode TEXT NOT NULL,
          decision_threshold DOUBLE PRECISION NOT NULL,
          updated_at TIMESTAMPTZ DEFAULT NOW())"""))

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
    threshold_source: str | None,
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
          threshold_source TEXT,
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
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS threshold_source TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS ensemble_weights_json TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS weights_source TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS inference_device TEXT"))
        c.execute(text("ALTER TABLE web_inference_runs ADD COLUMN IF NOT EXISTS overall_elapsed_seconds DOUBLE PRECISION"))
        c.execute(text("""INSERT INTO web_inference_runs(
          run_id,status,classification,ensemble_score,threshold,threshold_source,ensemble_weights_json,weights_source,inference_device,overall_elapsed_seconds,
          gmic_score,nyu_score,glam_score,minio_bucket,minio_prefix,minio_status,artifact_path)
          VALUES(:run_id,:status,:classification,:ensemble_score,:threshold,:threshold_source,:ensemble_weights_json,:weights_source,:inference_device,:overall_elapsed_seconds,
                 :gmic_score,:nyu_score,:glam_score,:minio_bucket,:minio_prefix,:minio_status,:artifact_path)
          ON CONFLICT(run_id) DO UPDATE SET
            status=EXCLUDED.status,
            classification=EXCLUDED.classification,
            ensemble_score=EXCLUDED.ensemble_score,
            threshold=EXCLUDED.threshold,
            threshold_source=EXCLUDED.threshold_source,
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
            "threshold_source": threshold_source,
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


def load_web_evaluation_settings(settings_key: str = "default") -> dict | None:
    """Load the persisted Web-only UI configuration. Never reads or mutates batch YAML."""
    if not URL:
        return None
    e = create_engine(URL)
    with e.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS web_evaluation_settings(
          settings_key TEXT PRIMARY KEY,
          inference_device TEXT NOT NULL,
          weight_mode TEXT NOT NULL,
          gmic_weight DOUBLE PRECISION NOT NULL,
          nyu_weight DOUBLE PRECISION NOT NULL,
          glam_weight DOUBLE PRECISION NOT NULL,
          threshold_mode TEXT NOT NULL,
          decision_threshold DOUBLE PRECISION NOT NULL,
          updated_at TIMESTAMPTZ DEFAULT NOW())"""))
        row = c.execute(text("""SELECT settings_key,inference_device,weight_mode,gmic_weight,nyu_weight,glam_weight,
          threshold_mode,decision_threshold,updated_at
          FROM web_evaluation_settings WHERE settings_key=:settings_key"""), {"settings_key": settings_key}).mappings().first()
    if row is None:
        return None
    return {
        "settings_key": str(row["settings_key"]),
        "inference_device": str(row["inference_device"]),
        "weight_mode": str(row["weight_mode"]),
        "weights": {
            "gmic": float(row["gmic_weight"]),
            "nyu": float(row["nyu_weight"]),
            "glam": float(row["glam_weight"]),
        },
        "threshold_mode": str(row["threshold_mode"]),
        "decision_threshold": float(row["decision_threshold"]),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] is not None else None,
    }


def save_web_evaluation_settings(
    *,
    inference_device: str,
    weight_mode: str,
    weights: dict[str, float],
    threshold_mode: str,
    decision_threshold: float,
    settings_key: str = "default",
) -> dict | None:
    """Persist Web UI defaults in PostgreSQL without changing any batch configuration source."""
    if not URL:
        return None
    e = create_engine(URL)
    payload = {
        "settings_key": settings_key,
        "inference_device": str(inference_device).lower(),
        "weight_mode": str(weight_mode).upper(),
        "gmic_weight": float(weights["gmic"]),
        "nyu_weight": float(weights["nyu"]),
        "glam_weight": float(weights["glam"]),
        "threshold_mode": str(threshold_mode).upper(),
        "decision_threshold": float(decision_threshold),
    }
    with e.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS web_evaluation_settings(
          settings_key TEXT PRIMARY KEY,
          inference_device TEXT NOT NULL,
          weight_mode TEXT NOT NULL,
          gmic_weight DOUBLE PRECISION NOT NULL,
          nyu_weight DOUBLE PRECISION NOT NULL,
          glam_weight DOUBLE PRECISION NOT NULL,
          threshold_mode TEXT NOT NULL,
          decision_threshold DOUBLE PRECISION NOT NULL,
          updated_at TIMESTAMPTZ DEFAULT NOW())"""))
        c.execute(text("""INSERT INTO web_evaluation_settings(
          settings_key,inference_device,weight_mode,gmic_weight,nyu_weight,glam_weight,threshold_mode,decision_threshold,updated_at)
          VALUES(:settings_key,:inference_device,:weight_mode,:gmic_weight,:nyu_weight,:glam_weight,:threshold_mode,:decision_threshold,NOW())
          ON CONFLICT(settings_key) DO UPDATE SET
            inference_device=EXCLUDED.inference_device,
            weight_mode=EXCLUDED.weight_mode,
            gmic_weight=EXCLUDED.gmic_weight,
            nyu_weight=EXCLUDED.nyu_weight,
            glam_weight=EXCLUDED.glam_weight,
            threshold_mode=EXCLUDED.threshold_mode,
            decision_threshold=EXCLUDED.decision_threshold,
            updated_at=NOW()"""), payload)
    return load_web_evaluation_settings(settings_key)


def load_latest_web_inference_settings() -> dict | None:
    """Best-effort migration source for v0.34.x: recover the most recently applied Web run configuration."""
    if not URL:
        return None
    e = create_engine(URL)
    with e.begin() as c:
        exists = c.execute(text("SELECT to_regclass('public.web_inference_runs')")).scalar()
        if not exists:
            return None
        row = c.execute(text("""SELECT inference_device,ensemble_weights_json,weights_source,threshold,threshold_source,created_at
          FROM web_inference_runs
          WHERE status='SUCCESS' AND threshold IS NOT NULL AND ensemble_weights_json IS NOT NULL
          ORDER BY created_at DESC LIMIT 1""")).mappings().first()
    if row is None:
        return None
    try:
        weights = json.loads(row["ensemble_weights_json"] or "{}")
        weights = {k: float(weights[k]) for k in ("gmic", "nyu", "glam")}
    except Exception:
        return None
    return {
        "inference_device": str(row["inference_device"] or "cpu").lower(),
        "weight_mode": "CUSTOM" if str(row["weights_source"] or "").upper() == "WEB_OVERRIDE" else "BASELINE",
        "weights": weights,
        "threshold_mode": "CUSTOM" if str(row["threshold_source"] or "").upper() == "WEB_OVERRIDE" else "BASELINE",
        "decision_threshold": float(row["threshold"]),
        "migrated_from_run": True,
        "source_created_at": row["created_at"].isoformat() if row["created_at"] is not None else None,
    }
