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
