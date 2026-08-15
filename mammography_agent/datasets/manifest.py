from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..workspace import safe_workspace_path

REQUIRED=["study_id","patient_id","ground_truth","l_cc","r_cc","l_mlo","r_mlo"]
OPTIONAL=["left_ground_truth","right_ground_truth","horizontal_flip"]

def read_manifest(path: str | Path) -> pd.DataFrame:
    p=safe_workspace_path(path)
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Manifest missing columns: {missing}")
    if not set(df.ground_truth.dropna().astype(int).unique()).issubset({0,1}):
        raise ValueError("ground_truth must be 0/1")
    if df.study_id.astype(str).duplicated().any(): raise ValueError("study_id must be unique")
    for c in ["l_cc","r_cc","l_mlo","r_mlo"]:
        for value in df[c].astype(str):
            if not safe_workspace_path(value).exists(): raise FileNotFoundError(f"Missing image in {c}: {value}")
    return df
