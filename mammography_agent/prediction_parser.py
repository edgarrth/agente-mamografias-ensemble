from __future__ import annotations
from pathlib import Path
import pandas as pd, re

def _study_from_image_index(value: str) -> str:
    s=str(value)
    return re.sub(r"[_-](L|R)[_-](CC|MLO)$","",s,flags=re.IGNORECASE)

def parse_image_level(csv_path: Path, model: str) -> pd.DataFrame:
    df=pd.read_csv(csv_path)
    if not {"image_index","malignant_pred"}.issubset(df.columns):
        raise ValueError(f"Unexpected {model} output columns: {list(df.columns)}")
    df["study_key"]=df.image_index.map(_study_from_image_index)
    out=df.groupby("study_key",as_index=False,sort=False).malignant_pred.max()
    out.columns=["study_key",f"{model}_score"]
    return out

def parse_nyu(csv_path: Path, study_order_path: Path) -> pd.DataFrame:
    df=pd.read_csv(csv_path)
    if not {"left_malignant","right_malignant"}.issubset(df.columns):
        raise ValueError(f"Unexpected NYU output columns: {list(df.columns)}")
    order=pd.read_csv(study_order_path)
    if len(df)!=len(order): raise ValueError(f"NYU prediction count {len(df)} != study count {len(order)}")
    return pd.DataFrame({"study_id":order.study_id.astype(str),"nyu_score":df[["left_malignant","right_malignant"]].max(axis=1).astype(float)})
