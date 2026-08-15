from __future__ import annotations
import pandas as pd, numpy as np
from ..config import load_yaml
from .metrics import evaluate

def all_configurations(df: pd.DataFrame) -> pd.DataFrame:
    cfg=load_yaml("experiments.yaml"); rows=[]
    for wid,w in cfg["weights"].items():
        score=df.gmic_score*w[0]+df.nyu_score*w[1]+df.glam_score*w[2]
        for tid,t in cfg["thresholds"].items():
            m=evaluate(df.ground_truth,score,float(t))
            rows.append({"weight_id":wid,"threshold_id":tid,"w_gmic":w[0],"w_nyu":w[1],"w_glam":w[2],**m})
    out=pd.DataFrame(rows)
    if len(out)!=80: raise AssertionError(f"Expected 80 configurations, got {len(out)}")
    return out

def select_configuration(results: pd.DataFrame) -> pd.Series:
    tol=float(load_yaml("experiments.yaml")["selection"]["sensitivity_tolerance"])
    # ROC-AUC is identical across thresholds for the same weights. Find best weight sets first.
    weight_auc=results.groupby("weight_id",as_index=False).roc_auc.max()
    best_auc=weight_auc.roc_auc.max()
    candidates=set(weight_auc[weight_auc.roc_auc==best_auc].weight_id)
    sub=results[results.weight_id.isin(candidates)].copy()
    min_fn=sub.fn.min(); sub=sub[sub.fn==min_fn]
    max_sens=sub.sensitivity.max(); sub=sub[sub.sensitivity>=max_sens-tol]
    min_fp=sub.fp.min(); sub=sub[sub.fp==min_fp]
    baseline=np.array([0.333333,0.333333,0.333334,0.50])
    sub["baseline_distance"]=sub.apply(lambda r: float(np.linalg.norm(np.array([r.w_gmic,r.w_nyu,r.w_glam,r.threshold])-baseline)),axis=1)
    return sub.sort_values(["baseline_distance","weight_id","threshold_id"]).iloc[0]
