from __future__ import annotations
import pandas as pd, numpy as np
from ..config import load_yaml
from ..score_analysis import derive_threshold_candidates, threshold_strategy_config
from .metrics import evaluate


def all_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate 16 weights x 5 adaptive thresholds on cached Configuration Set scores.

    Thresholds are derived independently for each weight combination from score
    quantiles in the Configuration Set. The derivation is label-independent; ground
    truth is used only after the candidates exist, to compute TN/FP/FN/TP/Sensitivity.
    """
    cfg = load_yaml("experiments.yaml")
    strategy = threshold_strategy_config()
    rows=[]
    for wid,w in cfg["weights"].items():
        score=df.gmic_score*float(w[0])+df.nyu_score*float(w[1])+df.glam_score*float(w[2])
        for candidate in derive_threshold_candidates(score, strategy):
            m=evaluate(df.ground_truth,score,float(candidate["threshold"]))
            rows.append({
                "weight_id":wid,
                "threshold_id":candidate["threshold_id"],
                "threshold_quantile":candidate["threshold_quantile"],
                "threshold_source":candidate["threshold_source"],
                "ground_truth_used_for_threshold_derivation":candidate["ground_truth_used_for_threshold_derivation"],
                "w_gmic":float(w[0]),"w_nyu":float(w[1]),"w_glam":float(w[2]),
                **m,
            })
    out=pd.DataFrame(rows)
    if len(out)!=80: raise AssertionError(f"Expected 80 configurations, got {len(out)}")
    return out


def select_configuration(results: pd.DataFrame) -> pd.Series:
    tol=float(load_yaml("experiments.yaml")["selection"]["sensitivity_tolerance"])
    if results.empty:
        raise ValueError("No experimental configurations to select from")
    # ROC-AUC is identical across thresholds for the same weights. Find best weight sets first.
    weight_auc=results.groupby("weight_id",as_index=False).roc_auc.max()
    if weight_auc.roc_auc.isna().all():
        raise ValueError("Configuration Set must contain both ground-truth classes to select by ROC-AUC")
    best_auc=weight_auc.roc_auc.max()
    candidates=set(weight_auc[weight_auc.roc_auc==best_auc].weight_id)
    sub=results[results.weight_id.isin(candidates)].copy()
    min_fn=sub.fn.min(); sub=sub[sub.fn==min_fn]
    max_sens=sub.sensitivity.max(); sub=sub[sub.sensitivity>=max_sens-tol]
    min_fp=sub.fp.min(); sub=sub[sub.fp==min_fp]
    baseline=np.array([0.333333,0.333333,0.333334,0.50])
    sub["baseline_distance"]=sub.apply(lambda r: float(np.linalg.norm(np.array([r.w_gmic,r.w_nyu,r.w_glam,r.threshold])-baseline)),axis=1)
    return sub.sort_values(["baseline_distance","weight_id","threshold_id"]).iloc[0]
