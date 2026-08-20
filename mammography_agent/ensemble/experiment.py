from __future__ import annotations
import pandas as pd, numpy as np
from ..config import load_yaml
from ..score_analysis import derive_threshold_candidates, threshold_strategy_config
from .metrics import evaluate


def all_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the configured weight x adaptive-threshold grid on cached Configuration Set scores.

    Threshold candidates are derived independently for each weight combination from
    Configuration Set score quantiles and do not use ground-truth labels. Labels are
    consulted only after candidate creation to compute research metrics.
    """
    cfg = load_yaml("experiments.yaml")
    strategy = threshold_strategy_config()
    rows=[]
    for wid,w in cfg["weights"].items():
        score=df.gmic_score*float(w[0])+df.nyu_score*float(w[1])+df.glam_score*float(w[2])
        for candidate in derive_threshold_candidates(
            score, strategy, threshold_source="configuration_score_quantile"
        ):
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
    expected=len(cfg["weights"])*len(strategy["quantiles"])
    if len(out)!=expected: raise AssertionError(f"Expected {expected} configurations, got {len(out)}")
    return out


def ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Return the human-readable v0.23 ranking used to inspect candidates.

    ROC-AUC chooses the score weighting/ranking quality. Balanced Accuracy then
    chooses a useful operating threshold instead of minimizing FN at any cost.
    Sensitivity and Specificity are deterministic tie-breakers.
    """
    if results.empty:
        return results.copy()
    out=results.copy()
    return out.sort_values(
        ["roc_auc","balanced_accuracy","sensitivity","specificity","fn","fp","weight_id","threshold_id"],
        ascending=[False,False,False,False,True,True,True,True],
        na_position="last",
    ).reset_index(drop=True)


def select_configuration(results: pd.DataFrame) -> pd.Series:
    """Select the frozen configuration without optimizing on the Final Test Set.

    v0.23 policy:
      1. choose the weight set(s) with highest ROC-AUC;
      2. among their thresholds, maximize Balanced Accuracy;
      3. within the configured tolerance, prefer higher Sensitivity;
      4. then prefer higher Specificity / fewer FP;
      5. deterministic tie-break by distance to the historical baseline.

    This removes the v0.22 behavior that minimized false negatives before considering
    specificity, which could select an almost-all-positive threshold.
    """
    selection=load_yaml("experiments.yaml")["selection"]
    tol=float(selection.get("balanced_accuracy_tolerance", 0.0))
    if results.empty:
        raise ValueError("No experimental configurations to select from")

    weight_auc=results.groupby("weight_id",as_index=False).roc_auc.max()
    if weight_auc.roc_auc.isna().all():
        raise ValueError("Configuration Set must contain both ground-truth classes to select by ROC-AUC")
    best_auc=float(weight_auc.roc_auc.max())
    candidates=set(weight_auc[np.isclose(weight_auc.roc_auc.astype(float),best_auc,rtol=0.0,atol=1e-12)].weight_id)
    sub=results[results.weight_id.isin(candidates)].copy()

    if sub.balanced_accuracy.isna().all():
        raise ValueError("Balanced Accuracy requires both ground-truth classes")
    best_balanced=float(sub.balanced_accuracy.max())
    sub=sub[sub.balanced_accuracy>=best_balanced-tol].copy()

    max_sens=sub.sensitivity.max()
    sub=sub[np.isclose(sub.sensitivity.astype(float),float(max_sens),rtol=0.0,atol=1e-12)].copy()
    max_spec=sub.specificity.max()
    sub=sub[np.isclose(sub.specificity.astype(float),float(max_spec),rtol=0.0,atol=1e-12)].copy()
    min_fp=sub.fp.min(); sub=sub[sub.fp==min_fp].copy()

    baseline=np.array([0.333333,0.333333,0.333334,0.50])
    sub["baseline_distance"]=sub.apply(
        lambda r: float(np.linalg.norm(np.array([r.w_gmic,r.w_nyu,r.w_glam,r.threshold])-baseline)),axis=1
    )
    return sub.sort_values(["baseline_distance","weight_id","threshold_id"]).iloc[0]
