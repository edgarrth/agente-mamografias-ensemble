from __future__ import annotations
import numpy as np
from ..domain import EnsembleResult

def vote(scores: dict[str,float], weights: dict[str,float], threshold: float, range_threshold: float=0.30) -> EnsembleResult:
    if set(scores)!={"gmic","nyu","glam"}: raise ValueError("Exactly gmic, nyu and glam scores are required")
    if any(not 0<=float(v)<=1 for v in scores.values()): raise ValueError("Scores must be in [0,1]")
    if abs(sum(weights.values())-1)>1e-6: raise ValueError("Weights must sum to 1")
    total=sum(float(scores[k])*float(weights[k]) for k in scores)
    vals=np.array(list(scores.values()),dtype=float)
    return EnsembleResult(ensemble_malignancy_score=float(total),
        classification="CANCER" if total>=threshold else "NO_CANCER",
        threshold=float(threshold),weights=weights,model_range=float(vals.max()-vals.min()),
        model_std=float(vals.std(ddof=0)),discordance=bool(vals.max()-vals.min()>=range_threshold))
