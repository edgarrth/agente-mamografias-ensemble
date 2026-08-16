from __future__ import annotations
import pandas as pd
from mammography_agent.score_analysis import stratified_bootstrap_auc


def test_stratified_bootstrap_auc_is_reproducible_and_keeps_classes():
    y = pd.Series([0,0,0,0,0,1,1,1,1,1])
    s = pd.Series([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    a = stratified_bootstrap_auc(y, s, replicates=200, seed=7)
    b = stratified_bootstrap_auc(y, s, replicates=200, seed=7)
    assert a == b
    assert a["bootstrap_replicates"] == 200
    assert a["roc_auc_ci95_low"] == 1.0
    assert a["roc_auc_ci95_high"] == 1.0
