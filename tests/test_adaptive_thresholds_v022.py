import pandas as pd
import numpy as np
from mammography_agent.ensemble.experiment import all_configurations
from mammography_agent.score_analysis import derive_threshold_candidates


def _df(labels=None):
    return pd.DataFrame({
        "study_id":[f"S{i}" for i in range(10)],
        "ground_truth": labels if labels is not None else [0,1]*5,
        "gmic_score":np.linspace(0.01,0.10,10),
        "nyu_score":np.linspace(0.02,0.11,10),
        "glam_score":np.linspace(0.005,0.095,10),
    })


def test_adaptive_grid_has_80_rows_and_is_not_legacy_fixed_grid():
    r=all_configurations(_df())
    assert len(r)==80
    assert r.weight_id.nunique()==16
    assert r.threshold_id.nunique()==5
    assert set(r.threshold_source)=={"configuration_score_quantile"}
    assert set(r.ground_truth_used_for_threshold_derivation)=={False}
    assert r.threshold.max() < 0.40


def test_threshold_derivation_is_label_independent():
    df1=_df([0,1]*5)
    df2=_df([1,0]*5)
    score1=df1.gmic_score*.333333+df1.nyu_score*.333333+df1.glam_score*.333334
    score2=df2.gmic_score*.333333+df2.nyu_score*.333333+df2.glam_score*.333334
    a=derive_threshold_candidates(score1)
    b=derive_threshold_candidates(score2)
    assert [(x["threshold_id"],x["threshold"]) for x in a] == [(x["threshold_id"],x["threshold"]) for x in b]
