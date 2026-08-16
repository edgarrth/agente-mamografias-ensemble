import pandas as pd
from mammography_agent.ensemble.experiment import select_configuration


def test_selection_prefers_balanced_operating_point_over_zero_fn_at_any_cost():
    # Same weight/AUC: v0.22 would choose T01 because fn=0 even though it predicts
    # almost everybody positive. v0.23 must choose T02 due to higher balanced accuracy.
    rows=[
        {
            "weight_id":"W01","threshold_id":"T01","roc_auc":0.70,
            "balanced_accuracy":0.55,"sensitivity":1.00,"specificity":0.10,
            "fn":0,"fp":9,"w_gmic":.333333,"w_nyu":.333333,"w_glam":.333334,"threshold":.02,
        },
        {
            "weight_id":"W01","threshold_id":"T02","roc_auc":0.70,
            "balanced_accuracy":0.75,"sensitivity":0.80,"specificity":0.70,
            "fn":2,"fp":3,"w_gmic":.333333,"w_nyu":.333333,"w_glam":.333334,"threshold":.04,
        },
    ]
    selected=select_configuration(pd.DataFrame(rows))
    assert selected.threshold_id=="T02"


def test_selection_uses_auc_to_choose_weights_before_threshold_metrics():
    rows=[
        {"weight_id":"W01","threshold_id":"T01","roc_auc":0.80,"balanced_accuracy":0.60,"sensitivity":.6,"specificity":.6,"fn":4,"fp":4,"w_gmic":.33,"w_nyu":.33,"w_glam":.34,"threshold":.04},
        {"weight_id":"W02","threshold_id":"T01","roc_auc":0.70,"balanced_accuracy":0.90,"sensitivity":.9,"specificity":.9,"fn":1,"fp":1,"w_gmic":.5,"w_nyu":.25,"w_glam":.25,"threshold":.04},
    ]
    selected=select_configuration(pd.DataFrame(rows))
    assert selected.weight_id=="W01"
