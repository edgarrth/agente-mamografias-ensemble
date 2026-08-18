import pytest
from mammography_agent.ensemble.metrics import evaluate


def test_metrics_include_threshold_tradeoff_measures():
    # tn=2 fp=1 fn=1 tp=2
    m=evaluate([0,0,0,1,1,1],[.1,.2,.8,.3,.7,.9],.5)
    assert m["tn"]==2 and m["fp"]==1 and m["fn"]==1 and m["tp"]==2
    assert m["sensitivity"]==pytest.approx(2/3)
    assert m["specificity"]==pytest.approx(2/3)
    assert m["precision_ppv"]==pytest.approx(2/3)
    assert m["npv"]==pytest.approx(2/3)
    assert m["fpr"]==pytest.approx(1/3)
    assert m["accuracy"]==pytest.approx(2/3)
    assert m["balanced_accuracy"]==pytest.approx(2/3)


def test_metrics_explain_undefined_ppv_and_npv():
    all_negative=evaluate([0,1],[.1,.2],.9)
    assert all_negative["precision_ppv"] is None
    assert all_negative["precision_ppv_unavailable_reason"]
    all_positive=evaluate([0,1],[.8,.9],.1)
    assert all_positive["npv"] is None
    assert all_positive["npv_unavailable_reason"]


def test_metrics_v0301_include_f1_and_average_precision_auprc():
    from sklearn.metrics import average_precision_score
    y=[0,0,0,1,1,1]
    scores=[.1,.2,.8,.3,.7,.9]
    m=evaluate(y,scores,.5)
    # tp=2, fp=1, fn=1 -> F1 = 2*2/(4+1+1)=2/3
    assert m["f1"]==pytest.approx(2/3)
    expected=average_precision_score(y,scores)
    assert m["average_precision"]==pytest.approx(expected)
    assert m["auprc"]==pytest.approx(expected)
    assert m["auprc_method"]=="average_precision_score"
