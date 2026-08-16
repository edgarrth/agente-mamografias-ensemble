from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd
from mammography_agent.upstream_reference_validation import evaluate_prediction_csv, _compare


def _data(tmp_path: Path):
    data=[]
    labels=[(0,0),(0,1),(0,0),(1,0)]
    for i,(l,r) in enumerate(labels):
        data.append({"L-CC":[f"{i}_L_CC"],"R-CC":[f"{i}_R_CC"],"L-MLO":[f"{i}_L_MLO"],"R-MLO":[f"{i}_R_MLO"],"cancer_label":{"left_malignant":l,"right_malignant":r},"horizontal_flip":"NO"})
    p=tmp_path/'data.pkl'
    with p.open('wb') as f: pickle.dump(data,f,protocol=4)
    return p


def test_nyu_breast_metrics_from_official_contract(tmp_path):
    p=_data(tmp_path)
    # Use the breast-level example scores printed in the upstream README.
    df=pd.DataFrame({"left_malignant":[0.0091,0.0012,0.2325,0.0909],"right_malignant":[0.0179,0.7258,0.1061,0.2579]})
    csv=tmp_path/'nyu.csv'; df.to_csv(csv,index=False)
    obs=evaluate_prediction_csv('nyu',csv,p)
    assert obs['breast_rows']==8
    assert 0 <= obs['breast_roc_auc'] <= 1
    assert 0 <= obs['breast_auprc'] <= 1


def test_image_model_aggregates_views_to_breast(tmp_path):
    p=_data(tmp_path)
    rows=[]
    for i in range(4):
        for lat in ('L','R'):
            label=[(0,0),(0,1),(0,0),(1,0)][i][0 if lat=='L' else 1]
            for view in ('CC','MLO'):
                rows.append({"image_index":f"{i}_{lat}_{view}","malignant_pred":0.9 if label else 0.1,"malignant_label":label})
    csv=tmp_path/'gmic.csv'; pd.DataFrame(rows).to_csv(csv,index=False)
    obs=evaluate_prediction_csv('gmic',csv,p)
    assert obs['image_roc_auc']==1.0
    assert obs['breast_roc_auc']==1.0
    assert obs['breast_rows']==8


def test_reference_comparison_uses_rounding_tolerance_only():
    obs={"image_roc_auc":0.8674,"image_auprc":0.8506,"breast_roc_auc":0.8665,"breast_auprc":0.8504}
    row=_compare('gmic',obs)
    assert row['reference_metrics_match'] is True
    obs['image_roc_auc']=0.80
    assert _compare('gmic',obs)['reference_metrics_match'] is False
