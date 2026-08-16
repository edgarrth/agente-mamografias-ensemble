from pathlib import Path
import json
import pandas as pd
import mammography_agent.breast_ensemble_analysis as be


def test_breast_aware_analysis_is_diagnostic_and_preserves_production_contract(tmp_path, monkeypatch):
    rows=[]
    # Six studies, balanced, two breasts each. NYU carries the useful signal.
    specs=[('S1',1,.8,.1),('S2',1,.7,.2),('S3',1,.6,.1),('S4',0,.2,.1),('S5',0,.3,.1),('S6',0,.25,.15)]
    for sid,gt,left_nyu,right_nyu in specs:
        for lat,nyu in [('LEFT',left_nyu),('RIGHT',right_nyu)]:
            bgt=1 if gt and lat=='LEFT' else 0
            for model,score in [('gmic',.2 if gt else .3),('nyu',nyu),('glam',.15 if gt else .2)]:
                rows.append({'study_id':sid,'patient_id':sid,'ground_truth':gt,'laterality':lat,'breast_ground_truth':bgt,'model':model,'breast_score':score})
    src=tmp_path/'breast_level_scores.csv'; pd.DataFrame(rows).to_csv(src,index=False)
    monkeypatch.setattr(be,'DEFAULT_ANALYSES',tmp_path/'analyses')
    out=be.analyze_breast_ensemble(src)
    summary=json.loads((out/'breast_ensemble_summary.json').read_text())
    assert summary['research_guards']['production_aggregation_changed'] is False
    assert summary['research_guards']['eligible_for_freeze'] is False
    ranking=pd.read_csv(out/'aggregation_strategy_ranking.csv')
    assert len(ranking)==16
    assert {'current_study_roc_auc','breast_aware_study_roc_auc','breast_level_roc_auc'} <= set(ranking.columns)
