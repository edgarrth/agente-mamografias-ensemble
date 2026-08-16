from pathlib import Path
import json
import pandas as pd

import mammography_agent.orientation_counterfactual as oc


def _selected():
    rows=[]
    for i,(sid,gt) in enumerate([('S1',0),('S2',1),('S3',0)]):
        rows.append({
            'study_id':sid,'patient_id':f'P{i}','ground_truth':gt,'dataset_source':'cbis_ddsm',
            'l_cc':f'/workspace/lcc{i}.png','r_cc':f'/workspace/rcc{i}.png',
            'l_mlo':f'/workspace/lmlo{i}.png','r_mlo':f'/workspace/rmlo{i}.png',
            'left_ground_truth':gt if sid=='S2' else 0,'right_ground_truth':0,'horizontal_flip':'NO',
        })
    return pd.DataFrame(rows)


def _raw():
    return pd.DataFrame([
        {'study_id':'S1','patient_id':'P0','ground_truth':0,'dataset_source':'cbis_ddsm','gmic_score':0.8,'nyu_score':0.7,'glam_score':0.6},
        {'study_id':'S2','patient_id':'P1','ground_truth':1,'dataset_source':'cbis_ddsm','gmic_score':0.6,'nyu_score':0.8,'glam_score':0.5},
        {'study_id':'S3','patient_id':'P2','ground_truth':0,'dataset_source':'cbis_ddsm','gmic_score':0.1,'nyu_score':0.2,'glam_score':0.1},
    ])


def _prep(study_ids, distance_by_study, flip='NO'):
    rows=[]
    for sid in study_ids:
        for model in ('gmic','nyu','glam'):
            for view in ('L-CC','R-CC','L-MLO','R-MLO'):
                d=float(distance_by_study.get(sid,0))
                rows.append({
                    'preprocessing_pickle_available':True,'study_id':sid,'model':model,'model_view':view,
                    'horizontal_flip':flip,'distance_from_starting_side':d,'distance_nonzero':d!=0,
                    'best_center_available':True,
                })
    return pd.DataFrame(rows)


def test_counterfactual_targets_only_four_view_suspects_and_prefers_geometry(tmp_path, monkeypatch):
    run=tmp_path/'normal'; run.mkdir()
    selected=_selected(); selected.to_csv(run/'selected_studies.csv',index=False)
    raw=_raw(); raw.to_csv(run/'raw_model_predictions.csv',index=False)
    original_prep=_prep(['S1','S2','S3'], {'S1':800,'S2':0,'S3':0}, 'NO')
    counter_prep=_prep(['S1'], {'S1':0}, 'YES')

    calls=[]
    def fake_audit(run_dir, selected_df):
        return original_prep if Path(run_dir).resolve()==run.resolve() else counter_prep
    def fake_infer(df, inference_dir, run_id):
        calls.append(df.copy())
        assert df.study_id.tolist()==['S1']
        assert df.horizontal_flip.tolist()==['YES']
        return pd.DataFrame([{
            'study_id':'S1','patient_id':'P0','ground_truth':0,'dataset_source':'cbis_ddsm',
            'gmic_score':0.2,'nyu_score':0.15,'glam_score':0.12,
        }])
    monkeypatch.setattr(oc,'_audit_model_preprocessing',fake_audit)
    monkeypatch.setattr(oc,'_infer_three',fake_infer)

    out=oc.audit_orientation_counterfactual(run, tmp_path/'out')
    assert len(calls)==1
    suspects=pd.read_csv(out/'suspect_studies.csv')
    assert suspects[suspects.suspect].study_id.tolist()==['S1']
    study=pd.read_csv(out/'orientation_study_comparison.csv')
    assert len(study)==3
    assert study.counterfactual_nonzero_views.eq(0).all()
    assert study.distance_improved.all()
    scores=pd.read_csv(out/'orientation_score_comparison.csv')
    assert set(scores.model)=={'gmic','nyu','glam'}
    summary=json.loads((out/'orientation_counterfactual_summary.json').read_text())
    assert summary['orientation_evidence'][0]['all_models_gap_eliminated_all_views'] is True
    assert summary['auc_impact_is_secondary_only'] is True
    assert summary['research_guards']['eligible_for_freeze'] is False


def test_counterfactual_does_not_infer_when_no_four_view_suspect(tmp_path, monkeypatch):
    run=tmp_path/'normal'; run.mkdir()
    selected=_selected(); selected.to_csv(run/'selected_studies.csv',index=False)
    _raw().to_csv(run/'raw_model_predictions.csv',index=False)
    # one nonzero view only after collapsing models
    prep=_prep(['S1','S2','S3'], {}, 'NO')
    prep.loc[(prep.study_id=='S1') & (prep.model_view=='L-CC'),'distance_from_starting_side']=20
    prep.loc[(prep.study_id=='S1') & (prep.model_view=='L-CC'),'distance_nonzero']=True
    monkeypatch.setattr(oc,'_audit_model_preprocessing',lambda *a,**k: prep)
    monkeypatch.setattr(oc,'_infer_three',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not infer')))
    out=oc.audit_orientation_counterfactual(run, tmp_path/'out')
    summary=json.loads((out/'orientation_counterfactual_summary.json').read_text())
    assert summary['suspect_studies']==0
    assert summary['new_inference_performed'] is False
