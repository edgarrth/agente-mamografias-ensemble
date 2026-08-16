from pathlib import Path
import json
import pandas as pd
import pickle

from mammography_agent import glam_runtime_differential as grd


def _sample_data(path: Path):
    exams=[]
    for i in range(4):
        # two malignant breasts total, enough for metrics
        exams.append({
            "L-CC":[f"{i}_L_CC"], "L-MLO":[f"{i}_L_MLO"],
            "R-CC":[f"{i}_R_CC"], "R-MLO":[f"{i}_R_MLO"],
            "cancer_label":{"left_malignant": int(i==3), "right_malignant": int(i==1)},
            "horizontal_flip":"NO",
        })
    with path.open('wb') as f: pickle.dump(exams,f)
    return exams


def _pred(path: Path, exams, scores):
    rows=[]; k=0
    for i,e in enumerate(exams):
        for view in ("L-CC","L-MLO","R-CC","R-MLO"):
            side='left' if view.startswith('L') else 'right'
            rows.append({"image_index":e[view][0], "malignant_pred":scores[k], "malignant_label":e['cancer_label'][f'{side}_malignant']})
            k+=1
    pd.DataFrame(rows).to_csv(path,index=False)


def test_runtime_differential_classifies_blackwell_mismatch(tmp_path, monkeypatch):
    meta=tmp_path/'runtime'/'mammography_metarepository'; (meta/'sample_data'/'images').mkdir(parents=True)
    for i in range(4):
        for v in ('L_CC','L_MLO','R_CC','R_MLO'): (meta/'sample_data'/'images'/f'{i}_{v}.png').write_bytes(b'x')
    exams=_sample_data(meta/'sample_data'/'data.pkl')
    monkeypatch.setattr(grd, 'WORKSPACE_ROOT', tmp_path)
    monkeypatch.setattr(grd, 'ensure_metarepository', lambda: {'path':str(meta)})

    # Legacy ordering chosen to match official metrics through a monkeypatched evaluator/comparator.
    def fake_legacy(**kw):
        _pred(Path(kw['output_file']), exams, [0.01,0.02,0.03,0.04, 0.02,0.8,0.03,0.7, 0.01,0.05,0.02,0.03, 0.7,0.1,0.6,0.05])
        return {'status':'SUCCESS'}
    def fake_bw(model, **kw):
        _pred(Path(kw['output_file']), exams, [0.01,0.02,0.03,0.04, 0.02,0.4,0.03,0.3, 0.01,0.05,0.02,0.03, 0.2,0.1,0.1,0.05])
        return {'status':'SUCCESS'}
    monkeypatch.setattr(grd, 'run_glam_legacy_cpu_reference', fake_legacy)
    monkeypatch.setattr(grd, 'run_model', fake_bw)

    calls={'n':0}
    def fake_compare(model, observed):
        calls['n']+=1
        return {'reference_metrics_match': calls['n']==1}
    monkeypatch.setattr(grd, '_compare', fake_compare)
    out=grd.run_glam_runtime_differential(tmp_path/'out')
    s=json.loads((out/'glam_runtime_differential_summary.json').read_text())
    assert s['decision']=='BLACKWELL_COMPATIBILITY_MISMATCH'
    assert (out/'glam_prediction_differential.csv').is_file()
    assert (out/'glam_runtime_metrics.csv').is_file()


def test_model_client_supports_device_override(monkeypatch):
    import mammography_agent.model_client as mc
    seen={}
    class R:
        ok=True; status_code=200; text='';
        def json(self): return {'status':'SUCCESS'}
    def post(url, json=None, timeout=None, params=None):
        seen['json']=json; return R()
    monkeypatch.setattr(mc.requests,'post',post)
    mc.run_model('glam','r','/workspace/i','/workspace/d','/workspace/o','/workspace/p',device='gpu')
    assert seen['json']['device']=='gpu'
