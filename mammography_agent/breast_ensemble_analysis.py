from __future__ import annotations
from pathlib import Path
import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import yaml

DEFAULT_ANALYSES = Path('/workspace/output/analyses')
CONFIG = Path('/app/config/experiments.yaml')
MODELS = ('gmic','nyu','glam')


def _utc_id(prefix):
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _load_weights(config_path: Path | None = None):
    candidates = [config_path] if config_path else []
    candidates += [CONFIG, Path(__file__).resolve().parents[1] / 'config' / 'experiments.yaml']
    for p in candidates:
        if p and Path(p).exists():
            data = yaml.safe_load(Path(p).read_text(encoding='utf-8'))
            return data['weights'], data['threshold_strategy']['quantiles']
    raise FileNotFoundError('Could not locate config/experiments.yaml')


def _cm_metrics(y, score, threshold):
    pred=(np.asarray(score)>=float(threshold)).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else None
    spec=tn/(tn+fp) if tn+fp else None
    ba=(sens+spec)/2 if sens is not None and spec is not None else None
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),sensitivity=sens,specificity=spec,balanced_accuracy=ba)


def analyze_breast_ensemble(breast_scores: str|Path, output: str|Path|None=None, config_path: str|Path|None=None) -> Path:
    src=Path(breast_scores).resolve()
    df=pd.read_csv(src)
    required={'study_id','patient_id','ground_truth','laterality','breast_ground_truth','model','breast_score'}
    missing=sorted(required-set(df.columns))
    if missing: raise ValueError(f'breast_level_scores.csv missing required columns: {missing}')
    pivot=df.pivot_table(index=['study_id','patient_id','ground_truth','laterality','breast_ground_truth'],columns='model',values='breast_score',aggfunc='first').reset_index()
    if any(m not in pivot.columns for m in MODELS):
        raise ValueError('All three models are required for breast-aware ensemble analysis')
    weights, quantiles=_load_weights(Path(config_path).resolve() if config_path else None)
    rows=[]
    study_score_rows=[]
    for wid,vals in weights.items():
        wg,wn,wl=map(float,vals)
        b=pivot.copy()
        b['breast_ensemble_score']=b.gmic*wg+b.nyu*wn+b.glam*wl
        b_auc=float(roc_auc_score(b.breast_ground_truth,b.breast_ensemble_score))
        # Strategy B: preserve laterality through voting, then reduce breasts to the study.
        aware=b.groupby(['study_id','patient_id','ground_truth'],as_index=False).breast_ensemble_score.max()
        aware_auc=float(roc_auc_score(aware.ground_truth,aware.breast_ensemble_score))
        # Current strategy: each model is reduced to study level first, then voted.
        current=pivot.groupby(['study_id','patient_id','ground_truth'],as_index=False)[list(MODELS)].max()
        current['study_score_current']=current.gmic*wg+current.nyu*wn+current.glam*wl
        current_auc=float(roc_auc_score(current.ground_truth,current.study_score_current))
        merged=current[['study_id','study_score_current']].merge(aware[['study_id','breast_ensemble_score']],on='study_id',validate='one_to_one')
        for _,sr in merged.iterrows():
            study_score_rows.append({'weight_id':wid,'study_id':sr.study_id,'current_study_score':sr.study_score_current,'breast_aware_study_score':sr.breast_ensemble_score})
        for tid,q in quantiles.items():
            threshold=float(aware.breast_ensemble_score.quantile(float(q)))
            m=_cm_metrics(aware.ground_truth,aware.breast_ensemble_score,threshold)
            rows.append({
                'weight_id':wid,'w_gmic':wg,'w_nyu':wn,'w_glam':wl,
                'threshold_id':tid,'threshold_quantile':float(q),'threshold':threshold,
                'breast_level_roc_auc':b_auc,
                'current_study_roc_auc':current_auc,
                'breast_aware_study_roc_auc':aware_auc,
                **m,'diagnostic_only':True,'eligible_for_freeze':False,
            })
    out=Path(output).resolve() if output else DEFAULT_ANALYSES/_utc_id('breast-ensemble')
    out.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(out/'aggregation_strategy_comparison.csv',index=False)
    pd.DataFrame(study_score_rows).to_csv(out/'aggregation_strategy_study_scores.csv',index=False)
    by_weight=result.groupby(['weight_id','w_gmic','w_nyu','w_glam'],as_index=False).agg(
        breast_level_roc_auc=('breast_level_roc_auc','first'),
        current_study_roc_auc=('current_study_roc_auc','first'),
        breast_aware_study_roc_auc=('breast_aware_study_roc_auc','first'),
        best_balanced_accuracy=('balanced_accuracy','max'),
    ).sort_values(['breast_aware_study_roc_auc','best_balanced_accuracy'],ascending=False)
    by_weight.to_csv(out/'aggregation_strategy_ranking.csv',index=False)
    best=by_weight.iloc[0].to_dict()
    summary={
        'source':str(src),'studies':int(pivot.study_id.nunique()),'breasts':int(len(pivot)),
        'strategies':{
            'current':'max breast per model -> weighted soft vote at study level',
            'breast_aware':'weighted soft vote within each breast -> max across breasts',
        },
        'best_breast_aware_weight':best,
        'research_guards':{'diagnostic_only':True,'eligible_for_freeze':False,'model_inference_performed':False,'production_aggregation_changed':False},
    }
    (out/'breast_ensemble_summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n',encoding='utf-8')
    report=[
        '# Breast-aware Ensemble Diagnostic','',
        '> CPU-only diagnostic. The production aggregation contract is not changed.','',
        f'- **source**: {src}',f'- **studies**: {pivot.study_id.nunique()}',f'- **breasts**: {len(pivot)}','',
        '## Strategies','',
        '- **current**: max across breasts separately for GMIC/NYU/GLAM, then weighted soft voting at study level.',
        '- **breast-aware**: weighted soft voting inside LEFT and RIGHT separately, then max across the two breast ensemble scores.','',
        '## Best breast-aware weighting','',
        f"- **weight_id**: {best['weight_id']}",
        f"- **weights**: GMIC={best['w_gmic']:.3f}, NYU={best['w_nyu']:.3f}, GLAM={best['w_glam']:.3f}",
        f"- **breast-level ROC-AUC**: {best['breast_level_roc_auc']:.4f}",
        f"- **current study ROC-AUC**: {best['current_study_roc_auc']:.4f}",
        f"- **breast-aware study ROC-AUC**: {best['breast_aware_study_roc_auc']:.4f}",
        f"- **best candidate Balanced Accuracy**: {best['best_balanced_accuracy']:.4f}",'',
        '## Research guard','',
        '- These 10-case diagnostic results are not eligible to freeze weights, threshold, or aggregation.',
    ]
    (out/'breast_ensemble_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return out
