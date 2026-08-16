from __future__ import annotations

from pathlib import Path
import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import load_yaml
from .ensemble.metrics import evaluate
from .input_fidelity import _audit_model_preprocessing
from .pipeline import _infer_three

DEFAULT_ANALYSES = Path('/workspace/output/analyses')
MODELS = ('gmic', 'nyu', 'glam')
MODEL_SCORE_COLUMNS = {'gmic': 'gmic_score', 'nyu': 'nyu_score', 'glam': 'glam_score'}
VIEWS = ('L-CC', 'R-CC', 'L-MLO', 'R-MLO')


def _utc_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def _normalize_flip(value: object) -> str:
    v = str(value if value is not None else 'NO').strip().upper()
    return 'YES' if v == 'YES' else 'NO'


def _toggle_flip(value: object) -> str:
    return 'NO' if _normalize_flip(value) == 'YES' else 'YES'


def _view_level_orientation(prep: pd.DataFrame) -> pd.DataFrame:
    """Collapse model-duplicated crop metadata to one row per study/view.

    GMIC, NYU and GLAM currently use the same NYU crop/orientation contract.  The
    diagnostic still preserves per-model rows separately later, but study
    discovery must not count the same view three times.
    """
    available = prep[prep.get('preprocessing_pickle_available', False) == True].copy()  # noqa: E712
    if available.empty:
        return pd.DataFrame(columns=['study_id', 'model_view', 'distance_nonzero', 'max_distance'])
    available['distance_from_starting_side'] = pd.to_numeric(available['distance_from_starting_side'], errors='coerce')
    rows = []
    for (study_id, model_view), group in available.groupby(['study_id', 'model_view'], dropna=False):
        distances = group['distance_from_starting_side'].dropna().astype(float)
        rows.append({
            'study_id': str(study_id),
            'model_view': str(model_view),
            'distance_nonzero': bool(group['distance_nonzero'].fillna(False).any()),
            'max_distance': float(distances.abs().max()) if len(distances) else None,
        })
    return pd.DataFrame(rows)


def _discover_suspects(prep: pd.DataFrame, min_nonzero_views: int) -> pd.DataFrame:
    view = _view_level_orientation(prep)
    rows = []
    if view.empty:
        return pd.DataFrame(columns=['study_id', 'nonzero_views', 'max_distance', 'suspect'])
    for study_id, group in view.groupby('study_id'):
        nonzero = int(group['distance_nonzero'].fillna(False).sum())
        distances = pd.to_numeric(group['max_distance'], errors='coerce').dropna()
        rows.append({
            'study_id': str(study_id),
            'nonzero_views': nonzero,
            'max_distance': float(distances.max()) if len(distances) else None,
            'suspect': bool(nonzero >= int(min_nonzero_views)),
        })
    return pd.DataFrame(rows).sort_values(['suspect', 'nonzero_views', 'study_id'], ascending=[False, False, True]).reset_index(drop=True)


def _prep_comparison(original: pd.DataFrame, counterfactual: pd.DataFrame, suspect_ids: set[str]) -> pd.DataFrame:
    cols = ['study_id', 'model', 'model_view', 'horizontal_flip', 'distance_from_starting_side', 'distance_nonzero', 'best_center_available']
    o = original[original.study_id.astype(str).isin(suspect_ids)][cols].copy()
    c = counterfactual[counterfactual.study_id.astype(str).isin(suspect_ids)][cols].copy()
    o = o.rename(columns={x: f'original_{x}' for x in cols if x not in {'study_id', 'model', 'model_view'}})
    c = c.rename(columns={x: f'counterfactual_{x}' for x in cols if x not in {'study_id', 'model', 'model_view'}})
    out = o.merge(c, on=['study_id', 'model', 'model_view'], how='outer', validate='one_to_one')
    for col in ['original_distance_from_starting_side', 'counterfactual_distance_from_starting_side']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['absolute_distance_reduction'] = out['original_distance_from_starting_side'].abs() - out['counterfactual_distance_from_starting_side'].abs()
    return out.sort_values(['study_id', 'model', 'model_view']).reset_index(drop=True)


def _study_orientation_summary(view_comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study_id, model), group in view_comparison.groupby(['study_id', 'model']):
        o = pd.to_numeric(group['original_distance_from_starting_side'], errors='coerce').fillna(0.0).abs()
        c = pd.to_numeric(group['counterfactual_distance_from_starting_side'], errors='coerce').fillna(0.0).abs()
        o_nonzero = int((o > 1e-9).sum())
        c_nonzero = int((c > 1e-9).sum())
        rows.append({
            'study_id': study_id,
            'model': model,
            'original_horizontal_flip': str(group['original_horizontal_flip'].dropna().iloc[0]) if group['original_horizontal_flip'].notna().any() else None,
            'counterfactual_horizontal_flip': str(group['counterfactual_horizontal_flip'].dropna().iloc[0]) if group['counterfactual_horizontal_flip'].notna().any() else None,
            'original_nonzero_views': o_nonzero,
            'counterfactual_nonzero_views': c_nonzero,
            'original_mean_abs_distance': float(o.mean()),
            'counterfactual_mean_abs_distance': float(c.mean()),
            'original_max_abs_distance': float(o.max()),
            'counterfactual_max_abs_distance': float(c.max()),
            'distance_improved': bool(c_nonzero < o_nonzero or (c_nonzero == o_nonzero and float(c.mean()) < float(o.mean()))),
            'counterfactual_eliminates_gap_all_views': bool(o_nonzero == len(group) and c_nonzero == 0),
        })
    return pd.DataFrame(rows).sort_values(['study_id', 'model']).reset_index(drop=True)


def _score_comparison(original_raw: pd.DataFrame, counterfactual_raw: pd.DataFrame, suspect_ids: set[str]) -> pd.DataFrame:
    rows = []
    orig = original_raw.copy(); orig['study_id'] = orig.study_id.astype(str)
    cf = counterfactual_raw.copy(); cf['study_id'] = cf.study_id.astype(str)
    for study_id in sorted(suspect_ids):
        o = orig[orig.study_id == study_id]
        c = cf[cf.study_id == study_id]
        if len(o) != 1 or len(c) != 1:
            raise ValueError(f'Expected exactly one original/counterfactual score row for {study_id}')
        ground_truth = int(o.iloc[0].ground_truth)
        for model, col in MODEL_SCORE_COLUMNS.items():
            ov = float(o.iloc[0][col]); cv = float(c.iloc[0][col])
            rows.append({
                'study_id': study_id,
                'ground_truth': ground_truth,
                'model': model,
                'original_score': ov,
                'counterfactual_score': cv,
                'score_delta': cv - ov,
                'absolute_score_delta': abs(cv - ov),
            })
    return pd.DataFrame(rows).sort_values(['study_id', 'model']).reset_index(drop=True)


def _auc_impact(original_raw: pd.DataFrame, counterfactual_raw: pd.DataFrame, suspect_ids: set[str]) -> pd.DataFrame:
    """Secondary/post-hoc impact only; never used to decide orientation."""
    original = original_raw.copy(); original['study_id'] = original.study_id.astype(str)
    replaced = original.copy()
    cf = counterfactual_raw.copy(); cf['study_id'] = cf.study_id.astype(str)
    cf_index = cf.set_index('study_id')
    for idx, row in replaced.iterrows():
        sid = str(row.study_id)
        if sid in suspect_ids:
            for col in MODEL_SCORE_COLUMNS.values():
                replaced.at[idx, col] = float(cf_index.loc[sid, col])

    rows = []
    for model, col in MODEL_SCORE_COLUMNS.items():
        oa = evaluate(original.ground_truth, original[col], 0.5).get('roc_auc')
        ca = evaluate(replaced.ground_truth, replaced[col], 0.5).get('roc_auc')
        rows.append({'score_system': model, 'original_roc_auc': oa, 'counterfactual_replacement_roc_auc': ca, 'delta_roc_auc': None if oa is None or ca is None else float(ca - oa)})

    weights = load_yaml('experiments.yaml')['weights']
    for wid in ('W01', 'W15'):
        if wid not in weights:
            continue
        w = [float(x) for x in weights[wid]]
        oscore = original.gmic_score*w[0] + original.nyu_score*w[1] + original.glam_score*w[2]
        cscore = replaced.gmic_score*w[0] + replaced.nyu_score*w[1] + replaced.glam_score*w[2]
        oa = evaluate(original.ground_truth, oscore, 0.5).get('roc_auc')
        ca = evaluate(replaced.ground_truth, cscore, 0.5).get('roc_auc')
        rows.append({'score_system': wid, 'original_roc_auc': oa, 'counterfactual_replacement_roc_auc': ca, 'delta_roc_auc': None if oa is None or ca is None else float(ca - oa)})
    return pd.DataFrame(rows)


def audit_orientation_counterfactual(
    run_dir: str | Path,
    output: str | Path | None = None,
    min_nonzero_views: int = 4,
) -> Path:
    """Toggle horizontal_flip only for studies with a 4-view starting-side gap.

    This is a targeted diagnostic. It intentionally performs new inference only on
    suspect studies so the upstream crop/center code can recompute its orientation
    metadata and model scores under the counterfactual orientation.  Dataset files,
    the original run, ensemble weights and thresholds are not modified.
    """
    run_dir = Path(run_dir).resolve()
    selected_path = run_dir / 'selected_studies.csv'
    raw_path = run_dir / 'raw_model_predictions.csv'
    if not selected_path.exists() or not raw_path.exists():
        raise FileNotFoundError('orientation counterfactual requires selected_studies.csv and raw_model_predictions.csv')
    selected = pd.read_csv(selected_path)
    raw = pd.read_csv(raw_path)
    required = {'study_id', 'ground_truth', 'l_cc', 'r_cc', 'l_mlo', 'r_mlo'}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f'selected_studies.csv missing required columns: {sorted(missing)}')
    if 'dataset_source' not in selected.columns:
        selected['dataset_source'] = 'unknown'
    if 'horizontal_flip' not in selected.columns:
        selected['horizontal_flip'] = 'NO'

    out = Path(output).resolve() if output else DEFAULT_ANALYSES / _utc_id('orientation-counterfactual')
    out.mkdir(parents=True, exist_ok=True)

    original_prep = _audit_model_preprocessing(run_dir, selected)
    suspects = _discover_suspects(original_prep, min_nonzero_views=min_nonzero_views)
    suspects.to_csv(out / 'suspect_studies.csv', index=False)
    suspect_ids = set(suspects[suspects.suspect].study_id.astype(str))

    if not suspect_ids:
        summary = {
            'source_run': str(run_dir), 'suspect_studies': 0, 'new_inference_performed': False,
            'decision': 'No study met the orientation-counterfactual trigger.',
            'research_guards': {'diagnostic_only': True, 'eligible_for_freeze': False, 'dataset_modified': False, 'original_run_modified': False},
        }
        _write_json(out / 'orientation_counterfactual_summary.json', summary)
        (out / 'orientation_counterfactual_report.md').write_text(
            '# Orientation Counterfactual Diagnostic\n\nNo study met the configured orientation trigger; no new inference was performed.\n',
            encoding='utf-8',
        )
        return out

    variant = selected[selected.study_id.astype(str).isin(suspect_ids)].copy().reset_index(drop=True)
    variant['original_horizontal_flip'] = variant['horizontal_flip'].map(_normalize_flip)
    variant['horizontal_flip'] = variant['horizontal_flip'].map(_toggle_flip)
    variant.to_csv(out / 'counterfactual_selected_studies.csv', index=False)

    inference_dir = out / 'counterfactual_inference'
    inference_dir.mkdir(parents=True, exist_ok=True)
    # Preserve a manifest inside the diagnostic run so all downstream audit helpers
    # can inspect this run exactly like a normal direct-layout inference.
    variant.to_csv(inference_dir / 'selected_studies.csv', index=False)
    cf_raw = _infer_three(variant, inference_dir, _utc_id('orientation-cf'))

    counter_prep = _audit_model_preprocessing(inference_dir, variant)
    view_cmp = _prep_comparison(original_prep, counter_prep, suspect_ids)
    study_cmp = _study_orientation_summary(view_cmp)
    score_cmp = _score_comparison(raw, cf_raw, suspect_ids)
    auc_impact = _auc_impact(raw, cf_raw, suspect_ids)

    view_cmp.to_csv(out / 'orientation_view_comparison.csv', index=False)
    study_cmp.to_csv(out / 'orientation_study_comparison.csv', index=False)
    score_cmp.to_csv(out / 'orientation_score_comparison.csv', index=False)
    auc_impact.to_csv(out / 'orientation_auc_impact.csv', index=False)

    per_study = []
    for sid in sorted(suspect_ids):
        sg = study_cmp[study_cmp.study_id == sid]
        per_study.append({
            'study_id': sid,
            'models_distance_improved': int(sg.distance_improved.fillna(False).sum()),
            'models_gap_eliminated_all_views': int(sg.counterfactual_eliminates_gap_all_views.fillna(False).sum()),
            'all_models_distance_improved': bool(len(sg) == len(MODELS) and sg.distance_improved.fillna(False).all()),
            'all_models_gap_eliminated_all_views': bool(len(sg) == len(MODELS) and sg.counterfactual_eliminates_gap_all_views.fillna(False).all()),
        })

    summary = {
        'source_run': str(run_dir),
        'trigger': {'minimum_nonzero_unique_views': int(min_nonzero_views), 'suspect_studies': len(suspect_ids), 'study_ids': sorted(suspect_ids)},
        'counterfactual': 'Toggle horizontal_flip per suspect study; keep image bytes and all model/checkpoint settings unchanged.',
        'new_inference_performed': True,
        'inferred_studies': len(variant),
        'models': list(MODELS),
        'orientation_evidence': per_study,
        'auc_impact_is_secondary_only': True,
        'orientation_decision_rule': 'Prefer upstream geometric evidence (distance_from_starting_side reduction/elimination) over post-hoc AUC changes.',
        'research_guards': {
            'diagnostic_only': True, 'eligible_for_freeze': False,
            'dataset_modified': False, 'original_run_modified': False,
            'ensemble_weights_changed': False, 'threshold_changed': False,
            'model_weights_changed': False,
        },
    }
    _write_json(out / 'orientation_counterfactual_summary.json', summary)

    lines = [
        '# Orientation Counterfactual Diagnostic', '',
        '> Targeted diagnostic. It toggles only `horizontal_flip` for studies whose four unique views showed a starting-side gap. The source dataset and original run are not mutated.', '',
        f'- **source_run**: {run_dir}',
        f'- **suspect studies**: {len(suspect_ids)} ({", ".join(sorted(suspect_ids))})',
        f'- **new inference**: {len(variant)} studies × 3 models',
        '- **decision criterion**: upstream geometric orientation evidence first; AUC impact is secondary/post-hoc only.', '',
        '## Orientation evidence', '',
        '| study | model | original nonzero views | counterfactual nonzero views | original mean distance | counterfactual mean distance | improved |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for _, r in study_cmp.iterrows():
        lines.append(f"| {r.study_id} | {r.model} | {int(r.original_nonzero_views)} | {int(r.counterfactual_nonzero_views)} | {r.original_mean_abs_distance:.2f} | {r.counterfactual_mean_abs_distance:.2f} | {bool(r.distance_improved)} |")
    lines += ['', '## Score impact (secondary)', '', '| study | model | original | counterfactual | delta |', '|---|---|---:|---:|---:|']
    for _, r in score_cmp.iterrows():
        lines.append(f"| {r.study_id} | {r.model} | {r.original_score:.6f} | {r.counterfactual_score:.6f} | {r.score_delta:+.6f} |")
    lines += ['', '## Research guard', '', '- These results are diagnostic and are **not eligible for freeze**.', '- Do not choose orientation because it improves ROC-AUC; use the upstream geometric orientation signal.']
    (out / 'orientation_counterfactual_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out
