# Migration v0.30.2 → v0.31.0

## Purpose

v0.31.0 expands Configuration-only ensemble selection without changing the already prepared RSNA data, the 10-study diagnostic exclusion, the deterministic 30% Configuration / 70% Final split, model checkpoints, preprocessing, orientation policy, canonical `malignancy_score`, or the Final-Test isolation boundary.

The release is designed to reuse the completed v0.30.2 Configuration inference. GMIC, NYU and GLAM do **not** need to be executed again when `configuration_inference/raw_model_predictions.csv` already exists and matches the preserved experiment manifests.

## Expanded search space

- Weight combinations: 40 unique candidates.
  - W01-W16 preserve the v0.30.2 identifiers and values.
  - W17-W40 add the missing 0.10-step simplex combinations while keeping every model weight >= 0.10.
- Adaptive threshold quantiles: 17 candidates per weight: q10, q20, q30, q40, q50, q60, q70, q80, q85, q90, q92, q94, q95, q96, q97, q98, q99.
- Total distinct candidate configurations: 40 × 17 = 680.

Threshold values remain score-derived and label-independent. Ground truth is used only after threshold derivation to compute research metrics.

## Cross-validated selection

`python -m experiments.reselect_configuration --experiment <EXPERIMENT_ID>` performs deterministic stratified 5-fold cross-validation with seed 42 over the existing Configuration predictions.

For each fold:

1. Four folds derive the numeric threshold associated with each configured quantile using ensemble scores only.
2. The held-out fold evaluates the already-derived threshold.
3. This is repeated for all 40 weights × 17 quantiles.

This produces 680 distinct candidates and 3,400 held-out fold evaluations. The predeclared selection order is preserved:

1. highest mean held-out ROC-AUC by weight;
2. highest mean Balanced Accuracy;
3. highest mean Sensitivity;
4. highest mean Specificity;
5. lowest mean false positives;
6. nearest historical baseline as deterministic tie-break.

After the winner `(weight_id, threshold_quantile)` is selected, its final numeric threshold is refit once using all 3,570 Configuration scores. Labels are not used for this threshold refit.

## Additive artifacts

The original v0.30.2 artifacts are not overwritten. v0.31.0 writes:

`workspace/output/experiments/<EXPERIMENT_ID>/configuration_selection_v0310/`

with:

- `fold_assignments.csv`
- `fold_metrics.csv`
- `candidate_cv_summary.csv`
- `ranking_cv.csv`
- `best_configuration.json`
- `selection_protocol.json`

`experiments.freeze` prefers this expanded-CV selection when it exists. If it does not exist, the legacy selection remains available for backward compatibility.

## Leakage guards

The reselection command refuses to run if:

- the experiment is already frozen;
- Final-Test inference/results already exist;
- the Final manifest hash differs from the original experiment plan;
- Configuration scores do not match the Configuration manifest identity/order;
- the live Configuration predictions differ from the preserved v0.30.2 audit copy when that copy is present.

No Final-Test scores are read or generated during reselection.
