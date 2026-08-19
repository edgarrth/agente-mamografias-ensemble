# Migration v0.30.0 → v0.30.1

## Scope

v0.30.1 does **not** change RSNA raw data, the 47,652 prepared PNG files, DICOM conversion, model checkpoints, preprocessing, orientation policy, canonical score semantics, or the 16 weight combinations / 5 adaptive thresholds. Reuse the existing `workspace` from v0.30.0.

## Formal RSNA population

The prepared RSNA manifest contains 11,913 studies. The 10 studies already used by `normal-20260818T001309Z` were selected before inference with `balanced + seed=42` and frozen in:

`/workspace/datasets/manifests/rsna_diagnostic_exclusion_v1.csv`

v0.30.1 requires this file for a formal RSNA experiment. Those 10 observed studies (5 benign, 5 malignant) are excluded **before** splitting. The formal pool is therefore expected to contain 11,903 studies: 11,422 benign and 481 malignant.

With `--configuration-ratio 0.30 --seed 42`, scikit-learn's deterministic stratified patient split is expected to assign:

- Configuration Set: 3,570 studies = 3,426 benign + 144 malignant.
- Final Test: 8,333 studies = 7,996 benign + 337 malignant.
- Overlap: 0 patients / 0 studies.
- Coverage: 11,903 / 11,903 formal studies (100%).

The phrase **100% formal pool** refers to all still-unobserved studies after the predeclared diagnostic exclusion. The 10 diagnostic studies remain available for descriptive/development analyses but are not eligible for the formal Configuration or Final subsets.

## Experimental sequence

1. `python -m experiments.run --datasets rsna --configuration-ratio 0.30 --seed 42`
2. Inspect `split_summary.json`, `configuration_report.md`, `ranking.csv`, `best_configuration.json` and score-analysis evidence.
3. `python -m experiments.freeze --experiment <ID>`
4. `python -m experiments.final_evaluation --experiment <ID>`

Only Configuration Set is inferred during step 1. Final Test inference occurs only in step 4 after freeze.

## New evidence

- `formal_pool_manifest.csv`
- `formal_exclusions_applied.csv`
- `split_summary.json`
- SHA-256 integrity values for Configuration and Final manifests in `experiment_plan.json`
- `average_precision` / `auprc` and `f1` in metrics
- `pr_points.csv` in score-analysis directories
- `final_model_comparison.csv` after Final Test

## Selection policy

v0.30.1 does not change the selection policy merely because RSNA is imbalanced. The predeclared selector remains:

1. highest ROC-AUC by weight set;
2. highest Balanced Accuracy among candidate thresholds;
3. higher Sensitivity;
4. higher Specificity / fewer FP;
5. shortest distance to the neutral baseline.

AUPRC/AP and F1 are added as required evidence. A later change to use AUPRC as a selection objective would require an explicit methodological decision and a new version before observing Final Test results.
