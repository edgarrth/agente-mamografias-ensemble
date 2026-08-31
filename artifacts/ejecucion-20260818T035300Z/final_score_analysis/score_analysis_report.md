# Score Analysis Report

> Research-only diagnostic. Scores are not calibrated clinical probabilities.

- **source**: /workspace/output/experiments/experiment-20260818T035300Z/final_inference/raw_model_predictions.csv
- **studies**: 8333
- **class_distribution**: {'BENIGN': 7996, 'MALIGNANT': 337}
- **baseline_threshold**: 0.5
- **baseline_score_range**: [0.007900, 0.934500]
- **threshold_strategy**: score quantiles {'T01': 0.1, 'T02': 0.2, 'T03': 0.3, 'T04': 0.4, 'T05': 0.5, 'T06': 0.6, 'T07': 0.7, 'T08': 0.8, 'T09': 0.85, 'T10': 0.9, 'T11': 0.92, 'T12': 0.94, 'T13': 0.95, 'T14': 0.96, 'T15': 0.97, 'T16': 0.98, 'T17': 0.99} derived independently for each weight combination
- **candidate_thresholds_generated**: False
- **threshold_derivation_uses_ground_truth**: False
- **score_inversion_or_calibration**: None

## Baseline classification metrics

- **TN / FP / FN / TP**: 7877 / 119 / 290 / 47
- **Sensitivity**: 0.1394658753709199
- **Specificity**: 0.9851175587793897
- **Precision / PPV**: 0.28313253012048195
- **NPV**: 0.9644912452552957
- **FPR**: 0.014882441220610306
- **Balanced Accuracy**: 0.5622917170751548
- **F1**: 0.18687872763419483
- **AUPRC / Average Precision**: 0.17346284886383062

## Diagnostic candidate evaluation

- **diagnostic_only**: True
- **eligible_for_freeze**: False
- When candidate thresholds are enabled, see `diagnostic_configurations.csv` and `diagnostic_ranking.csv` for the 16×5 CPU-only preview on this analysis set.

## Per-model discrimination

- **gmic**: ROC-AUC=0.7149346186446339; AUPRC/AP=0.07181220762264895 (ROC-AUC stratified-bootstrap 95% CI 0.6885412244326912–0.7389980552219731, n=2000)
- **nyu**: ROC-AUC=0.6837357848063497; AUPRC/AP=0.12923157214499306 (ROC-AUC stratified-bootstrap 95% CI 0.651964251784646–0.7144286813287949, n=2000)
- **glam**: ROC-AUC=0.7051749168352722; AUPRC/AP=0.14330696792197525 (ROC-AUC stratified-bootstrap 95% CI 0.6682098282078724–0.7390834883317029, n=2000)
- **baseline_ensemble**: ROC-AUC=0.7437594910214751; AUPRC/AP=0.17346284886383062 (ROC-AUC stratified-bootstrap 95% CI 0.7149040117239629–0.7702049977139905, n=2000)

## Statistical uncertainty

- **benign / malignant studies**: 7996 / 337
- **one strictly ordered positive-negative pair changes AUC by approximately**: 3.711054340226493e-07
- **CI method**: 2,000-replicate stratified bootstrap preserving class counts.
- These intervals describe uncertainty only. They are not used to choose weights, thresholds, orientation, or aggregation.
- With a 5/5 diagnostic set, modest AUC changes can correspond to only a few pairwise order changes and must not be treated as conclusive.

## Warnings

- None
