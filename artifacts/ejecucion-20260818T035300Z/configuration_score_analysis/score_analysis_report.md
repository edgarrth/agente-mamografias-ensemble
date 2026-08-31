# Score Analysis Report

> Research-only diagnostic. Scores are not calibrated clinical probabilities.

- **source**: /workspace/output/experiments/experiment-20260818T035300Z/configuration_set_predictions.csv
- **studies**: 3570
- **class_distribution**: {'BENIGN': 3426, 'MALIGNANT': 144}
- **baseline_threshold**: 0.5
- **baseline_score_range**: [0.009333, 0.821867]
- **threshold_strategy**: score quantiles {'T01': 0.1, 'T02': 0.3, 'T03': 0.5, 'T04': 0.7, 'T05': 0.9} derived independently for each weight combination
- **candidate_thresholds_generated**: True
- **threshold_derivation_uses_ground_truth**: False
- **score_inversion_or_calibration**: None

## Baseline classification metrics

- **TN / FP / FN / TP**: 3382 / 44 / 126 / 18
- **Sensitivity**: 0.125
- **Specificity**: 0.9871570344424986
- **Precision / PPV**: 0.2903225806451613
- **NPV**: 0.9640820980615735
- **FPR**: 0.01284296555750146
- **Balanced Accuracy**: 0.5560785172212492
- **F1**: 0.17475728155339806
- **AUPRC / Average Precision**: 0.1385067490755592

## Diagnostic candidate evaluation

- **diagnostic_only**: True
- **eligible_for_freeze**: False
- When candidate thresholds are enabled, see `diagnostic_configurations.csv` and `diagnostic_ranking.csv` for the 16×5 CPU-only preview on this analysis set.

## Per-model discrimination

- **gmic**: ROC-AUC=0.6955086106246351; AUPRC/AP=0.06714514106697536 (ROC-AUC stratified-bootstrap 95% CI 0.6535284203314523–0.7339776606181488, n=2000)
- **nyu**: ROC-AUC=0.6875000000000001; AUPRC/AP=0.10591646369040267 (ROC-AUC stratified-bootstrap 95% CI 0.6400937783777647–0.7361417692320166, n=2000)
- **glam**: ROC-AUC=0.6816166812609457; AUPRC/AP=0.12696454185579822 (ROC-AUC stratified-bootstrap 95% CI 0.6279104184909515–0.7350372660861386, n=2000)
- **baseline_ensemble**: ROC-AUC=0.7141375997275735; AUPRC/AP=0.1385067490755592 (ROC-AUC stratified-bootstrap 95% CI 0.668593972968152–0.7557515141564506, n=2000)

## Statistical uncertainty

- **benign / malignant studies**: 3426 / 144
- **one strictly ordered positive-negative pair changes AUC by approximately**: 2.0269832003632355e-06
- **CI method**: 2,000-replicate stratified bootstrap preserving class counts.
- These intervals describe uncertainty only. They are not used to choose weights, thresholds, orientation, or aggregation.
- With a 5/5 diagnostic set, modest AUC changes can correspond to only a few pairwise order changes and must not be treated as conclusive.

## Warnings

- None
