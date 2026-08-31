# Experimental Configuration Report

> Research-only result. Not a clinical diagnosis.

- **run_id**: experiment-20260818T035300Z
- **prepared_studies**: 11913
- **formal_excluded_studies**: 10
- **formal_pool_studies**: 11903
- **configuration_studies**: 3570
- **final_test_reserved**: 8333
- **configurations**: 80
- **threshold_strategy**: {'mode': 'score_quantiles', 'derive_per_weight': True, 'ground_truth_used_for_derivation': False, 'quantiles': {'T01': 0.1, 'T02': 0.3, 'T03': 0.5, 'T04': 0.7, 'T05': 0.9}}
- **formal_chunk_size**: 25
- **selected_weight_id**: W16
- **selected_threshold**: 0.04942
- **selected_roc_auc**: 0.7176138759161964
- **selected_auprc**: 0.14537281111262915
- **selected_f1**: 0.12337998963193364
- **selected_balanced_accuracy**: 0.6700537555944737
- **selected_sensitivity**: 0.8263888888888888
- **selected_specificity**: 0.5137186223000584
- **selected_fp**: 1666
- **selected_fn**: 25
- **next_step**: python -m experiments.freeze --experiment experiment-20260818T035300Z
