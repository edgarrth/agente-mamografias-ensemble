# Migración v0.26

v0.26 añade un diagnóstico contrafactual de `horizontal_flip` para estudios donde las cuatro vistas presentan `distance_from_starting_side != 0` durante el preprocessing upstream.

No cambia modelos, checkpoints, pesos, thresholds, dataset preparado ni agregación productiva. El comando genera un batch temporal dentro de `workspace/output/analyses/orientation-counterfactual-*` y ejecuta inferencia solo para los estudios sospechosos con `horizontal_flip` invertido.

Artefactos principales:
- `suspect_studies.csv`
- `counterfactual_selected_studies.csv`
- `orientation_view_comparison.csv`
- `orientation_study_comparison.csv`
- `orientation_score_comparison.csv`
- `orientation_auc_impact.csv`
- `orientation_counterfactual_summary.json`
- `orientation_counterfactual_report.md`

Los cambios de AUC son secundarios y no deben usarse como criterio para escoger la orientación. La evidencia primaria es la reducción/eliminación del hueco geométrico registrado por el crop upstream.
