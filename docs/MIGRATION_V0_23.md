# Migración v0.23 — métricas de operación y selección balanceada

v0.23 no modifica GMIC, NYU, GLAM, checkpoints, `build_revision`, CUDA/PyTorch, Soft Voting, XAI ni los 105 estudios/420 PNG CBIS-DDSM ya preparados. La migración afecta únicamente código de aplicación/experimentos y requiere reconstruir los servicios de aplicación.

## Motivo

El análisis v0.22 confirmó que el threshold baseline 0.50 está fuera de la escala observada y que bajar el threshold puede recuperar Sensitivity a costa de muchos falsos positivos. La política v0.22 seleccionaba `lowest_false_negatives` antes de considerar el coste en Specificity, lo que podía favorecer un clasificador casi-all-positive. Además, `Sensitivity` después de `FN` era redundante cuando la cantidad de malignos del Configuration Set es fija.

## Cambios

- `evaluate()` añade `specificity`, `precision_ppv`, `npv`, `fpr`, `accuracy` y `balanced_accuracy`, con razones explícitas cuando una métrica no puede calcularse.
- `all_configurations.csv`, `ranking.csv`, `best_configuration.json` y `final_metrics.json` heredan estas métricas.
- Selección v0.23: mejor ROC-AUC para escoger pesos; mejor Balanced Accuracy para escoger threshold; Sensitivity, Specificity/FP y distancia al baseline como desempates determinísticos.
- `candidate_thresholds.csv` generado por análisis diagnóstico usa `threshold_source=analysis_score_quantile`; el experimento formal usa `configuration_score_quantile`.
- `score_summary.json` y `score_analysis_report.md` incluyen métricas de clasificación del baseline threshold.
- El análisis CPU genera además `diagnostic_configurations.csv` y `diagnostic_ranking.csv` para inspeccionar las 80 combinaciones sobre un set diagnóstico; ambos quedan marcados como no elegibles para `freeze`.
- No hay inversión de score, calibración, entrenamiento ni uso del Final Test Set para seleccionar configuración.

## Migración

Conserve `.env`, `workspace/`, CBIS-DDSM preparado y las imágenes GPU existentes. Ejecute:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi bootstrap streamlit
docker compose up -d
```

Estos comandos recrean/reconstruyen servicios de aplicación. **No modifican, limpian, convierten ni eliminan el dataset** y no requieren repetir `download`, `inspect`, `prepare`, `ensure_gpu`, `gpu_probe` ni smoke tests porque v0.23 no cambia esos componentes.

## Validación científica siguiente

1. Reanalizar el `raw_model_predictions.csv` de los 10 estudios v0.21 para confirmar las nuevas métricas sin GPU.
2. Ejecutar `experiments.run` sobre los 105 estudios. Esta fase infiere únicamente el Configuration Set (~30%) y deja el Final Test Set reservado.
3. Revisar `ranking.csv` y `best_configuration.json` antes de congelar.
4. Congelar con `experiments.freeze`.
5. Ejecutar `experiments.final_evaluation` una sola vez sobre el Final Test Set reservado; reejecuciones reutilizan su cache compatible.
