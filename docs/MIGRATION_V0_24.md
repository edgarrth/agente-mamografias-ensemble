# Migración v0.24 — auditoría de procedencia y agregación de scores

v0.24 no cambia GMIC, NYU, GLAM, checkpoints, CUDA/PyTorch, XAI, Soft Voting, pesos, thresholds ni la agregación productiva. Agrega una auditoría CPU sobre los artefactos nativos ya producidos por un `normal_test` para comprobar cómo cada score viaja desde vista/mama hasta estudio.

## Motivo

La preview v0.23 mejoró Balanced Accuracy con W15/T02, pero el ROC-AUC máximo siguió siendo 0.44. Antes de ejecutar los 105 estudios es necesario descartar errores de semántica, lateralidad o agregación.

## Nuevo comando

```bash
python -m experiments.score_provenance \
  --run-dir /workspace/output/normal_tests/normal-20260815T195006Z
```

No ejecuta GPU. Descubre y reutiliza los outputs nativos del run existente en cualquiera de los layouts que genera `normal_test`: `model_batch/{gmic,nyu,glam}.csv` para ejecución directa o `chunks/<NNNN>/model_batch/{gmic,nyu,glam}.csv` para ejecución con `max_runtime_minutes`. También usa `selected_studies.csv` y `raw_model_predictions.csv`.

## Salidas

- `native_model_scores.csv`: scores originales por vista (GMIC/GLAM) o mama (NYU), con lateralidad.
- `breast_level_scores.csv`: reconstrucción de scores por mama y ground truth lateral.
- `study_score_reconstruction.csv`: prueba de que el score de estudio almacenado corresponde a la agregación actual.
- `model_provenance_metrics.csv`: ROC-AUC por mama y por estudio, más alineamiento entre la mama de máximo score y la mama maligna.
- `score_provenance_summary.json`.
- `score_provenance_report.md`.

## Contrato de agregación auditado

- GMIC: `malignant_pred` por vista -> `max` por mama -> `max` entre mamas.
- NYU: `left_malignant` / `right_malignant` -> `max` entre mamas.
- GLAM: `malignant_pred` por vista -> `max` por mama -> `max` entre mamas.

La auditoría no cambia esa agregación y queda marcada `diagnostic_only=true`, `eligible_for_freeze=false`.


## v0.24.1 backward-compatibility fix

`model_batch/study_order.csv` is no longer mandatory for historical runs. When it is absent, the score-provenance audit reconstructs the study order from `selected_studies.csv` using the same deterministic study-key sanitization contract as `build_batch()`. The generated audit records whether the mapping was read or reconstructed. No inference, score, aggregation, threshold, or ensemble weight is changed by this compatibility fallback.


## v0.24.2 chunked-run compatibility fix

v0.24.0 y v0.24.1 asumían que los outputs nativos estaban en `<run>/model_batch/`. El flujo real de `normal_test` usa una estructura distinta cuando se especifica `max_runtime_minutes`: cada lote se ejecuta bajo `<run>/chunks/<NNNN>/model_batch/`, mientras el `raw_model_predictions.csv` agregado queda en el nivel superior.

v0.24.2 elimina esa suposición fija y descubre automáticamente ambos layouts. Para ejecuciones chunked:

- combina todos los chunks completos;
- usa `chunks/<NNNN>/model_batch/study_order.csv` cuando existe;
- si falta, reconstruye el orden exacto desde `chunks/<NNNN>/raw_model_predictions.csv`;
- valida que los chunks no se solapen y que cubran todos los estudios procesados;
- registra en `score_provenance_summary.json` y el reporte el layout, cantidad de batches, rutas y fuente del orden;
- soporta también runs parciales donde `selected_studies.csv` contiene más estudios que `raw_model_predictions.csv`.

La corrección fue validada con pruebas unitarias para layout directo, layout legacy sin `study_order.csv`, un chunk y múltiples chunks, más una ejecución CLI sintética con la misma jerarquía de archivos que produce `normal_test(max_runtime_minutes=...)`.
