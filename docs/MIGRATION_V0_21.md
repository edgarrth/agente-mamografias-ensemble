# Migración v0.21 — sampling reproducible y evidencia operacional

## Motivo

v0.20 completó por primera vez el flujo real CBIS-DDSM → GMIC + NYU + GLAM → Soft Voting para 5 estudios. Los cinco primeros estudios resultaron benignos, por lo que `sensitivity` y `ROC-AUC` quedaron correctamente no disponibles. También se confirmó que `resource_metrics.samples` representaba lecturas del monitor y podía confundirse con casos del dataset.

## Cambios

- `tests_flow.normal --sampling sequential|random|stratified|balanced --seed <N>`.
- `stratified` conserva aproximadamente la proporción de clases del universo seleccionado; para los 105 estudios CBIS-DDSM actuales, 10 estudios producen una cuota objetivo 7 benignos / 3 malignos.
- `balanced` solicita cuotas iguales; 10 estudios producen 5 benignos / 5 malignos si hay disponibilidad.
- `selected_studies.csv` conserva los IDs exactos usados en la corrida.
- `configuration_used.yaml` incorpora metadata de sampling.
- nuevo `run_summary.json` con `processed_studies`, `processed_images` y `overall_elapsed_seconds`.
- `resource_metrics.samples` cambia a `monitoring_samples` para eliminar ambigüedad semántica.
- `metrics.json` explica por qué Sensitivity/ROC-AUC son `null` cuando faltan clases.

## Qué conservar

Conserve `.env`, `workspace/`, CBIS-DDSM raw/processed, manifests y las tres imágenes Blackwell. v0.21 no cambia modelos ni `build_revision`, por lo que no requiere `ensure_gpu`, `gpu_probe`, smoke test, `download`, `inspect` ni `prepare` antes del siguiente normal test.

## Próxima validación sugerida

Ejecutar 10 estudios con `--sampling balanced --seed 42` para garantizar presencia de ambas clases en una prueba de integración. Para una muestra más representativa de la distribución 72/33, usar `--sampling stratified --seed 42`.
