# Implementation status — v0.23

## Validado en workstation objetivo

- NVIDIA GeForce RTX 5060 Ti visible desde Docker/WSL.
- GMIC Blackwell runtime PyTorch 2.7.1 + CUDA 12.8: `gpu_probe` y smoke test upstream aprobados antes de v0.16.
- DMV-CNN/NYU Blackwell runtime: `gpu_probe` y smoke test aprobados.
- GLAM Blackwell runtime: `gpu_probe` y smoke test aprobados.
- CBIS-DDSM descargado mediante TCIA/NBIA y metadata oficial verificada.
- CBIS-DDSM inspect v0.15: 10,239 DICOM válidos; 1,566 pacientes; 105 estudios completos de cuatro vistas; 72 benignos y 33 malignos.
- CBIS-DDSM prepare v0.15: 105/105 estudios convertidos, 420 vistas; tiempo medido ~8m20s.
- Primer normal test de 5 estudios: falló rápidamente en GMIC por incompatibilidad de división entera del runtime moderno (`top_k_prop_y <= 0.0`).

## Cambio pendiente de validación en workstation

v0.16 corrige exclusivamente esa incompatibilidad GMIC mediante preservación de la semántica histórica de índices en `get_max_window`. El runtime GMIC usa `build_revision=2`, por lo que debe reconstruirse una vez y renovar `gpu_probe` antes de repetir el normal test de 5 estudios.

## Dataset / archivos

- La actualización v0.15→v0.16 no requiere re-descargar CBIS-DDSM.
- No requiere repetir `prepare`; los 105 estudios/420 PNG existentes se reutilizan.
- `.env.example` conserva la configuración GPU que ya fue validada para los tres modelos.

## Garantías metodológicas

No se realiza entrenamiento, fine-tuning, cambio de arquitectura, reemplazo de pesos ni cambio de la fórmula de Soft Voting. El parche GMIC afecta únicamente aritmética de índices para reproducir el comportamiento esperado por código escrito para PyTorch 1.1.

## v0.17

- `model_tools.validate_gpu`: implementado para `--models all` o cualquier subconjunto.
- Flujo: ensure current GPU revision → CUDA probe → upstream smoke test.
- `--force-rebuild`, `--fail-fast` y guard de device GPU implementados.
- Reporte persistente de validación bajo `workspace/output/model_validation/`.
- Sin cambios científicos en modelos/datasets/ensemble.


## v0.18

- GMIC Blackwell build revision 3: compatible con el contrato `cancer_label` malignancy-only del metarepository.
- El fix v0.16 de índices fue confirmado en CBIS-DDSM: el forward ya no falla con `top_k_prop_y`.
- La ausencia de etiquetas benignas independientes ya no aborta GMIC; se registra como `NaN` solo en metadata de salida.
- No se modifican datasets preparados ni se requiere repetir `prepare`.

## v0.19

La prueba real CBIS-DDSM posterior a v0.18 confirmó que GMIC ya completa el batch de 5 estudios: genera CSV de predicciones y 20 visualizaciones XAI. El fallo observado era exclusivamente de orquestación: el Model Runner fusionaba metadata de imagen `status=READY` después del estado operativo `SUCCESS`, por lo que el pipeline interpretaba una inferencia terminada como fallida. v0.19 corrige la precedencia del estado sin cambiar ningún runtime de modelo. La siguiente validación obligatoria es repetir el normal test de 5 estudios y comprobar que el flujo avanza a NYU, GLAM y Soft Voting.


## v0.20

La corrida real v0.19 confirmó GMIC SUCCESS (~112 s desde start/completed) y NYU SUCCESS (~21 s) en 5 estudios CBIS-DDSM. GLAM falló por `KeyError: left_benign` al copiar metadata de etiquetas, no por el forward. v0.20 corrige ese contrato en GLAM y endurece la trazabilidad del pipeline antes de repetir la prueba: aislamiento de XAI por modelo, mapping explícito de identidad de estudio y evidencia root de chunks.

## v0.21

v0.20 quedó validado end-to-end en la workstation: 5 estudios CBIS-DDSM completaron GMIC, NYU, GLAM, Soft Voting, XAI y reportes en ~3m52s. La selección first-N produjo 5/5 casos benignos. v0.21 no toca modelos ni datasets; añade sampling reproducible (`stratified`/`balanced`), evidencia `selected_studies.csv`, `run_summary.json`, razones explícitas para métricas no calculables y renombra el contador ambiguo de recursos a `monitoring_samples`.

## v0.22

La workstation validó v0.21 con 10 estudios CBIS-DDSM balanceados (5 benignos/5 malignos): los tres modelos, XAI y Soft Voting terminaron correctamente en 441.8 s. El resultado baseline fue TN=5, FP=0, FN=5, TP=0, Sensitivity=0 y ROC-AUC=0.36; los scores observados quedaron muy por debajo del threshold 0.50. v0.22 no modifica modelos: añade análisis reproducible de scores cacheados y reemplaza la grilla experimental absoluta 0.40-0.60 por cinco quantiles label-independent derivados del Configuration Set para cada combinación de pesos. El Final Test Set se mantiene reservado hasta freeze y su inferencia se reutiliza si ya existe.


## v0.23

v0.23 conserva modelos/dataset y corrige la evaluación experimental: se incorporan Specificity, PPV, NPV, FPR, Accuracy y Balanced Accuracy. La selección ya no minimiza FN antes que cualquier otro criterio; ahora escoge pesos por ROC-AUC y threshold por Balanced Accuracy, usando Sensitivity y Specificity como desempates. `threshold_source` distingue análisis diagnóstico (`analysis_score_quantile`) de Configuration Set (`configuration_score_quantile`). El Final Test Set sigue completamente aislado hasta freeze.
