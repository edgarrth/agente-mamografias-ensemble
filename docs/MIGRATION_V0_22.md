# Migración v0.22 — análisis de scores y thresholds adaptativos

v0.22 no modifica GMIC, NYU, GLAM, sus checkpoints, sus `build_revision`, CUDA/PyTorch ni los 105 estudios/420 PNG CBIS-DDSM preparados. La migración requiere reconstruir únicamente los servicios de aplicación.

## Motivo

La corrida real v0.21 de 10 estudios balanceados terminó `SUCCESS` en 441.8 s, pero produjo TN=5, FN=5, Sensitivity=0 y ROC-AUC=0.36 con threshold baseline 0.50. Los scores del ensemble estuvieron aproximadamente entre 0.024 y 0.102, por lo que la grilla histórica 0.40-0.60 no es informativa para este dataset/runtime.

## Cambios

- Nuevo `python -m experiments.score_analysis --input <raw_model_predictions.csv>`: analiza scores ya calculados sin lanzar contenedores de modelos.
- Evidencia: `score_summary.json`, `model_metrics.csv`, `score_distribution.csv`, `model_correlations.csv`, `roc_points.csv`, `candidate_thresholds.csv` y `score_analysis_report.md`.
- El experimento conserva 16 pesos × 5 thresholds = 80 configuraciones, pero cada peso deriva sus 5 thresholds de quantiles 10/30/50/70/90% de sus scores en el **Configuration Set**.
- La derivación de thresholds no usa `ground_truth`; las etiquetas se usan después para evaluar las 80 configuraciones.
- El Final Test Set continúa completamente reservado hasta `experiments.freeze`.
- `final_evaluation` reutiliza el cache de inferencia final si ya existe y es compatible, evitando ejecutar nuevamente modelos sobre los mismos estudios.

## Migración

Conserve `.env`, `workspace/`, CBIS-DDSM y las imágenes Blackwell existentes. Ejecute `docker compose down --remove-orphans`, reconstruya `model-runner fastapi bootstrap streamlit` y levante con `docker compose up -d`.

No ejecute `dataset_pipeline.prepare`, `ensure_gpu`, `gpu_probe` ni smoke tests por esta migración: v0.22 no cambia esos componentes.
