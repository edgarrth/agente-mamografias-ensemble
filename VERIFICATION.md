# Verification — v0.22

## Alcance

v0.22 no cambia GMIC, NYU, GLAM, sus checkpoints, `build_revision`, imágenes Blackwell ni CBIS-DDSM preparado. La evolución se concentra en análisis CPU de scores y diseño experimental.

## Evidencia real heredada de workstation

- v0.20: normal test 5 estudios end-to-end `SUCCESS` en ~3m52s.
- v0.21: normal test balanceado 10 estudios (5 benignos/5 malignos), 40 imágenes, `SUCCESS` en 441.796 s.
- v0.21 por modelo: GMIC 212.254 s, NYU 32.990 s, GLAM 178.762 s.
- v0.21 baseline: TN=5, FP=0, FN=5, TP=0, Sensitivity=0.0, ROC-AUC=0.36; score baseline observado ~0.0236..0.1021.

## Validación estática/unitaria del paquete v0.22

- `python -m compileall` para código Python.
- `pytest`: **80/80 PASS**; score analysis, adaptive threshold grid, 80 configurations, final inference cache, sampling, soft voting y regresiones anteriores.
- Parse de todos los YAML y `docker-compose.yml`.
- `bash -n` de scripts.
- Versión `0.22.0` en `VERSION`, package, FastAPI y Model Runner.
- `.env.example` conserva el perfil workstation validado: GMIC/NYU/GLAM GPU, `DEFAULT_MODEL_DEVICE=cpu`, `ALLOW_GPU=true`, `GPU_NUMBER=0`.
- `experiments.score_analysis` se ejecuta contra el `raw_model_predictions.csv` real de la corrida v0.21 sin GPU y reproduce AUC baseline 0.36, rango ~0.0236..0.1021 y 80 thresholds candidatos.
- Threshold derivation es label-independent y no ejecuta inversión/calibración/training.
- Final Test Set se mantiene sin inferencia antes de freeze y `final_evaluation` reutiliza cache compatible.

## Validación recomendada en workstation

No repita `ensure_gpu`, `gpu_probe`, smoke test, `download`, `inspect` ni `prepare`: v0.22 no cambia esos componentes.

1. Reconstruya únicamente servicios de aplicación.
2. Ejecute `experiments.score_analysis` contra el raw score v0.21 para validar el análisis CPU.
3. Abra el experimento formal completo sobre los 105 estudios con `experiments.run`; esta fase infiere solo ~31 estudios del Configuration Set.
4. Revise `configuration_score_analysis/`, `all_configurations.csv`, `ranking.csv` y `best_configuration.json`.
5. Congele con `experiments.freeze`.
6. Solo entonces ejecute `experiments.final_evaluation` sobre los ~74 estudios reservados.

El entorno de empaquetado no dispone de Docker Engine/GPU; la inferencia real de las fases 3 y 6 debe validarse en la workstation objetivo.
