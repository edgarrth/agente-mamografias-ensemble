# Implementation status — v0.16

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
