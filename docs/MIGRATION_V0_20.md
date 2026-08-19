# Migración v0.20 — GLAM CBIS-DDSM y endurecimiento de integración

## Motivo

La ejecución v0.19 demostró que GMIC y NYU completaron 5 estudios reales. GLAM alcanzó su classifier pero falló al copiar `left_benign`, una etiqueta opcional no incluida en el batch malignancy-only.

## Cambios

- GLAM Blackwell `build_revision=2`: etiqueta benigna ausente → `NaN` metadata, nunca sintetizada.
- Preprocessing/XAI aislado por modelo.
- Mapping explícito `study_key` ↔ `study_id` con detección de colisiones.
- Agregación root de XAI/resource metrics en chunk mode.
- Logs de ciclo de inferencia del Model Runner visibles en stdout.

## Qué conservar

Conserve `.env`, `workspace/`, datasets preparados y las imágenes GMIC/NYU. No repita `download`, `inspect` ni `prepare`.

## Qué reconstruir

Reconstruya servicios de aplicación. Luego `./scripts/validate-models.sh glam` detectará `GLAM build_revision=2`, reconstruirá solo GLAM, ejecutará GPU probe y smoke test. Después repita el normal test de 5 estudios.
