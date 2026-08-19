# Migración v0.14 — metadata CBIS-DDSM y perfil `.env.example` validado

v0.14 no cambia GMIC, DMV-CNN/NYU, GLAM, sus pesos, checkpoints, arquitectura, perfiles Blackwell ni la fórmula de Soft Voting.

## Cambios

1. `.env.example` ahora refleja la configuración ya validada en la workstation RTX 5060 Ti: GMIC, NYU y GLAM en GPU, `ALLOW_GPU=true`, `GPU_NUMBER=0`.
2. `dataset_pipeline.inspect --datasets cbis_ddsm` ya no lanza traceback cuando faltan los cuatro CSV de clasificación. Devuelve `METADATA_REQUIRED`, lista exactamente los archivos faltantes y genera `METADATA_INSTRUCTIONS.md`.
3. `prepare` también se detiene limpiamente con `METADATA_REQUIRED`; no intenta leer un `source_manifest.csv` inexistente.
4. El adapter acepta tanto los nombres canónicos de los cuatro CSV como los nombres alternativos derivados de las etiquetas de TCIA `Mass/Calc-Training/Test-Description.csv`.
5. Mientras falte metadata, `inspect` no inicia el índice DICOM costoso.

## Archivos requeridos

Bajo `/workspace/datasets/raw/cbis_ddsm/` deben existir los DICOM descargados por NBIA y los cuatro CSV oficiales:

- `mass_case_description_train_set.csv`
- `mass_case_description_test_set.csv`
- `calc_case_description_train_set.csv`
- `calc_case_description_test_set.csv`

Los CSV pueden estar en cualquier subdirectorio, recomendado `metadata/`.

## Actualización

Conservar `.env`, `workspace/` y las imágenes Docker ya creadas.

```bash
docker compose down --remove-orphans
# reemplazar solo el código por v0.14
docker compose build --no-cache fastapi bootstrap streamlit model-runner
docker compose up -d
```

No es necesario repetir `ensure_gpu` ni reconstruir las tres imágenes `:blackwell-cu128` si continúan en Docker.
