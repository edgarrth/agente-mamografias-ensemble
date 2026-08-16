# Migración v0.18 — contrato de etiquetas GMIC / metarepository

## Motivo

La primera prueba end-to-end de CBIS-DDSM posterior al fix de índices v0.16 alcanzó correctamente el `forward()` de GMIC y falló después de la inferencia al copiar etiquetas al CSV:

`KeyError: 'left_benign'`

El `data.pkl` canónico generado por el prototipo sigue el contrato del mammography metarepository y contiene `left_malignant` y `right_malignant`. El runner standalone de GMIC intenta leer además `left_benign`/`right_benign` únicamente para rellenar las columnas de etiqueta de su CSV de salida.

## Cambio

v0.18 incrementa solo `GMIC gpu_compatibility.build_revision` de 2 a 3 y adapta el runner GMIC Blackwell para:

- conservar obligatoriamente `left_malignant` y `right_malignant`;
- usar `NaN` en `benign_label` cuando el dataset no proporciona una etiqueta benigna independiente;
- no sintetizar `benign = 1 - malignant`;
- no modificar el `forward()` del modelo, arquitectura, checkpoints, pesos, saliency maps ni scores.

NYU y GLAM no se reconstruyen.

## Dataset

No es necesario volver a ejecutar `download`, `inspect` ni `prepare`. Los DICOM raw, los 420 PNG procesados y el manifest CBIS-DDSM v0.15 se reutilizan sin cambios.

## Actualización

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi bootstrap streamlit
docker compose up -d
```

Luego validar solo GMIC con el flujo integrado:

```bash
./scripts/validate-models.sh gmic
```

`ensure_gpu` detectará `build_revision=3`, reconstruirá la imagen GMIC una vez, invalidará el probe anterior, ejecutará un nuevo GPU probe y finalmente el smoke test.

Si el resultado es `overall_status: READY`, repetir la integración CBIS-DDSM de 5 estudios:

```bash
docker compose exec fastapi \
  python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 5 \
  --max-runtime-minutes 120
```
