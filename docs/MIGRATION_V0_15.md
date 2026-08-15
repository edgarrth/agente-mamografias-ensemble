# Migración v0.15 — descarga CBIS-DDSM no destructiva y reconstrucción de vistas con metadata.csv

v0.15 no cambia GMIC, DMV-CNN/NYU, GLAM, sus pesos, checkpoints, arquitectura, perfiles Blackwell ni la fórmula de Soft Voting.

## Cambios principales

1. `dataset_pipeline.download --datasets cbis_ddsm` **nunca descarga ni vuelve a descargar la colección DICOM**. La transferencia DICOM continúa siendo una acción explícita mediante TCIA/NBIA Data Retriever.
2. El comando sí descarga automáticamente, cuando faltan, los cuatro CSV oficiales de clasificación y los valida por SHA-256 y columnas obligatorias. Copias válidas existentes se reutilizan.
3. `metadata.csv` es reconocido como metadata auxiliar de TCIA. Sus campos `PatientID`, `StudyInstanceUID` y `SeriesInstanceUID` pueden enriquecer paciente/lateralidad/vista del índice DICOM, pero nunca aportan `pathology` ni ground truth.
4. El índice DICOM ya existente en `workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv` se reutiliza por defecto. Si existe un cache válido, v0.15 no reabre los DICOM ni recorre el árbol completo para reconstruir headers.
5. Si el árbol DICOM cambió después del último `inspect`, se debe solicitar explícitamente `--force-dicom-index`.
6. `inspect` agrega inventario explícito de objetos DICOM y casos: full mammograms, cropped images, ROI masks, otros DICOM, pacientes, estudios completos de cuatro vistas, estudios incompletos, vistas seleccionadas y ground truth a nivel del conjunto completo de cuatro vistas.
7. `.env.example` conserva exactamente el perfil de tres GPU ya validado en la workstation objetivo.

## Estado actual del workspace al migrar desde v0.14

Conservar sin cambios:

```text
.env
workspace/
Docker images mammography-model-*:blackwell-cu128
```

En particular, **no borrar**:

```text
workspace/datasets/raw/cbis_ddsm/
workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv
```

## Actualización

```bash
docker compose down --remove-orphans
# reemplazar únicamente el código por v0.15
docker compose build --no-cache fastapi bootstrap streamlit model-runner
docker compose up -d
```

No es necesario repetir `ensure_gpu` ni reconstruir las tres imágenes `:blackwell-cu128` si continúan en Docker.

## Descarga / verificación después de migrar

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.download \
  --datasets cbis_ddsm
```

Con un dataset DICOM ya descargado y los cuatro CSV presentes, el resultado esperado es:

```text
dicom_download_performed = false
dicom_reused = true
metadata_downloaded = []
metadata_reused = [4 archivos]
status = READY_FOR_INSPECT
```

Si falta alguno de los cuatro CSV, solo ese metadata pequeño será descargado. Los DICOM existentes no se tocan.

## Reinspección optimizada

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.inspect \
  --datasets cbis_ddsm
```

Por defecto reutiliza el índice DICOM existente y lo enriquece con `metadata.csv` si está disponible.

Solo si se modificó, completó o sustituyó el árbol DICOM después del último índice:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.inspect \
  --datasets cbis_ddsm \
  --force-dicom-index
```

Ese último comando sí vuelve a recorrer y leer headers DICOM.
