# Migración v0.13 — Adapter oficial de CBIS-DDSM

## Objetivo

v0.12 ya ejecuta GMIC, DMV-CNN/NYU y GLAM con perfiles Blackwell independientes. v0.13 no cambia modelos, pesos, checkpoints, ensemble ni runtimes GPU; cambia exclusivamente la ingestión de CBIS-DDSM para eliminar el `source_manifest.csv` manual.

El adapter v0.13 usa el release oficial TCIA:

- árbol DICOM descargado mediante TCIA/NBIA Data Retriever;
- `mass_case_description_train_set.csv`;
- `mass_case_description_test_set.csv`;
- `calc_case_description_train_set.csv`;
- `calc_case_description_test_set.csv`.

## Qué conservar

Conserve sin cambios:

```text
.env
workspace/
imágenes Docker de GMIC/NYU/GLAM ya construidas
```

No borre la descarga CBIS-DDSM que ya esté en progreso.

## Dónde dejar CBIS-DDSM

El root reconocido por el proyecto es:

```text
/workspace/datasets/raw/cbis_ddsm
```

En el host del proyecto:

```text
./workspace/datasets/raw/cbis_ddsm
```

El adapter es recursivo; puede conservar la jerarquía que genere NBIA, por ejemplo:

```text
workspace/datasets/raw/cbis_ddsm/
├── CBIS-DDSM/
│   └── ... estructura NBIA / DICOM ...
└── metadata/
    ├── mass_case_description_train_set.csv
    ├── mass_case_description_test_set.csv
    ├── calc_case_description_train_set.csv
    └── calc_case_description_test_set.csv
```

Los CSV pueden estar en cualquier subdirectorio bajo `cbis_ddsm`; `metadata/` es solo la ubicación recomendada.

Si NBIA descargó los datos en otra carpeta, mueva **la carpeta completa** debajo de `workspace/datasets/raw/cbis_ddsm/`. No aplane ni renombre los DICOM.

## Actualización del código

```bash
docker compose down --remove-orphans
# reemplazar solo el código del proyecto por v0.13; conservar .env y workspace/
docker compose build --no-cache fastapi bootstrap streamlit model-runner
docker compose up -d
```

Las imágenes `mammography-model-*:blackwell-cu128` existentes no necesitan reconstruirse.

## Inspección antes de convertir

Cuando la descarga y los cuatro CSV estén presentes:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.inspect \
  --datasets cbis_ddsm
```

La inspección:

1. valida la presencia de los cuatro CSV oficiales;
2. normaliza paciente, lateralidad, vista y `pathology`;
3. resuelve `image file path` contra el árbol NBIA por sufijo/UID;
4. crea un índice cacheado de cabeceras DICOM cuando hace falta;
5. detecta las cuatro vistas estándar por paciente;
6. genera automáticamente `source_manifest.csv`;
7. registra filas no resueltas y estudios incompletos.

No decodifica píxeles para construir el índice de cabeceras y no ejecuta modelos.

## Ground truth

Únicamente la columna oficial `pathology` se transforma:

```text
MALIGNANT               = 1
BENIGN                  = 0
BENIGN_WITHOUT_CALLBACK = 0
```

No se deriva malignidad desde BI-RADS/`assessment`.

## Compuerta de cuatro vistas

El flujo actual del ensemble usa DMV-CNN/NYU exam-level y por tanto necesita:

```text
L-CC
R-CC
L-MLO
R-MLO
```

v0.13 no rellena una vista faltante con duplicados, imágenes contralaterales incorrectas ni imágenes sintéticas. Los casos incompletos se escriben en:

```text
/workspace/datasets/rejected/cbis_ddsm_incomplete_studies.csv
```

Por esta razón, revise primero `complete_four_view_studies` y `ensemble_compatible` en la salida de `dataset_pipeline.inspect`.

## Preparación

Si hay estudios compatibles:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.prepare \
  --datasets cbis_ddsm
```

Se generan:

```text
/workspace/datasets/processed/cbis_ddsm/images/
/workspace/datasets/manifests/cbis_ddsm.csv
```

Si no existen estudios completos de cuatro vistas, la preparación devuelve:

```text
INSUFFICIENT_FOUR_VIEW_STUDIES
```

sin fabricar datos.
