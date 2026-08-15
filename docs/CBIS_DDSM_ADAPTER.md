# CBIS-DDSM adapter — v0.16

## Purpose

This adapter ingests the official CBIS-DDSM release from TCIA without requiring a researcher-authored mapping file. It is designed for auditability: every metadata row, image resolution, rejected row and incomplete study is persisted.

## Raw input contract

All downloaded content must remain under:

```text
/workspace/datasets/raw/cbis_ddsm
```

The directory tree may contain any nesting produced by NBIA Data Retriever. The adapter searches recursively.

The four official classification files are required. In v0.16, `dataset_pipeline.download` acquires missing copies automatically and verifies SHA-256 plus required columns; existing valid copies are reused:

```text
mass_case_description_train_set.csv
mass_case_description_test_set.csv
calc_case_description_train_set.csv
calc_case_description_test_set.csv
```

Recommended host layout:

```text
workspace/datasets/raw/cbis_ddsm/
├── CBIS-DDSM/
│   └── ... DICOM hierarchy ...
└── metadata/
    ├── metadata.csv                         # optional auxiliary TCIA series metadata
    ├── mass_case_description_train_set.csv
    ├── mass_case_description_test_set.csv
    ├── calc_case_description_train_set.csv
    └── calc_case_description_test_set.csv
```

## Mapping rules

The adapter uses these official fields:

- `patient_id`
- `left or right breast`
- `image view`
- `pathology`
- `image file path`

Ground truth mapping:

```text
MALIGNANT               -> 1
BENIGN                  -> 0
BENIGN_WITHOUT_CALLBACK -> 0
```

Unknown pathology values are rejected. `assessment`/BI-RADS is never converted to malignancy.

## Image resolution

Resolution order:

1. recursive path-suffix match against the official `image file path`;
2. DICOM Study/Series UID match from the metadata path;
3. deterministic patient/laterality/view fallback using DICOM headers and full-image size constraints.

A DICOM header index is cached at:

```text
/workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv
```

In v0.16, a completed cache is reused by default to avoid rescanning ~10k files through WSL/NTFS. If the DICOM tree changes after the cached inspection, rerun with `--force-dicom-index`. The optional TCIA `metadata.csv` is joined by `SeriesInstanceUID` to enrich patient/laterality/view identity without reopening DICOM files. It is never used for pathology.

ROI/cropped-image series are excluded from supplemental full-view discovery using the official cropped/ROI paths plus series/path descriptors.

## Four-view compatibility

The current ensemble uses the NYU/DMV-CNN exam-level classifier, whose input contract requires:

```text
L-CC
R-CC
L-MLO
R-MLO
```

The adapter never duplicates or synthesizes a missing view. It creates a study catalog and marks incomplete exams explicitly.

## Inspection

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.inspect \
  --datasets cbis_ddsm
```

Important output fields:

```text
metadata_rows
resolved_metadata_rows
unresolved_metadata_rows
dicom_files_indexed
dicom_headers_valid
dicom_objects
full_mammogram_images
cropped_images
roi_masks
other_dicom_images
auxiliary_metadata
supplemental_standard_views
patients
complete_four_view_studies
incomplete_studies
selected_full_view_images
complete_study_ground_truth_counts
ensemble_compatible
```

Artifacts:

```text
/workspace/datasets/manifests/cbis_ddsm_metadata_rows.csv
/workspace/datasets/manifests/cbis_ddsm_view_catalog.csv
/workspace/datasets/rejected/cbis_ddsm_unresolved_metadata_rows.csv
/workspace/datasets/rejected/cbis_ddsm_incomplete_studies.csv
/workspace/datasets/raw/cbis_ddsm/source_manifest.csv
```

## Preparation

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.prepare \
  --datasets cbis_ddsm
```

Only complete four-view studies are converted to 16-bit PNG and emitted into the canonical manifest:

```text
/workspace/datasets/processed/cbis_ddsm/images/
/workspace/datasets/manifests/cbis_ddsm.csv
```

If no compatible studies exist, preparation returns `INSUFFICIENT_FOUR_VIEW_STUDIES` and does not fabricate inputs.

## Scientific traceability

Primary dataset reference:

Lee RS, Gimenez F, Hoogi A, Miyake KK, Gorovoy M, Rubin DL. A curated mammography data set for use in computer-aided detection and diagnosis research. Scientific Data. 2017;4:170177. doi:10.1038/sdata.2017.177.

The adapter does not modify pathology labels, model weights, model architecture, or ensemble configuration.


## Preflight de metadata en v0.14

La descarga NBIA de imágenes puede existir sin las cuatro tablas CSV de clasificación. `inspect` verifica primero esas tablas. Si falta alguna, devuelve `METADATA_REQUIRED`, genera `METADATA_INSTRUCTIONS.md` y no inicia el índice DICOM. También reconoce aliases explícitos `Mass/Calc-Training/Test-Description.csv`.


## Política de descarga en v0.16

`dataset_pipeline.download --datasets cbis_ddsm` separa dos responsabilidades:

- **DICOM:** nunca se descarga automáticamente. Si ya existe bajo `raw/cbis_ddsm`, se reutiliza byte por byte. Si falta, el estado es `DICOM_DOWNLOAD_REQUIRED`.
- **Metadata clínica:** los cuatro CSV pequeños se descargan automáticamente solo si faltan; cada archivo se valida por SHA-256 y columnas.

Estados relevantes: `NOT_DOWNLOADED`, `DICOM_DOWNLOAD_REQUIRED`, `METADATA_REQUIRED`, `READY_FOR_INSPECT`, `INSPECTED_NOT_PREPARED`, `AVAILABLE`.

## metadata.csv auxiliar

El `metadata.csv` generado/descargado junto con la colección puede contener `PatientID`, `StudyInstanceUID` y `SeriesInstanceUID`. v0.16 lo usa únicamente como mapa de identidad de series. El texto de `PatientID` puede recuperar `P_XXXXX`, lateralidad y vista cuando el header DICOM no las expone. La etiqueta benigna/maligna sigue derivándose exclusivamente de `pathology` en los cuatro case-description CSV.
