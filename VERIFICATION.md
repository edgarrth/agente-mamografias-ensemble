# Verification — through v0.31.0

## Scope

- Targeted `horizontal_flip` counterfactual for studies with four-view starting-side gaps.
- Reuse of the original run as immutable baseline.
- New inference only for suspect studies.
- Geometry is the primary orientation criterion; AUC impact is secondary/post-hoc only.

## Required release checks

- Full pytest suite passes.
- Python compileall passes.
- YAML files parse.
- Shell scripts pass `bash -n`.
- Packaged ZIP is re-extracted and full pytest is repeated from extracted bytes.
- A synthetic orientation-counterfactual test proves that only 4-view suspects are inferred, `horizontal_flip` is toggled, geometric gaps are compared, and diagnostic results remain ineligible for freeze.


## v0.26.1 bootstrap regression validation

- `config/config_additions.yaml` schema: all entries require `id`, `name`, `value`, `reason`; IDs unique.
- `ADD-081` present for `orientation_counterfactual_diagnostic`.
- `log_configuration_additions()` smoke-tested against the packaged configuration.
- Exact bootstrap preamble (`ensure_workspace(); log_configuration_additions()`) smoke-tested before packaging and again from the unpacked ZIP.


## v0.27 orientation policy

- label-independent upstream NYU crop+center preflight before classifier inference
- strict four-view-gap trigger and four-view-zero counterfactual acceptance rule
- Configuration orientation resolved before configuration inference
- Final Test orientation unresolved until after freeze, then the same fixed policy is applied
- `python -m experiments.orientation_preflight --run-dir <existing-run>` validates the policy without classifier inference


## v0.27.1 PREPROCESS_ONLY PYTHONPATH regression

- Reproduces the exact Python import behavior that caused `ModuleNotFoundError: No module named 'src'` when an individual `src/cropping/...` script is launched without repository-root `PYTHONPATH`.
- Proves the same script succeeds when the repository root is exported in `PYTHONPATH`.
- Asserts the model-runner command exports `/home/bcc/breast_cancer_classifier` for NYU before crop/center preprocessing.

## v0.28.0 upstream reference runtime validation

- Adds `experiments.upstream_reference_validation`.
- Uses the official four-exam `sample_data/` bundled by `nyukat/mammography_metarepository`.
- Reproduces `evaluation/score.py` semantics exactly: ROC-AUC with `roc_auc_score`, PR AUC with `auc(recall, precision)`, and mean CC/MLO image scores for image-model breast-level evaluation.
- Compares GMIC/NYU/GLAM Blackwell runtime metrics against the three-decimal sample references published by the upstream metarepository.
- Diagnostic only; does not use CBIS-DDSM, alter model/ensemble weights, thresholds, or train models.


## v0.28.2 GLAM differential guard

- Official-sample GLAM mismatch must be isolated before formal ensemble freeze.
- Legacy reference path: pinned metarepository GLAM image, PyTorch 1.1.0, CPU.
- Blackwell path: pinned GLAM source/checkpoint, PyTorch 2.7.1 + CUDA 12.8, GPU.
- Compare per-image scores and pairwise ordering; a higher AUROC is not considered a reproduction pass.


## v0.29.0 CMMD/manual-acquisition guard

- CBIS-DDSM dataset acquisition is fully manual: no URL table, `urlopen`, or metadata auto-download path remains.
- CMMD requires manually placed DICOM + `metadata/CMMD_clinicaldata_revision.xlsx`.
- CMMD CC/MLO comes only from DICOM `ViewCodeSequence.CodeValue` (`399162004=CC`, `399368009=MLO`); `ImageLaterality` provides L/R.
- Exact four-view structure is required; missing/duplicate views are rejected, never synthesized.
- Binary CMMD benchmark is CMMD1/D1 only, with explicit bilateral clinical labels. Study label is malignant if either labelled breast is malignant.
- CMMD2 four-view cases are retained as nonbenchmark audit artifacts because the cohort is malignant/subtype-focused and would confound cohort with class.
- The audited real CMMD preflight is expected to reproduce 1,775 patients, 5,202 DICOM, 826 four-view and 949 two-view before benchmark filtering.

## v0.29.1 multi-dataset input-scale guard

- `input_scale_comparison` resolves one `dataset_source` from the selected run instead of labelling every dataset as CBIS-DDSM.
- Raw prepared PNG and official NYU-crop comparison remain classifier-free and label-blind.

## v0.29.2 DICOM presentation counterfactual

- Adds `experiments.dicom_presentation_counterfactual` and `scripts/audit-dicom-presentation.sh`.
- Reads only selected study/view identity and original DICOM paths; ground truth and model scores are not used.
- Compares the production adapter conversion against Modality LUT/rescale and VOI presentation branches.
- For identity rescale/no Modality LUT, the Modality branch must remain byte-identical to the production adapter, including the historical <16-bit left-shift convention.
- VOI differences are diagnostic only; no branch can be selected by AUC/model score and no raw/prepared dataset bytes are modified.
- Synthetic test verifies 8-bit MONOCHROME2 + identity rescale + SIGMOID WindowCenter/WindowWidth yields exact current-vs-Modality identity and a distinct VOI branch.
- Version metadata is coherent at `0.29.2` across `VERSION`, package metadata, `mammography_agent.__version__`, Model Runner API and version-exposure tests.

## v0.29.2 packaged-release evidence

- Source tree: `pytest -q` → **115 passed**.
- Python `compileall` → PASS.
- All `config/*.yaml` parsed → PASS.
- All `scripts/*.sh` pass `bash -n` → PASS.
- Package checksum manifest covers every packaged file except itself; verification requires zero missing, modified or extra files.
- The ZIP is re-extracted and the complete test suite is rerun from packaged bytes.

## v0.30.0 RSNA adapter evidence

- Added native `rsna` registration and `RSNADatasetAdapter`.
- Added `RSNA_REQUIRED_FOUR_VIEWS_V1` and deterministic label-blind repeated-view selection.
- Added explicit audit manifests for selected/unselected/non-standard/incomplete/conflicting RSNA records.
- Added pydicom compressed-pixel plugins pinned to the versions validated during RSNA preflight.
- Updated version metadata to `0.30.0`.
- Repository test suite executed against the packaged source: `118 passed`.
- This packaged source intentionally contains no current RSNA/CMMD/CBIS dataset payloads; research workspace data remains external on the host bind mount.

## v0.30.1 formal RSNA split and metrics evidence

- Full pytest suite: 123 passed.
- Added deterministic RSNA synthetic-contract test reproducing the expected post-diagnostic pool: 11,903 studies (11,422 benign / 481 malignant).
- Verified 30/70 seed-42 split: Configuration 3,570 (3,426 / 144), Final 8,333 (7,996 / 337), zero study/patient overlap and 100% formal-pool coverage.
- Verified missing required RSNA diagnostic-exclusion manifest fails closed.
- Verified F1 and Average Precision/AUPRC are emitted by the shared evaluator.
- Verified Final Test comparison artifact and final-manifest/exclusion guards are present.


## v0.30.2 resumable formal execution evidence

- Full source-tree pytest suite: **127 passed**.
- Added tests for deterministic formal chunking, interruption/restart, successful-chunk hash refusal, orientation chunk reuse, and default chunk size.
- Configuration and Final population/split semantics remain covered by the v0.30.1 tests.
- Python compilation of the updated pipeline and CLIs passes.
- FastAPI Dockerfile installs the `[test]` extra and copies repository static-contract inputs so pytest can be run inside the rebuilt application container.


## v0.31.0 expanded-grid / CV selection evidence

- `tests/test_expanded_cv_selection_v0310.py` contains five regression tests covering deterministic stratified fold assignment, 680 unique candidates, 3,400 held-out fold evaluations, additive evidence artifacts, label-independent threshold derivation, no-Final/no-model-rerun guards, and freeze preference for the expanded CV selection when present.
- Existing adaptive-threshold/grid tests now validate 40 unique weights × 17 configured quantiles.
- Legacy v0.30.2 artifacts remain readable; the new reselection command writes to `configuration_selection_v0310/` and does not overwrite the historical ranking or best configuration.

## v0.31.1 formal temporary cleanup

- Cleanup is post-success and does not modify canonical or chunk-level prediction CSVs.
- Orientation resume requires `orientation_chunk_status.json` + hashed `resolved_manifest.csv`; heavyweight preflight workdirs are not part of the resume contract.
- Inference resume requires `chunk_status.json` + hashed `raw_model_predictions.csv`; heavyweight copied images/preprocessed workdirs are not part of the resume contract.
- Native `{gmic,nyu,glam}.csv` and `study_order.csv` are preserved for provenance reconstruction.
- A compact deterministic XAI sample is retained outside the pruned preprocessing tree.
- Existing Configuration cleanup defaults to dry-run and validates all SUCCESS caches before `--apply` can prune them.
