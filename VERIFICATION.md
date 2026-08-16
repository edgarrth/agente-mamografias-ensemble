# Verification — through v0.29.0

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
