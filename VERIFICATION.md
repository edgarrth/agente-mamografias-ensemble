# Verification — through v0.30.2

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

## Web unit-inference + MinIO extension verification (2026-08-18)

- Full source-tree pytest suite in the packaging environment after the UI/metadata revision: **136 passed**.
- Web-specific coverage includes four-view inspection, manual projection recovery, conservative metadata fallbacks, cached presentation previews, preview PNG rendering, label-free inference, non-blocking MinIO persistence failure, MinIO object layout, and API request contract without ground-truth/training fields.
- `docker-compose.yml` parses successfully as YAML and FastAPI retains the MinIO Web environment variables.
- FastAPI Web routes import successfully: `/single-cases/inspect`, `/single-cases/previews`, `/single-cases/run`, `/single-cases/storage-status`.
- Critical validated batch files were SHA-256 compared against the original uploaded v0.30.2 resumable batch package and remain byte-identical: `mammography_agent/pipeline.py`, `mammography_agent/datasets/adapters.py`, `mammography_agent/datasets/rsna.py`, `mammography_agent/orientation_policy.py`, `mammography_agent/ensemble/soft_voting.py`, `experiments/run.py`, `experiments/final_evaluation.py`, `tests_flow/normal.py`, `config/experiments.yaml`, `config/ensemble.yaml`, and `config/models.yaml`.
- The packaging sandbox does not expose Docker Engine/GPU and does not contain every declared runtime package. Repository tests were therefore executed with the same minimal external LangGraph test stub used for prior package verification. A live GMIC+NYU+GLAM + MinIO GPU/container end-to-end run remains environment-dependent and was not available in this sandbox.

## v0.31.0 Web ensemble configuration and runtime preflight

Validation date: 2026-08-18.

Scope of this revision:

- Add per-case Web ensemble weights for GMIC, NYU/DMV-CNN and GLAM.
- Keep `config/ensemble.yaml` and `config/experiments.yaml` read-only from the Web route.
- Keep the Web threshold sourced from `config/ensemble.yaml -> baseline.threshold`.
- Record effective Web weights, source and elapsed time with the inference result.
- Present model-level execution times when Model Runner returns resource metrics.
- Report elapsed time when a Web evaluation terminates with an error.
- Treat a configured GPU model as unavailable until its GPU image exists and `gpu_probe` has passed.
- Add explicit CC/MLO descriptions to the visual projection-resolution control.
- Align release metadata to `0.31.0` while retaining the validated v0.30.2 formal batch protocol implementation unchanged.

Automated validation executed from the project root with the local LangGraph test stub used because LangGraph is not installed in this sandbox runtime:

```bash
PYTHONPATH=/mnt/data/test_stubs:. pytest -q
```

Result:

```text
145 passed in 30.83s
```

Focused Web/GPU/soft-voting validation:

```text
25 passed in 0.87s
```

The following batch-critical files were compared byte-for-byte with the original v0.30.2 source used for this Web work and remain identical:

- `mammography_agent/pipeline.py`
- `mammography_agent/datasets/rsna.py`
- `mammography_agent/datasets/adapters.py`
- `mammography_agent/orientation_policy.py`
- `mammography_agent/ensemble/soft_voting.py`
- `experiments/run.py`
- `experiments/final_evaluation.py`
- `tests_flow/normal.py`
- `config/experiments.yaml`
- `config/ensemble.yaml`
- `config/models.yaml`

The screenshot-reported `GPU_PROBE_REQUIRED` condition is an infrastructure precondition, not a DICOM or soft-voting error. The prescribed workstation validation remains:

```bash
docker compose exec fastapi python -m model_tools.validate_gpu --models all
```

A real Docker + NVIDIA GPU end-to-end inference was not executed in this sandbox because Docker Engine/GPU devices are not exposed here. The project retains its fail-safe behavior: GPU inference is refused until the configured runtime has passed its probe.


## v0.32.0 Web device isolation and configuration tab

Validation date: 2026-08-18.

Scope:

- Move Web ensemble weights out of the operational evaluation tab into **Configuración y estado**.
- Add per-request `inference_device=cpu|gpu`, defaulting to CPU from `WEB_INFERENCE_DEVICE`.
- Do not require GPU probe for Web CPU evaluations.
- Preserve GPU preflight only when Web GPU is selected.
- Record the effective Web device in result JSON and `web_inference_runs`.
- Keep batch device configuration independent.

Full source-tree validation with the local LangGraph stub:

```bash
PYTHONPATH=/mnt/data/test_stubs:. pytest -q
```

Result before packaging:

```text
153 passed in 29.43s
```

Batch-isolation verification:

- `experiments/run.py`, `experiments/final_evaluation.py`, `tests_flow/normal.py`, RSNA adapters, orientation policy, soft voting and all batch YAML files remain byte-identical to the v0.31.0 input package.
- `mammography_agent/pipeline.py` has one backward-compatible shared change: `_infer_three(..., device=None)`. If `device is None`, it invokes `run_model()` with the exact historical argument list. Existing batch call sites do not pass a device override.
- A dedicated test asserts that `model_client.run_model(..., device=None)` does not serialize a `device` field, while explicit Web CPU/GPU selection does.

The sandbox still does not expose Docker Engine/NVIDIA GPU, therefore a live model inference cannot be executed here. CPU/GPU routing, API contracts, persistence contracts and the full repository regression suite are covered by automated tests.

## v0.32.1 Web live progress, evidence presentation and CPU label-contract compatibility

Validation date: 2026-08-18.

Scope:

- Hide the Streamlit deployment toolbar from the research application UI.
- Replace the raw MinIO sidebar status with a value-oriented evidence-traceability status and explain PostgreSQL/MinIO roles under **Configuración y estado**.
- Generate a client-visible `run_id`, scroll to the progress region when evaluation starts, and poll `/single-cases/progress/{run_id}` while inference runs.
- Report preparation, orientation, GMIC, NYU/DMV-CNN, GLAM, ensemble integration and evidence persistence as live states, including model elapsed time when the Model Runner returns it.
- Fix Web CPU execution against historical GMIC/GLAM runners that require optional benign-label keys only while writing their output CSV. The Web-only compatibility layer adds `left_benign/right_benign = NaN`; it does not add diagnostic ground truth and does not change the canonical batch pickle contract.
- Keep MinIO non-blocking with respect to the mathematical inference result.

Full source-tree validation with the local LangGraph stub:

```bash
PYTHONPATH=/tmp/langgraph_stub:. pytest -q
```

Result:

```text
158 passed in 29.28s
```

Batch-isolation checks:

- `mammography_agent/metarepo_format.py`, `experiments/run.py`, `experiments/final_evaluation.py`, `tests_flow/normal.py`, `config/experiments.yaml`, `config/ensemble.yaml` and `config/models.yaml` remain byte-identical to the v0.32.0 input package.
- `mammography_agent/pipeline.py` changes only through optional Web parameters/helpers. Formal/batch call sites still invoke `_infer_three(...)` with the historical defaults.
- Dedicated tests confirm that the canonical batch `data.pkl` does not contain `left_benign/right_benign`, while the opt-in Web compatibility transform adds only NaN values for those optional metadata keys.

Docker Engine and an NVIDIA device are not exposed in this packaging sandbox, therefore the exact real GMIC CPU run that produced the reported workstation traceback cannot be re-executed here. The failing contract is reproduced and guarded by structural/unit tests, while the complete repository regression suite passes.



## v0.32.2 Web wall-clock timing, duplicate-run guard and MinIO navigation

- Web-visible model duration is measured around the complete `run_model(...)` request with `time.monotonic()`; runtime resource metrics remain diagnostic-only.
- Progress records independent durations for study preparation, orientation, model-input preparation, GMIC, NYU/DMV-CNN, GLAM, ensemble integration and result persistence.
- Streamlit disables the evaluation action while `web_eval_running=true` and preserves a single client-visible run request.
- `Resultados por modelo`, `Tiempos de ejecución`, `Preparación del estudio` and `Normalización de orientación` are collapsed by default.
- The sidebar no longer presents a generic MinIO evidence banner. The final result exposes the MinIO bucket/prefix and an optional console link through `MINIO_CONSOLE_PUBLIC_URL`.
- Batch entrypoints continue calling `_infer_three(...)` without Web callbacks; experiments, formal resume, normal test, voting and dataset configuration files remain unchanged.
- Regression executed in the packaged source environment: `163 passed`.


## v0.33.0 Web persistence isolation

Validation date: 2026-08-18.

Scope:

- Move Web case staging/runtime from the project `workspace/` bind mount to the dedicated Docker volume `web_scratch:/web-scratch`.
- Keep Web progress in bounded in-memory FastAPI state instead of `web_progress.json`.
- Remove each Web run directory and staged DICOMs after the request finishes, on success or failure.
- Persist durable Web results only to PostgreSQL and MinIO.
- Preserve the batch workspace, resume paths, formal experiment entrypoints and configuration files.

Regression with the local LangGraph test stub:

```bash
PYTHONPATH=/tmp/test_stubs:. pytest -q
```

Result:

```text
167 passed in 29.14s
```

Isolation evidence:

- `experiments/run.py`, `experiments/final_evaluation.py`, `tests_flow/normal.py`, RSNA adapters, `ensemble/soft_voting.py`, `config/experiments.yaml`, `config/ensemble.yaml` and `config/models.yaml` are byte-identical to v0.32.2.
- A dedicated Web-storage test places a sentinel under the batch experiment workspace, executes a Web single-case simulation, and verifies that the sentinel is unchanged, `workspace/output/single_cases` is not created, the Web scratch run is removed, staged uploads are removed, and the durable artifact reference is `minio://...`.
- The default `build_batch(...)` output was compared between v0.32.2 and v0.33.0 using identical synthetic inputs: image SHA-256, `study_order.csv` SHA-256 and `data.pkl` SHA-256 are identical.
- Shared helpers (`pipeline.py`, `orientation_policy.py`, `metarepo_format.py`) contain only opt-in Web resolver branches; batch/default calls retain their historical signatures.

The packaging sandbox does not expose Docker Engine/NVIDIA hardware, so a live Docker model inference cannot be executed here.


## v0.34.0 Web threshold override and Docker observability

Validation date: 2026-08-18.

- `decision_threshold` is optional and validated in `[0,1]` by the Web API.
- The override is resolved only inside `single_case.run_single_case`; no YAML or batch environment variable is mutated.
- `threshold_source` is persisted in the Web-only PostgreSQL table and result payload.
- Web Docker stdout now includes run-correlated events for configuration, stage/model timing, scores, ensemble, persistence and scratch cleanup.
- Model Runner emits Web-only runtime preparation and total runner wall-clock diagnostics; the historical batch success payload/log fields remain unchanged.
- `scripts/web-debug-logs.sh <run_id>` extracts FastAPI + Model Runner events for one Web evaluation.

Regression with the local LangGraph stub:

```bash
PYTHONPATH=/tmp/langgraph_stub:. pytest -q
```

Result:

```text
173 passed in 29.56s
```

Batch-isolation evidence:

- `experiments/run.py`, `experiments/final_evaluation.py`, `tests_flow/normal.py`, `mammography_agent/pipeline.py`, RSNA adapters, orientation policy, soft voting, `metarepo_format.py`, `config/experiments.yaml`, `config/ensemble.yaml` and `config/models.yaml` are byte-identical to v0.33.0.
- The threshold override exists only in `WebDicomCaseRequest` / `single_case.run_single_case` and is not serialized into any batch call.
- Model Runner detailed timing is gated by `run_id.startswith("web-")`; non-Web runs retain the historical `MODEL_RUN_SUCCESS` fields and response contract.

Docker Engine/NVIDIA hardware are not exposed in the packaging sandbox, therefore live GMIC/NYU/GLAM execution is not part of this package validation.


## v0.35.0 Web settings persistence

- Added PostgreSQL table `web_evaluation_settings` for the active Web-only configuration.
- Added `GET /single-cases/web-settings` and `PUT /single-cases/web-settings`.
- Streamlit restores the persisted device, weights and decision threshold after reruns/new browser sessions and automatically saves valid changes.
- No batch YAML or batch entrypoint is mutated by this feature.
- Added one-time fallback migration from the latest successful `web_inference_runs` configuration when `web_evaluation_settings` is initially empty.
