# Architecture change log

## 2026-08-15 — v0.10: GPU profile owned by each model; per-model device routing

- Removes global `GPU_RUNTIME_PROFILE` from `.env` and Docker Compose.
- Resolves each compatibility profile exclusively from `config/models.yaml`.
- Replaces global `MODEL_DEVICE` with `DEFAULT_MODEL_DEVICE` plus `GMIC_DEVICE`, `NYU_DEVICE` and `GLAM_DEVICE`.
- Removes redundant `ALLOW_LEGACY_GPU`; GPU inference can only use the model-specific validated GPU image and still requires `gpu_probe=GPU_READY`.
- Records the successful real GMIC Blackwell GPU smoke test supplied by the researcher: prediction CSV, 16 XAI artifacts, 86.91 s elapsed, 8.25% sampled average GPU utilization and 2424 MiB peak sampled GPU memory.
- No model architecture, checkpoint, learned weight, ensemble rule or training behavior changes.

**Reason:** a GPU runtime profile is a technical characteristic of a model, while CPU/GPU selection is a deployment decision. The previous global profile/device configuration could incorrectly imply that NYU and GLAM were ready for the GMIC Blackwell runtime.

## 2026-08-15 — v0.9: GPU probe success logging fix

- Fixes a duplicate `model` keyword in the structured log call executed after a successful GPU probe.
- Keeps the runner-managed timeout and unconditional cleanup introduced in v0.8.
- Adds a regression test so this success-path logging bug cannot silently return HTTP 500 again.
- No model source, architecture, weights, checkpoint, PyTorch/CUDA profile or inference semantics change.

**Reason:** the real GMIC Blackwell probe reached the success path but the response was converted into HTTP 500 by `log(..., model=model, **result)` because `result` already contained `model`.

## 2026-08-15 — v0.7: Fedora WSL2 Docker GPU/CDI preflight

- Records successful CPU build + real smoke tests for GMIC, DMV-CNN/NYU and GLAM on the research workstation.
- Adds a host-side GPU doctor that checks WSL, `nvidia-smi`, `nvidia-ctk`, CDI specs and Docker-discovered NVIDIA devices.
- Adds an explicit Fedora Remix / WSL2 NVIDIA Container Toolkit setup helper.
- The helper never installs a Linux NVIDIA display driver; it installs only the container toolkit, configures Docker and generates/refreshes CDI metadata.
- No model code, weights, checkpoints, preprocessing or ensemble behavior changes.

**Reason:** WSL sees the RTX 5060 Ti, but Docker 29 fails `--gpus all` with `failed to discover GPU vendor from CDI: no known GPU vendor found`. This is a host Docker/NVIDIA integration issue that must be resolved before testing the legacy PyTorch/CUDA stacks on GPU.

## 2026-08-15 — v0.6: NVIDIA legacy APT signing-key rotation compatibility

- Keeps the v0.5 CUDA 10.1 / Ubuntu 18.04 base-image compatibility layer.
- Detects whether the upstream Dockerfile already contains the NVIDIA CUDA signing-key refresh.
- Preserves the upstream workaround when present (DMV-CNN/NYU).
- Injects the same narrow signing-key refresh before `apt-get update` when absent (GMIC/GLAM).
- Logs `NVIDIA_APT_KEY_ROTATION_COMPATIBILITY_APPLIED` and records the status in compatibility metadata.
- Does not change model source, checkpoints, weights, training or prediction logic.

**Reason:** the real GMIC build progressed beyond the v0.5 base-image fix and failed at `apt-get update` with `NO_PUBKEY A4B469963BF863CC`, a historical NVIDIA repository key-rotation issue.

## 2026-08-15 — v0.5: auditable legacy CUDA base-image compatibility

- Keeps one lightweight `model-runner` + three isolated model images.
- Automatically trusts only the host-mounted NYU metarepository path with Git `safe.directory` when required.
- Adds a narrow compatibility layer for GMIC, DMV-CNN/NYU and GLAM that changes only the historical Docker `FROM` image from `nvidia/cuda:10.1-base-ubuntu18.04` to `nvidia/cudagl:10.1-devel-ubuntu18.04`.
- Generates the patched Dockerfile at runtime from the exact upstream Dockerfile and records original/generated SHA-256 hashes.
- Refuses the patch if the upstream first line changed, preventing silent drift.
- CLI commands now surface the Model Runner HTTP error detail instead of a generic `500 Server Error`.
- No model source commit, checkpoint, training, inference logic or ensemble logic is changed.

**Reason:** real workstation execution reached the upstream GMIC Docker build and proved that the historical `nvidia/cuda:10.1-base-ubuntu18.04` tag could no longer be resolved, while an NVIDIA CUDA 10.1 / Ubuntu 18.04 `cudagl` image was available.

## 2026-08-15 — v0.4: robust Docker Desktop/WSL2 Model Runner boundary

- Architecture remains one lightweight `model-runner` + three isolated model images.
- Replaced Debian `docker.io` in the runner with Docker's official `docker:29-cli` image.
- Added explicit `DOCKER_HOST=unix:///var/run/docker.sock`.
- Added `/doctor` endpoint with socket ping, Docker CLI version and daemon info diagnostics.
- Updated Compose socket mount to explicit bind syntax.
- Added host-side `scripts/doctor.sh`.
- No ML framework, checkpoint or inference algorithm was changed.

**Reason:** v0.3 could remain unhealthy on a recent Docker Desktop because the Debian-packaged Docker CLI could be older than the daemon's minimum API. The change is infrastructure-only and preserves the model isolation and thesis methodology.

## 2026-08-15 — v0.3: single Model Runner + three isolated model images

### Change
The three persistent controller services from v0.2 were consolidated into one persistent service:

- `model-runner` / `mammography-model-runner`

The actual model environments remain separate Docker images:

- `mammography-model-gmic:research`
- `mammography-model-nyu:research`
- `mammography-model-glam:research`

### Why
Dependency isolation is required between the three legacy models, but that isolation already exists in their separate model images. Maintaining three additional FastAPI controller containers duplicated routing, health and Docker control logic without adding meaningful isolation. A single runner is simpler and better suited to a master’s thesis prototype.

### Model Runner responsibility
The runner performs only technical orchestration:

- routes the requested model;
- clones/verifies the NYU metarepository;
- builds/reuses the selected model image;
- creates a temporary inference container;
- assigns GPU access when enabled;
- serializes GPU inference with a shared lock;
- captures resource metrics and logs;
- removes the temporary inference container when finished.

The runner intentionally does **not** contain PyTorch, TensorFlow, CUDA Toolkit, cuDNN or model checkpoints.

### Model responsibility
Each model image contains the upstream environment required by that model, including its own Python/framework/runtime dependencies and model artifacts as defined by the upstream research repository.

### Security boundary
Only `model-runner` receives `/var/run/docker.sock`. FastAPI, Streamlit and bootstrap do not. The temporary model containers also do not receive the Docker socket.

### GPU policy
Only one GPU inference executes at a time by default. The runner starts the selected child container with `--gpus device=<GPU_NUMBER>` and releases the lock after that inference finishes.

---

## 2026-08-15 — v0.2: per-model controller services (superseded)

v0.2 introduced `gmic-runtime`, `nyu-runtime` and `glam-runtime` as three persistent controller services. This made each model boundary visible but duplicated the controller layer. v0.3 supersedes that topology while preserving the three isolated model images.


## v0.8 — Runtime GPU Blackwell separado para GMIC

- Runtime legacy `:research` se conserva para CPU.
- Runtime `:blackwell-cu128` usa PyTorch 2.7.1/CUDA 12.8.
- Nuevos `ensure-gpu` y `gpu-probe` con limpieza fail-safe.
- Inferencia GPU nunca reutiliza silenciosamente la imagen legacy.

## 2026-08-15 — v0.11: runtime Blackwell específico para DMV-CNN / NYU

- Se agrega `mammography-model-nyu:blackwell-cu128` como imagen GPU independiente.
- El perfil `blackwell-cu128` permanece como metadato del modelo en `config/models.yaml`; no vuelve a `.env`.
- Se preserva el commit NYU `de2b0855d02984df0f516008bb4513ff71460e21` y los checkpoints upstream.
- Se modernizan únicamente dependencias de ejecución a Python 3.10, PyTorch 2.7.1, TorchVision 0.22.1 y CUDA 12.8.
- Los parches de compatibilidad se declaran por modelo en `compatibility_code_patches`, eliminando el texto hardcodeado específico de GMIC en la auditoría genérica del runner.
- NYU continúa en CPU por defecto hasta completar `ensure_gpu`, `gpu_probe` y smoke test real en la workstation.

## 2026-08-15 — v0.12: runtime Blackwell específico para GLAM

- Se agrega `mammography-model-glam:blackwell-cu128` como imagen GPU independiente.
- El perfil sigue perteneciendo a GLAM en `config/models.yaml`; `.env` solo selecciona `GLAM_DEVICE=cpu|gpu`.
- Se conserva el commit GLAM `17a0019860441e2ea8d7b7c7e0aaeada735e871f`, los checkpoints y la arquitectura.
- Se moderniza únicamente el runtime a Python 3.10, PyTorch 2.7.1, TorchVision 0.22.1 y CUDA 12.8.
- Los parches declarados cubren API cuDNN, backend gráfico headless, colocación de tensores y preservación de semánticas históricas de índice/sampling.
- GLAM permanece en CPU por defecto hasta completar `ensure_gpu`, `gpu_probe` y smoke test real en la workstation.


## 2026-08-15 — v0.13: adapter oficial TCIA para CBIS-DDSM

- Reemplaza el `source_manifest.csv` manual de CBIS-DDSM por un adapter dataset-specific.
- Descubre recursivamente el árbol DICOM de NBIA y los cuatro CSV oficiales de clasificación.
- Usa exclusivamente `pathology` como ground truth de malignidad; BI-RADS/assessment no se convierte a etiqueta.
- Resuelve imágenes por sufijo de ruta/UID y usa un índice cacheado de cabeceras DICOM como fallback.
- Genera automáticamente `source_manifest.csv`, catálogo de metadata, catálogo de vistas y reportes de filas/estudios rechazados.
- Introduce una compuerta explícita de cuatro vistas para el ensemble actual: L-CC, R-CC, L-MLO y R-MLO. No duplica ni sintetiza vistas faltantes.
- Agrega `python -m dataset_pipeline.inspect --datasets cbis_ddsm` y `POST /datasets/inspect`.
- La escritura PNG de 16 bits ahora entrega filas al encoder de forma incremental para evitar materializar toda la mamografía como listas Python; no cambia los valores de píxel.
- No cambia GMIC, DMV-CNN/NYU, GLAM, checkpoints, pesos, perfiles Blackwell ni fórmula de soft voting.

## 2026-08-15 — v0.14: guardas de metadata CBIS-DDSM y configuración validada

- `.env.example` pasa a representar la configuración de tres modelos en GPU que ya superó `gpu_probe` y smoke tests en la RTX 5060 Ti.
- El `inspect` de CBIS-DDSM retorna `METADATA_REQUIRED` de forma estructurada si faltan tablas de clasificación; no inicia el índice DICOM ni produce traceback.
- `prepare` conserva la misma guarda y no intenta convertir imágenes sin metadata oficial.
- Se aceptan aliases de nombres `Mass/Calc-Training/Test-Description.csv` sin alterar el contenido de las tablas ni la fuente de ground truth.
- No se modifica ningún modelo, peso, checkpoint, arquitectura ni regla de ensemble.

## 2026-08-15 — v0.15: descarga CBIS-DDSM no destructiva, cache reutilizable y metadata.csv auxiliar

- `dataset_pipeline.download` nunca descarga ni re-descarga DICOM CBIS-DDSM; la adquisición de imágenes continúa en TCIA/NBIA.
- Se automatiza únicamente la adquisición de los cuatro CSV oficiales de clasificación, con validación SHA-256 y columnas requeridas.
- Copias válidas existentes se reutilizan; la operación es idempotente respecto del dataset DICOM.
- `metadata.csv` pasa a ser una fuente auxiliar opcional de identidad SeriesInstanceUID/StudyInstanceUID/PatientID; no participa en ground truth.
- El índice DICOM existente se reutiliza por defecto y puede enriquecerse sin abrir nuevamente los DICOM. `--force-dicom-index` hace explícita una reconstrucción completa.
- `inspect` agrega inventario de objetos DICOM y conteos de estudios/vistas seleccionadas para evitar mezclar imágenes, anomalías, pacientes y unidades de evaluación.
- `.env.example` conserva el perfil GPU realmente validado para los tres modelos.

## 2026-08-15 — v0.16: compatibilidad de índices GMIC, health state logs y catálogo operativo

- GMIC Blackwell preserva semántica PyTorch 1.1 de cociente/remainder en `get_max_window` mediante `torch.div(..., rounding_mode="floor")`.
- `build_revision=2` reconstruye solo GMIC y obliga a renovar su `gpu_probe`; NYU/GLAM no cambian.
- Access logs de `/health` pasan a ser por transición: primer estado, fallo y recuperación; se suprimen probes 200 repetidos.
- `VERSION` se expone dentro de FastAPI y Model Runner y corrige el `0.10.0` stale de `/doctor`/`/health`.
- README centraliza comandos Docker y documenta efectos destructivos/no destructivos, en particular `dataset_pipeline.prepare`.

## 2026-08-15 — v0.17: validación GPU integrada y parametrizada

- Se agrega `model_tools.validate_gpu` para uno, varios o todos los modelos.
- La secuencia de validación es determinística: primero `ensure_gpu` de todos los seleccionados, después `gpu_probe`, finalmente smoke tests.
- La operación continúa con los demás modelos si uno falla, salvo `--fail-fast`, y deja un reporte JSON en `workspace/output/model_validation/`.
- `--force-rebuild` permite reconstruir explícitamente imágenes GPU aunque el `build_revision` ya coincida; por defecto no se reconstruyen imágenes sin cambios.
- El smoke integrado exige `device=gpu` para evitar falsos positivos de validación sobre CPU.
- No cambia arquitectura, pesos, checkpoints, datasets, fórmula de ensemble ni perfiles CUDA de GMIC/NYU/GLAM.


## 2026-08-15 — v0.18: compatibilidad del contrato de etiquetas GMIC

- La integración CBIS-DDSM confirmó que el fix de índices v0.16 permitió completar el `forward()` de GMIC.
- Se detectó después del forward un `KeyError: left_benign` al generar metadata del CSV de salida.
- El batch canónico conserva exclusivamente las etiquetas malignas breast-level exigidas por el metarepository; v0.18 no inventa etiquetas benignas.
- El runner GMIC Blackwell emite `NaN` en `benign_label` cuando esa etiqueta opcional no está disponible, preservando los scores benign/malignant producidos por el modelo.
- `GMIC build_revision=3`; NYU/GLAM permanecen en revisión 1.

## 2026-08-15 — v0.19: contrato de estado de inferencia Model Runner

- Corrige una colisión de claves en `/models/{model}/run`: la metadata `status=READY` de `ensure_gpu_image()` sobrescribía accidentalmente `status=SUCCESS` después de una inferencia real completada.
- `READY` queda reservado para disponibilidad de imagen/runtime; `GPU_READY` para probe CUDA; `SUCCESS` para una inferencia que produjo salida.
- El pipeline conserva la guarda estricta `status == SUCCESS`; no acepta `READY` como éxito de inferencia.
- La evidencia real de v0.18 mostró que GMIC terminó 5 estudios CBIS-DDSM, produjo `gmic.csv` y 20 XAI antes del falso fallo de orquestación.
- No cambia modelos, pesos, checkpoints, build revisions, datasets, ground truth, XAI ni Soft Voting.


## 2026-08-15 — v0.20: GLAM dataset contract, artifact isolation and run logging

- GLAM Blackwell acepta el contrato malignancy-only sin fabricar etiquetas benignas; `build_revision=2`.
- Se separa `preprocessed/<model>` para evitar atribución cruzada de XAI.
- Se elimina el acoplamiento por orden entre outputs image-level y estudios: mapping explícito `study_key` con validación one-to-one.
- Chunk mode agrega XAI y resource metrics al root de la corrida.
- Model Runner emite eventos de ciclo de inferencia a stdout; healthchecks repetidos continúan suprimidos.
- FastAPI/Model Runner migran startup a `lifespan`, eliminando warnings de `on_event` deprecado.

## 2026-08-15 — v0.21: reproducible class-aware normal-test sampling and clearer evidence

- Se añade sampling normal parametrizable y determinístico: `sequential`, `random`, `stratified` proporcional y `balanced`.
- Cada normal test conserva `selected_studies.csv` y metadata de sampling en configuración/resumen.
- `samples` de resource monitoring se renombra a `monitoring_samples` para no confundirlo con casos del dataset.
- Métricas no calculables incluyen motivo explícito.
- Se añade `run_summary.json` con estudios/imágenes procesados y tiempo end-to-end.
- No cambia ningún modelo, peso, checkpoint, runtime Blackwell, dataset preparado, ground truth ni fórmula de Soft Voting.

## 2026-08-15 — v0.22: cached score analysis, adaptive thresholds and final-set isolation

- Se agrega análisis CPU de `raw_model_predictions.csv` sin reinferencia GPU: distribución por clase, ROC-AUC por modelo, correlaciones, puntos ROC y advertencias de direccionalidad.
- La grilla fija 0.40/0.45/0.50/0.55/0.60 deja de utilizarse en el experimento porque quedó fuera de la escala observada en la primera corrida balanceada real.
- Se mantienen exactamente 80 configuraciones: 16 pesos × 5 thresholds, derivados como quantiles 10/30/50/70/90 de los scores del Configuration Set para cada combinación de pesos.
- La derivación de thresholds es independiente de `ground_truth`; ninguna inversión de score, calibración, entrenamiento o ajuste de modelo se realiza automáticamente.
- El Final Test Set no se infiere antes de congelar configuración. La inferencia final queda cacheada y se reutiliza en reejecuciones del mismo experimento.
- No cambian modelos, pesos preentrenados, runtimes Blackwell, dataset preparado ni regla de Soft Voting.


## 2026-08-15 — v0.23: balanced operating-point selection and richer metrics

- Se añaden Specificity, PPV, NPV, FPR, Accuracy y Balanced Accuracy a la evaluación threshold-dependent.
- La selección experimental deja de priorizar `FN=0` a cualquier costo: ROC-AUC selecciona pesos y Balanced Accuracy selecciona threshold; Sensitivity/Specificity/FP resuelven empates.
- Los thresholds de un análisis diagnóstico se etiquetan `analysis_score_quantile`; `configuration_score_quantile` queda reservado al Configuration Set formal.
- `score_summary.json` y el reporte diagnóstico exponen métricas del baseline threshold.
- Se preserva el aislamiento Configuration Set → freeze → Final Test Set y el cache de inferencia final.
- No se modifican modelos, checkpoints, runtimes GPU, dataset preparado, XAI ni fórmula de Soft Voting.

## v0.24 — score provenance audit

- Se agrega auditoría CPU de outputs nativos GMIC/NYU/GLAM antes de la evaluación formal.
- Reconstruye vista → mama → estudio y valida que el score persistido coincida con la agregación actual `max`.
- Compara ROC-AUC a nivel mama y estudio y registra alineamiento de la mama de score máximo con la lateralidad maligna.
- No modifica inferencia, modelos, pesos, thresholds, calibración ni reglas de agregación.


## v0.24.2 — compatibilidad de auditoría con runs chunked

- `score_provenance` deja de asumir `<run>/model_batch/` como única ubicación de outputs nativos.
- Descubre `chunks/<NNNN>/model_batch/`, combina múltiples chunks y conserva la procedencia exacta de cada CSV.
- Cuando falta `study_order.csv`, reconstruye el orden desde el `raw_model_predictions.csv` local al chunk para preservar el contrato posicional de NYU.
- Valida cobertura total, solapamientos y duplicados antes de calcular métricas.
- No cambia modelos, scores, ensemble, threshold, agregación ni dataset.


## v0.25.0 — input fidelity before full experimental inference

- Adds a CPU-only breast-aware aggregation diagnostic; it never changes the production aggregation contract.
- Adds an input-fidelity audit over prepared PNG headers, original DICOM presentation metadata when available, and model-native preprocessing metadata.
- Keeps all 10-study diagnostic outputs explicitly ineligible for configuration freeze.
- Synchronizes package/API version metadata to 0.25.0.

## v0.26.0 — targeted orientation counterfactual
- Adds a diagnostic-only horizontal orientation counterfactual for studies where all four unique views report `distance_from_starting_side != 0` in upstream preprocessing.
- The original run and prepared dataset remain immutable; only suspect studies are copied into a diagnostic batch with `horizontal_flip` toggled.
- GMIC/NYU/GLAM are rerun only for those studies to compare preprocessing geometry and score impact.
- Orientation decisions must use upstream geometric evidence first. Any AUC change is recorded as secondary/post-hoc evidence and is not eligible for freeze.


## v0.26.1 — bootstrap configuration-addition schema fix

Corrige la entrada de `orientation_counterfactual_diagnostic` para registrar `id: ADD-081`. Añade validación explícita de campos requeridos antes de escribir `configuration_additions.log`, evitando `KeyError` opacos durante bootstrap y cubriendo el contrato con tests de regresión. No cambia modelos, scores, pesos, threshold, dataset ni lógica del contrafactual v0.26.

## 2026-08-16 — v0.29.1: generic multi-dataset input-scale comparison

- `input_scale_comparison` detects a single `dataset_source` from the selected run instead of hardcoding CBIS-DDSM.
- Preserves the classifier-free comparison against the official NYU sample before and after upstream crop/optimal-center preprocessing.
- Does not use labels/model scores and does not modify normalization, datasets, model weights, ensemble weights or thresholds.

## 2026-08-16 — v0.29.2: DICOM presentation counterfactual

- Adds a label-blind, classifier-free comparison of the existing adapter conversion, DICOM Modality LUT/rescale and VOI/window presentation transforms on already-inspected studies.
- Raw and prepared dataset bytes remain immutable; transformed image copies are optional and are written only under analysis output.
- No branch may be selected by AUC or model scores. The result is evidence about DICOM presentation semantics, not a configuration-selection experiment.
- Aligns release-version metadata across the package and Model Runner API at `0.29.2`.

## 2026-08-17 — v0.30.0: RSNA dataset-aware adapter

- Adds `RSNADatasetAdapter` for the manually acquired RSNA Breast Cancer Detection train distribution.
- Requires the four standard views but retains studies with repeated CC/MLO images.
- Resolves repeated canonical views using deterministic label-blind SHA-256 selection and writes explicit selected/unselected audit manifests.
- Uses only `train.csv:cancer` for breast/study ground truth and rejects within-breast label conflicts.
- Adds reproducible JPEG Lossless/JPEG 2000 pydicom plugins validated against the audited RSNA runtime sample.
- Does not download RSNA automatically and never modifies raw DICOM/CSV inputs.
