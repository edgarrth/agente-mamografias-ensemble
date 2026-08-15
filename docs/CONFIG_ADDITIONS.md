# Configuration additions

Toda configuración agregada durante la implementación se registra aquí y en `config/config_additions.yaml`. Al arrancar el proyecto, el bootstrap copia estas decisiones a `workspace/logs/configuration_additions.log` con su razón y estado.

## ADD-001 — MODEL_DEVICE — **superseded by ADD-036**

**Valor:** `cpu|gpu; default=cpu`

**Por qué:** The upstream model images use legacy CUDA/PyTorch stacks. CPU is the safe first-run mode until the target GPU passes a real smoke test.

## ADD-002 — ALLOW_LEGACY_GPU — **superseded by ADD-037**

**Valor:** `false by default`

**Por qué:** Prevents accidental use of an unverified legacy CUDA stack on a recent GPU.

## ADD-003 — MODEL_BOOTSTRAP_MODE — **active**

**Valor:** `lazy|eager; default=lazy`

**Por qué:** Building the three upstream model images can take time; a thesis prototype should start quickly and build the required model image on first use.

## ADD-004 — study_level_aggregation — **active**

**Valor:** `GMIC=max image; NYU=max breast; GLAM=max image`

**Por qué:** The three official outputs have different native granularities but soft voting requires one deterministic score per study.

## ADD-005 — explicit_dataset_download — **active**

**Valor:** `dataset selection is mandatory`

**Por qué:** Datasets are large and may require license/credential steps. Docker startup must never unexpectedly download hundreds of GB.

## ADD-006 — max_runtime_minutes — **active**

**Valor:** `optional; development only`

**Por qué:** Pilot tests need a predictable time budget. The final thesis test set must never be truncated by this option.

## ADD-007 — docker_socket_isolated_to_model_runner — **active**

**Valor:** `/var/run/docker.sock only in model-runner`

**Por qué:** Only the lightweight model-runner needs Docker daemon access to build and start the isolated upstream model images. FastAPI, Streamlit, bootstrap and model images do not receive the Docker socket.

## ADD-008 — no_automatic_prediction_fallback — **active**

**Valor:** `fail explicitly`

**Por qué:** The thesis forbids mocks and simulated probabilities.

## ADD-009 — source_manifest_boundary — **active**

**Valor:** `source_manifest.csv required for raw datasets`

**Por qué:** Native public datasets use different label schemas. Requiring an explicit mapping prevents the prototype from inventing or incorrectly deriving cancer ground truth.

## ADD-010 — xai_official_flag — **active**

**Valor:** `enable official visualization option for GMIC/GLAM when supported`

**Por qué:** The architecture requires real XAI. The model-runner only requests visualization already implemented by the upstream GMIC/GLAM image; it never synthesizes a heatmap.

## ADD-011 — minio_image_tag — **active**

**Valor:** `minio/minio:latest during prototype development`

**Por qué:** A verified immutable MinIO tag was not part of the research specification. Before definitive measurements, the resolved image digest must be recorded/pinned.

## ADD-012 — dicom_conversion_mode — **active**

**Valor:** `preserve native ordering; MONOCHROME1 inversion; BitsStored -> 16-bit left shift`

**Por qué:** The official metarepository expects 16-bit PNGs. A deterministic conversion is needed, but arbitrary learned/image-enhancement transformations are avoided. Dataset-specific VOI LUT needs must be validated separately before definitive measurements.

## ADD-013 — RESOURCE_SAMPLE_SECONDS — **active**

**Valor:** `2 seconds by default`

**Por qué:** CPU/RAM/GPU resource metrics need periodic sampling while each real model container is executing. A small configurable interval keeps the prototype simple without a monitoring platform.

## ADD-014 — explicit_per_model_runtime_services — **superseded** por `ADD-017`

**Valor:** `gmic-runtime:8011 | nyu-runtime:8012 | glam-runtime:8013`

**Por qué:** This v0.2 decision exposed three controller services. It is superseded because model dependency isolation already occurs in the three upstream model images; three extra controllers were redundant for a thesis prototype.

## ADD-015 — single_runner_gpu_semaphore — **active**

**Valor:** `one GPU inference at a time by default, enforced by model-runner`

**Por qué:** The workstation has one GPU and 16 GB VRAM. The model-runner serializes GPU inference so GMIC, NYU and GLAM do not compete for GPU memory.

## ADD-016 — model_runtime_urls — **superseded** por `ADD-017`

**Valor:** `GMIC_RUNTIME_URL, NYU_RUNTIME_URL, GLAM_RUNTIME_URL`

**Por qué:** Per-model runtime URLs are no longer needed because FastAPI communicates with one model-runner endpoint and identifies the target model in the API path.

## ADD-017 — single_model_runner — **active**

**Valor:** `model-runner:8010`

**Por qué:** A single lightweight controller is sufficient to orchestrate all three isolated model images. This removes redundant controller containers while preserving separate model dependencies and checkpoints.

## ADD-018 — model_image_names — **active**

**Valor:** `mammography-model-gmic:research | mammography-model-nyu:research | mammography-model-glam:research`

**Por qué:** Explicit local image names make the distinction between the controller and the actual model environments clear in Docker image listings and thesis documentation.

## ADD-019 — model_runner_dependency_boundary — **active**

**Valor:** `no PyTorch, TensorFlow, CUDA Toolkit or cuDNN in model-runner`

**Por qué:** The runner schedules Docker jobs and controls GPU access but does not perform neural-network computation. ML/CUDA dependencies remain isolated inside each model image, avoiding incompatible framework stacks in the controller.

## ADD-020 — model_runner_api_contract — **active**

**Valor:** `/models, /models/{model}/info, /ensure, /smoke-test, /run`

**Por qué:** One uniform API keeps the application simple while retaining explicit model selection, health/status, provisioning and inference operations.

## ADD-021 — model_runner_official_docker_cli — **active**

**Valor:** `docker:29-cli` por defecto; configurable mediante `DOCKER_CLI_IMAGE`.

**Por qué:** la versión anterior instalaba `docker.io` desde Debian dentro del Model Runner. Ese paquete puede quedar detrás de la API mínima aceptada por Docker Desktop/Engine. El Runner ahora parte de la imagen oficial de Docker CLI y sigue sin contener frameworks ML.

## ADD-022 — model_runner_docker_host — **active**

**Valor:** `unix:///var/run/docker.sock`.

**Por qué:** el Model Runner necesita acceder al Docker Engine del host para construir y ejecutar las imágenes aisladas de GMIC, NYU y GLAM. El socket continúa expuesto únicamente a este servicio y `DOCKER_HOST` queda explícito para hacer el comportamiento reproducible y diagnosticable en WSL2/Docker Desktop.

## ADD-023 — model_runner_doctor_endpoint — **active**

**Valor:** `GET /doctor`.

**Por qué:** cuando Docker no es accesible, `/doctor` devuelve diagnóstico del socket, ping directo, `docker version` y `docker info`. Así se diferencia un problema de montaje/permisos de uno de compatibilidad CLI/API sin ocultar la causa.


## ADD-024 — git_safe_directory_for_host_workspace — **active**

**Valor:** `/workspace/runtime/mammography_metarepository`.

**Por qué:** los bind mounts WSL2/Docker pueden presentar ownership distinto dentro del Model Runner. Git rechaza por seguridad repositorios con ownership inesperado. El Runner agrega únicamente la ruta exacta del metarepositorio cuando hace falta y registra `GIT_SAFE_DIRECTORY_ADDED`.

## ADD-025 — legacy_cuda_base_image_compatibility — **active**

**Valor:** `nvidia/cudagl:10.1-devel-ubuntu18.04`.

**Por qué:** los Dockerfiles actuales del metarepositorio para GMIC, DMV-CNN/NYU y GLAM referencian `nvidia/cuda:10.1-base-ubuntu18.04`, que no se pudo resolver durante la construcción real. Se genera un Dockerfile auditable que cambia solo la línea `FROM`; no cambia código, pesos, checkpoints ni entrenamiento.

## ADD-026 — compatibility_patch_drift_guard — **active**

**Valor:** se exige coincidencia exacta de la línea `FROM` histórica antes de aplicar el reemplazo.

**Por qué:** si el upstream cambia, el prototipo debe fallar y solicitar revisión en vez de aplicar silenciosamente una corrección diseñada para otra versión.

## ADD-027 — model_runner_error_detail_cli — **active**

**Valor:** los comandos `ensure` y `smoke-test` muestran el `detail` HTTP devuelto por el Model Runner.

**Por qué:** evita ocultar errores reales de build/inferencia detrás de un traceback genérico `500 Server Error`, mejorando trazabilidad y depuración del prototipo.

## ADD-028 — legacy_nvidia_repository_key_rotation — **active**

**Valor:** `auto`: si el Dockerfile upstream ya contiene el refresh `3bf863cc.pub`, se conserva; si no, el Runner inserta el refresh de claves NVIDIA antes del primer `apt-get update`.

**Por qué:** la construcción real de GMIC con la base CUDA 10.1 compatible avanzó hasta `apt-get update`, donde el repositorio CUDA de Ubuntu 18.04 falló con `NO_PUBKEY A4B469963BF863CC`. NVIDIA rotó las claves de firma de sus repositorios CUDA. El Dockerfile upstream de `nyu_model` ya contiene una corrección equivalente, pero los Dockerfiles históricos de GMIC y GLAM no. El cambio afecta únicamente la construcción del entorno; no modifica código, pesos, checkpoints, entrenamiento ni lógica de inferencia. El evento se registra como `NVIDIA_APT_KEY_ROTATION_COMPATIBILITY_APPLIED`.

## ADD-029 — fedora_wsl_nvidia_container_toolkit_preflight — **active**

**Valor:** configuración explícita de NVIDIA Container Toolkit/CDI en el host Fedora Remix WSL2 antes de habilitar los modelos en GPU.

**Por qué:** la RTX 5060 Ti es visible desde WSL mediante `nvidia-smi`, pero Docker Engine 29 devolvió `failed to discover GPU vendor from CDI: no known GPU vendor found`. Esto demuestra que el bloqueo actual está en la integración Docker↔NVIDIA/CDI del host, no en los tres modelos. Se agregan `scripts/gpu-doctor.sh` y `scripts/setup-nvidia-container-toolkit-fedora-wsl.sh`. El setup es deliberadamente host-only y manual: Compose/bootstrap no instala drivers ni altera el host automáticamente.


## ADD-030 — blackwell_legacy_cuda_guard — **active**

**Valor:** la imagen legacy no se usa para GPU después del bloqueo observado en la primera asignación CUDA.

**Por qué:** PyTorch 1.1.0/CUDA 9.0 detecta la RTX 5060 Ti pero se bloquea en `tensor.cuda()`.

## ADD-031 — gmic_blackwell_runtime_profile — **active**

**Valor:** `mammography-model-gmic:blackwell-cu128`, Python 3.10, PyTorch 2.7.1, torchvision 0.22.1 y CUDA 12.8.

**Por qué:** PyTorch 2.7 introdujo soporte oficial para Blackwell con CUDA 12.8; se conserva el commit y checkpoints de GMIC.

## ADD-032 — gpu_probe_fail_safe — **active**

**Valor:** Model Runner administra timeout y elimina siempre el contenedor de prueba.

**Por qué:** un timeout externo puede dejar contenedores ejecutándose.

## ADD-033 — dual_runtime_model_images — **active**

**Valor:** `:research` para CPU y `:blackwell-cu128` para GPU.

**Por qué:** separa reproducibilidad histórica de compatibilidad con hardware moderno.


## ADD-034 — gpu_probe_success_logging_duplicate_model_fix — **active**

**Valor:** el registro estructurado del resultado exitoso de `gpu_probe` recibe el campo `model` una sola vez.

**Por qué:** el probe real de la RTX 5060 Ti llegó a `GPU_READY`, pero v0.8 devolvió HTTP 500 al duplicar el keyword `model` en el logger. La corrección solo afecta observabilidad y manejo de errores.

## ADD-035 — gpu_runtime_profile_model_metadata — **active**

**Valor:** `gpu_compatibility.profile` se define únicamente en `config/models.yaml` para cada modelo.

**Por qué:** El perfil describe compatibilidad técnica propia del modelo (Python, framework y CUDA). No es una decisión del despliegue y por eso ya no se configura mediante `.env`.

## ADD-036 — per_model_device_selection — **active**

**Valor:** `DEFAULT_MODEL_DEVICE` y overrides `GMIC_DEVICE`, `NYU_DEVICE`, `GLAM_DEVICE`, cada uno `cpu|gpu`.

**Por qué:** La elección CPU/GPU sí es una decisión del despliegue y puede ser distinta por modelo. Permite ejecutar GMIC en su runtime Blackwell ya validado y conservar NYU/GLAM en CPU mientras se validan sus propios perfiles.

## ADD-037 — validated_gpu_image_only — **active**

**Valor:** Toda inferencia GPU resuelve exclusivamente `gpu_compatibility.image` del modelo y exige `gpu_probe=GPU_READY`; se elimina `ALLOW_LEGACY_GPU`.

**Por qué:** La separación `:research` (CPU reproducible) / imagen GPU compatible ya impide usar accidentalmente el runtime legacy en GPU. El flag adicional era redundante.

## ADD-038 — `nyu_blackwell_runtime_profile`

Se incorpora `mammography-model-nyu:blackwell-cu128` como runtime GPU separado para DMV-CNN/NYU. Mantiene el commit upstream y los checkpoints; actualiza únicamente el stack de ejecución a Python 3.10, PyTorch 2.7.1, TorchVision 0.22.1 y CUDA 12.8 para validar la RTX 5060 Ti.

## ADD-039 — `model_owned_gpu_compatibility_patch_audit`

Los parches de compatibilidad GPU se describen ahora dentro del bloque `gpu_compatibility` de cada modelo. Esto evita que la auditoría genérica del Model Runner atribuya a NYU un parche específico de GMIC y conserva trazabilidad precisa por runtime.

## ADD-042 — cbis_ddsm_official_tcia_metadata_adapter — **active**

**Valor:** ingestión recursiva de los cuatro CSV oficiales de CBIS-DDSM más el árbol DICOM descargado con TCIA/NBIA.

**Por qué:** elimina la construcción manual de `source_manifest.csv`. El adapter deriva paciente, lateralidad, vista y patología desde los metadatos oficiales y resuelve las rutas contra los DICOM descargados.

## ADD-043 — cbis_ddsm_pathology_only_ground_truth — **active**

**Valor:** `MALIGNANT=1`, `BENIGN=0`, `BENIGN_WITHOUT_CALLBACK=0`; patologías desconocidas se rechazan; BI-RADS no se convierte a etiqueta de cáncer.

**Por qué:** la clasificación de cáncer debe usar el ground truth patológico del dataset y no una regla heurística basada en assessment/BI-RADS.

## ADD-044 — cbis_ddsm_recursive_raw_layout — **active**

**Valor:** cualquier jerarquía generada por NBIA es válida debajo de `/workspace/datasets/raw/cbis_ddsm`; los CSV oficiales se descubren recursivamente.

**Por qué:** evita aplanar o renombrar manualmente cientos de GB. El adapter usa coincidencia de sufijos/UID y, cuando hace falta, un índice DICOM de cabeceras cacheado.

## ADD-045 — cbis_ddsm_four_view_ensemble_gate — **active**

**Valor:** el ensemble actual exige `L-CC`, `R-CC`, `L-MLO`, `R-MLO`; estudios incompletos se registran y excluyen, sin duplicar ni sintetizar vistas.

**Por qué:** el flujo exam-level del DMV-CNN/NYU seleccionado requiere las cuatro vistas estándar. Completar vistas artificialmente invalidaría la entrada científica.

## ADD-046 — cbis_ddsm_inspection_artifacts — **active**

**Valor:** catálogos de metadata/vistas, reporte de filas no resueltas, reporte de estudios incompletos, `source_manifest.csv` autogenerado e índice DICOM cacheado.

**Por qué:** permite auditar exactamente cómo se transformó el release oficial antes de convertir imágenes o ejecutar inferencia.

## ADD-047 — validated_three_gpu_env_example — **active**

**Valor:** `.env.example` selecciona GMIC, NYU y GLAM en GPU con `ALLOW_GPU=true`, `GPU_NUMBER=0`; el perfil técnico Blackwell sigue exclusivamente en `config/models.yaml`.

**Por qué:** los tres runtimes ya fueron comprobados en la workstation objetivo y el ejemplo debe reflejar el estado realmente probado.

## ADD-048 — cbis_ddsm_metadata_preflight_guard — **active**

**Valor:** `inspect` y `prepare` devuelven `METADATA_REQUIRED` con archivos faltantes e instrucciones; no construyen el índice DICOM mientras la metadata oficial esté incompleta.

**Por qué:** evita tracebacks y trabajo costoso inútil cuando NBIA ha descargado imágenes pero aún faltan los CSV separados de TCIA.

## ADD-049 — cbis_ddsm_metadata_filename_aliases — **active**

**Valor:** se reconocen los nombres canónicos y aliases `Mass/Calc-Training/Test-Description.csv`.

**Por qué:** desacopla el adapter del nombre de descarga sin modificar contenido, etiquetas ni procedencia de la metadata.

## ADD-050 — cbis_ddsm_non_destructive_dicom_download_policy — **active**

**Valor:** `dataset_pipeline.download` nunca transfiere bytes DICOM. Si el árbol raw ya existe, se reutiliza; si falta, devuelve `DICOM_DOWNLOAD_REQUIRED`.

**Por qué:** evita duplicar ~163 GB, consumir ancho de banda o sobrescribir insumos de investigación al repetir un comando idempotente del prototipo.

## ADD-051 — cbis_ddsm_automatic_official_metadata_acquisition — **active**

**Valor:** solo los cuatro CSV oficiales mass/calc train/test pueden descargarse automáticamente cuando faltan. Cada archivo se valida por SHA-256 y columnas obligatorias; copias válidas existentes se reutilizan.

**Por qué:** esos cuatro archivos pequeños son necesarios para `pathology` y están separados del transfer DICOM de NBIA. Automatizar este paso elimina trabajo manual sin automatizar ni eludir la adquisición de imágenes.

## ADD-052 — cbis_ddsm_auxiliary_series_metadata — **active**

**Valor:** `metadata.csv` es opcional y se une por `SeriesInstanceUID` único para enriquecer `PatientID`, `StudyInstanceUID`, lateralidad y vista. Nunca se usa como ground truth.

**Por qué:** la exportación TCIA puede codificar identidad de paciente/vista en `PatientID` aunque el header DICOM o la ruta NBIA no la expongan de forma utilizable. La fuente de etiqueta sigue siendo exclusivamente `pathology` de los case-description CSV.

## ADD-053 — cbis_ddsm_reuse_completed_dicom_index — **active**

**Valor:** se reutiliza por defecto `/workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv`; `--force-dicom-index` reconstruye explícitamente el índice cuando cambió el árbol DICOM.

**Por qué:** el primer `inspect` real sobre `/mnt/d` tardó ~11m33s. Una inspección repetida sobre un dataset sin cambios no debe volver a recorrer/abrir 10k+ DICOM innecesariamente.

## ADD-054 — cbis_ddsm_explicit_object_inventory — **active**

**Valor:** `inspect` reporta `dicom_objects`, `full_mammogram_images`, `cropped_images`, `roi_masks`, `other_dicom_images`, `selected_full_view_images` y `complete_study_ground_truth_counts`.

**Por qué:** la tesis debe diferenciar claramente archivos/imágenes, filas de anormalidad, participantes y unidades de evaluación de cuatro vistas.

## ADD-055 — validated_env_example_retained_v015 — **active**

**Valor:** `.env.example` conserva `DEFAULT_MODEL_DEVICE=cpu`, `GMIC_DEVICE=gpu`, `NYU_DEVICE=gpu`, `GLAM_DEVICE=gpu`, `ALLOW_GPU=true`, `GPU_NUMBER=0`.

**Por qué:** v0.15 cambia solo la capa de datasets; no introduce una configuración de modelos no probada.

## ADD-056 — gmic_blackwell_legacy_integer_index_semantics — **active**

**Valor:** GMIC Blackwell usa `torch.div(max_linear_idx, W_map, rounding_mode="floor")` en `get_max_window` y `build_revision=2`.

**Por qué:** la primera integración con CBIS-DDSM real expuso una coordenada ROI negativa causada por semántica moderna de división verdadera. La corrección preserva el cálculo entero histórico; no modifica arquitectura, pesos, checkpoints ni entrenamiento.

## ADD-057 — healthcheck_access_log_state_transitions — **active**

**Valor:** primer estado `/health`, primer fallo y primera recuperación se registran; probes repetidos con el mismo estado se suprimen.

**Por qué:** reduce ruido de `docker compose logs` sin ocultar pérdida o recuperación de salud.

## ADD-058 — container_version_exposure — **active**

**Valor:** `VERSION` existe en `/app/VERSION` y `/runner/VERSION`; `/health`/`/doctor` exponen la versión actual (`0.18.0` en este paquete).

**Por qué:** permite verificar exactamente qué build está corriendo y elimina versiones hardcodeadas obsoletas.

## ADD-059 — parameterized_gpu_release_validation — **active**

**Valor:** `python -m model_tools.validate_gpu --models <uno|varios|all>` ejecuta `ensure_gpu` para todos los seleccionados, después `gpu_probe`, y finalmente smoke tests. Persiste un reporte JSON por corrida.

**Por qué:** reduce pasos manuales repetitivos y deja una evidencia única de que la revisión configurada de cada runtime existe, ejecuta CUDA y completa inferencia upstream.

## ADD-060 — explicit_gpu_force_rebuild — **active**

**Valor:** `--force-rebuild` fuerza un rebuild de las imágenes GPU seleccionadas. Sin el flag, `ensure_gpu` conserva semántica idempotente y solo reconstruye si falta la imagen o cambió `build_revision`.

**Por qué:** evita rebuilds costosos por defecto, pero permite demostrar una reconstrucción desde cero cuando sea metodológicamente necesario.

## ADD-061 — gpu_validation_device_guard — **active**

**Valor:** el smoke test de la validación integrada exige `<MODEL>_DEVICE=gpu`; `--allow-cpu-smoke` es una excepción explícita.

**Por qué:** impide confundir un smoke test CPU exitoso con evidencia de que la imagen Blackwell recién asegurada realmente fue utilizada.


## ADD-062 — gmic_metarepository_malignant_only_label_contract — **active**

**Valor:** el runtime GMIC Blackwell acepta `cancer_label` con `left_malignant` y `right_malignant`; si `left_benign`/`right_benign` no existen, solo las columnas `benign_label` del CSV se registran como `NaN`.

**Por qué:** el metarepository define las etiquetas canónicas breast-level mediante presencia de malignidad. El runner standalone de GMIC intentaba leer etiquetas benignas adicionales después del forward, provocando `KeyError` con datasets reales adaptados. No se debe inventar `benign = 1 - malignant`; la ausencia se representa explícitamente como dato no disponible.
