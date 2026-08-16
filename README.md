# Mammography AI Agent — prototipo de tesis de maestría

Prototipo local y reproducible que orquesta **GMIC + DMV-CNN / NYU Breast Cancer Classifier + GLAM** y combina sus `malignancy_score` mediante **Weighted Soft Voting**.

> **Uso exclusivo de investigación.** No es un dispositivo médico, no emite diagnóstico autónomo y requiere revisión humana.

## Principios del prototipo

- No entrena modelos, no hace fine-tuning, LoRA ni nuevas classifier heads.
- No usa LLM, RAG ni APIs externas de inferencia.
- No contiene un modelo sustituto ni predicciones simuladas: si un modelo real no puede ejecutarse, la corrida falla explícitamente.
- Usa LangGraph únicamente como máquina de estados determinística.
- Mantiene inputs, datasets, modelos/runtime, resultados y logs en una carpeta visible desde el host: `./workspace`.
- **Docker no descarga datasets al arrancar.** El investigador elige explícitamente `cbis_ddsm`, `vindr` o `all`.
- La descarga, preparación y cantidad de casos de una prueba son decisiones independientes.

## Arquitectura simple

```text
Streamlit
   |
FastAPI / LangGraph
   |
   v
Model Runner (controlador único, sin frameworks ML)
   |
   +----------------+----------------+
   |                |                |
   v                v                v
mammography-     mammography-     mammography-
model-gmic       model-nyu        model-glam
(imagen aislada) (imagen aislada) (imagen aislada)
   |                |                |
   +----------------+----------------+
                    |
           Weighted Soft Voting
                    |
PostgreSQL + workspace/output + logs
```

Docker Compose mantiene **un solo servicio persistente `model-runner`**. Ese servicio es un controlador técnico liviano: recibe las solicitudes de FastAPI/LangGraph, selecciona el modelo, construye/reutiliza su imagen Docker y crea un contenedor de inferencia temporal.

Las dependencias de IA **no están en el Model Runner**. Las imágenes reales de los modelos son independientes:

```text
mammography-model-gmic:research
mammography-model-nyu:research
mammography-model-glam:research
```

Cada imagen conserva su propio Python/framework/dependencias legacy definidos por el metarepositorio de NYU. Cuando una referencia histórica de infraestructura ya no puede resolverse, v0.6 aplica una **capa de compatibilidad auditable** que modifica únicamente la línea `FROM` configurada; no cambia el código del modelo, sus commits, checkpoints ni lógica de inferencia. El Model Runner no instala PyTorch, TensorFlow, CUDA Toolkit ni cuDNN. Esto evita mezclar versiones incompatibles en un mismo entorno.

Durante una inferencia el flujo es:

```text
FastAPI/LangGraph
      |
      v
model-runner
      |
      +--> crea mammography-inference-gmic-<run>  -> ejecuta GMIC -> elimina
      +--> crea mammography-inference-nyu-<run>   -> ejecuta NYU  -> elimina
      +--> crea mammography-inference-glam-<run>  -> ejecuta GLAM -> elimina
```

Si se habilita GPU, el Model Runner aplica un lock compartido en `/workspace/runtime/locks/gpu_inference.lock`: **solo un modelo ejecuta inferencia GPU a la vez**. El runner decide cuándo asignar la GPU al contenedor hijo mediante Docker/NVIDIA, pero las librerías CUDA/PyTorch necesarias para el cálculo pertenecen a la imagen de cada modelo.

## 1. Requisitos

- Linux x86-64. WSL2 es válido si Docker está correctamente integrado.
- Docker Engine + Docker Compose v2.
- Internet solo durante la primera construcción de las imágenes oficiales de los modelos y adquisición autorizada de datasets.
- Para GPU: NVIDIA Driver + NVIDIA Container Toolkit.

### RTX 50 / Blackwell

Las imágenes/entornos oficiales de estos modelos usan versiones antiguas de CUDA/PyTorch. La selección de dispositivo es una decisión de despliegue **por modelo**, mientras que el perfil técnico GPU pertenece a `config/models.yaml`. En v0.16, `.env.example` refleja la configuración que ya fue validada en la workstation objetivo RTX 5060 Ti:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=gpu
NYU_DEVICE=gpu
GLAM_DEVICE=gpu
ALLOW_GPU=true
GPU_NUMBER=0
```

Cada modelo posee su propio `gpu_compatibility.profile` en `config/models.yaml` y ya pasó `gpu_probe` y smoke test completo en esa workstation. No existe un `GPU_RUNTIME_PROFILE` global y la imagen legacy `:research` nunca se selecciona para GPU. En hardware distinto, vuelva temporalmente los `*_DEVICE` a `cpu` hasta validar el perfil correspondiente.

## 2. Inicio rápido

```bash
cp .env.example .env
docker compose up -d --build
```

> Use `-d` para dejar la plataforma ejecutándose en segundo plano. Si se usa `docker compose up --build` sin `-d` y se interrumpe con `Ctrl+C`, los servicios dejan de estar disponibles para los comandos posteriores.

Estado:

```bash
docker compose ps -a
./scripts/status.sh
```

Validación específica del Model Runner:

```bash
curl http://localhost:8010/doctor
curl http://localhost:8010/health
```

`/doctor` siempre intenta devolver el diagnóstico de la frontera Docker: existencia del socket, ping directo, `docker version` y `docker info`. `/health` solo devuelve 200 cuando el Runner puede utilizar realmente el Docker Engine del host.

Servicios:

- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000/docs
- Model Runner: http://localhost:8010/docs
- MinIO: http://localhost:9001



## 2.1 Catálogo operativo de comandos Docker / Docker Compose

Esta sección centraliza los comandos usados durante la validación del prototipo. Antes de ejecutar un comando de dataset, revise **Qué modifica**: `raw/` contiene la copia autorizada original; `processed/` contiene derivados reproducibles; `manifests/` contiene índices/labels; `output/` contiene resultados de pruebas.

| Comando | Para qué sirve | Qué modifica | Qué esperar |
|---|---|---|---|
| `docker compose up -d --build` | Construye/levanta la plataforma. | Imágenes de servicios y volúmenes de infraestructura; no descarga datasets. | Servicios `Up`/`healthy`. |
| `docker compose up -d --force-recreate model-runner fastapi` | Recrea servicios para releer `.env`/código. | Contenedores, no datasets ni resultados. | Runner y FastAPI vuelven `healthy`. |
| `docker compose build --no-cache model-runner fastapi bootstrap streamlit` | Reconstruye servicios de aplicación desde cero. | Imágenes Docker de aplicación; no toca `workspace/`. | Builds exitosos. |
| `docker compose down --remove-orphans` | Detiene la plataforma y elimina contenedores huérfanos. | Contenedores; conserva bind mount `workspace/` y volúmenes nombrados salvo que se añada `-v`. | Servicios detenidos. |
| `docker compose ps -a` | Muestra estado/health de servicios. | Nada. | `healthy` en dependencias persistentes. |
| `docker compose logs -f` | Sigue logs de todos los servicios. | Nada. | Logs en vivo. Los `/health` 200 repetidos se suprimen desde v0.16. |
| `docker compose logs -f model-runner` | Sigue el ciclo de inferencia del Model Runner: espera/adquisición GPU, inicio de contenedor temporal, inicio/fin de comando, éxito/fallo y métricas resumidas. | Nada. | Eventos de inferencia visibles sin spam de healthchecks. |
| `./scripts/logs.sh` | Wrapper para seguir en vivo los logs de `fastapi` y `model-runner` con un tail inicial configurable (`TAIL_LINES=200 ./scripts/logs.sh`). | Nada. | Logs operativos de servicios; eventos CLI persisten además en `workspace/logs/*.jsonl`. |
| `tail -f workspace/logs/audit.jsonl workspace/logs/model_runner.jsonl` | Sigue la auditoría persistente de CLI/pipeline y Runner desde el host. | Nada. | Eventos JSONL incluso cuando el comando se ejecutó mediante `docker compose exec`. |
| `docker compose exec fastapi cat /app/VERSION` | Verifica versión del código dentro de FastAPI. | Nada. | `0.29.2`. |
| `docker compose exec model-runner cat /runner/VERSION` | Verifica versión del Model Runner. | Nada. | `0.29.2`. |
| `docker compose exec model-runner docker version` | Verifica cliente/daemon Docker desde el Runner. | Nada. | Client/Server accesibles. |
| `docker compose exec model-runner docker info` | Diagnóstico detallado del daemon desde el Runner. | Nada. | Información del Engine sin error. |
| `docker compose exec fastapi python -m model_tools.status` | Estado de imágenes, perfiles GPU y device por modelo. | Nada. | GMIC/NYU/GLAM `device=gpu` en workstation validada. |
| `docker compose exec fastapi python -m model_tools.ensure --models gmic nyu glam` | Construye/reutiliza imágenes legacy `:research`. | Imágenes Docker; no cambia pesos/arquitectura. | `READY`. |
| `docker compose exec fastapi python -m model_tools.ensure_gpu --models gmic nyu glam` | Construye/reutiliza runtimes Blackwell para uno o más modelos. Revisiones actuales: GMIC=3, NYU=1, GLAM=2. Puede usarse con un subconjunto. | Imágenes GPU seleccionadas; invalida el probe de un modelo solo si ese runtime se reconstruye. | `READY` por modelo y `rebuild_performed` explícito. |
| `docker compose exec fastapi python -m model_tools.gpu_probe --models gmic nyu glam` | Prueba asignación y kernel CUDA de uno o más runtimes GPU seleccionados. | Actualiza evidencia `workspace/models/gpu_compatibility/<model>.probe.json`; no toca datasets. | `GPU_READY` por modelo. |
| `docker compose exec fastapi python -m model_tools.smoke_test --models gmic nyu glam` | Prueba los modelos con sample data upstream. | `workspace/output/smoke_tests/`; no toca datasets raw. | `SUCCESS` por inferencia completada. |
| `docker compose exec fastapi python -m model_tools.validate_gpu --models all` | Orquestación de release/validación: asegura la revisión GPU configurada de GMIC+NYU+GLAM, ejecuta `gpu_probe` de todos y luego smoke test de todos. También acepta uno o más modelos (`--models gmic nyu`). | Puede reconstruir solo las imágenes cuyo `build_revision` cambió; renueva sus probes y escribe evidencia JSON en `workspace/output/model_validation/`; no modifica datasets. | `overall_status=READY` y `PASS` en `ensure_gpu`, `gpu_probe` y `smoke_test` por modelo. |
| `docker compose exec fastapi python -m model_tools.validate_gpu --models all --force-rebuild` | Igual que el anterior, pero fuerza reconstrucción de todas las imágenes GPU seleccionadas aunque su revisión ya coincida. Úselo solo cuando se quiera validar bytes nuevos explícitamente. | Rebuild de imágenes seleccionadas, invalida/renueva probes y escribe reporte; no toca datasets. | `rebuild_performed=true` por modelo y validación completa. |
| `./scripts/validate-models.sh all` o `./scripts/validate-models.sh gmic nyu` | Wrapper host del comando integrado anterior; evita escribir el `docker compose exec ...` completo y acepta uno o más modelos. | Los mismos efectos de `model_tools.validate_gpu`; no modifica datasets. | Mismo JSON/resumen de validación. |
| `docker compose exec fastapi python -m dataset_pipeline.status` | Consulta estado de datasets. | Nada. | `AVAILABLE`, `READY_FOR_INSPECT`, etc. |
| `docker compose exec fastapi python -m dataset_pipeline.download --datasets cbis_ddsm` | Solo escribe/actualiza instrucciones y valida archivos ya presentes. **No descarga DICOM ni CSV**. | Puede escribir `DOWNLOAD_INSTRUCTIONS.md`; no modifica raw. | Estado manual (`READY_FOR_INSPECT`, `METADATA_REQUIRED`, `DICOM_DOWNLOAD_REQUIRED`, etc.). |
| `docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cmmd --force-dicom-index` | Audita CMMD DICOM + XLSX clínico manual, resuelve CC/MLO desde `ViewCodeSequence`, separa four-view y construye el subconjunto binario CMMD1/D1. | Escribe índices/manifests/rejected; no modifica raw. | Conteos DICOM, cohortes, four-view y benchmark D1. |
| `docker compose exec fastapi python -m dataset_pipeline.prepare --datasets cmmd` | Convierte solo el subconjunto CMMD1/D1 four-view con labels bilaterales explícitos a PNG 16-bit. | `processed/cmmd/images/` + `manifests/cmmd.csv`; no modifica DICOM/XLSX raw. | `AVAILABLE` y `converted_studies=...`. |
| `docker compose exec fastapi python -m tests_flow.normal --datasets cmmd --samples 10 --sampling balanced --seed 42 --max-runtime-minutes 30` | Prueba diagnóstica balanceada sobre el subconjunto CMMD1/D1 preparado. | Solo outputs/logs. | 5 benignos/5 malignos si hay al menos cinco de cada clase. |
| `./scripts/audit-dicom-presentation.sh /workspace/output/normal_tests/<RUN>` | Contrafactual DICOM label-blind sobre los mismos estudios ya diagnosticados: compara conversión actual vs Modality LUT/rescale vs VOI/Window. No ejecuta clasificadores. | Solo `workspace/output/analyses/dicom-presentation-.../`; no modifica raw ni prepared. | Reporte Markdown + JSON/CSV de diferencias; no selecciona una rama por AUC. |
| `docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cbis_ddsm` | Cruza metadata, reutiliza índice DICOM y construye catálogos/manifiesto de estudios completos. | `manifests/`, `rejected/`, `source_manifest.csv`, cache de índice; no modifica pixels raw. | Conteos de pacientes/vistas y `ensemble_compatible`. |
| `docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cbis_ddsm --force-dicom-index` | Igual que `inspect`, pero reconstruye headers DICOM. | Reescribe cache del índice; no modifica DICOM. | Mucho más lento; usar solo si cambió el árbol raw. |
| `docker compose exec fastapi python -m dataset_pipeline.prepare --datasets cbis_ddsm` | Convierte **solo estudios de 4 vistas compatibles** a PNG 16-bit y escribe manifiesto canónico. | Escribe/regenera `processed/cbis_ddsm/images/*.png` y `manifests/cbis_ddsm.csv`; **no limpia, borra ni modifica DICOM raw**. No elimina derivados antiguos no referenciados. | `AVAILABLE`, `converted_studies=...`. |
| `docker compose exec fastapi python -m tests_flow.normal --datasets cbis_ddsm --samples 10 --sampling stratified --seed 42 --max-runtime-minutes 30` | Prueba end-to-end con sampling proporcional reproducible. En el CBIS-DDSM preparado actual (72 benignos/33 malignos), 10 seleccionan objetivo 7/3. | Solo `workspace/output/normal_tests/<run>/` y logs; no modifica dataset preparado. | `selected_studies.csv`, `run_summary.json`, `NORMAL_TEST_COMPLETED` y predicciones/métricas. |
| `docker compose exec fastapi python -m tests_flow.normal --datasets cbis_ddsm --samples 10 --sampling balanced --seed 42 --max-runtime-minutes 30` | Prueba de integración balanceada que fuerza cuotas iguales por clase cuando hay disponibilidad; útil para ejercitar TN/FP/FN/TP. | Solo outputs/logs de la nueva corrida; no limpia ni reconvierte datasets. | Para 10 estudios, objetivo 5 benignos/5 malignos y métricas calculables si todos completan. |
| `docker compose exec fastapi python -m experiments.score_analysis --input /workspace/output/normal_tests/<RUN>/raw_model_predictions.csv` | Analiza scores ya calculados sin ejecutar GMIC/NYU/GLAM otra vez: ROC-AUC por modelo, distribuciones, correlaciones, puntos ROC, métricas baseline (Sensitivity/Specificity/PPV/NPV/FPR/Balanced Accuracy) y preview de thresholds adaptativos. | Solo crea `workspace/output/analyses/score-analysis-.../`; no modifica datasets, modelos ni el run de origen. | `score_summary.json`, `model_metrics.csv`, `candidate_thresholds.csv`, `diagnostic_configurations.csv`, `diagnostic_ranking.csv`, `score_analysis_report.md`. |
| `./scripts/analyze-scores.sh /workspace/output/normal_tests/<RUN>/raw_model_predictions.csv` | Wrapper host del análisis anterior. | Mismos outputs CPU-only; no usa GPU. | Ruta del directorio de análisis. |
| `docker compose exec fastapi python -m experiments.run --datasets cbis_ddsm --configuration-ratio 0.30 --seed 42` | Fase de configuración experimental. Divide por paciente antes de inferir, ejecuta modelos solo sobre Configuration Set y evalúa 16 pesos × 5 thresholds adaptativos (=80) en CPU. v0.23 selecciona pesos por ROC-AUC y threshold por Balanced Accuracy, con Sensitivity/Specificity como desempate. El Final Test Set queda reservado sin scores. | `workspace/output/experiments/`; no modifica dataset. | `experiment_plan.json`, manifests config/final, score analysis, 80 configuraciones, ranking y best configuration. |
| `docker compose exec fastapi python -m experiments.freeze --experiment <ID>` | Congela pesos/threshold seleccionados. | Crea `frozen_configuration.yaml`; no se sobreescribe con contenido distinto. | Configuración frozen. |
| `docker compose exec fastapi python -m experiments.final_evaluation --experiment <ID>` | Evalúa el Final Test Set reservado. | Resultados finales en el experimento; no reoptimiza. | Métricas selected vs baseline. |
| `docker run --rm --gpus all nvidia/cudagl:10.1-devel-ubuntu18.04 nvidia-smi` | Diagnóstico de exposición GPU a Docker/WSL. | Nada persistente. | RTX visible dentro del contenedor. |

### Semántica exacta de `dataset_pipeline.prepare`

`prepare` **no limpia el dataset original**. Para CBIS-DDSM: (1) ejecuta/reutiliza `inspect`; (2) lee `source_manifest.csv`; (3) abre únicamente los DICOM seleccionados para los estudios completos de cuatro vistas; (4) aplica la conversión DICOM→PNG 16-bit requerida por el metarepositorio NYU, incluyendo `MONOCHROME1` cuando corresponda; (5) escribe los PNG derivados en `workspace/datasets/processed/cbis_ddsm/images/`; y (6) reescribe el manifiesto canónico `workspace/datasets/manifests/cbis_ddsm.csv`. Los DICOM de `raw/` permanecen byte-a-byte intactos. Ejecutar `prepare` de nuevo regenera los destinos seleccionados, pero no borra archivos raw ni realiza una limpieza global de `processed/`.

## Corrección v0.4 para Docker Desktop / WSL2

La arquitectura sigue siendo **un Model Runner + tres imágenes aisladas de modelos**. v0.4 corrige únicamente la comunicación del Runner con Docker Desktop/Engine.

La versión anterior instalaba `docker.io` desde Debian dentro del Runner. v0.4 usa por defecto la imagen oficial:

```env
DOCKER_CLI_IMAGE=docker:29-cli
```

y hace explícito:

```text
DOCKER_HOST=unix:///var/run/docker.sock
```

Si vienes de v0.3, conserva `workspace/` y los volúmenes y ejecuta:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner
docker compose up -d
```

Después verifica:

```bash
docker compose ps -a
curl http://localhost:8010/doctor
curl http://localhost:8010/health
```

Para diagnosticar desde dentro del servicio usa el nombre de servicio de Compose, no un nombre de contenedor escrito manualmente:

```bash
docker compose exec model-runner docker version
docker compose exec model-runner docker info
```

También puedes ejecutar:

```bash
./scripts/doctor.sh
```

Ver `docs/MIGRATION_V0_4.md`.

## Corrección v0.5 para imágenes CUDA legacy

El `ensure` real de GMIC confirmó que el Dockerfile upstream usa la referencia histórica:

```text
nvidia/cuda:10.1-base-ubuntu18.04
```

y el entorno actual ya no puede resolver ese tag. Los Dockerfiles upstream de GMIC, DMV-CNN/NYU y GLAM comparten esa misma base. v0.5 genera, para cada modelo, un Dockerfile de compatibilidad que reemplaza **solo** esa línea por:

```text
nvidia/cudagl:10.1-devel-ubuntu18.04
```

El Dockerfile generado queda bajo `.thesis_compat/` dentro de la copia local del metarepositorio. Se registran SHA-256 del Dockerfile original y generado, imagen original/reemplazo, motivo y las garantías:

```text
model_code_changed=false
model_weights_changed=false
training_performed=false
```

Además, v0.5 registra automáticamente el metarepositorio host-mounted como `git safe.directory` cuando sea necesario, evitando el workaround manual observado en WSL2.

Para migrar desde v0.4 conserva `.env` y `workspace/`:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Luego prueba solo GMIC:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure \
  --models gmic
```

Si el build vuelve a fallar, el CLI ahora muestra el detalle devuelto por el Model Runner y el evento completo sigue disponible en `workspace/logs/model_runner.jsonl`.

Ver `docs/MIGRATION_V0_5.md`.


## Corrección v0.6 para rotación de claves NVIDIA legacy

Con v0.5 el build real de GMIC avanzó hasta `apt-get update`, donde el repositorio CUDA de Ubuntu 18.04 falló con:

```text
NO_PUBKEY A4B469963BF863CC
```

v0.6 mantiene la sustitución de imagen base de v0.5 y añade `nvidia_repository_key_rotation_fix: auto`. Si el Dockerfile upstream ya contiene el refresh de clave NVIDIA, se conserva. Si no lo contiene, el Model Runner lo inserta antes del `apt-get update`. Esto permite que GMIC/GLAM utilicen el mismo tipo de workaround que el Dockerfile upstream actual de DMV-CNN/NYU ya incorpora.

La corrección es exclusivamente del entorno de construcción:

```text
model_code_changed=false
model_weights_changed=false
training_performed=false
```

La evidencia queda en `workspace/logs/model_runner.jsonl` y `workspace/models/compatibility/<model>.json`.

Para migrar desde v0.5 conserva `.env` y `workspace/`:

```bash
docker compose down --remove-orphans
docker compose build --no-cache model-runner fastapi
docker compose up -d
```

Luego vuelve a probar únicamente GMIC:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure \
  --models gmic
```

Ver `docs/MIGRATION_V0_6.md`.

## Migración desde v0.2

La arquitectura introducida en v0.3 (y mantenida en v0.4/v0.5/v0.6) simplifica los tres controladores `gmic-runtime`, `nyu-runtime` y `glam-runtime` a **un único `model-runner`**. Las imágenes de los modelos siguen separadas.

Si v0.2 está ejecutándose:

```bash
docker compose down --remove-orphans
```

No uses `-v` si deseas conservar PostgreSQL/MinIO. Conserva también tu `workspace/`. Sustituye los archivos del proyecto, revisa `.env` y elimina si existieran:

```text
GMIC_RUNTIME_URL
NYU_RUNTIME_URL
GLAM_RUNTIME_URL
```

La nueva variable es:

```env
MODEL_RUNNER_URL=http://model-runner:8010
```

Luego:

```bash
docker compose up -d --build
docker compose ps
```

Debes ver `mammography-model-runner` como único controlador persistente de modelos. Ver `docs/MIGRATION_V0_3.md`.

## 3. ¿Dónde están mis archivos?

Por defecto:

```text
<directorio-del-proyecto>/workspace/
```

Dentro de los contenedores la misma carpeta aparece como:

```text
/workspace/
```

Linux:

```bash
pwd
xdg-open ./workspace
```

Windows/PowerShell, desde la carpeta del proyecto:

```powershell
explorer .\workspace
```

Se puede cambiar en `.env`:

```env
HOST_WORKSPACE=./workspace
```

Mapa:

```text
workspace/
├── input/
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── rejected/
├── models/
├── runtime/
├── output/
│   ├── analyses/
│   ├── normal_tests/
│   ├── experiments/
│   ├── final_evaluations/
│   ├── xai/
│   └── reports/
└── logs/
```

## 4. Descarga de datasets: siempre explícita

Consultar estado:

```bash
docker compose exec fastapi python -m dataset_pipeline.status
```

Solicitar solo CBIS-DDSM para una prueba inicial:

```bash
docker compose exec fastapi python -m dataset_pipeline.download --datasets cbis_ddsm
```

Solicitar solo VinDr-Mammo:

```bash
docker compose exec fastapi python -m dataset_pipeline.download --datasets vindr
```

Solicitar todos los datasets configurados que aún falten:

```bash
docker compose exec fastapi python -m dataset_pipeline.download --datasets all
```

**Importante:** los dos datasets pueden requerir pasos de acceso/licencia fuera del programa. Cuando eso ocurra, el comando devuelve `MANUAL_DOWNLOAD_REQUIRED` y crea `DOWNLOAD_INSTRUCTIONS.md` dentro de `workspace/datasets/raw/<dataset>/`. El agente nunca intenta eludir esos controles.

`docker compose up -d` no invoca ninguno de estos comandos.

## 5. Preparación de datasets

### CBIS-DDSM — adapter oficial TCIA (v0.16)

CBIS-DDSM ya **no requiere que el investigador construya manualmente `source_manifest.csv`**. El adapter descubre recursivamente el árbol DICOM descargado con NBIA Data Retriever y los cuatro CSV oficiales:

```text
mass_case_description_train_set.csv
mass_case_description_test_set.csv
calc_case_description_train_set.csv
calc_case_description_test_set.csv
```

Deje todo bajo el raw directory del workspace. No es necesario aplanar ni renombrar la estructura que genere NBIA:

```text
workspace/datasets/raw/cbis_ddsm/
├── CBIS-DDSM/                 # nombre/jerarquía de NBIA puede variar
│   └── ... DICOM ...
└── metadata/                  # recomendado; también puede estar en otro subdirectorio
    ├── mass_case_description_train_set.csv
    ├── mass_case_description_test_set.csv
    ├── calc_case_description_train_set.csv
    └── calc_case_description_test_set.csv
```

Antes de convertir imágenes, inspeccione el release descargado:

```bash
docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cbis_ddsm
```

La inspección no inventa etiquetas ni vistas. Genera y conserva:

```text
workspace/datasets/manifests/cbis_ddsm_metadata_rows.csv
workspace/datasets/manifests/cbis_ddsm_view_catalog.csv
workspace/datasets/rejected/cbis_ddsm_unresolved_metadata_rows.csv
workspace/datasets/rejected/cbis_ddsm_incomplete_studies.csv
workspace/datasets/raw/cbis_ddsm/source_manifest.csv
workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv
```

Ground truth se obtiene **exclusivamente** de la columna oficial `pathology`:

```text
MALIGNANT                -> 1
BENIGN                   -> 0
BENIGN_WITHOUT_CALLBACK  -> 0
```

BI-RADS/`assessment` no se convierte a cáncer/no-cáncer. Valores de patología desconocidos se reportan como no resueltos.

El ensemble actual incluye el clasificador exam-level DMV-CNN/NYU, por lo que el manifiesto canónico admite únicamente estudios con las cuatro vistas estándar:

```text
L-CC, R-CC, L-MLO, R-MLO
```

Si una vista falta, el estudio queda en `cbis_ddsm_incomplete_studies.csv`; v0.16 **no duplica ni sintetiza** una vista faltante. El comando `inspect` informa `complete_four_view_studies` y `ensemble_compatible` antes de realizar la conversión costosa.

Cuando la inspección sea satisfactoria:

```bash
docker compose exec fastapi python -m dataset_pipeline.prepare --datasets cbis_ddsm
```

La preparación convierte únicamente los estudios compatibles a PNG de 16 bits y crea:

```text
workspace/datasets/processed/cbis_ddsm/images/
workspace/datasets/manifests/cbis_ddsm.csv
```

Si no existe ningún estudio completo de cuatro vistas, el resultado es `INSUFFICIENT_FOUR_VIEW_STUDIES` y no se fabrica un dataset compatible.

### Otros adapters

VinDr conserva en esta versión el contrato genérico de `source_manifest.csv`. No se modificó su comportamiento en v0.16.

## 6. Modelos reales: un Runner y tres imágenes aisladas

Después de `docker compose up -d --build`, `docker compose ps` debe mostrar de forma permanente:

```text
mammography-model-runner
```

Las tres imágenes reales de los modelos se construyen localmente bajo demanda a partir de los Dockerfiles del metarepositorio de NYU:

```text
mammography-model-gmic:research
mammography-model-nyu:research
mammography-model-glam:research
```

Antes de construir modelos, confirma que el Runner puede utilizar Docker Desktop/Engine:

```bash
curl http://localhost:8010/health
```

Debe devolver `"docker_daemon": true`. Si no ocurre, ejecuta `./scripts/doctor.sh`.

Estado de los tres modelos a través del runner:

```bash
docker compose exec fastapi python -m model_tools.status
```

Construir/verificar las tres imágenes sin ejecutar inferencia:

```bash
docker compose exec fastapi python -m model_tools.ensure --models gmic nyu glam
```

Smoke test con los datos de muestra del metarepository:

```bash
docker compose exec fastapi python -m model_tools.smoke_test --models gmic nyu glam
```

Con `MODEL_BOOTSTRAP_MODE=lazy`, `docker compose up` construye el Model Runner pero **no construye todavía las tres imágenes legacy**. `ensure`, `smoke-test` o la primera inferencia construyen/reutilizan la imagen específica necesaria.

El Model Runner no contiene frameworks ML. Su función es: routing por modelo, construcción/reutilización de imágenes, ejecución temporal, serialización GPU, logging y métricas.

### Score canónico a nivel de estudio

Se requiere un único score por estudio para el voting. Las reglas están congeladas en `config/models.yaml`:

- GMIC: máximo `malignant_pred` de las imágenes del estudio.
- GLAM: máximo `malignant_pred` de las imágenes del estudio.
- DMV-CNN/NYU: máximo entre `left_malignant` y `right_malignant`.

Son reglas determinísticas; no hay calibración aprendida.

## 7. Prueba normal

La prueba normal usa por defecto `--sampling sequential` para compatibilidad con versiones anteriores. v0.21 añade selección reproducible consciente de clase:

- `sequential`: primeros N estudios del manifest (legacy).
- `random`: N estudios aleatorios reproducibles mediante `--seed`.
- `stratified`: mantiene aproximadamente la proporción de clases del universo disponible. En los 105 estudios CBIS-DDSM actuales (72 benignos/33 malignos), `--samples 10` produce objetivo **7 benignos / 3 malignos**.
- `balanced`: cuotas iguales por clase; `--samples 10` produce **5 benignos / 5 malignos** si existen suficientes casos.

Toda corrida escribe `selected_studies.csv`, por lo que los IDs exactos usados quedan auditables. `configuration_used.yaml` registra sampling/seed/distribuciones y `run_summary.json` registra estudios/imágenes procesados y tiempo total.

Prueba representativa estratificada:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 10 \
  --sampling stratified \
  --seed 42 \
  --max-runtime-minutes 30
```

Prueba de integración balanceada (recomendada para comprobar ambas clases antes de aumentar el volumen):

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 10 \
  --sampling balanced \
  --seed 42 \
  --max-runtime-minutes 30
```

Pesos manuales:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 10 \
  --sampling stratified \
  --seed 42 \
  --weights 0.50 0.30 0.20 \
  --threshold 0.45
```

Configuración congelada de un experimento anterior:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 10 \
  --sampling stratified \
  --seed 42 \
  --config /workspace/output/experiments/<ID>/frozen_configuration.yaml
```

El límite `--max-runtime-minutes` se evalúa **entre chunks completos**; no mata una inferencia ya iniciada. Si se alcanza después de al menos un chunk, la corrida queda `PARTIAL_TIME_LIMIT` y conserva evidencia de los chunks completados. No modifica ni limpia el dataset preparado.

`resource_metrics.csv` usa desde v0.21 el campo `monitoring_samples`: es el número de lecturas periódicas del monitor de CPU/GPU, **no** el número de estudios o imágenes.

## 8. Prueba experimental

v0.23 mantiene la separación metodológica **Configuration Set → freeze → Final Test Set**. No se deben generar scores del Final Test Set antes de congelar pesos/threshold. La política es inferir cada estudio como máximo una vez dentro del experimento: Configuration Set durante `experiments.run`; Final Test Set recién durante `final_evaluation`, con reutilización del cache si se repite ese comando.

### 8.1 Análisis CPU de scores ya existentes

Antes de abrir un experimento formal puede analizar cualquier `raw_model_predictions.csv` ya generado:

```bash
docker compose exec fastapi python -m experiments.score_analysis \
  --input /workspace/output/normal_tests/<RUN>/raw_model_predictions.csv
```

Este comando **no ejecuta modelos ni usa GPU**. No modifica el run de origen. Escribe bajo `workspace/output/analyses/`:

- `score_summary.json`: rango observado, AUC baseline, métricas threshold-dependent del baseline, warnings y research guards.
- `model_metrics.csv`: ROC-AUC y estadísticos benign/malignant por GMIC, NYU, GLAM y ensemble baseline.
- `score_distribution.csv`: min/quantiles/media/std por clase.
- `model_correlations.csv`: Pearson/Spearman entre scores.
- `roc_points.csv`: puntos de curva ROC cuando hay ambas clases.
- `candidate_thresholds.csv`: preview 16×5 de thresholds adaptativos con `threshold_source=analysis_score_quantile`; no es una selección formal.
- `diagnostic_configurations.csv`: métricas CPU de los 80 candidatos sobre el set analizado; incluye `diagnostic_only=true` y `eligible_for_freeze=false`.
- `diagnostic_ranking.csv`: ranking diagnóstico con la política v0.23; sirve para entender el comportamiento antes del experimento formal, no para congelar configuración.
- `score_analysis_report.md`.

v0.23 **no invierte scores con AUC < 0.5, no calibra y no entrena**; solo registra la evidencia. Además, el análisis diagnóstico usa `threshold_source=analysis_score_quantile`; `configuration_score_quantile` queda reservado al experimento formal.

### 8.2 Thresholds adaptativos del Configuration Set

La grilla fija histórica `0.40,0.45,0.50,0.55,0.60` queda documentada como legacy pero no se usa en el experimento v0.23. Para cada una de las 16 combinaciones de pesos se calculan cinco candidatos a partir de sus scores en el Configuration Set:

```text
T01 = quantile 10%
T02 = quantile 30%
T03 = quantile 50%
T04 = quantile 70%
T05 = quantile 90%
```

La derivación usa **solo scores**, nunca `ground_truth`. Después de fijar los cinco valores se usan las etiquetas del Configuration Set para calcular TN/FP/FN/TP, Sensitivity, Specificity, PPV, NPV, FPR, Accuracy, Balanced Accuracy y ROC-AUC. Por tanto siguen existiendo exactamente **16 × 5 = 80 configuraciones** y el Final Test Set no participa en la optimización.

### 8.3 Abrir el experimento formal completo

```bash
docker compose exec fastapi python -m experiments.run \
  --datasets cbis_ddsm \
  --configuration-ratio 0.30 \
  --seed 42
```

Sin `--samples`, se parte de los 105 estudios preparados. Primero se divide por paciente y de forma estratificada. Aproximadamente 30% queda como Configuration Set y 70% como Final Test Set reservado. **Solo el Configuration Set se infiere en esta fase.**

Archivos principales:

```text
experiment_plan.json
configuration_set_manifest.csv
final_test_manifest.csv
configuration_set_predictions.csv
configuration_score_analysis/
all_configurations.csv
ranking.csv
best_configuration.json
configuration_report.md
```

La política de selección v0.23 evita escoger un threshold solo porque produzca `FN=0`. Primero selecciona la combinación de pesos con mejor ROC-AUC en el Configuration Set; después elige el threshold con mayor **Balanced Accuracy**. En empates se prioriza mayor Sensitivity, luego mayor Specificity/menor FP y finalmente cercanía determinística al baseline histórico. El Final Test Set no interviene en ninguna de estas decisiones.

### 8.4 Congelar configuración

```bash
docker compose exec fastapi python -m experiments.freeze --experiment <ID>
```

Crea `frozen_configuration.yaml`. Si ya existe con contenido diferente, el proceso falla: no se permite reoptimizar silenciosamente la configuración.

### 8.5 Evaluar el Final Test Set reservado

```bash
docker compose exec fastapi python -m experiments.final_evaluation --experiment <ID>
```

Solo después del freeze se ejecutan GMIC+NYU+GLAM sobre el Final Test Set. Si `final_inference/raw_model_predictions.csv` ya existe y es compatible, v0.23 lo reutiliza en vez de volver a ejecutar los modelos. La evaluación final no cambia pesos ni threshold.

## 9. Configuraciones agregadas durante la implementación

Cualquier configuración que fue necesaria para convertir la especificación en software ejecutable está registrada en dos lugares:

```text
config/config_additions.yaml
docs/CONFIG_ADDITIONS.md
```

Al arrancar el servicio, esas decisiones también se escriben en:

```text
workspace/logs/configuration_additions.log
```

Así queda documentado **qué se agregó y por qué**.

## 10. Tests

```bash
pytest -q
```

Los tests unitarios usan únicamente fixtures numéricos para probar voting, métricas, selección y filtros. **No simulan la inferencia de GMIC/NYU/GLAM.** La inferencia solo se considera validada cuando el smoke test real de la imagen oficial del modelo, lanzada por el Model Runner, termina correctamente.

## 11. Limitaciones conscientes del prototipo

- La conversión automática desde la estructura nativa completa de cada dataset no se adivina: el adapter requiere un manifiesto de origen trazable. Esto evita introducir ground truth incorrecto.
- Las imágenes oficiales de los modelos usan stacks legacy y deben validarse en la estación de investigación antes de medir tiempos o métricas de tesis.
- XAI solo se conserva cuando la ejecución oficial del modelo produce un artefacto real. Para GMIC/GLAM el Model Runner habilita la opción oficial de visualización dentro de la imagen correspondiente; si no se genera, se registra como ausencia de evidencia, nunca se crea una imagen falsa.
- GMIC y GLAM son arquitectónicamente relacionados, por lo que sus errores pueden estar correlacionados.

## 12. Evidencia de implementación

Ver `VERIFICATION.md` y `docs/ARCHITECTURE_CHANGE_LOG.md` para distinguir lo validado en este paquete de lo que debe ejecutarse necesariamente en la workstation con Docker/GPU.

## v0.7 — GPU host preflight para Fedora Remix / WSL2

Si `nvidia-smi` funciona en WSL pero Docker responde:

```text
failed to discover GPU vendor from CDI: no known GPU vendor found
```

la GPU todavía no está disponible para contenedores. No habilite ningún `<MODEL>_DEVICE=gpu` todavía. Primero configure NVIDIA Container Toolkit/CDI en la misma distribución Fedora WSL que ejecuta Docker Engine:

```bash
docker compose down --remove-orphans
./scripts/setup-nvidia-container-toolkit-fedora-wsl.sh
./scripts/gpu-doctor.sh
```

Después valide:

```bash
docker run --rm --gpus all \
  nvidia/cudagl:10.1-devel-ubuntu18.04 \
  nvidia-smi
```

Solo cuando esa prueba funcione se debe iniciar la prueba GPU de GMIC/NYU/GLAM. No instale un driver NVIDIA Linux dentro de WSL; el driver GPU es el de Windows y el host WSL solo requiere la integración de contenedores. Ver `docs/MIGRATION_V0_7.md`.


## v0.11 — Runtime Blackwell de DMV-CNN / NYU

DMV-CNN/NYU dispone de un perfil GPU propio en `config/models.yaml`:

```text
mammography-model-nyu:blackwell-cu128
Python 3.10
PyTorch 2.7.1
TorchVision 0.22.1
CUDA 12.8
```

El perfil conserva el commit upstream `de2b0855d02984df0f516008bb4513ff71460e21` y no cambia los checkpoints ni la arquitectura. La imagen legacy `mammography-model-nyu:research` continúa siendo la referencia CPU.

Valide el nuevo runtime en este orden:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure_gpu \
  --models nyu

docker compose exec fastapi \
  python -m model_tools.gpu_probe \
  --models nyu
```

Solo después de `GPU_READY`, establezca `NYU_DEVICE=gpu`, recree `model-runner`/`fastapi` y ejecute:

```bash
docker compose exec fastapi \
  python -m model_tools.smoke_test \
  --models nyu
```

El perfil no se configura en `.env`; `.env` únicamente decide `NYU_DEVICE=cpu|gpu`. Ver `docs/MIGRATION_V0_11.md`.

## v0.10 — Perfil GPU como metadato del modelo y dispositivo por modelo

`GPU_RUNTIME_PROFILE` fue eliminado de `.env` y de Docker Compose. El perfil de compatibilidad pertenece ahora exclusivamente a cada modelo en `config/models.yaml`. La selección CPU/GPU sigue siendo una decisión de despliegue y se resuelve por modelo mediante `GMIC_DEVICE`, `NYU_DEVICE` y `GLAM_DEVICE`, con `DEFAULT_MODEL_DEVICE` como fallback.

Estado validado en la workstation al crear esta versión:

```text
GMIC  -> GPU / blackwell-cu128  (gpu_probe + smoke test reales PASS)
NYU   -> CPU                    (smoke test real PASS)
GLAM  -> CPU                    (smoke test real PASS)
```

Para reproducir esa combinación:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=gpu
NYU_DEVICE=cpu
GLAM_DEVICE=cpu
ALLOW_GPU=true
GPU_NUMBER=0
```

El smoke test GPU de GMIC reportado por la workstation produjo `predictions.csv`, 16 artefactos XAI, `elapsed_seconds=86.9096`, `avg_gpu_util_percent=8.25` y `max_gpu_memory_mib=2424`. Estas cifras son evidencia de smoke test, no un benchmark definitivo.

See `docs/MIGRATION_V0_10.md` and `docs/WORKSTATION_VALIDATION.md`.

## v0.9 — GPU probe success-path fix

The GPU compatibility architecture from v0.8 is unchanged. v0.9 fixes a Model Runner logging bug
that could turn a completed `GPU_READY` probe into HTTP 500 because `model` was supplied twice to
the structured logger. Rebuild only `model-runner` and `fastapi`; the already-built model images
and host workspace can be reused. See `docs/MIGRATION_V0_9.md`.

## v0.8 — GMIC GPU Blackwell compatibility runtime

The legacy GMIC image remains the reproducible CPU baseline:

```text
mammography-model-gmic:research
```

A separate GPU compatibility image is used for the RTX 5060 Ti / Blackwell path:

```text
mammography-model-gmic:blackwell-cu128
```

This profile keeps the same GMIC source commit and checkpoints but updates the execution runtime to Python 3.10, PyTorch 2.7.1 and CUDA 12.8 wheels. It is opt-in and must pass the runner-managed GPU probe before real GPU inference.

Build the compatibility image:

```bash
docker compose exec fastapi \
  python -m model_tools.ensure_gpu \
  --models gmic
```

Then run the fail-safe allocation/kernel probe:

```bash
docker compose exec fastapi \
  python -m model_tools.gpu_probe \
  --models gmic
```

Expected result:

```text
status: GPU_READY
allocation_ok: true
kernel_ok: true
```

Only after that result, enable **only that model** in `.env`:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=gpu
NYU_DEVICE=cpu
GLAM_DEVICE=cpu
ALLOW_GPU=true
GPU_NUMBER=0
```

The `blackwell-cu128` profile itself is **not** an environment variable. It is model metadata defined under `models.gmic.gpu_compatibility.profile` in `config/models.yaml`.

Then recreate the application services:

```bash
docker compose up -d --force-recreate model-runner fastapi streamlit
```

The runner refuses GPU inference if the selected model has no configured GPU compatibility profile or has not passed `gpu_probe`.

## v0.12 — Runtime Blackwell específico para GLAM

v0.12 completa la definición de perfiles GPU por modelo agregando `mammography-model-glam:blackwell-cu128`. El perfil técnico está en `config/models.yaml`; no existe un perfil GPU global en `.env`.

Después de validar GMIC y DMV-CNN/NYU en GPU, GLAM se habilita de forma independiente:

```bash
docker compose exec fastapi python -m model_tools.ensure_gpu --models glam
docker compose exec fastapi python -m model_tools.gpu_probe --models glam
```

Solo después de `GPU_READY`:

```env
GMIC_DEVICE=gpu
NYU_DEVICE=gpu
GLAM_DEVICE=gpu
ALLOW_GPU=true
GPU_NUMBER=0
```

Luego:

```bash
docker compose up -d --force-recreate model-runner fastapi
docker compose exec fastapi python -m model_tools.smoke_test --models glam
```

El runtime Blackwell de GLAM conserva el commit/checkpoints/arquitectura upstream. Los cambios declarados son únicamente de compatibilidad de ejecución y preservación de semánticas históricas del framework. Ver `docs/MIGRATION_V0_12.md`.


## v0.13 — Adapter real de CBIS-DDSM

v0.13 reemplaza el mapping manual de CBIS-DDSM por un adapter específico del release oficial TCIA. Descubre los cuatro archivos de clasificación, resuelve sus `image file path` contra el árbol DICOM de NBIA por ruta/UID y usa un índice de cabeceras DICOM cacheado como fallback. Genera `source_manifest.csv` automáticamente y conserva catálogos de resolución/rechazo.

Flujo recomendado:

```bash
docker compose exec fastapi python -m dataset_pipeline.status
docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cbis_ddsm
docker compose exec fastapi python -m dataset_pipeline.prepare --datasets cbis_ddsm
```

El adapter usa únicamente `pathology` como ground truth y mantiene una compuerta de cuatro vistas para el ensemble actual. Ver `docs/MIGRATION_V0_13.md`.

## v0.14 — Metadata preflight de CBIS-DDSM y `.env.example` validado

`.env.example` refleja ahora la configuración que ya pasó pruebas reales en la workstation RTX 5060 Ti:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=gpu
NYU_DEVICE=gpu
GLAM_DEVICE=gpu
ALLOW_GPU=true
GPU_NUMBER=0
```

Si los DICOM de CBIS-DDSM existen pero faltan uno o más de los cuatro CSV oficiales, `inspect` devuelve `METADATA_REQUIRED`, crea `workspace/datasets/raw/cbis_ddsm/METADATA_INSTRUCTIONS.md` y **no inicia** el índice DICOM. Después de colocar los cuatro CSV en cualquier subdirectorio bajo `raw/cbis_ddsm`, se vuelve a ejecutar `inspect`.

## v0.15 — CBIS-DDSM sin re-descarga DICOM

La política histórica v0.15 separó DICOM de metadata. **Desde v0.29.0 la política es más estricta:** el prototipo no descarga ningún archivo de dataset. Tanto los DICOM como los cuatro CSV oficiales se adquieren manualmente y el adapter solo los localiza/valida.

```text
dataset_pipeline.download --datasets cbis_ddsm
        │
        ├── DICOM (~163 GB) -> NUNCA auto-download
        └── 4 case-description CSV -> NUNCA auto-download

Missing file -> instrucción/estado accionable; no red.
```

### `metadata.csv` auxiliar

Si existe `workspace/datasets/raw/cbis_ddsm/**/metadata.csv`, v0.15 puede relacionar `SeriesInstanceUID` con el identificador textual TCIA para recuperar `P_XXXXX`, lateralidad y `CC/MLO` cuando el header DICOM no sea suficiente. Esto no modifica labels: `pathology` de los cuatro CSV oficiales sigue siendo la única fuente de ground truth.

### Reutilización del índice DICOM

Después de una inspección exitosa se conserva:

```text
workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv
```

La siguiente inspección usa ese índice por defecto y no necesita reabrir los 10k+ DICOM. Si el árbol DICOM cambió después del último índice, fuerza una reconstrucción:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.inspect \
  --datasets cbis_ddsm \
  --force-dicom-index
```

`inspect` ahora reporta además `full_mammogram_images`, `cropped_images`, `roi_masks`, `other_dicom_images`, `selected_full_view_images` y `complete_study_ground_truth_counts`.


## v0.16 — integración CBIS-DDSM→GMIC, health logs y operación documentada

- Corrige la incompatibilidad de semántica de división entera de GMIC al ejecutar el commit upstream fijado con PyTorch 2.7.1. `src/utilities/tools.py` conserva el cociente/remainder entero histórico mediante `torch.div(..., rounding_mode="floor")`; no se elimina el assert ni se cambian arquitectura, pesos, saliency maps o función de selección de ROI.
- `GMIC gpu_compatibility.build_revision=2` obliga a reconstruir una sola vez la imagen Blackwell existente y a invalidar el probe previo, de modo que la workstation deba repetir `gpu_probe` sobre la imagen realmente corregida. NYU/GLAM permanecen en revisión 1 y no se reconstruyen por este cambio.
- Uvicorn registra el primer `/health`, el primer fallo y la primera recuperación; los `200 OK` repetidos del mismo estado se suprimen. Otros access logs no se filtran.
- `VERSION` queda copiado dentro de `/app/VERSION` y `/runner/VERSION`; `/health` y `/doctor` usan la versión real `0.16.0`.
- `.env.example` conserva exactamente la configuración de tres GPUs validada en la RTX 5060 Ti.
- Este README centraliza los comandos Docker usados y documenta efectos/expectativas, incluyendo la semántica no destructiva de `prepare`.


## v0.17 — validación GPU integrada y parametrizada

Se agrega un único comando para validar la revisión GPU vigente de uno, varios o todos los modelos:

```bash
docker compose exec fastapi \
  python -m model_tools.validate_gpu \
  --models all
```

También puede ejecutarse un subconjunto:

```bash
docker compose exec fastapi \
  python -m model_tools.validate_gpu \
  --models gmic nyu
```

El flujo siempre ejecuta por fases: (1) `ensure_gpu` de todos los modelos seleccionados; (2) `gpu_probe` de los runtimes que quedaron listos; (3) smoke test upstream de los que pasaron el probe. `ensure_gpu` es idempotente: reconstruye solo cuando falta la imagen o cambia `gpu_compatibility.build_revision`. `--force-rebuild` permite reconstruir explícitamente todos los seleccionados.

Por seguridad, el smoke test integrado exige que cada modelo seleccionado esté configurado con `<MODEL>_DEVICE=gpu`; de lo contrario lo marca `SKIPPED/FAILED` para evitar presentar como validación GPU un smoke test que en realidad se ejecutó en CPU. Cada corrida genera evidencia JSON en `workspace/output/model_validation/`.


## v0.18 — contrato de etiquetas GMIC con datasets reales

La prueba CBIS-DDSM de v0.17 confirmó que GMIC ya supera el `forward()` que fallaba por aritmética de índices. El siguiente fallo ocurrió después de inferencia al intentar copiar `left_benign` desde `cancer_label`.

El batch canónico del prototipo, alineado con el metarepository, conserva `left_malignant` y `right_malignant`. v0.18 no inventa una etiqueta benigna complementaria: el runtime GMIC registra `NaN` únicamente para `benign_label` cuando esa verdad de referencia independiente no está disponible. Los `benign_pred` y `malignant_pred` del modelo no cambian.

Para migrar desde v0.17 no se repite la preparación CBIS-DDSM. Después de reconstruir los servicios de aplicación, ejecute `./scripts/validate-models.sh gmic`; el cambio de `build_revision` reconstruye solo GMIC, hace GPU probe y smoke test.

## v0.19 — corrección del estado de ejecución real

La prueba de 5 estudios CBIS-DDSM en v0.18 demostró que GMIC completaba inferencia, escribía su CSV y generaba 20 artefactos XAI, pero el pipeline lo marcaba como fallido porque la respuesta `/run` terminaba con `status=READY`. La causa era una colisión de metadata: `READY` describe que la imagen del modelo está disponible, mientras que la operación real debe devolver `SUCCESS` después de verificar el CSV.

v0.19 corrige únicamente esa precedencia de estado. No cambia GMIC/NYU/GLAM, checkpoints, pesos, build revisions, datasets ni Soft Voting. Por ello, al migrar desde v0.18 **no es necesario reconstruir modelos ni repetir `download`, `inspect` o `prepare`**; basta reconstruir los servicios de aplicación y repetir el normal test.


## v0.20 — GLAM dataset contract, isolation and stronger integration guards

La prueba real de v0.19 confirmó que GMIC y NYU completan los 5 estudios CBIS-DDSM; GLAM falló únicamente al copiar una etiqueta opcional `left_benign` después del forward. v0.20 aplica a GLAM la misma política científica ya validada para GMIC: la etiqueta maligna continúa siendo obligatoria y una etiqueta benigna independiente ausente se representa como `NaN`, nunca como `1-malignant`. `GLAM build_revision=2` fuerza una única reconstrucción del runtime GLAM.

Antes de empaquetar se revisó además el flujo completo y se corrigieron dos riesgos de integración que todavía no habían producido un traceback: (1) cada modelo usa ahora un directorio `preprocessed/<model>/` separado, evitando que XAI de GMIC sea reportado como XAI de NYU/GLAM; (2) GMIC/GLAM se relacionan con el `study_id` original mediante un `study_key` sanitizado explícito y validado, en lugar de asumir que el orden de `groupby` coincide con el manifest. Se detectan colisiones de IDs sanitizados antes de inferencia. En ejecución por chunks, XAI y métricas de recursos se agregan también al directorio raíz de la corrida.

El Model Runner escribe ahora a stdout eventos operativos de alto valor (`MODEL_RUN_STARTED`, lock GPU, contenedor temporal, comando de modelo, éxito/fallo y métricas), además de conservar `workspace/logs/model_runner.jsonl`. Los healthchecks repetitivos continúan suprimidos por transición de estado.


## v0.21 — sampling reproducible, métricas explicables y conteos inequívocos

v0.20 completó por primera vez 5 estudios CBIS-DDSM reales con GMIC + NYU + GLAM + Soft Voting en aproximadamente 3m52s. Como el comportamiento legacy tomaba los primeros N registros, los cinco casos fueron benignos y Sensitivity/ROC-AUC quedaron correctamente no disponibles. v0.21 incorpora sampling reproducible (`stratified` y `balanced`), conserva los `study_id` seleccionados, explica métricas `null`, registra `processed_studies`/`processed_images`/`overall_elapsed_seconds` y renombra las observaciones del monitor de recursos a `monitoring_samples`. No cambia modelos, pesos, checkpoints, datasets ni Soft Voting.

## v0.22 — análisis de scores, thresholds adaptativos y aislamiento del Final Test Set

La corrida real v0.21 de 10 estudios balanceados completó 40 mamografías en ~7m22s, pero el baseline threshold 0.50 dejó 5/5 malignos como FN y produjo ROC-AUC 0.36. v0.22 no cambia ningún modelo. Añade análisis CPU de scores cacheados, reporta AUC por modelo/correlaciones/distribuciones y reemplaza la grilla experimental fija 0.40-0.60 por cinco quantiles label-independent del Configuration Set para cada peso. El Final Test Set permanece sin inferencia hasta freeze y su cache se reutiliza si `final_evaluation` se ejecuta nuevamente.


## v0.23 — métricas de operación y selección balanceada

v0.23 mantiene intactos GMIC, NYU y GLAM. Añade Specificity, PPV, NPV, FPR, Accuracy y Balanced Accuracy a la evaluación threshold-dependent; corrige `threshold_source` para distinguir análisis diagnóstico de Configuration Set; y reemplaza el selector v0.22 `min FN → Sensitivity → FP` por `ROC-AUC por pesos → Balanced Accuracy por threshold → Sensitivity → Specificity/FP`. Esto evita premiar automáticamente thresholds casi-all-positive. El aislamiento **Configuration Set → freeze → Final Test Set** se conserva sin cambios.

## v0.24 — auditoría de procedencia de scores (CPU, sin reinferencia)

Antes de ejecutar los 105 estudios, audite el run diagnóstico existente:

```bash
docker compose exec fastapi python -m experiments.score_provenance \
  --run-dir /workspace/output/normal_tests/normal-20260815T195006Z
```

La auditoría reconstruye score por vista/mama/estudio, valida lateralidad y compara ROC-AUC breast-level vs study-level sin cambiar modelos, pesos, threshold ni agregación.

### v0.24.2 — compatibilidad con `normal_test` chunked

`score_provenance` descubre automáticamente los dos layouts reales del pipeline: ejecución directa en `<run>/model_batch/` y ejecución con `max_runtime_minutes` en `<run>/chunks/<NNNN>/model_batch/`. En modo chunked combina todos los batches, preserva el orden de NYU mediante `study_order.csv` o el `raw_model_predictions.csv` local al chunk y registra la procedencia exacta en el reporte de auditoría.


### v0.25.0 — diagnóstico de fidelidad de entrada y agregación breast-aware

Antes de ejecutar el Configuration Set completo, v0.25 agrega dos diagnósticos CPU-only:

```bash
docker compose exec fastapi python -m experiments.input_fidelity --run-dir <normal-run>
```

Audita contrato 16-bit PNG, metadata DICOM relevante para presentación y metadata producida por el crop/optimal-center oficial de GMIC/NYU/GLAM (`distance_from_starting_side`, `best_center`). No modifica imágenes ni ejecuta inferencia.

```bash
docker compose exec fastapi python -m experiments.breast_ensemble_analysis --breast-level-scores <score-provenance>/breast_level_scores.csv
```

Compara, sin cambiar producción, la agregación actual `max por modelo -> vote` contra `vote por mama -> max entre mamas`. Ambos resultados son diagnósticos y no elegibles para freeze.

### v0.26.0 — contrafactual dirigido de orientación

v0.25 confirmó 40/40 PNG 16-bit grayscale y 40/40 DICOM sin metadata VOI/Window/Rescale que requiera revisión, pero localizó `distance_from_starting_side != 0` en las cuatro vistas de estudios concretos. El upstream NYU/GMIC documenta esa señal como útil para detectar un posible `horizontal_flip` incorrecto según el dataset.

v0.26 ejecuta una prueba dirigida únicamente sobre estudios con las 4 vistas afectadas. No modifica el dataset ni el run original: crea un batch diagnóstico, invierte solo `horizontal_flip`, reejecuta GMIC/NYU/GLAM para esos estudios y compara la geometría del preprocessing y los scores.

```bash
docker compose exec fastapi python -m experiments.orientation_counterfactual \
  --run-dir /workspace/output/normal_tests/normal-20260815T195006Z
```

La decisión de orientación se basa primero en la evidencia geométrica upstream (`distance_from_starting_side`); cualquier cambio de ROC-AUC se registra solo como impacto secundario/post-hoc y no es elegible para freeze.


## v0.27 automatic orientation resolution

Before classifier inference, v0.27 runs the pinned NYU crop + optimal-center preprocessing only (no classifier) as a label-independent orientation preflight. A study is considered for correction only when all four views have non-zero `distance_from_starting_side`; the study-level `horizontal_flip` toggle is accepted only when the counterfactual makes all four distances zero. Ground truth, model scores and AUC are not used. Evidence is persisted under `orientation_resolution/`. The same fixed rule is applied to Configuration and, only after freeze, Final Test inputs.


## v0.27.1 hotfix

The label-independent NYU `PREPROCESS_ONLY` endpoint now exports the upstream repository root (`/home/bcc/breast_cancer_classifier`) in `PYTHONPATH` before invoking `src/cropping/crop_mammogram.py` and `src/optimal_centers/get_optimal_centers.py` directly. This follows the upstream NYU requirement for individual-script execution and fixes the v0.27.0 `ModuleNotFoundError: No module named 'src'` seen before orientation preprocessing began. No model weights, images, labels, orientation policy, ensemble weights, thresholds, or aggregation rules are changed.

## v0.28 upstream reference runtime validation

After CBIS-DDSM input fidelity, aggregation and orientation diagnostics, v0.28 adds a separate runtime-reproduction gate using the official four-exam sample bundled by the NYU mammography metarepository. Run:

```bash
docker compose exec fastapi python -m experiments.upstream_reference_validation
```

The command runs the existing Blackwell GMIC/NYU/GLAM images on `sample_data/`, computes image/breast ROC-AUC and AUPRC using the metarepository label contract, and compares them with the reproduction references published upstream. It does not use CBIS-DDSM, ensemble weights or thresholds and is not eligible for freeze.


## v0.28.2 GLAM runtime differential

When the official upstream reference validation passes GMIC/NYU but fails GLAM, run `python -m experiments.glam_runtime_differential`. It executes the pinned upstream GLAM PyTorch 1.1 runtime on CPU and the Blackwell PyTorch 2.7/CUDA 12.8 runtime on the same official 4-exam sample, then compares raw image scores, ordering, AUROC and AUPRC. The legacy path changes only the matplotlib backend from TkAgg to Agg for headless execution; it does not alter model architecture, checkpoint or intended inference semantics.


## v0.29.0 — CMMD adapter + adquisición manual estricta

v0.29.0 incorpora `cmmd` como dataset nativo y elimina la adquisición automática de metadata CBIS-DDSM. Los cuatro CSV de CBIS y `CMMD_clinicaldata_revision.xlsx` se colocan manualmente; el proyecto no usa URLs ni `urlopen` para descargar datasets.

Hallazgos de preflight CMMD usados para diseñar el adapter (descarga TCIA auditada el 16-08-2026):

- 1,775 pacientes / 1,775 estudios / 5,202 DICOM.
- 949 pacientes con 2 imágenes y 826 con 4 imágenes.
- `ViewPosition` vacío; CC/MLO se resuelve por `ViewCodeSequence.CodeValue`: `399162004=CC`, `399368009=MLO`.
- `ImageLaterality` resuelve L/R.
- 5,200 imágenes con `BitsStored=8`; dos imágenes de `D1-1343` usan 16 bits y pertenecen al grupo no four-view observado.
- El XLSX tiene 1,872 filas para 1,775 pacientes; 97 IDs tienen dos filas bilaterales y 30 tienen una mama benigna y la otra maligna. A nivel estudio, `MALIGNANT` significa al menos una mama explícitamente maligna.
- Entre los 826 four-view: 81 son D1 y 745 D2. El preflight observado produjo D1=61 benignos, 2 malignos consistentes y 18 bilaterales mixtos; D2=733 malignos consistentes y 12 bilaterales mixtos.

### Política de benchmark CMMD

La clave `cmmd` **no mezcla D1 y D2 como benchmark binario**. D2 es un cohort de malignidad/subtipos y usar los 826 four-view juntos haría que clase y cohort estuvieran fuertemente confundidos. El manifiesto canónico de `cmmd` se limita a **CMMD1/D1, cuatro vistas exactas y labels clínicos explícitos para ambas mamas**. Los D2 four-view se conservan en `cmmd_nonbenchmark_four_view.csv` para análisis posterior de dominio/malignidad, sin entrar al benchmark binario.

Archivos esperados manualmente:

```text
workspace/datasets/raw/cmmd/
├── ... árbol DICOM TCIA ...
└── metadata/
    └── CMMD_clinicaldata_revision.xlsx
```

Inspección limpia:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.inspect \
  --datasets cmmd \
  --force-dicom-index
```

Preparación:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.prepare \
  --datasets cmmd
```

Primera inferencia permitida: diagnóstica, `10` estudios balanceados, `seed=42`; no es elegible para freeze.

## v0.29.1 — comparación de escala multi-dataset

`input_scale_comparison` deja de asumir CBIS-DDSM y detecta `dataset_source` desde el run seleccionado. Permite comparar CMMD (y futuros adapters) contra el sample oficial antes/después del crop NYU sin ground truth ni inferencia. No modifica normalización, pesos, threshold ni datasets.

## v0.29.2 — contrafactual de presentación DICOM

v0.29.2 añade un único diagnóstico final de fidelidad de presentación para los DICOM ya inspeccionados. Compara tres ramas sin usar labels, scores o clasificadores:

1. `current_adapter`: conversión productiva actual a PNG 16-bit.
2. `modality_lut`: aplica Modality LUT/Rescale antes de la presentación.
3. `voi_presentation`: además aplica VOI LUT o `WindowCenter`/`WindowWidth` con `VOILUTFunction` cuando exista.

Ejecutar sobre un run diagnóstico existente:

```bash
./scripts/audit-dicom-presentation.sh \
  /workspace/output/normal_tests/normal-20260816T054908Z
```

El comando escribe `dicom_presentation_report.md`, `dicom_presentation_summary.json` y CSVs bajo `workspace/output/analyses/dicom-presentation-<timestamp>/`. Por defecto **no persiste copias transformadas de las imágenes**; `--write-images` es opcional para inspección visual. El resultado no puede usarse para elegir una rama por AUC ni para congelar pesos/thresholds.

Este release también alinea la metadata de versión (`VERSION`, `pyproject.toml`, `mammography_agent.__version__` y Model Runner API) en `0.29.2`.

