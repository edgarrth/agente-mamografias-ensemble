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
| `docker compose exec fastapi cat /app/VERSION` | Verifica versión del código dentro de FastAPI. | Nada. | `0.18.0`. |
| `docker compose exec model-runner cat /runner/VERSION` | Verifica versión del Model Runner. | Nada. | `0.18.0`. |
| `docker compose exec model-runner docker version` | Verifica cliente/daemon Docker desde el Runner. | Nada. | Client/Server accesibles. |
| `docker compose exec model-runner docker info` | Diagnóstico detallado del daemon desde el Runner. | Nada. | Información del Engine sin error. |
| `docker compose exec fastapi python -m model_tools.status` | Estado de imágenes, perfiles GPU y device por modelo. | Nada. | GMIC/NYU/GLAM `device=gpu` en workstation validada. |
| `docker compose exec fastapi python -m model_tools.ensure --models gmic nyu glam` | Construye/reutiliza imágenes legacy `:research`. | Imágenes Docker; no cambia pesos/arquitectura. | `READY`. |
| `docker compose exec fastapi python -m model_tools.ensure_gpu --models gmic` | Construye/reutiliza runtime Blackwell de GMIC. v0.16 detecta `build_revision=2` y reconstruye GMIC una vez para aplicar el fix de índices. | Imagen `mammography-model-gmic:blackwell-cu128`; invalida el probe GPU anterior solo si reconstruye. | `READY`, `build_revision=2`. |
| `docker compose exec fastapi python -m model_tools.gpu_probe --models gmic` | Prueba asignación y kernel CUDA del runtime reconstruido. | Solo actualiza evidencia `workspace/models/gpu_compatibility/gmic.probe.json`. | `GPU_READY`. |
| `docker compose exec fastapi python -m model_tools.smoke_test --models gmic nyu glam` | Prueba los modelos con sample data upstream. | `workspace/output/smoke_tests/`; no toca datasets raw. | `READY` por modelo. |
| `docker compose exec fastapi python -m model_tools.validate_gpu --models all` | Orquestación de release/validación: asegura la revisión GPU configurada de GMIC+NYU+GLAM, ejecuta `gpu_probe` de todos y luego smoke test de todos. También acepta uno o más modelos (`--models gmic nyu`). | Puede reconstruir solo las imágenes cuyo `build_revision` cambió; renueva sus probes y escribe evidencia JSON en `workspace/output/model_validation/`; no modifica datasets. | `overall_status=READY` y `PASS` en `ensure_gpu`, `gpu_probe` y `smoke_test` por modelo. |
| `docker compose exec fastapi python -m model_tools.validate_gpu --models all --force-rebuild` | Igual que el anterior, pero fuerza reconstrucción de todas las imágenes GPU seleccionadas aunque su revisión ya coincida. Úselo solo cuando se quiera validar bytes nuevos explícitamente. | Rebuild de imágenes seleccionadas, invalida/renueva probes y escribe reporte; no toca datasets. | `rebuild_performed=true` por modelo y validación completa. |
| `./scripts/validate-models.sh all` o `./scripts/validate-models.sh gmic nyu` | Wrapper host del comando integrado anterior; evita escribir el `docker compose exec ...` completo y acepta uno o más modelos. | Los mismos efectos de `model_tools.validate_gpu`; no modifica datasets. | Mismo JSON/resumen de validación. |
| `docker compose exec fastapi python -m dataset_pipeline.status` | Consulta estado de datasets. | Nada. | `AVAILABLE`, `READY_FOR_INSPECT`, etc. |
| `docker compose exec fastapi python -m dataset_pipeline.download --datasets cbis_ddsm` | Reutiliza DICOM existentes y descarga/verifica solo metadata oficial pequeña si falta. | Puede crear CSV en `raw/cbis_ddsm/metadata`; **nunca re-descarga los DICOM**. | `READY_FOR_INSPECT`, `dicom_reused=true`. |
| `docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cbis_ddsm` | Cruza metadata, reutiliza índice DICOM y construye catálogos/manifiesto de estudios completos. | `manifests/`, `rejected/`, `source_manifest.csv`, cache de índice; no modifica pixels raw. | Conteos de pacientes/vistas y `ensemble_compatible`. |
| `docker compose exec fastapi python -m dataset_pipeline.inspect --datasets cbis_ddsm --force-dicom-index` | Igual que `inspect`, pero reconstruye headers DICOM. | Reescribe cache del índice; no modifica DICOM. | Mucho más lento; usar solo si cambió el árbol raw. |
| `docker compose exec fastapi python -m dataset_pipeline.prepare --datasets cbis_ddsm` | Convierte **solo estudios de 4 vistas compatibles** a PNG 16-bit y escribe manifiesto canónico. | Escribe/regenera `processed/cbis_ddsm/images/*.png` y `manifests/cbis_ddsm.csv`; **no limpia, borra ni modifica DICOM raw**. No elimina derivados antiguos no referenciados. | `AVAILABLE`, `converted_studies=...`. |
| `docker compose exec fastapi python -m tests_flow.normal --datasets cbis_ddsm --samples 5 --max-runtime-minutes 120` | Prueba end-to-end dataset→3 modelos→Soft Voting. | Solo `workspace/output/normal_tests/<run>/` y logs; no modifica dataset preparado. | `NORMAL_TEST_COMPLETED` y `predictions.csv`, o fallo explícito del modelo. |
| `docker compose exec fastapi python -m experiments.run ...` | Fase de configuración experimental; ejecuta modelos una vez y evalúa 80 combinaciones CPU. | `workspace/output/experiments/`; no modifica dataset. | ranking + best configuration. |
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

Baseline:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 50
```

Pesos manuales:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 50 \
  --weights 0.50 0.30 0.20 \
  --threshold 0.45
```

Configuración congelada de un experimento anterior:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 50 \
  --config /workspace/output/experiments/<ID>/frozen_configuration.yaml
```

Prueba piloto con máximo aproximado de 120 minutos:

```bash
docker compose exec fastapi python -m tests_flow.normal \
  --datasets cbis_ddsm \
  --samples 500 \
  --max-runtime-minutes 120
```

El límite de tiempo solo se evalúa entre lotes completos. No mata una inferencia en curso. Si se alcanza, se guardan los resultados terminados y el estado queda `PARTIAL_TIME_LIMIT`.

## 8. Prueba experimental

```bash
docker compose exec fastapi python -m experiments.run \
  --datasets cbis_ddsm \
  --samples 200 \
  --configuration-ratio 0.30 \
  --seed 42
```

El flujo:

1. separa por paciente `Configuration Set` y `Final Test Set`;
2. ejecuta GMIC + NYU + GLAM **una sola vez por estudio**;
3. guarda los tres scores;
4. evalúa 16 configuraciones de pesos × 5 thresholds = 80 combinaciones sobre el Configuration Set;
5. selecciona una configuración usando ROC-AUC y luego Sensitivity/FN/FP;
6. **no ejecuta todavía el Final Test Set**.

Congelar la configuración seleccionada:

```bash
docker compose exec fastapi python -m experiments.freeze --experiment <ID>
```

Solo después del freeze, ejecutar la evaluación final:

```bash
docker compose exec fastapi python -m experiments.final_evaluation --experiment <ID>
```

La evaluación final ejecuta los tres modelos una sola vez sobre cada estudio reservado y compara la configuración congelada contra el baseline, sin reoptimizar.

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

La política de adquisición queda separada:

```text
dataset_pipeline.download --datasets cbis_ddsm
        │
        ├── DICOM (~163 GB) -> NUNCA auto-download
        │                     ├── si existe: REUSE
        │                     └── si falta: DICOM_DOWNLOAD_REQUIRED
        │
        └── 4 case-description CSV
                              ├── si existen y son válidos: REUSE
                              └── si faltan: descarga TCIA + SHA-256 + validación de columnas
```

Por tanto, ejecutar nuevamente:

```bash
docker compose exec fastapi \
  python -m dataset_pipeline.download \
  --datasets cbis_ddsm
```

**no vuelve a descargar la colección DICOM**. Con un workspace ya poblado, solo verifica/reutiliza los DICOM y los cuatro CSV. Si falta un CSV pequeño, descarga únicamente ese metadata.

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
