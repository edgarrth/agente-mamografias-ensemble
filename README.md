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

Las imágenes/entornos oficiales de estos modelos usan versiones antiguas de CUDA/PyTorch. La selección de dispositivo es una decisión de despliegue **por modelo**, mientras que el perfil técnico GPU pertenece a `config/models.yaml`. El prototipo inicia de forma conservadora con:

```env
DEFAULT_MODEL_DEVICE=cpu
GMIC_DEVICE=cpu
NYU_DEVICE=cpu
GLAM_DEVICE=cpu
ALLOW_GPU=false
```

Un modelo solo puede pasar a GPU cuando tiene un `gpu_compatibility.profile` propio en `config/models.yaml` y ese runtime ha pasado `gpu_probe`. No existe un `GPU_RUNTIME_PROFILE` global y la imagen legacy `:research` nunca se selecciona para GPU.

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

Para mantener el prototipo simple y auditable, cada adapter acepta un `source_manifest.csv` dentro de la carpeta raw del dataset. Este manifiesto vincula el ground truth autorizado con las cuatro vistas. No se infieren etiquetas a partir de BI-RADS.

Columnas mínimas:

```text
study_id,patient_id,ground_truth,l_cc,r_cc,l_mlo,r_mlo
```

Opcionales:

```text
left_ground_truth,right_ground_truth,horizontal_flip
```

Las rutas pueden apuntar a DICOM o PNG dentro de `/workspace`. La preparación convierte DICOM a PNG de 16 bits y genera el manifiesto canónico.

```bash
docker compose exec fastapi python -m dataset_pipeline.prepare --datasets cbis_ddsm
```

```bash
docker compose exec fastapi python -m dataset_pipeline.prepare --datasets all
```

Si falta el `source_manifest.csv`, el adapter falla explícitamente y deja instrucciones; no inventa ground truth.

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
