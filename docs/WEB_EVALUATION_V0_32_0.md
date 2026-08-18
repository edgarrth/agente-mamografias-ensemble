# Web evaluation v0.32.0

## Alcance

La versión 0.32.0 separa explícitamente la configuración de inferencia Web de la configuración utilizada por los procesos batch. La interfaz agrupa dispositivo, pesos, disponibilidad de modelos y persistencia en **Configuración y estado**.

## Dispositivo de inferencia

- `WEB_INFERENCE_DEVICE=cpu` define el valor inicial de la interfaz.
- La UI permite seleccionar `CPU` o `GPU`.
- `/single-cases/run` recibe `inference_device` por petición.
- CPU no requiere `gpu_probe`.
- GPU conserva `ensure_gpu_image` y `gpu_probe` del Model Runner.
- La UI no modifica `GMIC_DEVICE`, `NYU_DEVICE`, `GLAM_DEVICE` ni `config/models.yaml`.

## Aislamiento respecto del batch

`pipeline._infer_three()` acepta `device=None`. Los entrypoints batch existentes siguen invocándolo sin override. En ese caso, `run_model()` se llama con la firma histórica y el Model Runner resuelve el dispositivo mediante su configuración actual. La ruta Web es la única que suministra un override explícito.

Los pesos Web siguen siendo overrides por petición y no escriben `config/ensemble.yaml` ni `config/experiments.yaml`.

## Persistencia

El resultado registra `inference_device` junto con los pesos, scores, tiempos, clasificación y referencias de MinIO. PostgreSQL añade la columna `inference_device` únicamente en `web_inference_runs`; `research_runs` permanece sin cambios.
