# Web evaluation v0.31.0

La ruta Web ejecuta inferencia unitaria sobre un estudio DICOM de cuatro proyecciones. No recibe etiquetas diagnósticas y no ejecuta entrenamiento.

## Pesos del ensemble

La configuración base continúa en `config/ensemble.yaml`. La interfaz puede enviar un override por caso con las claves `gmic`, `nyu` y `glam`; cada valor debe encontrarse entre 0 y 1 y la suma debe ser 1. El override no modifica archivos de configuración y no es utilizado por `experiments.run`, `experiments.freeze` ni `experiments.final_evaluation`.

`config/experiments.yaml` conserva W01-W16 para el flujo experimental masivo. La Web no importa ni escribe ese archivo.

## Umbral

El umbral Web permanece definido por `config/ensemble.yaml -> baseline.threshold` y se presenta como valor de solo lectura. La versión 0.31.0 separa deliberadamente el análisis de pesos de la selección del operating point.

## Precondición GPU

Cuando un modelo está configurado con `*_DEVICE=gpu`, la interfaz requiere que la imagen GPU correspondiente exista y que el probe de compatibilidad esté aprobado. La validación recomendada es:

```bash
docker compose exec fastapi python -m model_tools.validate_gpu --models all
```

La evaluación se habilita después de obtener `overall_status=READY` y actualizar el estado en Streamlit.

## Tiempo de ejecución

El resultado registra el tiempo de inferencia/preparación, el tiempo total de la ruta Web y las métricas de tiempo reportadas por cada Model Runner. Si una ejecución se interrumpe antes de producir resultado, Streamlit muestra el tiempo transcurrido hasta el error.
