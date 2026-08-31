# Fusión de las ramas `web` y `batch` — versión 1.0.0

Este árbol unifica las dos ramas del proyecto en un único proyecto ejecutable.
Sustituye a las releases `web-v1.0.0` (código 0.36.0) y `batch-v1.0.0` (código 0.31.1).

## Punto de partida

La rama `web` es la línea más reciente (0.36.0) y aporta la mayor parte del código:
la interfaz de caso único, la persistencia en PostgreSQL y MinIO, el despliegue de
producción y la selección de dispositivo de inferencia.

La rama `batch` (0.31.1) conserva, en cambio, el espacio de búsqueda realmente
utilizado en la evaluación formal y el subsistema de limpieza de temporales.

## Conflictos resueltos y criterio aplicado

### Espacio de búsqueda: prevalece `batch`

`config/experiments.yaml` divergía entre ramas. La rama `web` conservaba la grilla
anterior de **16 pesos × 5 cuantiles = 80 candidatos**; la rama `batch` tiene la de
**40 pesos × 17 cuantiles = 680 candidatos**, que es la utilizada en la evaluación
formal y la que se documenta en la tesis. El identificador `W17`, correspondiente a
la configuración congelada, no existía en el archivo de la rama web.

Se adopta la de `batch`. Con ella se incorporan también los módulos que la validan
de forma generalizada:

- `mammography_agent/ensemble/experiment.py` — la versión de `web` comprobaba
  `len(out) != 80`; la de `batch` calcula el total esperado a partir de la
  configuración.
- `mammography_agent/score_analysis.py` — la versión de `web` exigía exactamente
  cinco cuantiles; la de `batch` acepta cualquier número.
- Las cinco pruebas asociadas, que en `web` afirmaban 80 configuraciones y en
  `batch` afirman 680.

Sin este criterio, el proyecto fusionado habría fallado al arrancar con la grilla
real.

### Archivos exclusivos de `batch`

Se incorporan cinco que la rama `web` no tenía:

- `mammography_agent/ensemble/cv_selection.py` — validación cruzada estratificada
- `experiments/reselect_configuration.py`
- `experiments/cleanup_formal_temporaries.py`
- `tests/test_expanded_cv_selection_v0310.py`
- `tests/test_formal_cleanup_v0311.py`

### `pipeline.py`: fusión manual

Las dos ramas modificaron el mismo archivo en direcciones distintas.

De `web` se conservan la selección de dispositivo, los callbacks de progreso por
modelo y por etapa, y la compatibilidad de caso único.

De `batch` se restituyen tres funciones que `web` había eliminado, junto con sus dos
sitios de llamada y las dos claves de política que las gobiernan:

- `_evenly_spaced`
- `_retain_xai_and_cleanup_model_batch`
- `_cleanup_orientation_temporaries`
- `cleanup_successful_chunk_temporaries` y `xai_retention_per_model_per_chunk`

Sin ellas, una ejecución masiva conserva todos los temporales por bloque, lo que en
el run de 334 bloques significó cientos de gigabytes de datos derivados.

### Resto de archivos

En los demás casos la rama `web` es un superconjunto y prevalece: `.env.example`
(añade MinIO y las variables del ámbito web), `docker-compose.yml` (añade el volumen
`web_scratch`), `config.py`, `metarepo_format.py`, `logging_utils.py`,
`orientation_policy.py`, `api.py`, `storage.py`, `graph.py`, `ui/streamlit_app.py`,
`model_runner/api.py`, `requirements.txt` y `pyproject.toml`.

## Versión

Se unifica en **1.0.0** en `VERSION`, `pyproject.toml`, `mammography_agent/__init__.py`,
`model_runner/api.py`, los sellos de `chunk_status.json` y la prueba de exposición de
versión. Las ramas traían 0.36.0 y 0.31.1, que ya no aplican al árbol unificado.

## Comprobaciones realizadas

- Todos los archivos `.py` compilan.
- `config/experiments.yaml` produce 40 pesos y 17 cuantiles; los pesos suman 1 y
  ninguno baja de 0.10; los cuantiles están ordenados.
- `all_configurations` devuelve 680 filas con 40 pesos y 17 umbrales.
- `select_configuration` opera sobre la grilla completa.
- `pipeline.py` no contiene definiciones duplicadas y conserva las cuatro funciones
  relevantes de ambas ramas.
- Finales de línea normalizados a LF.

## Pendiente de decisión

1. **Punto de operación de la interfaz web.** Arranca en la línea base uniforme
   (0.333 / 0.333 / 0.333, umbral 0.50), cuya sensibilidad sobre el conjunto
   reservado es de 13.9 %. La configuración congelada es W17/T05, con umbral
   0.040985. Conviene alinear el valor por defecto antes de publicar.
2. **`config/hipotesis-config-web`.** Nota suelta con pesos 0.30 / 0.20 / 0.50 y
   umbrales 0.30 y 0.02 que no corresponden a ninguna configuración congelada.
   Conviene eliminarla o documentar a qué pertenece.
3. **README.** Se conserva el de `web` (81 KB). El de `batch` (71 KB) documenta la
   ejecución masiva; hay que fundirlos o remitir explícitamente de uno a otro.

## Cómo validar la fusión

El árbol incluye `scripts/validate_merge.py`. Reproduce la selección de
configuración **sin ejecutar ningún modelo**, porque opera sobre los puntajes ya
persistidos del run:

```bash
python scripts/validate_merge.py ruta/a/configuration_set_predictions.csv
```

Comprueba tres cosas: que el espacio de búsqueda declarado es el de 40 × 17, que
las funciones restituidas de la rama batch están presentes, y que el código
fusionado vuelve a elegir W17/T05 con las mismas métricas informadas en la tesis.

Ejecutado sobre los artefactos del run `experiment-20260818T035300Z`, las
dieciocho comprobaciones pasan: 3,570 estudios con 144 malignos, 680 candidatos,
3,400 evaluaciones candidato-partición, y W17/T05 con ROC-AUC 0.7242, exactitud
balanceada 0.6736, sensibilidad 0.8340 y especificidad 0.5131.

## Publicación de artefactos al control de versiones

`workspace/` es directorio de trabajo y no debe versionarse: contiene cientos de
gigabytes de derivados. Para llevar al repositorio solo los artefactos que
sustentan las cifras de la tesis:

```bash
python scripts/publish_run_artifacts.py experiment-20260818T035300Z            # simulación
python scripts/publish_run_artifacts.py experiment-20260818T035300Z --apply    # ejecutar
python scripts/publish_run_artifacts.py experiment-20260818T035300Z --verify-only
```

Excluye directorios de trabajo (`model_batch`, `preprocessed`, `images`,
`xai_retained`), formatos derivados de píxeles o pesos de modelo, y cualquier
archivo mayor de 20 MB. Genera `SHA256SUMS` y verifica los veinticuatro
resúmenes que el Anexo 1 declara: si alguno no coincide, la copia no corresponde
al run congelado y el guion termina con código de error.

Sobre el run `experiment-20260818T035300Z` publica 68 archivos y 33,176,975
bytes, que son exactamente las cifras que el Anexo 1 informa.
