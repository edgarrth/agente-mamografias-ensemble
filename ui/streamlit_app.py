from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components

from mammography_agent.streamlit_compat import image_fill_width_kwargs

API = os.getenv("API_URL", "http://fastapi:8000").rstrip("/")
WEB_SCRATCH_ROOT = Path(os.getenv("WEB_SCRATCH_ROOT", "/web-scratch")).resolve()
WEB_SCRATCH_TTL_MINUTES = int(os.getenv("WEB_SCRATCH_TTL_MINUTES", "60"))
RUN_TIMEOUT_SECONDS = 60 * 60 * 24
REQUIRED_VIEWS = ("L_CC", "R_CC", "L_MLO", "R_MLO")
DEFAULT_WEB_DEVICE = str(os.getenv("WEB_INFERENCE_DEVICE", "cpu")).strip().lower()
if DEFAULT_WEB_DEVICE not in {"cpu", "gpu"}:
    DEFAULT_WEB_DEVICE = "cpu"

st.set_page_config(
    page_title="Evaluación mamográfica",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1380px; padding-top: 1.35rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(127,127,127,.20); border-radius: 18px;
       margin-bottom: 1rem; background: linear-gradient(135deg, rgba(88,101,242,.09), rgba(127,127,127,.02));}
.hero h1 {margin: 0; font-size: 2.05rem; line-height: 1.15;}
.hero p {margin: .6rem 0 0 0; opacity: .78; max-width: 920px;}
.note {padding:.9rem 1rem; border:1px solid rgba(127,127,127,.18); border-radius:12px; margin:.45rem 0 1rem 0;}
.method-card {padding: .9rem 1rem; border: 1px solid rgba(127,127,127,.18); border-radius: 12px; min-height: 112px;}
.method-card strong {display:block; margin-bottom:.35rem;}
.method-card span {opacity:.72; font-size:.88rem; line-height:1.35;}
.preview-title {font-weight: 650; margin: .15rem 0 .3rem 0;}
.preview-meta {opacity:.74; font-size:.86rem; margin-bottom:.45rem;}
.small-muted {opacity:.70; font-size:.84rem;}
/* The hosted Streamlit deployment control is not part of the research application UI. */
[data-testid="stToolbar"] {display:none !important;}
[data-testid="stDecoration"] {display:none !important;}
[data-testid="stStatusWidget"] button {cursor: default;}
</style>
""",
    unsafe_allow_html=True,
)


def _api_json(method: str, path: str, *, timeout: int = 30, **kwargs) -> Any:
    response = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
    if not response.ok:
        try:
            body = response.json()
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except Exception:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def _safe_name(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return cleaned[:150] or fallback


def _display_name(staged_name: str) -> str:
    return re.sub(r"^\d{2}_", "", str(staged_name or ""), count=1)


def _upload_signature(files: list[Any]) -> str:
    h = hashlib.sha256()
    for uploaded in files:
        raw = uploaded.getvalue()
        h.update(uploaded.name.encode("utf-8", errors="ignore"))
        h.update(len(raw).to_bytes(8, "big"))
        h.update(hashlib.sha256(raw).digest())
    return h.hexdigest()




def _purge_stale_web_uploads() -> None:
    cutoff = time.time() - max(1, WEB_SCRATCH_TTL_MINUTES) * 60
    for root in (WEB_SCRATCH_ROOT / "uploads", WEB_SCRATCH_ROOT / "previews"):
        if not root.exists():
            continue
        for child in list(root.iterdir()):
            try:
                if child.stat().st_mtime >= cutoff:
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception:
                pass

def _stage_uploads(files: list[Any]) -> list[str]:
    _purge_stale_web_uploads()
    signature = _upload_signature(files)
    cached = st.session_state.get("web_staged_upload")
    if cached and cached.get("signature") == signature:
        paths = cached.get("paths") or []
        if paths and all(Path(p).is_file() for p in paths):
            return list(paths)

    session_id = st.session_state.setdefault("web_session_id", uuid.uuid4().hex[:10])
    stage = WEB_SCRATCH_ROOT / "uploads" / session_id / signature[:16]
    stage.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, uploaded in enumerate(files, start=1):
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in {".dcm", ".dicom"}:
            suffix = ".dcm"
        name = _safe_name(uploaded.name, f"image_{idx:02d}{suffix}")
        target = stage / f"{idx:02d}_{name}"
        target.write_bytes(uploaded.getvalue())
        paths.append(str(target))
    st.session_state["web_staged_upload"] = {"signature": signature, "paths": paths}
    return paths


def _status_payload() -> dict[str, Any] | None:
    try:
        return _api_json("GET", "/workspace/status", timeout=20)
    except Exception:
        return None


def _storage_payload() -> dict[str, Any] | None:
    try:
        return _api_json("GET", "/single-cases/storage-status", timeout=20)
    except Exception:
        return None


def _scroll_to_progress() -> None:
    components.html(
        """
        <script>
        setTimeout(() => {
          const el = window.parent.document.getElementById('web-evaluation-progress');
          if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
        }, 120);
        </script>
        """,
        height=0,
    )


def _progress_states(payload: dict[str, Any] | None) -> dict[str, str]:
    payload = payload or {}
    explicit = payload.get("stages") or {}
    keys = ["PREPARATION", "ORIENTATION", "MODEL_INPUT_PREPARATION", "ENSEMBLE", "PERSISTENCE"]
    if explicit:
        return {key: str((explicit.get(key) or {}).get("state") or "PENDING") for key in keys}

    # Backward-compatible fallback for progress payloads generated by earlier Web versions.
    stage = str(payload.get("stage") or "QUEUED").upper()
    order = ["PREPARATION", "ORIENTATION", "MODEL_INPUT_PREPARATION", "MODELS", "ENSEMBLE", "PERSISTENCE", "COMPLETED"]
    states = {key: "PENDING" for key in keys}
    if stage == "COMPLETED":
        return {key: "SUCCESS" for key in states}
    if stage == "FAILED":
        return states
    if stage in order:
        idx = order.index(stage)
        for key in keys:
            if key in order and order.index(key) < idx:
                states[key] = "SUCCESS"
        if stage in states:
            states[stage] = "RUNNING"
    return states


def _progress_text(label: str, state: str, elapsed: float | None = None) -> str:
    icon = {"SUCCESS": "✅", "RUNNING": "⏳", "FAILED": "❌", "PENDING": "○"}.get(state, "○")
    suffix = f" · {elapsed:.1f} s" if elapsed is not None and elapsed >= 0 else ""
    return f"{icon} **{label}**{suffix}"


def _stage_elapsed(payload: dict[str, Any], key: str) -> float | None:
    value = ((payload.get("stages") or {}).get(key) or {}).get("elapsed_seconds")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _render_live_progress(payload: dict[str, Any] | None, slots: dict[str, Any], elapsed: float) -> None:
    payload = payload or {}
    stages = _progress_states(payload)
    overall_state = str(payload.get("state") or "RUNNING").upper()
    overall_stage = str(payload.get("stage") or "").upper()
    if overall_stage == "COMPLETED" or overall_state == "SUCCESS":
        heading = f"**Evaluación completada · {elapsed:.1f} s**"
    elif overall_stage == "FAILED" or overall_state == "FAILED":
        heading = f"**Evaluación interrumpida · {elapsed:.1f} s**"
    else:
        heading = f"**Evaluación en curso · {elapsed:.1f} s**"
    message = payload.get("message") or "Inicializando la evaluación."
    slots["summary"].markdown(f"{heading}  \n{message}")

    slots["preparation"].markdown(_progress_text(
        "Preparación del estudio", stages.get("PREPARATION", "PENDING"), _stage_elapsed(payload, "PREPARATION")
    ))
    slots["orientation"].markdown(_progress_text(
        "Normalización de orientación", stages.get("ORIENTATION", "PENDING"), _stage_elapsed(payload, "ORIENTATION")
    ))
    slots["model_input"].markdown(_progress_text(
        "Preparación de entradas para modelos", stages.get("MODEL_INPUT_PREPARATION", "PENDING"),
        _stage_elapsed(payload, "MODEL_INPUT_PREPARATION")
    ))
    models = payload.get("models") or {}
    for key, label in (("gmic", "GMIC"), ("nyu", "NYU / DMV-CNN"), ("glam", "GLAM")):
        item = models.get(key) or {}
        slots[key].markdown(_progress_text(label, str(item.get("state") or "PENDING"), item.get("elapsed_seconds")))
    slots["ensemble"].markdown(_progress_text(
        "Integración ponderada del ensemble", stages.get("ENSEMBLE", "PENDING"), _stage_elapsed(payload, "ENSEMBLE")
    ))
    slots["persistence"].markdown(_progress_text(
        "Registro de resultados", stages.get("PERSISTENCE", "PENDING"), _stage_elapsed(payload, "PERSISTENCE")
    ))


def _queue_web_evaluation() -> None:
    st.session_state["web_eval_running"] = True
    st.session_state["web_eval_pending"] = True
    st.session_state.pop("last_web_result", None)
    st.session_state.pop("last_web_progress", None)
    st.session_state.pop("last_web_progress_elapsed", None)
    st.session_state.pop("last_web_error", None)

def _ensemble_config_payload() -> dict[str, Any] | None:
    try:
        return _api_json("GET", "/single-cases/ensemble-config", timeout=20)
    except Exception:
        return None


def _web_settings_payload(*, attempts: int = 2) -> tuple[dict[str, Any] | None, str | None]:
    """Load the persisted Web settings without confusing transport failures with baseline settings.

    A failed GET must never be interpreted as "no persisted settings" because doing so can
    overwrite a valid PostgreSQL configuration with UI defaults during Streamlit startup.
    """
    last_error: str | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            return _api_json("GET", "/single-cases/web-settings", timeout=20), None
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(0.25)
    return None, last_error or "No fue posible recuperar la configuración Web guardada."


def _persist_web_settings(*, device: str, weight_mode: str, weights: dict[str, float], threshold_mode: str, threshold: float) -> dict[str, Any] | None:
    payload = {
        "inference_device": str(device).lower(),
        "weight_mode": "CUSTOM" if weight_mode == "Configuración personalizada" else "BASELINE",
        "ensemble_weights": {k: float(v) for k, v in weights.items()},
        "threshold_mode": "CUSTOM" if threshold_mode == "Configuración personalizada" else "BASELINE",
        "decision_threshold": float(threshold),
    }
    return _api_json("PUT", "/single-cases/web-settings", timeout=20, json=payload)


def _hydrate_web_settings(settings: dict[str, Any] | None, ensemble_config: dict[str, Any] | None) -> bool:
    if st.session_state.get("_web_settings_hydrated"):
        return True
    # Hydration is allowed only after a successful API response. A transport/API failure is
    # not equivalent to baseline configuration and must never trigger a settings write.
    if settings is None:
        return False
    baseline_weights = ((ensemble_config or {}).get("weights") or {"gmic": 0.333333, "nyu": 0.333333, "glam": 0.333334})
    baseline_threshold = float((ensemble_config or {}).get("threshold", 0.50))
    persisted = settings
    weights = persisted.get("weights") or baseline_weights
    device = str(persisted.get("inference_device") or DEFAULT_WEB_DEVICE).lower()
    if device not in {"cpu", "gpu"}:
        device = DEFAULT_WEB_DEVICE
    st.session_state["web_inference_device"] = device
    st.session_state["web_weight_mode"] = (
        "Configuración personalizada" if str(persisted.get("weight_mode") or "BASELINE").upper() == "CUSTOM" else "Configuración base"
    )
    st.session_state["web_weight_gmic"] = float(weights.get("gmic", baseline_weights.get("gmic", 0.333333)))
    st.session_state["web_weight_nyu"] = float(weights.get("nyu", baseline_weights.get("nyu", 0.333333)))
    st.session_state["web_weight_glam"] = float(weights.get("glam", baseline_weights.get("glam", 0.333334)))
    st.session_state["web_threshold_mode"] = (
        "Configuración personalizada" if str(persisted.get("threshold_mode") or "BASELINE").upper() == "CUSTOM" else "Configuración base"
    )
    st.session_state["web_decision_threshold"] = float(persisted.get("decision_threshold", baseline_threshold))
    st.session_state["_web_settings_persisted_at"] = persisted.get("updated_at")
    st.session_state["_web_settings_hydrated"] = True
    return True


def _model_runtime_ready(model: dict[str, Any], requested_device: str | None = None) -> tuple[bool, str]:
    if not bool(model.get("reachable", False)):
        return False, "Servicio no disponible"
    device = str(requested_device or model.get("device") or "cpu").lower()
    if device == "gpu":
        if not bool(model.get("gpu_built", False)):
            return False, "Runtime GPU no preparado"
        if not bool(model.get("gpu_probe_passed", False)):
            return False, "Validación GPU pendiente"
    status = str(model.get("status", model.get("runtime_status", ""))).upper()
    if status in {"FAILED", "ERROR", "UNAVAILABLE"}:
        return False, "Runtime no disponible"
    return True, "Disponible"


def _all_models_ready(status_payload: dict[str, Any] | None, requested_device: str | None = None) -> bool:
    models = (status_payload or {}).get("models") or []
    return len(models) == 3 and all(_model_runtime_ready(model, requested_device)[0] for model in models)


def _model_badge(model: dict[str, Any], requested_device: str | None = None) -> tuple[str, str]:
    ready, label = _model_runtime_ready(model, requested_device)
    return ("●", label) if ready else ("○", label)

def _source_label(source: str | None) -> str:
    value = str(source or "")
    if not value or value == "unresolved":
        return "No disponible"
    labels = {
        "ViewPosition": "View Position",
        "SeriesDescription": "Series Description",
        "ProtocolName": "Protocol Name",
        "RequestedProcedureDescription": "Requested Procedure",
        "StudyDescription": "Study Description",
        "ImageComments": "Image Comments",
        "ImageType": "Image Type",
        "ViewCodeSequence:CodeMeaning": "View Code Sequence",
    }
    if value.startswith("ViewCodeSequence:"):
        return "View Code Sequence"
    if value.startswith("PerformedProtocolCodeSequence:"):
        return "Performed Protocol Code"
    if value.startswith("ProcedureCodeSequence:"):
        return "Procedure Code"
    return labels.get(value, value)


def _render_detection_table(inspection: dict[str, Any]) -> None:
    rows = []
    for item in inspection.get("files", []):
        assignment = item.get("assigned_view")
        rows.append({
            "Archivo": _display_name(item.get("name") or ""),
            "Lateralidad": item.get("laterality") or "No determinada",
            "Proyección": item.get("view") or "No determinada",
            "Asignación": assignment.replace("_", "-") if assignment else "Pendiente",
            "Estado": "Seleccionada" if item.get("selected") else "Pendiente",
            "Fuente": _source_label(item.get("view_source")),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _manual_options(item: dict[str, Any]) -> tuple[list[str], int]:
    side = str(item.get("laterality") or "").upper()
    detected = str(item.get("detected_view") or "")
    if side in {"L", "R"}:
        options = ["AUTO", "CC", "MLO", "IGNORE"]
        projection = detected.split("_", 1)[1] if detected.startswith(f"{side}_") else ""
        return options, options.index(projection) if projection in options else 0
    options = ["AUTO", *REQUIRED_VIEWS, "IGNORE"]
    return options, options.index(detected) if detected in options else 0


def _manual_override(item: dict[str, Any], choice: str) -> str | None:
    if choice == "AUTO":
        return None
    if choice == "IGNORE":
        return "IGNORE"
    side = str(item.get("laterality") or "").upper()
    if choice in {"CC", "MLO"} and side in {"L", "R"}:
        return f"{side}_{choice}"
    return choice


def _choice_label(value: str) -> str:
    return {
        "AUTO": "Automática",
        "IGNORE": "Omitir archivo",
        "CC": "CC · cráneo-caudal",
        "MLO": "MLO · medio-lateral oblicua",
        "L_CC": "L-CC",
        "R_CC": "R-CC",
        "L_MLO": "L-MLO",
        "R_MLO": "R-MLO",
    }.get(value, value.replace("_", "-"))


def _render_manual_resolution(
    automatic: dict[str, Any],
    staged_paths: list[str],
) -> dict[str, str]:
    preview_by_name: dict[str, dict[str, Any]] = {}
    try:
        payload = _api_json(
            "POST",
            "/single-cases/previews",
            json={"dicom_paths": staged_paths, "view_assignments": {}},
            timeout=120,
        )
        preview_by_name = {item["name"]: item for item in payload.get("previews", [])}
    except Exception as exc:
        st.warning(f"No fue posible generar las imágenes de revisión: {exc}")

    overrides: dict[str, str] = {}
    files = automatic.get("files", [])
    cols = st.columns(2)
    for idx, item in enumerate(files):
        col = cols[idx % 2]
        name = str(item.get("name") or Path(staged_paths[idx]).name)
        display_name = _display_name(name)
        side = str(item.get("laterality") or "")
        side_label = {"L": "Izquierda (L)", "R": "Derecha (R)"}.get(side, "No determinada")
        projection = item.get("view") or "No determinada"
        preview = preview_by_name.get(name) or {}

        with col.container(border=True):
            st.markdown(f'<div class="preview-title">{display_name}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="preview-meta">Lateralidad: <b>{side_label}</b> · Proyección detectada: <b>{projection}</b></div>',
                unsafe_allow_html=True,
            )
            preview_path = preview.get("preview_path")
            if preview_path and Path(preview_path).is_file():
                st.image(preview_path, **image_fill_width_kwargs(st.image))
                st.caption("Vista previa para verificación visual. No se utiliza como entrada del modelo.")
            elif preview.get("error"):
                st.warning("La vista previa no pudo generarse para este archivo. La asignación manual continúa disponible.")
            else:
                st.info("Vista previa no disponible.")

            options, default = _manual_options(item)
            choice = st.selectbox(
                "Proyección",
                options,
                index=default,
                format_func=_choice_label,
                key=f"view-{name}",
                help=(
                    "Seleccione únicamente CC o MLO cuando la proyección no esté disponible en la metadata DICOM. "
                    "La lateralidad detectada se mantiene."
                    if side in {"L", "R"}
                    else "La lateralidad no pudo determinarse; seleccione la combinación lateralidad-proyección."
                ),
            )
            if side in {"L", "R"}:
                st.caption(
                    "CC · cráneo-caudal: proyección superior-inferior, habitualmente con contorno mamario más redondeado. "
                    "MLO · medio-lateral oblicua: proyección oblicua que suele incluir el músculo pectoral en la región superior posterior."
                )
            override = _manual_override(item, choice)
            if override is not None:
                overrides[name] = override
    return overrides


def _render_result(result: dict[str, Any]) -> None:
    classification = str(result.get("classification"))
    score = float(result.get("ensemble_malignancy_score", 0.0))
    threshold = float(result.get("threshold", 0.0))

    st.divider()
    st.subheader("Resultado de la evaluación")
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    c1.metric("Clasificación", "CÁNCER" if classification == "CANCER" else "NO CÁNCER")
    c2.metric("Valor del ensemble", f"{score:.4f}")
    c3.metric("Umbral de decisión", f"{threshold:.4f}")
    c4.metric("Tiempo de ejecución", f"{float(result.get('overall_elapsed_seconds', 0.0)):.1f} s")

    if classification == "CANCER":
        st.warning(
            "El valor combinado se encuentra en o por encima del umbral de decisión configurado. "
            "La interpretación corresponde exclusivamente al protocolo experimental."
        )
    else:
        st.info(
            "El valor combinado se encuentra por debajo del umbral de decisión configurado. "
            "Un resultado negativo no excluye patología en un contexto clínico."
        )

    weights = result.get("weights") or {}
    with st.expander("Configuración aplicada", expanded=False):
        st.write(f"**Dispositivo de inferencia Web:** {str(result.get('inference_device', 'cpu')).upper()}")
        st.write(f"**Origen de los pesos:** {result.get('weights_source', 'BASELINE')}")
        st.write(f"**Umbral de decisión:** {float(result.get('threshold', 0.0)):.4f}")
        st.write(f"**Origen del umbral:** {result.get('threshold_source', 'BASELINE')}")
        st.write(f"**GMIC:** {float(weights.get('gmic', 0.0)):.4f}")
        st.write(f"**NYU / DMV-CNN:** {float(weights.get('nyu', 0.0)):.4f}")
        st.write(f"**GLAM:** {float(weights.get('glam', 0.0)):.4f}")
        st.write(f"**Suma:** {sum(float(v) for v in weights.values()):.4f}")
        st.caption("La configuración aplicada a este caso no modifica los archivos de configuración del flujo batch.")

    with st.expander("Resultados por modelo", expanded=False):
        scores = result.get("model_scores") or {}
        cols = st.columns(3)
        for col, key, label in zip(cols, ("gmic", "nyu", "glam"), ("GMIC", "NYU / DMV-CNN", "GLAM")):
            value = float(scores.get(key, 0.0))
            col.metric(label, f"{value:.4f}")
            col.progress(max(0.0, min(1.0, value)))

        if result.get("discordance"):
            st.warning("La dispersión entre los modelos supera el criterio de discordancia configurado.")
        else:
            st.info("Los modelos no superan el criterio de discordancia configurado. Este indicador describe consistencia entre scores y no implica corrección de la clasificación.")

    timings = result.get("execution_timings") or {}
    stages = timings.get("stages") or {}
    models = timings.get("models") or {}
    with st.expander("Tiempos de ejecución", expanded=False):
        timing_rows = []
        for key, label in (
            ("PREPARATION", "Preparación del estudio"),
            ("ORIENTATION", "Normalización de orientación"),
            ("MODEL_INPUT_PREPARATION", "Preparación de entradas para modelos"),
        ):
            elapsed = (stages.get(key) or {}).get("elapsed_seconds")
            if elapsed is not None:
                timing_rows.append({"Etapa": label, "Tiempo observado (s)": round(float(elapsed), 2)})
        for key, label in (("gmic", "GMIC"), ("nyu", "NYU / DMV-CNN"), ("glam", "GLAM")):
            elapsed = (models.get(key) or {}).get("elapsed_seconds")
            if elapsed is not None:
                timing_rows.append({"Etapa": label, "Tiempo observado (s)": round(float(elapsed), 2)})
        for key, label in (("ENSEMBLE", "Integración del ensemble"), ("PERSISTENCE", "Registro de resultados")):
            elapsed = (stages.get(key) or {}).get("elapsed_seconds")
            if elapsed is not None:
                timing_rows.append({"Etapa": label, "Tiempo observado (s)": round(float(elapsed), 2)})
        if timing_rows:
            st.dataframe(timing_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("La ejecución no contiene tiempos de pared por etapa; corresponde a un resultado generado por una versión anterior.")
        st.caption(
            "Los tiempos mostrados son tiempos de pared observados por la ruta Web, desde el inicio hasta el retorno de cada etapa. "
            "Las métricas internas del runtime se conservan como diagnóstico técnico y no se utilizan como tiempo visible de la evaluación."
        )
        st.metric("Tiempo total", f"{float(result.get('overall_elapsed_seconds', 0.0)):.2f} s")

    prep = result.get("input_preparation") or {}
    with st.expander("Preparación del estudio", expanded=False):
        st.write("**Modalidad de ejecución:** Inferencia unitaria")
        st.write("**Etiqueta diagnóstica externa:** No requerida")
        selected = prep.get("selected_views") or {}
        if selected:
            st.dataframe([
                {
                    "Proyección": view.replace("_", "-"),
                    "Archivo": _display_name(info.get("name") or ""),
                    "Representación canónica": info.get("canonical_png"),
                }
                for view, info in selected.items()
            ], use_container_width=True, hide_index=True)

    orientation = result.get("orientation") or {}
    with st.expander("Normalización de orientación", expanded=False):
        st.write("**Política aplicada:**", orientation.get("summary", {}).get("policy_id", "—"))
        resolution = orientation.get("resolution") or {}
        if resolution:
            st.write("**Orientación modificada:**", "Sí" if bool(resolution.get("orientation_changed")) else "No")
            st.write("**Criterio registrado:**", resolution.get("decision_reason", "—"))

    persistence = result.get("persistence") or {}
    with st.expander("Registro y artefactos de la evaluación", expanded=False):
        st.caption("La ruta Web no conserva artefactos del caso en el workspace del proyecto; los archivos locales de trabajo son temporales y se eliminan al finalizar.")
        p1, p2 = st.columns(2)
        p1.metric("Registro PostgreSQL", (persistence.get("postgresql") or {}).get("status", "—"))
        minio = persistence.get("minio") or {}
        p2.metric("Artefactos MinIO", minio.get("status", "—"))
        st.code(f"run_id: {result.get('run_id')}")
        if minio.get("status") == "SUCCESS":
            st.write("**Ubicación de artefactos en MinIO**")
            st.code(
                f"bucket: {minio.get('bucket')}\n"
                f"prefix: {minio.get('prefix')}\n"
                f"objects: {minio.get('object_count')}"
            )
            console_url = str(minio.get("console_public_url") or "").strip()
            if console_url:
                st.link_button("Abrir consola MinIO", console_url)
            st.caption("En la consola, abra el bucket indicado y navegue al prefijo runs/<run_id> para consultar los DICOM, imágenes canónicas, resultado y manifiesto.")
        elif minio.get("error"):
            st.warning("La evaluación finalizó correctamente, pero no fue posible guardar los artefactos en MinIO.")

def _current_web_settings(ensemble_config: dict[str, Any] | None) -> tuple[str, str, dict[str, float], str, float, bool]:
    baseline = ((ensemble_config or {}).get("weights") or {"gmic": 0.333333, "nyu": 0.333333, "glam": 0.333334})
    baseline_threshold = float((ensemble_config or {}).get("threshold", 0.50))
    device = str(st.session_state.get("web_inference_device", DEFAULT_WEB_DEVICE)).lower()
    if device not in {"cpu", "gpu"}:
        device = DEFAULT_WEB_DEVICE
    mode = str(st.session_state.get("web_weight_mode", "Configuración base"))
    if mode == "Configuración personalizada":
        weights = {
            "gmic": float(st.session_state.get("web_weight_gmic", baseline.get("gmic", 0.333333))),
            "nyu": float(st.session_state.get("web_weight_nyu", baseline.get("nyu", 0.333333))),
            "glam": float(st.session_state.get("web_weight_glam", baseline.get("glam", 0.333334))),
        }
    else:
        weights = {k: float(v) for k, v in baseline.items()}
    threshold_mode = str(st.session_state.get("web_threshold_mode", "Configuración base"))
    if threshold_mode == "Configuración personalizada":
        threshold = float(st.session_state.get("web_decision_threshold", baseline_threshold))
    else:
        threshold = baseline_threshold
    valid = (
        abs(sum(weights.values()) - 1.0) <= 1e-6
        and all(0.0 <= v <= 1.0 for v in weights.values())
        and 0.0 <= threshold <= 1.0
    )
    return device, mode, weights, threshold_mode, threshold, valid


def _web_settings_fingerprint(ensemble_config: dict[str, Any] | None) -> tuple[Any, ...]:
    device, mode, weights, threshold_mode, threshold, _ = _current_web_settings(ensemble_config)
    return (
        device, mode,
        round(float(weights["gmic"]), 8), round(float(weights["nyu"]), 8), round(float(weights["glam"]), 8),
        threshold_mode, round(float(threshold), 8),
    )


def _render_methodology_summary() -> None:
    items = [
        ("1. Recepción y verificación", "Validación del estudio DICOM, identidad del examen, lateralidad y proyección."),
        ("2. Normalización", "Conversión canónica a PNG de 16 bits y aplicación de la política de orientación vigente."),
        ("3. Inferencia ensemble", "Ejecución de GMIC, NYU/DMV-CNN y GLAM, seguida de combinación mediante soft voting ponderado."),
        ("4. Trazabilidad", "Registro estructurado de resultados en PostgreSQL y conservación de artefactos de auditoría en MinIO."),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, items):
        col.markdown(
            f'<div class="method-card"><strong>{title}</strong><span>{body}</span></div>',
            unsafe_allow_html=True,
        )


st.markdown(
    """
<div class="hero">
  <h1>Evaluación mamográfica por ensemble</h1>
  <p>Interfaz de investigación para la evaluación inferencial de un estudio mamográfico DICOM de cuatro proyecciones.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.warning("Prototipo de investigación. Los resultados no sustituyen la evaluación clínica ni constituyen un diagnóstico médico.")

status = _status_payload()
storage = _storage_payload()
ensemble_config = _ensemble_config_payload()
persisted_web_settings, web_settings_load_error = _web_settings_payload()
just_hydrated = False
if not st.session_state.get("_web_settings_hydrated") and persisted_web_settings is not None:
    just_hydrated = _hydrate_web_settings(persisted_web_settings, ensemble_config)
elif st.session_state.get("_web_settings_hydrated"):
    just_hydrated = True

# Defaults are display-only until a successful settings hydration occurs. Autosave and
# evaluation remain disabled while PostgreSQL/FastAPI settings cannot be recovered.
st.session_state.setdefault("web_inference_device", DEFAULT_WEB_DEVICE)
st.session_state.setdefault("web_weight_mode", "Configuración base")
st.session_state.setdefault("web_threshold_mode", "Configuración base")
web_device, config_mode, web_weights, threshold_mode, threshold_value, weights_valid = _current_web_settings(ensemble_config)
settings_hydrated = bool(st.session_state.get("_web_settings_hydrated"))
if just_hydrated and settings_hydrated and "_web_settings_last_saved_fingerprint" not in st.session_state:
    # The successfully loaded PostgreSQL value is the active configuration. The UI never
    # writes settings during hydration; persistence is explicit through the configuration tab.
    st.session_state["_web_settings_last_saved_fingerprint"] = _web_settings_fingerprint(ensemble_config)
with st.sidebar:
    st.header("Sesión de evaluación")
    st.caption(f"Modo de inferencia Web · {web_device.upper()}")
    if not settings_hydrated:
        st.warning("Configuración Web pendiente de recuperación. No se ejecutarán evaluaciones ni se guardarán valores base hasta recuperar PostgreSQL.")
        if st.button("Reintentar configuración", use_container_width=True, key="retry_web_settings_sidebar"):
            st.rerun()
    if status is None:
        st.error("Servicio de evaluación no disponible")
    else:
        ready_models = sum(1 for model in status.get("models", []) if _model_runtime_ready(model, web_device)[0])
        total_models = len(status.get("models", []))
        if total_models == 3 and ready_models == 3:
            st.success("Motor inferencial disponible")
        else:
            st.warning(f"Modelos disponibles · {ready_models}/{total_models or 3}")
        for model in status.get("models", []):
            dot, label = _model_badge(model, web_device)
            st.write(f"{dot} **{str(model.get('model', 'model')).upper()}** · {label}")
    if st.button("Actualizar estado", use_container_width=True):
        st.rerun()

main_tab, config_tab, method_tab = st.tabs(["Evaluación del estudio", "Configuración y estado", "Metodología y trazabilidad"])

with main_tab:
    st.subheader("1. Carga del estudio")
    st.markdown(
        """
<div class="note">
<b>Requisitos de entrada.</b> Adjunte archivos DICOM correspondientes a un único estudio mamográfico.
El procedimiento requiere las proyecciones <b>L-CC, R-CC, L-MLO y R-MLO</b>.
No se solicitan etiquetas diagnósticas ni archivos de entrenamiento.
</div>
""",
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Archivos DICOM del estudio",
        type=["dcm", "dicom"],
        accept_multiple_files=True,
        help="Adjunte al menos las cuatro proyecciones estándar del mismo examen. Se admiten archivos adicionales cuando existen adquisiciones duplicadas.",
    )
    count = len(uploads or [])
    a, b, c = st.columns(3)
    a.metric("Archivos DICOM", count)
    b.metric("Proyecciones requeridas", "4")
    c.metric("Etiqueta diagnóstica", "No solicitada")

    inspection = None
    staged_paths: list[str] = []
    overrides: dict[str, str] = {}
    if count >= 4:
        try:
            staged_paths = _stage_uploads(list(uploads or []))
            automatic = _api_json(
                "POST",
                "/single-cases/inspect",
                json={"dicom_paths": staged_paths, "view_assignments": {}},
                timeout=60,
            )
            st.subheader("2. Verificación del estudio")
            _render_detection_table(automatic)

            if automatic.get("unresolved_files") or automatic.get("missing_views"):
                st.info(
                    "No fue posible determinar automáticamente todas las proyecciones CC/MLO a partir de la metadata disponible. "
                    "Revise las imágenes y complete únicamente la proyección faltante; cuando la lateralidad está disponible, se conserva automáticamente."
                )
                overrides = _render_manual_resolution(automatic, staged_paths)
                inspection = _api_json(
                    "POST",
                    "/single-cases/inspect",
                    json={"dicom_paths": staged_paths, "view_assignments": overrides},
                    timeout=60,
                )
                st.markdown("**Resultado de la verificación**")
                _render_detection_table(inspection)
            else:
                inspection = automatic

            for warning in inspection.get("warnings", []):
                st.warning(warning)
            for error in inspection.get("errors", []):
                st.error(error)
        except Exception as exc:
            st.error(f"No fue posible verificar el estudio DICOM: {exc}")
    elif count:
        st.info("Adjunte al menos cuatro archivos DICOM para completar las proyecciones L-CC, R-CC, L-MLO y R-MLO.")

    st.subheader("3. Evaluación inferencial")
    st.caption(
        "El procedimiento ejecuta la preparación vigente, los tres modelos y la combinación ponderada del ensemble. "
        "Durante la ejecución se informa el estado y el tiempo observado de cada etapa."
    )
    study_ready = bool(inspection and inspection.get("ready"))
    runtimes_ready = _all_models_ready(status, web_device)
    if study_ready and web_device == "gpu" and not runtimes_ready:
        st.warning(
            "El modo GPU seleccionado para la evaluación Web requiere una validación vigente de los runtimes. "
            "Puede cambiar el dispositivo Web a CPU en «Configuración y estado» o ejecutar `docker compose exec fastapi python -m model_tools.validate_gpu --models all`."
        )
    elif study_ready and web_device == "cpu" and not runtimes_ready:
        st.warning("Uno o más servicios de modelos no están disponibles para la evaluación Web en CPU.")

    # Current Web controls are immediately valid for the current evaluation.
    # "Actualizar configuración" persists them in PostgreSQL for future sessions,
    # but an unsaved edit does not block the current inference.
    ready = study_ready and weights_valid and runtimes_ready and settings_hydrated
    if study_ready and not settings_hydrated:
        st.warning(
            "La evaluación está temporalmente bloqueada porque no fue posible recuperar la configuración Web guardada. "
            "Use «Reintentar configuración» para evitar ejecutar el estudio con valores base no confirmados."
        )
    st.session_state.setdefault("web_eval_running", False)
    st.markdown('<div id="web-evaluation-progress"></div>', unsafe_allow_html=True)
    running = bool(st.session_state.get("web_eval_running", False))
    st.button(
        "Evaluación en curso…" if running else "Ejecutar evaluación",
        type="primary",
        use_container_width=True,
        disabled=(not ready) or running,
        on_click=_queue_web_evaluation,
        key="execute_web_evaluation",
    )
    pending = bool(st.session_state.pop("web_eval_pending", False))
    if pending:
        request_started = time.monotonic()
        run_id = f"web-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        _scroll_to_progress()
        progress_box = st.container(border=True)
        with progress_box:
            summary_slot = st.empty()
            st.markdown("**Progreso de la evaluación**")
            preparation_slot = st.empty()
            orientation_slot = st.empty()
            model_input_slot = st.empty()
            gmic_slot = st.empty()
            nyu_slot = st.empty()
            glam_slot = st.empty()
            ensemble_slot = st.empty()
            persistence_slot = st.empty()
        slots = {
            "summary": summary_slot,
            "preparation": preparation_slot,
            "orientation": orientation_slot,
            "model_input": model_input_slot,
            "gmic": gmic_slot,
            "nyu": nyu_slot,
            "glam": glam_slot,
            "ensemble": ensemble_slot,
            "persistence": persistence_slot,
        }
        request_payload = {
            "dicom_paths": staged_paths,
            "view_assignments": overrides,
            "ensemble_weights": (web_weights if config_mode == "Configuración personalizada" else None),
            "decision_threshold": (threshold_value if threshold_mode == "Configuración personalizada" else None),
            "inference_device": web_device,
            "run_id": run_id,
        }
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _api_json, "POST", "/single-cases/run", timeout=RUN_TIMEOUT_SECONDS, json=request_payload
                )
                last_progress: dict[str, Any] = {}
                while not future.done():
                    try:
                        last_progress = _api_json("GET", f"/single-cases/progress/{run_id}", timeout=10)
                    except Exception:
                        pass
                    _render_live_progress(last_progress, slots, time.monotonic() - request_started)
                    time.sleep(0.75)
                result = future.result()
            request_elapsed = time.monotonic() - request_started
            result["web_request_elapsed_seconds"] = float(request_elapsed)
            try:
                final_progress = _api_json("GET", f"/single-cases/progress/{run_id}", timeout=10)
            except Exception:
                final_progress = {
                    "stage": "COMPLETED",
                    "state": "SUCCESS",
                    "message": "Evaluación completada.",
                    "models": (result.get("execution_timings") or {}).get("models") or {},
                    "stages": (result.get("execution_timings") or {}).get("stages") or {},
                }
            _render_live_progress(final_progress, slots, request_elapsed)
            st.session_state["last_web_result"] = result
            st.session_state["last_web_progress"] = final_progress
            st.session_state["last_web_progress_elapsed"] = float(request_elapsed)
        except Exception as exc:
            request_elapsed = time.monotonic() - request_started
            message = str(exc)
            try:
                failed_progress = _api_json("GET", f"/single-cases/progress/{run_id}", timeout=10)
            except Exception:
                failed_progress = {"stage": "FAILED", "state": "FAILED", "message": "La evaluación se interrumpió."}
            _render_live_progress(failed_progress, slots, request_elapsed)
            st.session_state["last_web_progress"] = failed_progress
            st.session_state["last_web_progress_elapsed"] = float(request_elapsed)
            st.session_state["last_web_error"] = message
        finally:
            st.session_state["web_eval_running"] = False
            st.rerun()

    last_progress = st.session_state.get("last_web_progress")
    if last_progress and not bool(st.session_state.get("web_eval_running", False)):
        with st.container(border=True):
            summary_slot = st.empty()
            st.markdown("**Progreso de la evaluación**")
            preparation_slot = st.empty()
            orientation_slot = st.empty()
            model_input_slot = st.empty()
            gmic_slot = st.empty()
            nyu_slot = st.empty()
            glam_slot = st.empty()
            ensemble_slot = st.empty()
            persistence_slot = st.empty()
        static_slots = {
            "summary": summary_slot,
            "preparation": preparation_slot,
            "orientation": orientation_slot,
            "model_input": model_input_slot,
            "gmic": gmic_slot,
            "nyu": nyu_slot,
            "glam": glam_slot,
            "ensemble": ensemble_slot,
            "persistence": persistence_slot,
        }
        _render_live_progress(
            last_progress,
            static_slots,
            float(st.session_state.get("last_web_progress_elapsed", 0.0)),
        )

    if st.session_state.get("last_web_error"):
        message = str(st.session_state["last_web_error"])
        if "GPU_PROBE_REQUIRED" in message:
            st.error(
                "La evaluación se detuvo porque el modo GPU Web seleccionado no cuenta con una validación vigente. "
                "Cambie el dispositivo Web a CPU en «Configuración y estado» o valide los runtimes GPU antes de repetir la evaluación."
            )
        else:
            st.error("La evaluación no pudo completarse. Consulte el detalle técnico para revisar la causa.")
        with st.expander("Detalle técnico", expanded=False):
            st.code(message)
        st.caption(
            f"Tiempo transcurrido hasta la interrupción: "
            f"{float(st.session_state.get('last_web_progress_elapsed', 0.0)):.1f} s"
        )

    if st.session_state.get("last_web_result"):
        _render_result(st.session_state["last_web_result"])

with config_tab:
    st.subheader("Configuración de la evaluación Web")
    if not settings_hydrated:
        st.error(
            "No fue posible recuperar la configuración Web persistida en PostgreSQL. "
            "Los controles se muestran en modo seguro y no se guardará ninguna configuración hasta recuperar el estado persistido."
        )
        if web_settings_load_error:
            with st.expander("Detalle de conexión", expanded=False):
                st.code(web_settings_load_error)
        if st.button("Reintentar configuración", key="retry_web_settings_config"):
            st.rerun()
    st.markdown("**Dispositivo de inferencia**")
    st.radio(
        "Ejecución de los modelos",
        ["cpu", "gpu"],
        key="web_inference_device",
        horizontal=True,
        disabled=not settings_hydrated,
        format_func=lambda value: "CPU" if value == "cpu" else "GPU",
        help="CPU no requiere gpu_probe. GPU utiliza aceleración y conserva la validación GPU del Model Runner. Esta selección se envía solamente en la petición Web actual.",
    )
    selected_device = str(st.session_state.get("web_inference_device", DEFAULT_WEB_DEVICE)).lower()

    st.markdown("**Pesos del ensemble**")
    baseline_weights = ((ensemble_config or {}).get("weights") or {"gmic": 0.333333, "nyu": 0.333333, "glam": 0.333334})
    st.radio(
        "Origen de los pesos",
        ["Configuración base", "Configuración personalizada"],
        key="web_weight_mode",
        horizontal=True,
        disabled=not settings_hydrated,
        help="Los pesos personalizados se aplican únicamente a la evaluación Web y no se escriben en los YAML del proyecto.",
    )
    selected_mode = str(st.session_state.get("web_weight_mode", "Configuración base"))
    if selected_mode == "Configuración personalizada":
        w1, w2, w3 = st.columns(3)
        w1.number_input("GMIC", min_value=0.0, max_value=1.0, value=float(st.session_state.get("web_weight_gmic", baseline_weights.get("gmic", 0.333333))), step=0.05, format="%.6f", key="web_weight_gmic", disabled=not settings_hydrated)
        w2.number_input("NYU / DMV-CNN", min_value=0.0, max_value=1.0, value=float(st.session_state.get("web_weight_nyu", baseline_weights.get("nyu", 0.333333))), step=0.05, format="%.6f", key="web_weight_nyu", disabled=not settings_hydrated)
        w3.number_input("GLAM", min_value=0.0, max_value=1.0, value=float(st.session_state.get("web_weight_glam", baseline_weights.get("glam", 0.333334))), step=0.05, format="%.6f", key="web_weight_glam", disabled=not settings_hydrated)
        shown_weights = {
            "gmic": float(st.session_state.get("web_weight_gmic", baseline_weights.get("gmic", 0.333333))),
            "nyu": float(st.session_state.get("web_weight_nyu", baseline_weights.get("nyu", 0.333333))),
            "glam": float(st.session_state.get("web_weight_glam", baseline_weights.get("glam", 0.333334))),
        }
    else:
        shown_weights = {k: float(v) for k, v in baseline_weights.items()}
        w1, w2, w3 = st.columns(3)
        w1.metric("GMIC", f"{shown_weights['gmic']:.6f}")
        w2.metric("NYU / DMV-CNN", f"{shown_weights['nyu']:.6f}")
        w3.metric("GLAM", f"{shown_weights['glam']:.6f}")

    shown_sum = sum(shown_weights.values())
    st.metric("Suma de pesos", f"{shown_sum:.6f}")
    if abs(shown_sum - 1.0) > 1e-6:
        st.error("La suma de los pesos debe ser exactamente 1.000000 antes de ejecutar una evaluación.")

    st.markdown("**Umbral de decisión**")
    baseline_threshold = float((ensemble_config or {}).get("threshold", 0.50))
    st.radio(
        "Origen del umbral",
        ["Configuración base", "Configuración personalizada"],
        key="web_threshold_mode",
        horizontal=True,
        disabled=not settings_hydrated,
        help="El umbral personalizado se aplica únicamente a esta evaluación Web y no modifica config/ensemble.yaml ni config/experiments.yaml.",
    )
    selected_threshold_mode = str(st.session_state.get("web_threshold_mode", "Configuración base"))
    if selected_threshold_mode == "Configuración personalizada":
        shown_threshold = st.number_input(
            "Threshold", min_value=0.0, max_value=1.0,
            value=float(st.session_state.get("web_decision_threshold", baseline_threshold)),
            step=0.01, format="%.4f", key="web_decision_threshold", disabled=not settings_hydrated,
            help="Clasifica CÁNCER cuando el score combinado es mayor o igual a este valor. Solo afecta la ruta Web.",
        )
        st.caption("Umbral Web temporal para las evaluaciones iniciadas desde esta sesión. Los experimentos batch conservan su configuración independiente.")
    else:
        shown_threshold = baseline_threshold
        st.metric("Threshold base", f"{shown_threshold:.4f}")
        st.caption("Valor leído de config/ensemble.yaml. La interfaz no modifica ese archivo.")

    # The Web configuration is persisted only after an explicit user action. This avoids
    # writes during Streamlit reruns and prevents display defaults from overwriting a valid
    # PostgreSQL configuration after a transient read failure.
    stored_candidate_weights = {k: float(v) for k, v in shown_weights.items()}
    stored_candidate_threshold = float(shown_threshold)
    settings_valid = (
        abs(sum(stored_candidate_weights.values()) - 1.0) <= 1e-6
        and all(0.0 <= value <= 1.0 for value in stored_candidate_weights.values())
        and 0.0 <= stored_candidate_threshold <= 1.0
    )
    settings_fingerprint = (
        selected_device, selected_mode,
        round(stored_candidate_weights["gmic"], 8), round(stored_candidate_weights["nyu"], 8), round(stored_candidate_weights["glam"], 8),
        selected_threshold_mode, round(stored_candidate_threshold, 8),
    )
    persisted_fingerprint = st.session_state.get("_web_settings_last_saved_fingerprint")
    settings_dirty = bool(settings_hydrated and settings_fingerprint != persisted_fingerprint)

    if not settings_hydrated:
        st.caption("La configuración no puede actualizarse hasta recuperar correctamente el estado persistido desde PostgreSQL.")
    elif not settings_dirty:
        persisted_at = st.session_state.get("_web_settings_persisted_at")
        suffix = f" · {persisted_at}" if persisted_at else ""
        st.caption(f"Configuración guardada en PostgreSQL{suffix}.")

    action_col, reset_col = st.columns([2, 1])
    update_clicked = action_col.button(
        "Actualizar configuración",
        type="primary",
        use_container_width=True,
        disabled=(not settings_hydrated) or (not settings_valid) or (not settings_dirty),
        key="update_web_settings",
    )
    reset_clicked = reset_col.button(
        "Restaurar configuración base",
        use_container_width=True,
        disabled=not settings_hydrated,
        key="reset_web_settings",
    )

    if update_clicked:
        try:
            saved_settings = _persist_web_settings(
                device=selected_device,
                weight_mode=selected_mode,
                weights=stored_candidate_weights,
                threshold_mode=selected_threshold_mode,
                threshold=stored_candidate_threshold,
            )
            st.session_state["_web_settings_last_saved_fingerprint"] = settings_fingerprint
            st.session_state["_web_settings_persisted_at"] = (saved_settings or {}).get("updated_at")
            st.session_state.pop("_web_settings_save_error", None)
            st.success("Configuración actualizada correctamente.")
            st.rerun()
        except Exception as exc:
            st.session_state["_web_settings_save_error"] = str(exc)

    if reset_clicked:
        base_weights = {k: float(v) for k, v in baseline_weights.items()}
        try:
            saved_settings = _persist_web_settings(
                device=DEFAULT_WEB_DEVICE,
                weight_mode="Configuración base",
                weights=base_weights,
                threshold_mode="Configuración base",
                threshold=baseline_threshold,
            )
            # Re-hydrate from PostgreSQL on the next run. Widget-backed keys are intentionally
            # not mutated after their widgets have been instantiated in the current Streamlit run.
            st.session_state["_web_settings_hydrated"] = False
            st.session_state.pop("_web_settings_last_saved_fingerprint", None)
            st.session_state["_web_settings_persisted_at"] = (saved_settings or {}).get("updated_at")
            st.session_state.pop("_web_settings_save_error", None)
            st.rerun()
        except Exception as exc:
            st.session_state["_web_settings_save_error"] = str(exc)

    if st.session_state.get("_web_settings_save_error"):
        st.warning("No fue posible actualizar la configuración en PostgreSQL. Los cambios permanecen sin aplicar.")
        with st.expander("Detalle de persistencia", expanded=False):
            st.code(str(st.session_state["_web_settings_save_error"]))

    st.divider()
    st.subheader("Disponibilidad de modelos")
    if status and status.get("models"):
        model_rows = []
        for model in status["models"]:
            _, readiness = _model_runtime_ready(model, selected_device)
            row = {
                "Modelo": str(model.get("model", "")).upper(),
                "Dispositivo Web": selected_device.upper(),
                "Estado": readiness,
            }
            if selected_device == "gpu":
                row["Imagen GPU"] = "Preparada" if model.get("gpu_built") else "No preparada"
                row["GPU probe"] = "Aprobado" if model.get("gpu_probe_passed") else "Pendiente"
            model_rows.append(row)
        st.dataframe(model_rows, use_container_width=True, hide_index=True)
        if selected_device == "gpu" and not _all_models_ready(status, "gpu"):
            st.info("Para validar los runtimes GPU: `docker compose exec fastapi python -m model_tools.validate_gpu --models all`.")
    else:
        st.info("No hay información de disponibilidad de modelos.")

    st.subheader("Persistencia de resultados")
    st.caption("La Web conserva de forma permanente únicamente el registro estructurado en PostgreSQL y los artefactos del caso en MinIO. Los archivos locales utilizados durante la inferencia son temporales.")
    if storage:
        minio = storage.get("minio") or {}
        p1, p2 = st.columns(2)
        with p1.container(border=True):
            st.markdown("**PostgreSQL**")
            db_ready = bool((storage.get("postgresql") or {}).get("configured"))
            st.write("Disponible" if db_ready else "Requiere revisión")
            st.caption("Conserva el identificador de ejecución, configuración aplicada, scores, clasificación y tiempos.")
        with p2.container(border=True):
            st.markdown("**MinIO**")
            object_ready = minio.get("status") == "READY"
            st.write("Disponible" if object_ready else "Requiere revisión")
            st.caption("Conserva los DICOM de entrada, imágenes canónicas, resultado y manifiesto del run. El scratch local se elimina al finalizar.")
            console_url = str(minio.get("console_public_url") or "").strip()
            if object_ready and console_url:
                st.link_button("Abrir consola MinIO", console_url, use_container_width=True)
        if minio.get("status") == "READY":
            st.caption(f"Bucket Web: {minio.get('bucket', 'configurado')} · cada evaluación se almacena bajo runs/<run_id>.")
        with st.expander("Detalle técnico de persistencia", expanded=False):
            st.json(storage)
    else:
        st.warning("No fue posible consultar el estado de los servicios de persistencia.")

with method_tab:
    st.subheader("Protocolo de evaluación")
    st.write(
        "La interfaz Web implementa una ruta de inferencia unitaria independiente del flujo experimental masivo. "
        "Reutiliza los componentes vigentes de preparación, orientación, ejecución de modelos y combinación de resultados."
    )
    _render_methodology_summary()

    st.subheader("Criterios de implementación")
    st.markdown(
        """
- La entrada corresponde exclusivamente a imágenes DICOM del estudio; no se incorpora una etiqueta diagnóstica al proceso de inferencia.
- La proyección se obtiene de la metadata DICOM cuando está disponible. Si no puede resolverse, la interfaz permite una verificación visual y una asignación explícita de CC/MLO.
- La representación utilizada por los modelos se genera mediante el conversor canónico DICOM a PNG monocromático de 16 bits ya utilizado por los adapters del proyecto.
- La política de orientación se reutiliza sin cambios. El dispositivo de inferencia y los pesos del ensemble pueden definirse por caso Web sin modificar la configuración del flujo batch.
- PostgreSQL conserva el registro estructurado de la ejecución y MinIO mantiene los artefactos del caso. La ruta Web utiliza almacenamiento local únicamente como scratch temporal y lo elimina al finalizar; el workspace persistente queda reservado para el flujo batch.
- Las imágenes de vista previa se generan únicamente para facilitar la verificación de la proyección y no intervienen en la inferencia.
"""
    )
