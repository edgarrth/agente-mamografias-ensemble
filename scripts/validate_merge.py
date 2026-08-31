#!/usr/bin/env python3
"""Validación de la fusión de las ramas web y batch.

Comprueba que el árbol unificado reproduce las cifras congeladas de la
evaluación formal SIN ejecutar ningún modelo: la selección de configuración
opera sobre los puntajes ya persistidos, de modo que toda la cadena de
decisión es verificable en segundos y sin GPU.

Uso:
    python scripts/validate_merge.py <ruta-a-configuration_set_predictions.csv>
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ESPERADO = {
    "estudios": 3570,
    "malignos": 144,
    "pesos": 40,
    "cuantiles": 17,
    "candidatos": 680,
    "weight_id": "W17",
    "threshold_id": "T05",
    "w_gmic": 0.10, "w_nyu": 0.10, "w_glam": 0.80,
    "roc_auc_mean": 0.7242,
    "balanced_accuracy_mean": 0.6736,
    "sensitivity_mean": 0.8340,
    "specificity_mean": 0.5131,
    "umbral_congelado": 0.040985,
}

fallos = []
def check(nombre, obtenido, esperado, tol=None):
    ok = (abs(obtenido - esperado) <= tol) if tol else (obtenido == esperado)
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre:38s} {obtenido!r:>12}  esperado {esperado!r}")
    if not ok:
        fallos.append(nombre)

print("=" * 72)
print("VALIDACIÓN DE LA FUSIÓN — agente-mamografias-ensemble 1.0.0")
print("=" * 72)

# ---------------------------------------------------------------- 1. Config
print("\n1. Espacio de búsqueda declarado en config/experiments.yaml")
from mammography_agent.config import load_yaml
cfg = load_yaml("experiments.yaml")
pesos = cfg["weights"]
cuantiles = cfg["threshold_strategy"]["quantiles"]
check("combinaciones de pesos", len(pesos), ESPERADO["pesos"])
check("cuantiles de umbral", len(cuantiles), ESPERADO["cuantiles"])
check("candidatos", len(pesos) * len(cuantiles), ESPERADO["candidatos"])
w17 = pesos.get("W17")
check("W17 existe y es GMIC 0.10", float(w17[0]) if w17 else -1, ESPERADO["w_gmic"], 1e-9)
check("W17 GLAM 0.80", float(w17[2]) if w17 else -1, ESPERADO["w_glam"], 1e-9)
check("T05 es la mediana", float(cuantiles.get("T05", -1)), 0.50, 1e-9)
suma_ok = all(abs(sum(v) - 1) < 1e-6 for v in pesos.values())
minimo_ok = all(min(v) >= 0.10 - 1e-9 for v in pesos.values())
print(f"  [{'OK ' if suma_ok else 'FALLA'}] todos los pesos suman 1")
print(f"  [{'OK ' if minimo_ok else 'FALLA'}] ningún peso por debajo de 0.10")
if not suma_ok: fallos.append("suma de pesos")
if not minimo_ok: fallos.append("peso mínimo")

# ------------------------------------------------- 2. Módulos restituidos
print("\n2. Funciones que la rama web había eliminado")
import ast
src = (Path(__file__).resolve().parents[1] / "mammography_agent" / "pipeline.py").read_text(encoding="utf-8")
fns = {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
for f in ["_evenly_spaced", "_retain_xai_and_cleanup_model_batch",
          "_cleanup_orientation_temporaries", "_apply_web_label_blind_compat"]:
    ok = f in fns
    print(f"  [{'OK ' if ok else 'FALLA'}] {f}")
    if not ok: fallos.append(f)

# ------------------------------------------------------ 3. Reproducción
ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not ruta or not ruta.exists():
    print("\n3. Reproducción de la selección: OMITIDA")
    print("   Pase la ruta de configuration_set_predictions.csv para ejecutarla.")
else:
    print(f"\n3. Reproducción de la selección sobre {ruta.name}")
    import pandas as pd
    from mammography_agent.ensemble.cv_selection import evaluate_cv_grid
    df = pd.read_csv(ruta)
    check("estudios de configuración", len(df), ESPERADO["estudios"])
    check("casos malignos", int(df.ground_truth.sum()), ESPERADO["malignos"])
    asignaciones, por_fold, ranking = evaluate_cv_grid(df, n_splits=5, seed=42)
    check("estudios asignados a un fold", len(asignaciones), ESPERADO["estudios"])
    check("evaluaciones candidato-partición", len(por_fold), ESPERADO["candidatos"] * 5)
    check("candidatos evaluados", len(ranking), ESPERADO["candidatos"])
    fila = ranking[(ranking.weight_id == "W17") & (ranking.threshold_id == "T05")].iloc[0]
    for m in ["roc_auc_mean", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean"]:
        check(f"W17/T05 {m}", round(float(fila[m]), 4), ESPERADO[m], 1e-4)
    mejor = ranking.loc[ranking.roc_auc_mean.idxmax()]
    check("vector de pesos ganador", mejor.weight_id, ESPERADO["weight_id"])

print("\n" + "=" * 72)
if fallos:
    print(f"RESULTADO: {len(fallos)} comprobación(es) fallida(s): {', '.join(fallos)}")
    sys.exit(1)
print("RESULTADO: todas las comprobaciones pasaron.")
print("=" * 72)
