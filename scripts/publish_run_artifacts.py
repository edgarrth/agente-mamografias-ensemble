#!/usr/bin/env python3
"""Publica los artefactos ligeros de una ejecución desde workspace/ al repositorio.

workspace/ es directorio de trabajo y no debe versionarse entero: contiene
cientos de gigabytes de derivados. Este guion copia únicamente los artefactos
que sustentan las cifras de la tesis, genera SHA256SUMS y verifica los resúmenes
criptográficos que el Anexo 1 declara.

Uso:
    # ver qué haría, sin escribir nada
    python scripts/publish_run_artifacts.py experiment-20260818T035300Z

    # ejecutar
    python scripts/publish_run_artifacts.py experiment-20260818T035300Z --apply

    # solo verificar lo ya publicado
    python scripts/publish_run_artifacts.py experiment-20260818T035300Z --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN_BASE = RAIZ / "workspace" / "output" / "experiments"
DESTINO_BASE = RAIZ / "artefacts"

# Directorios de trabajo pesados: entradas de modelo, imágenes preprocesadas y
# temporales por bloque. Nunca se versionan.
DIRECTORIOS_EXCLUIDOS = {
    "model_batch", "preprocessed", "images", "xai_retained",
    "original", "counterfactual", "tmp", "cache", "__pycache__",
}

# Formatos que corresponden a datos derivados de píxeles o a pesos de modelo.
# Su exclusión no es solo de tamaño: redistribuirlos infringiría las licencias
# de RSNA y de NYU (véase NOTICE).
EXTENSIONES_EXCLUIDAS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom",
    ".h5", ".hdf5", ".pkl", ".pickle", ".pt", ".pth", ".npy", ".npz",
    ".zip", ".tar", ".gz", ".log",
}

LIMITE_POR_ARCHIVO_MB = 20

# Resúmenes declarados en el Anexo 1 de la tesis. Si alguno no coincide, la
# copia no corresponde al run congelado que sustenta el Capítulo VI.
HASHES_DECLARADOS = {
    "split_summary.json": "fb5938962e7abce2",
    "experiment_plan.json": "1a306e1382e7e7ba",
    "formal_exclusions_applied.csv": "98eca61d70c1752b",
    "configuration_selection_v0310/selection_protocol.json": "5cf349ebcd126e24",
    "configuration_selection_v0310/best_configuration.json": "9fdbe52baa704d46",
    "frozen_configuration.yaml": "100fcf814fd86aab",
    "final_metrics.json": "7add63ef11053309",
    "final_model_comparison.csv": "6330cf246e541c7e",
    "final_score_analysis/model_correlations.csv": "a495a9f4fa00cba7",
    "configuration_orientation/orientation_policy_summary.json": "eae88562ec413fa2",
    "configuration_report.md": "c81c2a50933be185",
    "final_predictions.csv": "f3abe8b64ec52770",
    "configuration_set_predictions.csv": "c20f06cf7d132fb2",
    "configuration_selection_v0310/ranking_cv.csv": "cb27b258c59f786c",
    "configuration_selection_v0310/fold_metrics.csv": "07875849ee7f8dcf",
    "configuration_selection_v0310/fold_assignments.csv": "3797fa818e007cbc",
    "formal_pool_manifest.csv": "bc10a24c0783f57f",
    "final_test_manifest.csv": "848eebd1d4044042",
    "final_inference/resource_metrics.csv": "aa7a498f00be9a11",
    "final_score_analysis/roc_points.csv": "26a884b4a8100368",
    "final_score_analysis/pr_points.csv": "4e28b9b04513871f",
    "final_inference/xai_artifacts.json": "50789ca07935eb87",
    "all_configurations.csv": "87093aa53f7f729d",
    "ranking.csv": "7a5aef601fef7636",
}


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def seleccionar(origen: Path):
    """Devuelve (incluidos, descartados) como listas de rutas relativas."""
    incluidos, descartados = [], []
    limite = LIMITE_POR_ARCHIVO_MB * 1024 * 1024
    for ruta in sorted(origen.rglob("*")):
        if not ruta.is_file():
            continue
        rel = ruta.relative_to(origen)
        if DIRECTORIOS_EXCLUIDOS & set(rel.parts[:-1]):
            descartados.append((rel, "directorio de trabajo"))
        elif ruta.suffix.lower() in EXTENSIONES_EXCLUIDAS:
            descartados.append((rel, f"formato {ruta.suffix}"))
        elif ruta.stat().st_size > limite:
            descartados.append((rel, f"{ruta.stat().st_size / 1048576:.1f} MB"))
        else:
            incluidos.append(rel)
    return incluidos, descartados


def verificar(destino: Path) -> list[str]:
    fallos = []
    for rel, prefijo in HASHES_DECLARADOS.items():
        ruta = destino / rel
        if not ruta.exists():
            fallos.append(f"{rel}: ausente")
            continue
        real = sha256(ruta)
        if not real.startswith(prefijo):
            fallos.append(f"{rel}: declarado {prefijo}…, obtenido {real[:16]}…")
    return fallos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", help="p. ej. experiment-20260818T035300Z")
    p.add_argument("--apply", action="store_true", help="escribir de verdad")
    p.add_argument("--verify-only", action="store_true",
                   help="solo verificar lo ya publicado")
    a = p.parse_args()

    origen = ORIGEN_BASE / a.run_id
    destino = DESTINO_BASE / a.run_id

    print("=" * 72)
    print(f"PUBLICACIÓN DE ARTEFACTOS — {a.run_id}")
    print("=" * 72)

    if a.verify_only:
        if not destino.exists():
            print(f"\nNo existe {destino}")
            return 2
        print(f"\nVerificando {destino}")
        fallos = verificar(destino)
        for f in fallos:
            print(f"  [FALLA] {f}")
        if fallos:
            print(f"\n{len(fallos)} de {len(HASHES_DECLARADOS)} resúmenes no coinciden.")
            return 1
        print(f"  [OK ] los {len(HASHES_DECLARADOS)} resúmenes del Anexo 1 coinciden")
        return 0

    if not origen.exists():
        print(f"\nNo existe {origen}")
        print("\nEjecuciones disponibles:")
        for d in sorted(ORIGEN_BASE.glob("experiment-*")):
            print(f"  {d.name}")
        return 2

    incluidos, descartados = seleccionar(origen)
    total = sum((origen / r).stat().st_size for r in incluidos)

    print(f"\nOrigen : {origen}")
    print(f"Destino: {destino}")
    print(f"\nSe publicarán {len(incluidos)} archivos, {total:,} bytes "
          f"({total / 1048576:.1f} MiB)")
    print(f"Se omitirán   {len(descartados)} archivos")

    if descartados:
        print("\nOmitidos (primeros 15):")
        for rel, motivo in descartados[:15]:
            print(f"  {str(rel)[:58]:60s} {motivo}")
        if len(descartados) > 15:
            print(f"  … y {len(descartados) - 15} más")

    if not a.apply:
        print("\nSimulación. Añada --apply para escribir.")
        return 0

    if destino.exists():
        print(f"\nEl destino ya existe. Se reemplaza: {destino}")
        shutil.rmtree(destino)
    for rel in incluidos:
        dst = destino / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen / rel, dst)
    print(f"\nCopiados {len(incluidos)} archivos.")

    lineas = [f"{sha256(destino / r)}  {r.as_posix()}" for r in sorted(incluidos)]
    (destino / "SHA256SUMS").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Generado SHA256SUMS con {len(lineas)} entradas.")

    print("\nVerificación contra los resúmenes declarados en el Anexo 1:")
    fallos = verificar(destino)
    for f in fallos:
        print(f"  [FALLA] {f}")
    if fallos:
        print(f"\n{len(fallos)} de {len(HASHES_DECLARADOS)} no coinciden. "
              f"La copia NO corresponde al run congelado.")
        return 1
    print(f"  [OK ] los {len(HASHES_DECLARADOS)} resúmenes coinciden")

    print("\nSiguiente paso:")
    print(f"  git add artefacts/{a.run_id}")
    print(f'  git commit -m "Artefactos del run {a.run_id}"')
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
