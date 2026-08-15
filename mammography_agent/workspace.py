from __future__ import annotations
from pathlib import Path
from .config import WORKSPACE_ROOT

REQUIRED = [
    "input", "datasets/raw", "datasets/processed", "datasets/manifests", "datasets/rejected",
    "models", "runtime", "output/analyses", "output/normal_tests", "output/experiments",
    "output/final_evaluations", "output/xai", "output/reports", "logs"
]

def ensure_workspace() -> list[Path]:
    created=[]
    for rel in REQUIRED:
        p=WORKSPACE_ROOT/rel
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(p)
    return created

def safe_workspace_path(value: str | Path) -> Path:
    p=Path(value)
    if not p.is_absolute(): p=WORKSPACE_ROOT/p
    p=p.resolve()
    root=WORKSPACE_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path outside workspace is forbidden: {p}")
    return p
