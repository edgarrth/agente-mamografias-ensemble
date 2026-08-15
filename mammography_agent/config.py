from __future__ import annotations
from pathlib import Path
import os, yaml

CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT", "/app/config" if Path("/app/config").exists() else "config"))
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/workspace" if Path("/workspace").exists() else "workspace"))

def load_yaml(name: str) -> dict:
    path = CONFIG_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
