from __future__ import annotations
from pathlib import Path
import json

def write_json(path: Path, value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding="utf-8")

def write_report(path: Path, title: str, data: dict):
    lines=[f"# {title}","","> Research-only result. Not a clinical diagnosis.",""]
    for k,v in data.items(): lines.append(f"- **{k}**: {v}")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
