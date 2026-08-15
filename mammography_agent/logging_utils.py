from __future__ import annotations
from .config import WORKSPACE_ROOT, load_yaml
from .workspace import ensure_workspace
from pathlib import Path
import json, logging, datetime, os

LOG = logging.getLogger("mammography-agent")
if not LOG.handlers:
    handler=logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(os.getenv("LOG_LEVEL","INFO"))

def audit(event: str, **data):
    ensure_workspace()
    record={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"event":event,**data}
    path=WORKSPACE_ROOT/"logs"/"audit.jsonl"
    with path.open("a",encoding="utf-8") as fh: fh.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
    LOG.info("%s %s", event, data)

def log_configuration_additions():
    ensure_workspace()
    cfg=load_yaml("config_additions.yaml")
    path=WORKSPACE_ROOT/"logs"/"configuration_additions.log"
    with path.open("w",encoding="utf-8") as fh:
        fh.write("Configuration additions required by implementation\n")
        fh.write("These entries do not add models or training.\n\n")
        for index, x in enumerate(cfg.get("additions", []), start=1):
            required = ("id", "name", "value", "reason")
            missing = [key for key in required if key not in x]
            if missing:
                raise ValueError(
                    f"Invalid config_additions.yaml entry #{index}: missing required field(s): {', '.join(missing)}"
                )
            status = x.get("status", "active")
            superseded = f" | superseded_by={x.get('superseded_by')}" if x.get("superseded_by") else ""
            fh.write(f"{x['id']} | {x['name']} | {x['value']} | status={status}{superseded}\nWHY: {x['reason']}\n\n")
    return path
