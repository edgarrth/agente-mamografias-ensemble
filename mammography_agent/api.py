from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .workspace import ensure_workspace
from .logging_utils import log_configuration_additions, audit
from .storage import init_db
from .datasets.manager import statuses, request_download, prepare, inspect
from .model_client import status as model_status
from .health_logging import install_healthcheck_access_filter
from . import __version__

app=FastAPI(title="Mammography AI Agent",version=__version__)

@app.on_event("startup")
def startup():
    install_healthcheck_access_filter()
    ensure_workspace(); log_configuration_additions(); init_db(); audit("APPLICATION_READY", version=__version__)

@app.get("/health")
def health(): return {"status":"ok","research_only":True,"version":__version__}

@app.get("/workspace/status")
def workspace_status(): return {"datasets":statuses(),"models":model_status()}

@app.get("/datasets")
def datasets(): return statuses()

class DatasetSelection(BaseModel): datasets:list[str]
@app.post("/datasets/download")
def download(req:DatasetSelection):
    try: return request_download(req.datasets)
    except Exception as e: raise HTTPException(400,str(e))
@app.post("/datasets/prepare")
def prep(req:DatasetSelection):
    try: return prepare(req.datasets)
    except Exception as e: raise HTTPException(400,str(e))

@app.post("/datasets/inspect")
def inspect_dataset(req:DatasetSelection):
    try: return inspect(req.datasets)
    except Exception as e: raise HTTPException(400,str(e))
