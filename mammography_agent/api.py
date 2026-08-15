from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .workspace import ensure_workspace
from .logging_utils import log_configuration_additions, audit
from .storage import init_db
from .datasets.manager import statuses, request_download, prepare
from .model_client import status as model_status

app=FastAPI(title="Mammography AI Agent",version="0.4.0")

@app.on_event("startup")
def startup():
    ensure_workspace(); log_configuration_additions(); init_db(); audit("APPLICATION_READY")

@app.get("/health")
def health(): return {"status":"ok","research_only":True}

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
