from __future__ import annotations
from ..config import load_yaml
from .adapters import CBISDDSMDatasetAdapter, VinDrDatasetAdapter
from ..logging_utils import audit

FACTORY={"cbis_ddsm":CBISDDSMDatasetAdapter,"vindr":VinDrDatasetAdapter}

def catalog(): return load_yaml("datasets.yaml").get("datasets",{})

def selected(keys: list[str]) -> list[str]:
    cfg=catalog(); keys=[k.lower() for k in keys]
    result=list(cfg) if "all" in keys else keys
    unknown=[k for k in result if k not in cfg]
    if unknown: raise ValueError(f"Unknown datasets: {unknown}. Configured: {list(cfg)}")
    return result

def adapter(key: str):
    cfg=catalog()[key]
    return FACTORY[key](key,cfg)

def statuses(): return [adapter(k).status() for k in catalog()]

def request_download(keys: list[str]):
    chosen=selected(keys); audit("DATASET_SELECTION",datasets=chosen,operation="download")
    return [adapter(k).download() for k in chosen]

def prepare(keys: list[str]):
    chosen=selected(keys); audit("DATASET_SELECTION",datasets=chosen,operation="prepare")
    results=[]
    for k in chosen:
        st=adapter(k).status()
        if st["status"]=="NOT_DOWNLOADED":
            results.append({"dataset":k,"status":"SKIPPED_NOT_DOWNLOADED"}); continue
        results.append(adapter(k).prepare())
    return results
