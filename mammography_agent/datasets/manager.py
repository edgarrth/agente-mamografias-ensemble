from __future__ import annotations
from ..config import load_yaml
from .adapters import VinDrDatasetAdapter
from .cbis_ddsm import CBISDDSMDatasetAdapter
from .cmmd import CMMDDatasetAdapter
from .rsna import RSNADatasetAdapter
from ..logging_utils import audit

FACTORY={"cbis_ddsm":CBISDDSMDatasetAdapter,"cmmd":CMMDDatasetAdapter,"vindr":VinDrDatasetAdapter,"rsna":RSNADatasetAdapter}

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


def inspect(keys: list[str], force_dicom_index: bool=False):
    chosen=selected(keys); audit("DATASET_SELECTION",datasets=chosen,operation="inspect")
    results=[]
    for k in chosen:
        a=adapter(k)
        if hasattr(a,"inspect"):
            results.append(a.inspect(force_dicom_index=force_dicom_index) if k in {"cbis_ddsm","cmmd","rsna"} else a.inspect())
        else:
            results.append(a.status())
    return results
