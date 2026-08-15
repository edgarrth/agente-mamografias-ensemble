from __future__ import annotations
from pathlib import Path
import pandas as pd, shutil, hashlib
from .base import DatasetAdapter
from .manifest import REQUIRED
from ..workspace import safe_workspace_path
from ..logging_utils import audit

class ManifestDatasetAdapter(DatasetAdapter):
    def _paths(self):
        return {k:safe_workspace_path(v) for k,v in {
            "raw":self.cfg["raw_dir"],"processed":self.cfg["processed_dir"],
            "source":self.cfg["source_manifest"],"canonical":self.cfg["canonical_manifest"]}.items()}

    def status(self) -> dict:
        p=self._paths()
        if p["canonical"].exists(): state="AVAILABLE"
        elif p["source"].exists(): state="DOWNLOADED_NOT_PREPARED"
        elif p["raw"].exists() and any(p["raw"].iterdir()): state="MANUAL_DOWNLOAD_REQUIRED"
        else: state="NOT_DOWNLOADED"
        return {"dataset":self.key,"name":self.cfg["name"],"status":state,
                "raw_dir":str(p["raw"]),"canonical_manifest":str(p["canonical"])}

    def download(self) -> dict:
        p=self._paths(); p["raw"].mkdir(parents=True,exist_ok=True)
        if p["canonical"].exists():
            audit("DATASET_REUSED",dataset=self.key,status="AVAILABLE")
            return {**self.status(),"action":"reused"}
        instructions=p["raw"]/"DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            f"# {self.cfg['name']}\n\n"
            "Use the dataset's official authorized access method. This prototype does not bypass login, license, or usage agreements.\n\n"
            f"Official information: {self.cfg['official_information']}\n\n"
            "Place the authorized raw files in this directory. Then create `source_manifest.csv` with columns:\n\n"
            "`study_id,patient_id,ground_truth,l_cc,r_cc,l_mlo,r_mlo`\n\n"
            "Optional: `left_ground_truth,right_ground_truth,horizontal_flip`.\n"
            "All image paths must point to files under the host workspace.\n",
            encoding="utf-8")
        audit("DATASET_DOWNLOAD_MANUAL_ACTION_REQUIRED",dataset=self.key,instructions=str(instructions))
        return {**self.status(),"status":"MANUAL_DOWNLOAD_REQUIRED","instructions":str(instructions)}

    def verify_integrity(self) -> dict:
        p=self._paths()
        if not p["source"].exists() and not p["canonical"].exists():
            return {"dataset":self.key,"valid":False,"reason":"source_manifest.csv/canonical manifest missing"}
        manifest=p["canonical"] if p["canonical"].exists() else p["source"]
        df=pd.read_csv(manifest)
        missing=[c for c in REQUIRED if c not in df.columns]
        return {"dataset":self.key,"valid":not missing,"rows":len(df),"missing_columns":missing}

    def _convert_to_png(self, source: Path, dest: Path):
        import numpy as np
        import png
        import pydicom
        dest.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix.lower()==".png":
            if source.resolve()!=dest.resolve(): shutil.copy2(source,dest)
            return
        if source.suffix.lower() not in {".dcm",".dicom",""}:
            raise ValueError(f"Unsupported image type for {source}; use DICOM or 16-bit PNG")
        ds=pydicom.dcmread(source)
        arr=np.asarray(ds.pixel_array)
        if arr.min() < 0:
            raise ValueError(f"Signed DICOM pixels require dataset-specific conversion: {source}")
        bits=int(getattr(ds,"BitsStored",16) or 16)
        if str(getattr(ds,"PhotometricInterpretation",""))=="MONOCHROME1":
            max_native=(1 << bits)-1
            arr=max_native-arr
        arr=np.clip(arr,0,(1 << bits)-1).astype(np.uint16)
        if bits < 16:
            arr=np.left_shift(arr,16-bits).astype(np.uint16)
        with dest.open("wb") as fh:
            png.Writer(width=arr.shape[1],height=arr.shape[0],greyscale=True,bitdepth=16).write(fh,arr.tolist())

    def prepare(self) -> dict:
        p=self._paths(); p["processed"].mkdir(parents=True,exist_ok=True); p["canonical"].parent.mkdir(parents=True,exist_ok=True)
        if not p["source"].exists():
            raise FileNotFoundError(f"{p['source']} not found. Ground truth mapping is never invented; create source_manifest.csv from authorized dataset metadata.")
        src=pd.read_csv(p["source"])
        missing=[c for c in REQUIRED if c not in src.columns]
        if missing: raise ValueError(f"source_manifest.csv missing {missing}")
        out=[]; image_dir=p["processed"]/"images"; image_dir.mkdir(parents=True,exist_ok=True)
        for _,r in src.iterrows():
            row={"study_id":str(r.study_id),"patient_id":str(r.patient_id),"ground_truth":int(r.ground_truth)}
            for col,view in [("l_cc","L_CC"),("r_cc","R_CC"),("l_mlo","L_MLO"),("r_mlo","R_MLO")]:
                source=safe_workspace_path(str(r[col]))
                dest=image_dir/f"{row['study_id']}_{view}.png"
                self._convert_to_png(source,dest)
                row[col]=str(dest)
            for col in ["left_ground_truth","right_ground_truth","horizontal_flip"]:
                if col in src.columns and pd.notna(r.get(col)): row[col]=r.get(col)
            row.setdefault("horizontal_flip","NO")
            out.append(row)
        df=pd.DataFrame(out)
        df.to_csv(p["canonical"],index=False)
        audit("DATASET_PREPARED",dataset=self.key,studies=len(df),manifest=str(p["canonical"]))
        return {"dataset":self.key,"status":"AVAILABLE","studies":len(df),"manifest":str(p["canonical"])}

class CBISDDSMDatasetAdapter(ManifestDatasetAdapter): pass
class VinDrDatasetAdapter(ManifestDatasetAdapter): pass
