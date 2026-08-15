from __future__ import annotations
from pathlib import Path
import pandas as pd, pickle, shutil, re
from .workspace import safe_workspace_path

def build_batch(df: pd.DataFrame, run_dir: Path):
    images=run_dir/"images"; images.mkdir(parents=True,exist_ok=True)
    data=[]; study_order=[]
    for _,r in df.reset_index(drop=True).iterrows():
        sid=re.sub(r"[^A-Za-z0-9_.-]","_",str(r.study_id))
        study_order.append(str(r.study_id))
        names={}
        for col,key,suffix in [
            ("l_cc","L-CC","L_CC"),("r_cc","R-CC","R_CC"),
            ("l_mlo","L-MLO","L_MLO"),("r_mlo","R-MLO","R_MLO")]:
            src=safe_workspace_path(str(r[col])); stem=f"{sid}_{suffix}"; dst=images/f"{stem}.png"
            if not dst.exists(): shutil.copy2(src,dst)
            names[key]=[stem]
        left = int(r["left_ground_truth"]) if "left_ground_truth" in r.index and pd.notna(r["left_ground_truth"]) else 0
        right = int(r["right_ground_truth"]) if "right_ground_truth" in r.index and pd.notna(r["right_ground_truth"]) else 0
        data.append({**names,"cancer_label":{"left_malignant":left,"right_malignant":right},
                     "horizontal_flip":str(r.get("horizontal_flip","NO"))})
    pkl=run_dir/"data.pkl"
    with pkl.open("wb") as fh: pickle.dump(data,fh,protocol=4)
    pd.DataFrame({"position":range(len(study_order)),"study_id":study_order}).to_csv(run_dir/"study_order.csv",index=False)
    for p in [run_dir, images]: p.chmod(0o777)
    pkl.chmod(0o666)
    return images,pkl
