from __future__ import annotations
from pathlib import Path
import pandas as pd, pickle, shutil, re
from .workspace import safe_workspace_path

def build_batch(df: pd.DataFrame, run_dir: Path, source_path_resolver=None):
    resolver = source_path_resolver or safe_workspace_path
    images=run_dir/"images"; images.mkdir(parents=True,exist_ok=True)
    data=[]; study_order=[]; sanitized_keys=[]
    for _,r in df.reset_index(drop=True).iterrows():
        original_sid=str(r.study_id)
        sid=re.sub(r"[^A-Za-z0-9_.-]","_",original_sid)
        study_order.append(original_sid)
        sanitized_keys.append(sid)
        names={}
        for col,key,suffix in [
            ("l_cc","L-CC","L_CC"),("r_cc","R-CC","R_CC"),
            ("l_mlo","L-MLO","L_MLO"),("r_mlo","R-MLO","R_MLO")]:
            src=resolver(str(r[col])); stem=f"{sid}_{suffix}"; dst=images/f"{stem}.png"
            if not dst.exists(): shutil.copy2(src,dst)
            names[key]=[stem]
        left = int(r["left_ground_truth"]) if "left_ground_truth" in r.index and pd.notna(r["left_ground_truth"]) else 0
        right = int(r["right_ground_truth"]) if "right_ground_truth" in r.index and pd.notna(r["right_ground_truth"]) else 0
        data.append({**names,"cancer_label":{"left_malignant":left,"right_malignant":right},
                     "horizontal_flip":str(r.get("horizontal_flip","NO"))})
    pkl=run_dir/"data.pkl"
    with pkl.open("wb") as fh: pickle.dump(data,fh,protocol=4)
    if len(set(sanitized_keys)) != len(sanitized_keys):
        duplicates=pd.Series(sanitized_keys)[pd.Series(sanitized_keys).duplicated(keep=False)].tolist()
        raise ValueError(f"Sanitized study_id collision in model batch: {duplicates[:10]}")
    pd.DataFrame({
        "position":range(len(study_order)),
        "study_id":study_order,
        "study_key":sanitized_keys,
    }).to_csv(run_dir/"study_order.csv",index=False)
    for p in [run_dir, images]: p.chmod(0o777)
    pkl.chmod(0o666)
    return images,pkl
