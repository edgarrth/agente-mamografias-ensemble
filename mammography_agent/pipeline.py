from __future__ import annotations
from pathlib import Path
import datetime, time, yaml, pandas as pd
from .config import WORKSPACE_ROOT, load_yaml
from .datasets.manager import selected, adapter
from .datasets.manifest import read_manifest
from .metarepo_format import build_batch
from .model_client import run_model
from .prediction_parser import parse_image_level, parse_nyu
from .ensemble.soft_voting import vote
from .ensemble.metrics import evaluate
from .ensemble.experiment import all_configurations, select_configuration
from .reporting import write_json, write_report
from .storage import save_run
from .logging_utils import audit

MODELS=["gmic","nyu","glam"]

def _id(prefix): return f"{prefix}-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

def load_datasets(keys: list[str], samples: int|None=None) -> pd.DataFrame:
    frames=[]
    for key in selected(keys):
        st=adapter(key).status()
        if st["status"]!="AVAILABLE": raise RuntimeError(f"Dataset {key} is not prepared/AVAILABLE: {st}")
        df=read_manifest(st["canonical_manifest"]).copy(); df["dataset_source"]=key
        frames.append(df)
    out=pd.concat(frames,ignore_index=True)
    if samples is not None: out=out.head(int(samples)).copy()
    return out

def _infer_three(df: pd.DataFrame, run_dir: Path, run_id: str) -> pd.DataFrame:
    batch=run_dir/"model_batch"; batch.mkdir(parents=True,exist_ok=True); batch.chmod(0o777)
    images,pkl=build_batch(df,batch)
    outputs={}; xai={}; resources=[]
    for model in MODELS:
        out=batch/f"{model}.csv"; out.touch(); out.chmod(0o666)
        pre=batch/"preprocessed"; pre.mkdir(exist_ok=True); pre.chmod(0o777)
        audit("MODEL_INFERENCE_STARTED",run_id=run_id,model=model,studies=len(df))
        result=run_model(model,f"{run_id}-{model}",str(images),str(pkl),str(out),str(pre))
        if result.get("status")!="SUCCESS": raise RuntimeError(f"Real model {model} failed: {result}")
        audit("MODEL_INFERENCE_COMPLETED",run_id=run_id,model=model,xai_count=len(result.get("xai_artifacts",[])))
        outputs[model]=out; xai[model]=result.get("xai_artifacts",[])
        resources.append({"model":model,**(result.get("resource_metrics") or {})})
    g=parse_image_level(outputs["gmic"],"gmic")
    l=parse_image_level(outputs["glam"],"glam")
    n=parse_nyu(outputs["nyu"],batch/"study_order.csv")
    base=df[["study_id","patient_id","ground_truth","dataset_source"]].copy(); base.study_id=base.study_id.astype(str)
    # image-level keys come from the sanitized study id used in filenames; align by order if exact keys differ.
    if len(g)!=len(base) or len(l)!=len(base): raise ValueError("Image-level prediction count does not match study count")
    g=g.copy(); l=l.copy(); g["study_id"]=base.study_id.values; l["study_id"]=base.study_id.values
    merged=base.merge(g[["study_id","gmic_score"]],on="study_id").merge(n,on="study_id").merge(l[["study_id","glam_score"]],on="study_id")
    if len(merged)!=len(base): raise ValueError("Prediction merge incomplete")
    merged.to_csv(run_dir/"raw_model_predictions.csv",index=False)
    write_json(run_dir/"xai_artifacts.json",xai)
    pd.DataFrame(resources).to_csv(run_dir/"resource_metrics.csv",index=False)
    return merged

def _resolve_config(weights,threshold,config_file):
    base=load_yaml("ensemble.yaml")["baseline"]
    if config_file:
        p=Path(config_file); data=yaml.safe_load(p.read_text(encoding="utf-8"))
        return {k:float(v) for k,v in data["weights"].items()},float(data["threshold"]),"CONFIG_FILE"
    if weights:
        if len(weights)!=3: raise ValueError("weights must contain GMIC NYU GLAM")
        return dict(zip(MODELS,map(float,weights))),float(threshold if threshold is not None else base["threshold"]),"MANUAL"
    return {k:float(v) for k,v in base["weights"].items()},float(threshold if threshold is not None else base["threshold"]),"BASELINE"

def normal_test(datasets, samples=None, weights=None, threshold=None, config_file=None, max_runtime_minutes=None):
    run_id=_id("normal"); run_dir=WORKSPACE_ROOT/"output"/"normal_tests"/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    audit("NORMAL_TEST_STARTED",run_id=run_id,datasets=datasets,samples=samples)
    try:
        df=load_datasets(datasets,samples); w,t,source=_resolve_config(weights,threshold,config_file)
        status="SUCCESS"
        if max_runtime_minutes is None:
            scores=_infer_three(df,run_dir,run_id)
        else:
            started=time.monotonic(); chunks=[]; chunk_size=10
            for i in range(0,len(df),chunk_size):
                if i>0 and time.monotonic()-started>=float(max_runtime_minutes)*60:
                    status="PARTIAL_TIME_LIMIT"; break
                sub=df.iloc[i:i+chunk_size].copy(); cdir=run_dir/"chunks"/f"{i//chunk_size:04d}"; cdir.mkdir(parents=True,exist_ok=True)
                chunks.append(_infer_three(sub,cdir,f"{run_id}-c{i//chunk_size:04d}"))
            if not chunks: raise RuntimeError("No complete chunk was processed")
            scores=pd.concat(chunks,ignore_index=True); scores.to_csv(run_dir/"raw_model_predictions.csv",index=False)
        dis=float(load_yaml("ensemble.yaml")["discordance"]["range_threshold"])
        rows=[]
        for _,r in scores.iterrows():
            er=vote({"gmic":r.gmic_score,"nyu":r.nyu_score,"glam":r.glam_score},w,t,dis)
            rows.append({**r.to_dict(),**er.model_dump()})
        pred=pd.DataFrame(rows); pred.to_csv(run_dir/"predictions.csv",index=False)
        m=evaluate(pred.ground_truth,pred.ensemble_malignancy_score,t); write_json(run_dir/"metrics.json",m)
        (run_dir/"configuration_used.yaml").write_text(yaml.safe_dump({"source":source,"weights":w,"threshold":t},sort_keys=False),encoding="utf-8")
        write_report(run_dir/"normal_test_report.md","Normal Test Report",{"run_id":run_id,"status":status,"processed_studies":len(pred),"requested_studies":len(df),"config_source":source,**m})
        save_run(run_id,"normal",status,str(run_dir)); audit("NORMAL_TEST_COMPLETED",run_id=run_id,status=status,studies=len(pred)); return run_dir
    except Exception as exc:
        write_json(run_dir/"error.json",{"type":type(exc).__name__,"message":str(exc)}); audit("NORMAL_TEST_FAILED",run_id=run_id,error=str(exc)); raise

def experimental_test(datasets, samples=None, configuration_ratio=0.30, seed=42):
    """Configuration phase only. The Final Test Set is reserved and not inferred here."""
    from sklearn.model_selection import train_test_split
    run_id=_id("experiment"); run_dir=WORKSPACE_ROOT/"output"/"experiments"/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    audit("EXPERIMENT_CONFIGURATION_STARTED",run_id=run_id,datasets=datasets,samples=samples,seed=seed)
    df=load_datasets(datasets,samples)
    patient=df.groupby("patient_id").ground_truth.max().reset_index()
    cp,fp=train_test_split(patient,train_size=float(configuration_ratio),random_state=int(seed),stratify=patient.ground_truth if patient.ground_truth.nunique()>1 else None)
    config_df=df[df.patient_id.isin(cp.patient_id)].copy(); final_df=df[df.patient_id.isin(fp.patient_id)].copy()
    config_df.to_csv(run_dir/"configuration_set_manifest.csv",index=False)
    final_df.to_csv(run_dir/"final_test_manifest.csv",index=False)
    # Critical methodological boundary: only Configuration Set is inferred before freeze.
    scores=_infer_three(config_df,run_dir/"configuration_inference",run_id)
    scores.to_csv(run_dir/"configuration_set_predictions.csv",index=False)
    results=all_configurations(scores); results.to_csv(run_dir/"all_configurations.csv",index=False)
    rank=results.sort_values(["roc_auc","fn","sensitivity","fp"],ascending=[False,True,False,True]); rank.to_csv(run_dir/"ranking.csv",index=False)
    sel=select_configuration(results); write_json(run_dir/"best_configuration.json",sel.to_dict())
    write_report(run_dir/"configuration_report.md","Experimental Configuration Report",{
        "run_id":run_id,"configuration_studies":len(config_df),"final_test_reserved":len(final_df),
        "configurations":80,"selected_weight_id":sel.weight_id,"selected_threshold":sel.threshold,
        "next_step":f"python -m experiments.freeze --experiment {run_id}"})
    save_run(run_id,"experimental_configuration","CONFIGURATION_SELECTED",str(run_dir))
    audit("EXPERIMENT_CONFIGURATION_COMPLETED",run_id=run_id,selected_weight_id=sel.weight_id,threshold=float(sel.threshold))
    return run_dir

def freeze_experiment(experiment_id: str) -> Path:
    run_dir=WORKSPACE_ROOT/"output"/"experiments"/experiment_id
    best=run_dir/"best_configuration.json"
    if not best.exists(): raise FileNotFoundError(f"best_configuration.json not found for {experiment_id}")
    import json
    sel=json.loads(best.read_text(encoding="utf-8"))
    frozen={"source_experiment":experiment_id,
            "weights":{"gmic":float(sel["w_gmic"]),"nyu":float(sel["w_nyu"]),"glam":float(sel["w_glam"])},
            "threshold":float(sel["threshold"]),
            "selection_policy":"ROC-AUC by weights -> FN/Sensitivity -> FP within tolerance -> baseline distance",
            "frozen":True}
    path=run_dir/"frozen_configuration.yaml"
    if path.exists():
        current=yaml.safe_load(path.read_text(encoding="utf-8"))
        if current != frozen: raise RuntimeError("Frozen configuration already exists with different content; it cannot be overwritten")
        return path
    path.write_text(yaml.safe_dump(frozen,sort_keys=False),encoding="utf-8")
    audit("CONFIGURATION_FROZEN",experiment_id=experiment_id,path=str(path))
    return path

def final_evaluation(experiment_id: str) -> Path:
    run_dir=WORKSPACE_ROOT/"output"/"experiments"/experiment_id
    frozen_path=run_dir/"frozen_configuration.yaml"; final_manifest=run_dir/"final_test_manifest.csv"
    if not frozen_path.exists(): raise RuntimeError("Freeze the selected configuration before final evaluation")
    if not final_manifest.exists(): raise FileNotFoundError(final_manifest)
    frozen=yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    final_df=pd.read_csv(final_manifest)
    audit("FINAL_TEST_STARTED",experiment_id=experiment_id,studies=len(final_df))
    scores=_infer_three(final_df,run_dir/"final_inference",f"{experiment_id}-final")
    w=frozen["weights"]; t=float(frozen["threshold"])
    selected_score=scores.gmic_score*w["gmic"]+scores.nyu_score*w["nyu"]+scores.glam_score*w["glam"]
    selected_metrics=evaluate(scores.ground_truth,selected_score,t)
    baseline_score=scores.gmic_score*0.333333+scores.nyu_score*0.333333+scores.glam_score*0.333334
    baseline_metrics=evaluate(scores.ground_truth,baseline_score,0.50)
    final=scores.copy(); final["selected_score"]=selected_score; final["baseline_score"]=baseline_score
    final.to_csv(run_dir/"final_predictions.csv",index=False)
    write_json(run_dir/"final_metrics.json",{"selected":selected_metrics,"baseline":baseline_metrics})
    write_report(run_dir/"final_report.md","Final Evaluation Report",{
        "experiment_id":experiment_id,"final_studies":len(final),"selected":selected_metrics,"baseline":baseline_metrics})
    save_run(experiment_id,"experimental_final","SUCCESS",str(run_dir)); audit("FINAL_TEST_COMPLETED",experiment_id=experiment_id)
    return run_dir
