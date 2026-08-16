from __future__ import annotations
from pathlib import Path
import datetime, time, yaml, pandas as pd, json, math
from .config import WORKSPACE_ROOT, load_yaml
from .datasets.manager import selected, adapter
from .datasets.manifest import read_manifest
from .metarepo_format import build_batch
from .model_client import run_model
from .prediction_parser import parse_image_level, parse_nyu
from .ensemble.soft_voting import vote
from .ensemble.metrics import evaluate
from .ensemble.experiment import all_configurations, ranking, select_configuration
from .reporting import write_json, write_report
from .storage import save_run
from .logging_utils import audit
from .score_analysis import analyze_score_frame, threshold_strategy_config
from .orientation_policy import resolve_orientation, POLICY_ID as ORIENTATION_POLICY_ID

MODELS=["gmic","nyu","glam"]
VIEW_COLUMNS=["l_cc","r_cc","l_mlo","r_mlo"]
SAMPLING_STRATEGIES=("sequential","random","stratified","balanced")

def _id(prefix): return f"{prefix}-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

def load_datasets(keys: list[str], samples: int|None=None) -> pd.DataFrame:
    """Load prepared canonical manifests.

    `samples` is retained for backwards compatibility with experimental/final helpers.
    Normal-test sampling is performed by `_sample_normal_dataset` so its strategy/seed
    can be persisted as research evidence.
    """
    frames=[]
    for key in selected(keys):
        st=adapter(key).status()
        if st["status"]!="AVAILABLE": raise RuntimeError(f"Dataset {key} is not prepared/AVAILABLE: {st}")
        df=read_manifest(st["canonical_manifest"]).copy(); df["dataset_source"]=key
        frames.append(df)
    out=pd.concat(frames,ignore_index=True)
    if samples is not None: out=out.head(int(samples)).copy()
    return out

def _class_distribution(df: pd.DataFrame) -> dict[str,int]:
    y=df["ground_truth"].astype(int)
    return {"BENIGN":int((y==0).sum()),"MALIGNANT":int((y==1).sum())}

def _proportional_quotas(df: pd.DataFrame, n: int) -> dict[int,int]:
    counts=df["ground_truth"].astype(int).value_counts().sort_index().to_dict()
    total=sum(counts.values())
    if total == 0 or n <= 0: return {int(k):0 for k in counts}
    raw={int(k):n*int(v)/total for k,v in counts.items()}
    quotas={k:min(int(counts[k]), int(math.floor(raw[k]))) for k in raw}
    remaining=n-sum(quotas.values())
    order=sorted(raw, key=lambda k:(-(raw[k]-math.floor(raw[k])), k))
    while remaining>0:
        progressed=False
        for k in order:
            if quotas[k] < int(counts[k]):
                quotas[k]+=1; remaining-=1; progressed=True
                if remaining==0: break
        if not progressed: break
    return quotas

def _balanced_quotas(df: pd.DataFrame, n: int) -> dict[int,int]:
    counts=df["ground_truth"].astype(int).value_counts().sort_index().to_dict()
    classes=sorted(int(k) for k in counts)
    if len(classes)<2:
        raise ValueError("balanced sampling requires both ground-truth classes")
    base=n//len(classes); remainder=n%len(classes)
    quotas={k:base+(1 if i<remainder else 0) for i,k in enumerate(classes)}
    insufficient={k:q for k,q in quotas.items() if int(counts.get(k,0))<q}
    if insufficient:
        raise ValueError(f"balanced sampling cannot satisfy requested class quotas: {insufficient}; available={counts}")
    return quotas

def _sample_normal_dataset(df: pd.DataFrame, samples: int|None, strategy: str="sequential", seed: int=42) -> tuple[pd.DataFrame,dict]:
    strategy=str(strategy or "sequential").lower()
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling must be one of {SAMPLING_STRATEGIES}")
    available=len(df); requested=None if samples is None else int(samples)
    if requested is not None and requested <= 0: raise ValueError("samples must be > 0")
    n=available if requested is None else min(requested,available)
    requested_distribution=None
    if n == available and samples is None:
        selected_df=df.copy()
    elif strategy == "sequential":
        selected_df=df.head(n).copy()
    elif strategy == "random":
        selected_df=df.sample(n=n,random_state=int(seed)).copy()
    else:
        quotas=_proportional_quotas(df,n) if strategy=="stratified" else _balanced_quotas(df,n)
        requested_distribution={"BENIGN":int(quotas.get(0,0)),"MALIGNANT":int(quotas.get(1,0))}
        pieces=[]
        for label,q in sorted(quotas.items()):
            if q:
                pieces.append(df[df.ground_truth.astype(int)==int(label)].sample(n=int(q),random_state=int(seed)+int(label)*1009))
        selected_df=pd.concat(pieces,ignore_index=False).sample(frac=1.0,random_state=int(seed)).copy() if pieces else df.head(0).copy()
    selected_df=selected_df.reset_index(drop=True)
    info={
        "strategy":strategy,
        "seed":int(seed) if strategy in {"random","stratified","balanced"} else None,
        "requested_samples":requested,
        "available_studies":available,
        "selected_studies":len(selected_df),
        "available_class_distribution":_class_distribution(df),
        "requested_class_distribution":requested_distribution,
        "actual_class_distribution":_class_distribution(selected_df),
    }
    return selected_df,info

def _infer_three(df: pd.DataFrame, run_dir: Path, run_id: str) -> pd.DataFrame:
    batch=run_dir/"model_batch"; batch.mkdir(parents=True,exist_ok=True); batch.chmod(0o777)
    images,pkl=build_batch(df,batch)
    outputs={}; xai={}; resources=[]
    for model in MODELS:
        out=batch/f"{model}.csv"; out.touch(); out.chmod(0o666)
        pre=batch/"preprocessed"/model; pre.mkdir(parents=True,exist_ok=True); pre.chmod(0o777)
        audit("MODEL_INFERENCE_STARTED",run_id=run_id,model=model,studies=len(df),preprocessed_dir=str(pre))
        result=run_model(model,f"{run_id}-{model}",str(images),str(pkl),str(out),str(pre))
        if result.get("status")!="SUCCESS": raise RuntimeError(f"Real model {model} failed: {result}")
        metrics=result.get("resource_metrics") or {}
        audit(
            "MODEL_INFERENCE_COMPLETED",run_id=run_id,model=model,
            xai_count=len(result.get("xai_artifacts",[])),
            elapsed_seconds=metrics.get("elapsed_seconds"),
            max_gpu_memory_mib=metrics.get("max_gpu_memory_mib"),
            output_file=result.get("output_file"),
        )
        outputs[model]=out; xai[model]=result.get("xai_artifacts",[])
        resources.append({"model":model,**(result.get("resource_metrics") or {})})
    g=parse_image_level(outputs["gmic"],"gmic")
    l=parse_image_level(outputs["glam"],"glam")
    n=parse_nyu(outputs["nyu"],batch/"study_order.csv")
    base=df[["study_id","patient_id","ground_truth","dataset_source"]].copy(); base.study_id=base.study_id.astype(str)
    order=pd.read_csv(batch/"study_order.csv",dtype={"study_id":str,"study_key":str})
    if not {"study_id","study_key"}.issubset(order.columns):
        raise ValueError("study_order.csv must contain study_id and study_key")
    if order.study_id.duplicated().any() or order.study_key.duplicated().any():
        raise ValueError("Duplicate study identity in study_order.csv")
    if len(g)!=len(base) or len(l)!=len(base):
        raise ValueError("Image-level prediction count does not match study count")
    g=g.merge(order[["study_id","study_key"]],on="study_key",how="left",validate="one_to_one")
    l=l.merge(order[["study_id","study_key"]],on="study_key",how="left",validate="one_to_one")
    if g.study_id.isna().any() or l.study_id.isna().any():
        raise ValueError("Image-level prediction contains an unknown sanitized study key")
    merged=base.merge(g[["study_id","gmic_score"]],on="study_id",validate="one_to_one").merge(n,on="study_id",validate="one_to_one").merge(l[["study_id","glam_score"]],on="study_id",validate="one_to_one")
    if len(merged)!=len(base): raise ValueError("Prediction merge incomplete")
    merged.to_csv(run_dir/"raw_model_predictions.csv",index=False)
    write_json(run_dir/"xai_artifacts.json",xai)
    pd.DataFrame(resources).to_csv(run_dir/"resource_metrics.csv",index=False)
    return merged

def _aggregate_chunk_evidence(run_dir: Path, chunk_dirs: list[tuple[int,Path]]) -> None:
    xai={m:[] for m in MODELS}; resources=[]
    for chunk_index,cdir in chunk_dirs:
        xp=cdir/"xai_artifacts.json"
        if xp.exists():
            payload=json.loads(xp.read_text(encoding="utf-8"))
            for model in MODELS:
                xai[model].extend(payload.get(model,[]))
        rp=cdir/"resource_metrics.csv"
        if rp.exists():
            rdf=pd.read_csv(rp)
            rdf.insert(0,"chunk",chunk_index)
            resources.append(rdf)
    write_json(run_dir/"xai_artifacts.json",xai)
    if resources:
        pd.concat(resources,ignore_index=True).to_csv(run_dir/"resource_metrics.csv",index=False)

def _resolve_config(weights,threshold,config_file):
    base=load_yaml("ensemble.yaml")["baseline"]
    if config_file:
        p=Path(config_file); data=yaml.safe_load(p.read_text(encoding="utf-8"))
        return {k:float(v) for k,v in data["weights"].items()},float(data["threshold"]),"CONFIG_FILE"
    if weights:
        if len(weights)!=3: raise ValueError("weights must contain GMIC NYU GLAM")
        return dict(zip(MODELS,map(float,weights))),float(threshold if threshold is not None else base["threshold"]),"MANUAL"
    return {k:float(v) for k,v in base["weights"].items()},float(threshold if threshold is not None else base["threshold"]),"BASELINE"

def normal_test(datasets, samples=None, weights=None, threshold=None, config_file=None, max_runtime_minutes=None, sampling="sequential", seed=42):
    overall_started=time.monotonic()
    run_id=_id("normal"); run_dir=WORKSPACE_ROOT/"output"/"normal_tests"/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    audit("NORMAL_TEST_STARTED",run_id=run_id,datasets=datasets,samples=samples,sampling=sampling,seed=seed)
    try:
        available_df=load_datasets(datasets)
        df,sampling_info=_sample_normal_dataset(available_df,samples,sampling,seed)
        if df.empty: raise RuntimeError("Sampling selected zero studies")
        df.to_csv(run_dir/"selected_studies_before_orientation.csv",index=False)
        df=resolve_orientation(df,run_dir/"orientation_resolution",run_id)
        df.to_csv(run_dir/"selected_studies.csv",index=False)
        audit("NORMAL_TEST_SAMPLING_SELECTED",run_id=run_id,orientation_policy=ORIENTATION_POLICY_ID,**sampling_info)
        w,t,source=_resolve_config(weights,threshold,config_file)
        status="SUCCESS"
        if max_runtime_minutes is None:
            scores=_infer_three(df,run_dir,run_id)
        else:
            started=time.monotonic(); chunks=[]; chunk_dirs=[]; chunk_size=10
            for i in range(0,len(df),chunk_size):
                if i>0 and time.monotonic()-started>=float(max_runtime_minutes)*60:
                    status="PARTIAL_TIME_LIMIT"; break
                chunk_index=i//chunk_size
                sub=df.iloc[i:i+chunk_size].copy(); cdir=run_dir/"chunks"/f"{chunk_index:04d}"; cdir.mkdir(parents=True,exist_ok=True)
                chunks.append(_infer_three(sub,cdir,f"{run_id}-c{chunk_index:04d}")); chunk_dirs.append((chunk_index,cdir))
            if not chunks: raise RuntimeError("No complete chunk was processed")
            scores=pd.concat(chunks,ignore_index=True); scores.to_csv(run_dir/"raw_model_predictions.csv",index=False)
            _aggregate_chunk_evidence(run_dir,chunk_dirs)
        dis=float(load_yaml("ensemble.yaml")["discordance"]["range_threshold"])
        rows=[]
        for _,r in scores.iterrows():
            er=vote({"gmic":r.gmic_score,"nyu":r.nyu_score,"glam":r.glam_score},w,t,dis)
            rows.append({**r.to_dict(),**er.model_dump()})
        pred=pd.DataFrame(rows); pred.to_csv(run_dir/"predictions.csv",index=False)
        m=evaluate(pred.ground_truth,pred.ensemble_malignancy_score,t); write_json(run_dir/"metrics.json",m)
        processed_images=int(len(pred)*len(VIEW_COLUMNS))
        overall_elapsed=float(time.monotonic()-overall_started)
        config_payload={"source":source,"weights":w,"threshold":t,"sampling":sampling_info}
        (run_dir/"configuration_used.yaml").write_text(yaml.safe_dump(config_payload,sort_keys=False),encoding="utf-8")
        run_summary={
            "run_id":run_id,"status":status,"datasets":list(datasets),
            "requested_samples":samples,"requested_studies":int(sampling_info["selected_studies"]),
            "available_studies":int(sampling_info["available_studies"]),
            "processed_studies":int(len(pred)),"processed_images":processed_images,
            "sampling_strategy":sampling_info["strategy"],"sampling_seed":sampling_info["seed"],
            "available_class_distribution":sampling_info["available_class_distribution"],
            "requested_class_distribution":sampling_info["requested_class_distribution"],
            "actual_class_distribution":_class_distribution(pred),
            "max_runtime_minutes":max_runtime_minutes,"overall_elapsed_seconds":overall_elapsed,
            "orientation_policy":ORIENTATION_POLICY_ID,
        }
        write_json(run_dir/"run_summary.json",run_summary)
        write_report(run_dir/"normal_test_report.md","Normal Test Report",{**run_summary,"config_source":source,**m})
        save_run(run_id,"normal",status,str(run_dir)); audit("NORMAL_TEST_COMPLETED",run_id=run_id,status=status,studies=len(pred),images=processed_images,overall_elapsed_seconds=overall_elapsed); return run_dir
    except Exception as exc:
        write_json(run_dir/"error.json",{"type":type(exc).__name__,"message":str(exc),"overall_elapsed_seconds":float(time.monotonic()-overall_started)}); audit("NORMAL_TEST_FAILED",run_id=run_id,error=str(exc)); raise

def experimental_test(datasets, samples=None, configuration_ratio=0.30, seed=42):
    """Configuration phase only. The Final Test Set is reserved and not inferred here."""
    from sklearn.model_selection import train_test_split
    run_id=_id("experiment"); run_dir=WORKSPACE_ROOT/"output"/"experiments"/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    audit("EXPERIMENT_CONFIGURATION_STARTED",run_id=run_id,datasets=datasets,samples=samples,seed=seed)
    df=load_datasets(datasets,samples)
    patient=df.groupby("patient_id").ground_truth.max().reset_index()
    cp,fp=train_test_split(patient,train_size=float(configuration_ratio),random_state=int(seed),stratify=patient.ground_truth if patient.ground_truth.nunique()>1 else None)
    config_df=df[df.patient_id.isin(cp.patient_id)].copy(); final_df=df[df.patient_id.isin(fp.patient_id)].copy()
    config_df.to_csv(run_dir/"configuration_set_manifest_before_orientation.csv",index=False)
    config_df=resolve_orientation(config_df,run_dir/"configuration_orientation",f"{run_id}-configuration")
    config_df.to_csv(run_dir/"configuration_set_manifest.csv",index=False)
    # Final Test Set remains untouched until after freeze. Orientation is resolved deterministically at final evaluation time.
    final_df.to_csv(run_dir/"final_test_manifest.csv",index=False)
    write_json(run_dir/"experiment_plan.json",{
        "run_id":run_id,"seed":int(seed),"configuration_ratio":float(configuration_ratio),
        "configuration_studies":int(len(config_df)),"final_test_reserved_studies":int(len(final_df)),
        "configuration_inference_before_freeze":True,"final_inference_before_freeze":False,
        "inference_policy":"Each study is inferred at most once within this experiment; final-test scores are created only after freeze and reused on repeated final_evaluation calls.",
        "threshold_policy":"Five label-independent score quantiles are derived per weight combination from Configuration Set scores only.",
        "selection_policy":"Highest ROC-AUC by weights -> highest Balanced Accuracy by threshold -> Sensitivity -> Specificity/FP -> baseline distance.",
        "orientation_policy":ORIENTATION_POLICY_ID,
        "orientation_policy_ground_truth_used":False,
        "final_orientation_resolution_before_freeze":False,
    })
    # Critical methodological boundary: only Configuration Set is inferred before freeze.
    scores=_infer_three(config_df,run_dir/"configuration_inference",run_id)
    scores.to_csv(run_dir/"configuration_set_predictions.csv",index=False)
    analyze_score_frame(scores, run_dir/"configuration_score_analysis", source=str(run_dir/"configuration_set_predictions.csv"))
    results=all_configurations(scores); results.to_csv(run_dir/"all_configurations.csv",index=False)
    rank=ranking(results); rank.to_csv(run_dir/"ranking.csv",index=False)
    sel=select_configuration(results); write_json(run_dir/"best_configuration.json",sel.to_dict())
    write_report(run_dir/"configuration_report.md","Experimental Configuration Report",{
        "run_id":run_id,"configuration_studies":len(config_df),"final_test_reserved":len(final_df),
        "configurations":80,"threshold_strategy":threshold_strategy_config(),
        "selected_weight_id":sel.weight_id,"selected_threshold":sel.threshold,
        "selected_roc_auc":sel.roc_auc,"selected_balanced_accuracy":sel.balanced_accuracy,
        "selected_sensitivity":sel.sensitivity,"selected_specificity":sel.specificity,
        "selected_fp":int(sel.fp),"selected_fn":int(sel.fn),
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
            "selection_policy":"ROC-AUC by weights -> Balanced Accuracy -> Sensitivity -> Specificity/FP -> baseline distance",
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
    final_inference_dir=run_dir/"final_inference"
    cached_scores=final_inference_dir/"raw_model_predictions.csv"
    orientation_marker=final_inference_dir/"orientation_resolution"/"orientation_policy_summary.json"
    if cached_scores.exists():
        if not orientation_marker.exists():
            raise RuntimeError("Existing final inference cache predates the required v0.27 orientation policy; refusing silent reuse")
        marker=json.loads(orientation_marker.read_text(encoding="utf-8"))
        if marker.get("policy_id") != ORIENTATION_POLICY_ID:
            raise RuntimeError("Existing final inference cache uses a different orientation policy; refusing silent reuse")
        scores=pd.read_csv(cached_scores)
        required={"study_id","patient_id","ground_truth","dataset_source","gmic_score","nyu_score","glam_score"}
        if not required.issubset(scores.columns) or len(scores)!=len(final_df):
            raise RuntimeError("Existing final inference cache is incomplete/incompatible; refusing silent re-inference")
        audit("FINAL_TEST_SCORES_REUSED",experiment_id=experiment_id,studies=len(scores),path=str(cached_scores))
    else:
        final_df.to_csv(run_dir/"final_test_manifest_before_orientation.csv",index=False)
        resolved_final=resolve_orientation(final_df,final_inference_dir/"orientation_resolution",f"{experiment_id}-final")
        resolved_final.to_csv(run_dir/"final_test_manifest_resolved.csv",index=False)
        scores=_infer_three(resolved_final,final_inference_dir,f"{experiment_id}-final")
    w=frozen["weights"]; t=float(frozen["threshold"])
    selected_score=scores.gmic_score*w["gmic"]+scores.nyu_score*w["nyu"]+scores.glam_score*w["glam"]
    selected_metrics=evaluate(scores.ground_truth,selected_score,t)
    baseline_score=scores.gmic_score*0.333333+scores.nyu_score*0.333333+scores.glam_score*0.333334
    baseline_metrics=evaluate(scores.ground_truth,baseline_score,0.50)
    final=scores.copy(); final["selected_score"]=selected_score; final["baseline_score"]=baseline_score
    final.to_csv(run_dir/"final_predictions.csv",index=False)
    analyze_score_frame(scores, run_dir/"final_score_analysis", source=str(run_dir/"final_inference"/"raw_model_predictions.csv"), include_candidate_thresholds=False)
    write_json(run_dir/"final_metrics.json",{"selected":selected_metrics,"baseline":baseline_metrics})
    write_report(run_dir/"final_report.md","Final Evaluation Report",{
        "experiment_id":experiment_id,"final_studies":len(final),"selected":selected_metrics,"baseline":baseline_metrics})
    save_run(experiment_id,"experimental_final","SUCCESS",str(run_dir)); audit("FINAL_TEST_COMPLETED",experiment_id=experiment_id)
    return run_dir
