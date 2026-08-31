from __future__ import annotations
from pathlib import Path
import datetime, time, yaml, pandas as pd, json, math, shutil, hashlib
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

def _apply_web_label_blind_compat(data_pickle: Path) -> None:
    """Add optional benign-output metadata required by historical CPU runners.

    This helper is opt-in and used only by the Web single-case route.  It does not
    alter the canonical batch contract or introduce clinical ground truth.
    """
    import pickle
    with data_pickle.open("rb") as fh:
        web_batch = pickle.load(fh)
    for exam in web_batch:
        labels = exam.setdefault("cancer_label", {})
        labels.setdefault("left_benign", float("nan"))
        labels.setdefault("right_benign", float("nan"))
    with data_pickle.open("wb") as fh:
        pickle.dump(web_batch, fh, protocol=4)
    data_pickle.chmod(0o666)


def _infer_three(
    df: pd.DataFrame,
    run_dir: Path,
    run_id: str,
    device: str | None = None,
    *,
    web_label_blind_compat: bool = False,
    progress_callback=None,
    stage_progress_callback=None,
) -> pd.DataFrame:
    batch=run_dir/"model_batch"; batch.mkdir(parents=True,exist_ok=True); batch.chmod(0o777)
    input_preparation_started = time.monotonic()
    if stage_progress_callback is not None:
        stage_progress_callback(stage="MODEL_INPUT_PREPARATION", state="RUNNING")
    try:
        if web_label_blind_compat:
            runtime_root = run_dir.resolve()
            def _web_source(value):
                path = Path(value).resolve()
                if path != runtime_root and runtime_root not in path.parents:
                    raise ValueError(f"Web model input escaped the single-case runtime: {path}")
                return path
            images,pkl=build_batch(df,batch,source_path_resolver=_web_source)
        else:
            images,pkl=build_batch(df,batch)
        if web_label_blind_compat:
            # Historical CPU GMIC/GLAM runners read optional benign-label keys only
            # when constructing their result CSV after inference.
            _apply_web_label_blind_compat(pkl)
    except Exception:
        if stage_progress_callback is not None:
            stage_progress_callback(
                stage="MODEL_INPUT_PREPARATION", state="FAILED",
                elapsed_seconds=time.monotonic() - input_preparation_started,
            )
        raise
    if stage_progress_callback is not None:
        stage_progress_callback(
            stage="MODEL_INPUT_PREPARATION", state="SUCCESS",
            elapsed_seconds=time.monotonic() - input_preparation_started,
        )
    outputs={}; xai={}; resources=[]
    for model in MODELS:
        out=batch/f"{model}.csv"; out.touch(); out.chmod(0o666)
        pre=batch/"preprocessed"/model; pre.mkdir(parents=True,exist_ok=True); pre.chmod(0o777)
        audit("MODEL_INFERENCE_STARTED",run_id=run_id,model=model,studies=len(df),preprocessed_dir=str(pre))
        model_started = time.monotonic()
        if progress_callback is not None:
            progress_callback(model=model, state="RUNNING")
        try:
            if device is None:
                result=run_model(model,f"{run_id}-{model}",str(images),str(pkl),str(out),str(pre))
            else:
                result=run_model(model,f"{run_id}-{model}",str(images),str(pkl),str(out),str(pre),device=device)
            if result.get("status")!="SUCCESS":
                raise RuntimeError(f"Real model {model} failed: {result}")
        except Exception:
            if progress_callback is not None:
                progress_callback(model=model, state="FAILED", elapsed_seconds=time.monotonic() - model_started)
            raise
        wall_elapsed = time.monotonic() - model_started
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
        if progress_callback is not None:
            audit("WEB_MODEL_WALL_TIME", run_id=run_id, model=model, elapsed_seconds=wall_elapsed)
            progress_callback(model=model, state="SUCCESS", elapsed_seconds=wall_elapsed)
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


def _frame_sha256(df: pd.DataFrame) -> str:
    """Stable SHA-256 for the exact ordered dataframe passed to a formal chunk."""
    payload=df.to_csv(index=False,lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _formal_inference_config(chunk_size: int|None=None) -> dict:
    cfg=load_yaml("experiments.yaml").get("formal_inference",{})
    resolved=int(chunk_size if chunk_size is not None else cfg.get("chunk_size",25))
    if resolved <= 0:
        raise ValueError("chunk_size must be > 0")
    return {
        "chunk_size":resolved,
        "resume_enabled":bool(cfg.get("resume_enabled",True)),
        "cache_policy":str(cfg.get("cache_policy","SUCCESS_MARKER_AND_HASHES")),
        "cleanup_successful_chunk_temporaries":bool(cfg.get("cleanup_successful_chunk_temporaries",True)),
        "xai_retention_per_model_per_chunk":int(cfg.get("xai_retention_per_model_per_chunk",4)),
    }


def _evenly_spaced(items: list[str], limit: int) -> list[str]:
    ordered=sorted(dict.fromkeys(str(x) for x in items))
    if limit <= 0 or not ordered:
        return []
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    indexes=sorted(set(round(i*(len(ordered)-1)/(limit-1)) for i in range(limit)))
    return [ordered[i] for i in indexes]


def _retain_xai_and_cleanup_model_batch(cdir: Path, per_model_limit: int) -> dict:
    """Retain a compact deterministic XAI sample and remove heavyweight model_batch temporaries.

    This function never changes predictions, chunk identity or resume hashes. The chunk-level
    xai_artifacts.json is rewritten to reference only retained copies outside model_batch.
    """
    xp=cdir/"xai_artifacts.json"
    source={m:[] for m in MODELS}
    if xp.exists():
        try:
            payload=json.loads(xp.read_text(encoding="utf-8"))
            for model in MODELS:
                source[model]=[str(x) for x in payload.get(model,[])]
        except Exception:
            source={m:[] for m in MODELS}
    retained={m:[] for m in MODELS}; generated={m:len(source[m]) for m in MODELS}
    root=cdir/"xai_retained"
    for model in MODELS:
        chosen=_evenly_spaced(source[model],per_model_limit)
        if chosen:
            dest_dir=root/model; dest_dir.mkdir(parents=True,exist_ok=True)
            for idx,raw in enumerate(chosen):
                src=Path(raw)
                if not src.exists() or not src.is_file():
                    continue
                suffix=src.suffix or ".bin"
                dest=dest_dir/f"{idx:04d}{suffix}"
                shutil.copy2(src,dest)
                retained[model].append(str(dest))
    write_json(xp,retained)
    batch=cdir/"model_batch"
    removed_paths=[]
    if batch.exists():
        for name in ("images","preprocessed","data.pkl"):
            target=batch/name
            if target.is_dir():
                shutil.rmtree(target)
                removed_paths.append(name)
            elif target.exists():
                target.unlink()
                removed_paths.append(name)
    summary={
        "policy":"deterministic_evenly_spaced_xai_sample_then_prune_heavy_model_batch_temporaries",
        "generated_xai_count":generated,
        "retained_xai_count":{m:len(retained[m]) for m in MODELS},
        "per_model_limit":int(per_model_limit),
        "removed_model_batch_paths":removed_paths,
        "native_model_csvs_preserved":all((batch/f"{m}.csv").exists() for m in MODELS) if batch.exists() else False,
        "study_order_preserved":(batch/"study_order.csv").exists() if batch.exists() else False,
        "predictions_modified":False,
    }
    write_json(cdir/"cleanup_summary.json",summary)
    return summary


def _cleanup_orientation_temporaries(cdir: Path) -> dict:
    removed=[]
    for name in ("original","counterfactual"):
        p=cdir/name
        if p.exists():
            shutil.rmtree(p)
            removed.append(name)
    summary={
        "policy":"remove_orientation_preflight_workdirs_after_persisted_evidence",
        "removed_directories":removed,
        "resolved_manifest_preserved":(cdir/"resolved_manifest.csv").exists(),
        "evidence_csvs_preserved":True,
    }
    write_json(cdir/"cleanup_summary.json",summary)
    return summary


def _chunk_cache_status(sub: pd.DataFrame, cdir: Path, chunk_index: int) -> tuple[bool,pd.DataFrame|None,str|None]:
    """Validate a completed chunk before reuse.

    A SUCCESS marker is never trusted by itself. The ordered input hash, prediction
    file hash, row count and study identity must all match. A mismatched SUCCESS
    cache is treated as a methodological integrity error rather than silently
    re-inferred.
    """
    status_path=cdir/"chunk_status.json"
    pred_path=cdir/"raw_model_predictions.csv"
    if not status_path.exists():
        return False,None,"NO_SUCCESS_MARKER"
    try:
        status=json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False,None,"UNREADABLE_SUCCESS_MARKER"
    if status.get("status") != "SUCCESS":
        return False,None,"NON_SUCCESS_MARKER"
    expected_input=_frame_sha256(sub)
    if status.get("input_manifest_sha256") != expected_input:
        raise RuntimeError(f"Chunk {chunk_index:04d} SUCCESS cache input hash differs from current manifest; refusing reuse")
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        raise RuntimeError(f"Chunk {chunk_index:04d} is marked SUCCESS but predictions are missing; refusing silent recovery")
    if status.get("predictions_sha256") != _sha256_file(pred_path):
        raise RuntimeError(f"Chunk {chunk_index:04d} SUCCESS cache prediction hash mismatch; refusing reuse")
    pred=pd.read_csv(pred_path,dtype={"study_id":str,"patient_id":str,"dataset_source":str})
    required={"study_id","patient_id","ground_truth","dataset_source","gmic_score","nyu_score","glam_score"}
    if not required.issubset(pred.columns):
        raise RuntimeError(f"Chunk {chunk_index:04d} is marked SUCCESS but prediction schema is incomplete; refusing reuse")
    if len(pred) != len(sub) or int(status.get("studies",-1)) != len(sub):
        raise RuntimeError(f"Chunk {chunk_index:04d} is marked SUCCESS but row count differs; refusing reuse")
    expected_ids=sub.study_id.astype(str).tolist()
    actual_ids=pred.study_id.astype(str).tolist()
    if actual_ids != expected_ids:
        raise RuntimeError(f"Chunk {chunk_index:04d} SUCCESS cache study order/identity differs; refusing reuse")
    return True,pred,None


def _write_formal_progress(run_dir: Path, *, run_id: str, chunk_size: int, total_studies: int,
                           total_chunks: int, completed_chunks: int, completed_studies: int,
                           status: str, current_chunk: int|None=None, reused_chunks: int=0,
                           error: str|None=None) -> None:
    write_json(run_dir/"chunk_progress.json",{
        "run_id":run_id,
        "status":status,
        "chunk_size":int(chunk_size),
        "total_studies":int(total_studies),
        "total_chunks":int(total_chunks),
        "completed_chunks":int(completed_chunks),
        "completed_studies":int(completed_studies),
        "remaining_studies":int(max(0,total_studies-completed_studies)),
        "current_chunk":current_chunk,
        "reused_chunks":int(reused_chunks),
        "resume_enabled":True,
        "updated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error":error,
    })


def _infer_three_chunked(df: pd.DataFrame, run_dir: Path, run_id: str, chunk_size: int|None=None) -> pd.DataFrame:
    """Formal inference with deterministic chunks, checkpoints and safe resume.

    Models remain sequential inside each chunk (GMIC -> NYU -> GLAM). Successful
    chunks are reused only after hash and identity validation. An interrupted or
    failed chunk is rerun from its beginning; earlier successful chunks are kept.
    """
    policy=_formal_inference_config(chunk_size)
    size=int(policy["chunk_size"])
    run_dir.mkdir(parents=True,exist_ok=True)
    if df.empty:
        raise ValueError("Cannot infer an empty formal set")

    # Small unit/diagnostic sets preserve the historical single-call path.
    if len(df) <= size:
        return _infer_three(df,run_dir,run_id)

    total_chunks=int(math.ceil(len(df)/size))
    chunks=[]; chunk_dirs=[]; completed_studies=0; reused=0
    _write_formal_progress(run_dir,run_id=run_id,chunk_size=size,total_studies=len(df),total_chunks=total_chunks,
                           completed_chunks=0,completed_studies=0,status="RUNNING")
    for start in range(0,len(df),size):
        chunk_index=start//size
        sub=df.iloc[start:start+size].copy().reset_index(drop=True)
        cdir=run_dir/"chunks"/f"{chunk_index:04d}"
        cdir.mkdir(parents=True,exist_ok=True)
        input_hash=_frame_sha256(sub)
        valid,cached,reason=_chunk_cache_status(sub,cdir,chunk_index)
        if valid:
            reused += 1
            chunks.append(cached)
            chunk_dirs.append((chunk_index,cdir))
            completed_studies += len(sub)
            audit("FORMAL_CHUNK_REUSED",run_id=run_id,chunk=chunk_index,studies=len(sub),path=str(cdir))
            _write_formal_progress(run_dir,run_id=run_id,chunk_size=size,total_studies=len(df),total_chunks=total_chunks,
                                   completed_chunks=len(chunks),completed_studies=completed_studies,status="RUNNING",
                                   current_chunk=chunk_index,reused_chunks=reused)
            continue

        # A non-successful partial chunk is safe to restart from its beginning.
        if any(cdir.iterdir()):
            audit("FORMAL_CHUNK_PARTIAL_RESET",run_id=run_id,chunk=chunk_index,reason=reason)
            shutil.rmtree(cdir)
            cdir.mkdir(parents=True,exist_ok=True)
        sub.to_csv(cdir/"chunk_manifest.csv",index=False)
        _write_formal_progress(run_dir,run_id=run_id,chunk_size=size,total_studies=len(df),total_chunks=total_chunks,
                               completed_chunks=len(chunks),completed_studies=completed_studies,status="RUNNING",
                               current_chunk=chunk_index,reused_chunks=reused)
        audit("FORMAL_CHUNK_STARTED",run_id=run_id,chunk=chunk_index,studies=len(sub),input_manifest_sha256=input_hash)
        try:
            pred=_infer_three(sub,cdir,f"{run_id}-c{chunk_index:04d}")
            pred_path=cdir/"raw_model_predictions.csv"
            if policy["cleanup_successful_chunk_temporaries"]:
                try:
                    summary=_retain_xai_and_cleanup_model_batch(cdir,policy["xai_retention_per_model_per_chunk"])
                    audit("FORMAL_CHUNK_TEMPORARIES_CLEANED",run_id=run_id,chunk=chunk_index,**summary)
                except Exception as cleanup_exc:
                    audit("FORMAL_CHUNK_CLEANUP_WARNING",run_id=run_id,chunk=chunk_index,error=str(cleanup_exc))
            status_payload={
                "status":"SUCCESS","run_id":run_id,"chunk":chunk_index,"studies":int(len(sub)),
                "input_manifest_sha256":input_hash,"predictions_sha256":_sha256_file(pred_path),
                "version":"1.0.0","completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            write_json(cdir/"chunk_status.json",status_payload)
            chunks.append(pred); chunk_dirs.append((chunk_index,cdir)); completed_studies += len(sub)
            audit("FORMAL_CHUNK_COMPLETED",run_id=run_id,chunk=chunk_index,studies=len(sub),predictions_sha256=status_payload["predictions_sha256"])
            _write_formal_progress(run_dir,run_id=run_id,chunk_size=size,total_studies=len(df),total_chunks=total_chunks,
                                   completed_chunks=len(chunks),completed_studies=completed_studies,status="RUNNING",
                                   current_chunk=chunk_index,reused_chunks=reused)
        except Exception as exc:
            write_json(cdir/"chunk_status.json",{
                "status":"FAILED","run_id":run_id,"chunk":chunk_index,"studies":int(len(sub)),
                "input_manifest_sha256":input_hash,"error_type":type(exc).__name__,"error":str(exc),
                "failed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            _write_formal_progress(run_dir,run_id=run_id,chunk_size=size,total_studies=len(df),total_chunks=total_chunks,
                                   completed_chunks=len(chunks),completed_studies=completed_studies,status="INTERRUPTED_OR_FAILED",
                                   current_chunk=chunk_index,reused_chunks=reused,error=str(exc))
            audit("FORMAL_CHUNK_FAILED",run_id=run_id,chunk=chunk_index,error=str(exc))
            raise

    scores=pd.concat(chunks,ignore_index=True)
    if len(scores) != len(df):
        raise RuntimeError(f"Chunk aggregation incomplete: expected={len(df)} observed={len(scores)}")
    if scores.study_id.astype(str).tolist() != df.study_id.astype(str).tolist():
        raise RuntimeError("Chunk aggregation changed formal study order/identity")
    scores.to_csv(run_dir/"raw_model_predictions.csv",index=False)
    _aggregate_chunk_evidence(run_dir,chunk_dirs)
    _write_formal_progress(run_dir,run_id=run_id,chunk_size=size,total_studies=len(df),total_chunks=total_chunks,
                           completed_chunks=total_chunks,completed_studies=len(df),status="SUCCESS",reused_chunks=reused)
    audit("FORMAL_CHUNKED_INFERENCE_COMPLETED",run_id=run_id,studies=len(df),chunks=total_chunks,reused_chunks=reused)
    return scores


def _orientation_chunk_cache_status(sub: pd.DataFrame, cdir: Path, chunk_index: int) -> tuple[bool,pd.DataFrame|None,str|None]:
    status_path=cdir/"orientation_chunk_status.json"
    resolved_path=cdir/"resolved_manifest.csv"
    if not status_path.exists():
        return False,None,"NO_SUCCESS_MARKER"
    try:
        status=json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False,None,"UNREADABLE_SUCCESS_MARKER"
    if status.get("status") != "SUCCESS":
        return False,None,"NON_SUCCESS_MARKER"
    if status.get("input_manifest_sha256") != _frame_sha256(sub):
        raise RuntimeError(f"Orientation chunk {chunk_index:04d} SUCCESS cache input hash differs; refusing reuse")
    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        raise RuntimeError(f"Orientation chunk {chunk_index:04d} is marked SUCCESS but resolved manifest is missing; refusing reuse")
    if status.get("resolved_manifest_sha256") != _sha256_file(resolved_path):
        raise RuntimeError(f"Orientation chunk {chunk_index:04d} resolved-manifest hash mismatch; refusing reuse")
    resolved=pd.read_csv(resolved_path,dtype={"study_id":str,"patient_id":str,"dataset_source":str})
    if len(resolved) != len(sub) or resolved.study_id.astype(str).tolist() != sub.study_id.astype(str).tolist():
        raise RuntimeError(f"Orientation chunk {chunk_index:04d} study identity/order mismatch; refusing reuse")
    return True,resolved,None


def _aggregate_orientation_evidence(output_dir: Path, chunk_dirs: list[tuple[int,Path]], resolved: pd.DataFrame, run_id: str) -> None:
    resolution_parts=[]; evidence_parts=[]; original_parts=[]; counter_parts=[]; changed=[]; triggered=0
    for chunk_index,cdir in chunk_dirs:
        rp=cdir/"orientation_resolution.csv"
        if rp.exists():
            r=pd.read_csv(rp,dtype={"study_id":str})
            r.insert(0,"chunk",chunk_index); resolution_parts.append(r)
            if not r.empty and "triggered" in r.columns:
                t=r["triggered"] if r["triggered"].dtype == bool else r["triggered"].astype(str).str.lower().eq("true")
                triggered += int(t.fillna(False).sum())
            if "orientation_changed" in r.columns:
                c=r["orientation_changed"] if r["orientation_changed"].dtype == bool else r["orientation_changed"].astype(str).str.lower().eq("true")
                changed.extend(r.loc[c.fillna(False),"study_id"].astype(str).tolist())
        ep=cdir/"orientation_view_evidence.csv"
        if ep.exists():
            e=pd.read_csv(ep,dtype={"study_id":str}); e.insert(0,"chunk",chunk_index); evidence_parts.append(e)
        op=cdir/"orientation_original_views.csv"
        if op.exists():
            o=pd.read_csv(op,dtype={"study_id":str}); o.insert(0,"chunk",chunk_index); original_parts.append(o)
        cp=cdir/"orientation_counterfactual_views.csv"
        if cp.exists():
            c=pd.read_csv(cp,dtype={"study_id":str}); c.insert(0,"chunk",chunk_index); counter_parts.append(c)
    if resolution_parts: pd.concat(resolution_parts,ignore_index=True).to_csv(output_dir/"orientation_resolution.csv",index=False)
    if evidence_parts: pd.concat(evidence_parts,ignore_index=True).to_csv(output_dir/"orientation_view_evidence.csv",index=False)
    if original_parts: pd.concat(original_parts,ignore_index=True).to_csv(output_dir/"orientation_original_views.csv",index=False)
    if counter_parts: pd.concat(counter_parts,ignore_index=True).to_csv(output_dir/"orientation_counterfactual_views.csv",index=False)
    resolved.to_csv(output_dir/"resolved_manifest.csv",index=False)
    summary={
        "policy_id":ORIENTATION_POLICY_ID,"studies":int(len(resolved)),"triggered_studies":int(triggered),
        "orientation_changed_studies":int(len(changed)),"changed_study_ids":sorted(set(changed)),
        "decision_rule":"trigger only when all 4 views have non-zero distance_from_starting_side; accept toggle only when all 4 become zero",
        "ground_truth_used":False,"model_scores_used":False,"auc_used":False,"classifier_inference_performed":False,
        "execution_mode":"chunked_resumable","run_id":run_id,
    }
    write_json(output_dir/"orientation_policy_summary.json",summary)
    (output_dir/"orientation_policy_report.md").write_text(
        "# Automatic Orientation Resolution\n\n"
        f"- **policy**: {ORIENTATION_POLICY_ID}\n- **studies**: {len(resolved)}\n"
        f"- **triggered**: {triggered}\n- **orientation changed**: {len(changed)}\n"
        "- **ground truth used**: False\n- **model scores/AUC used**: False\n"
        "- **classifier inference during preflight**: False\n- **execution mode**: chunked/resumable\n",
        encoding="utf-8",
    )


def _resolve_orientation_chunked(df: pd.DataFrame, output_dir: Path, run_id: str, chunk_size: int|None=None) -> pd.DataFrame:
    """Apply the unchanged orientation policy per deterministic chunk with safe resume."""
    size=int(_formal_inference_config(chunk_size)["chunk_size"])
    output_dir.mkdir(parents=True,exist_ok=True)
    if df.empty: raise ValueError("Cannot resolve orientation for an empty formal set")
    if len(df) <= size:
        return resolve_orientation(df,output_dir,run_id)
    total_chunks=int(math.ceil(len(df)/size)); results=[]; chunk_dirs=[]; reused=0; completed=0
    write_json(output_dir/"orientation_chunk_progress.json",{
        "run_id":run_id,"status":"RUNNING","chunk_size":size,"total_studies":len(df),"total_chunks":total_chunks,
        "completed_chunks":0,"completed_studies":0,"reused_chunks":0,
    })
    for start in range(0,len(df),size):
        idx=start//size; sub=df.iloc[start:start+size].copy().reset_index(drop=True)
        cdir=output_dir/"chunks"/f"{idx:04d}"; cdir.mkdir(parents=True,exist_ok=True)
        valid,cached,reason=_orientation_chunk_cache_status(sub,cdir,idx)
        if valid:
            reused+=1; results.append(cached); chunk_dirs.append((idx,cdir)); completed+=len(sub)
            audit("ORIENTATION_CHUNK_REUSED",run_id=run_id,chunk=idx,studies=len(sub))
        else:
            if any(cdir.iterdir()):
                audit("ORIENTATION_CHUNK_PARTIAL_RESET",run_id=run_id,chunk=idx,reason=reason)
                shutil.rmtree(cdir); cdir.mkdir(parents=True,exist_ok=True)
            sub.to_csv(cdir/"input_manifest.csv",index=False)
            audit("ORIENTATION_CHUNK_STARTED",run_id=run_id,chunk=idx,studies=len(sub))
            try:
                resolved=resolve_orientation(sub,cdir,f"{run_id}-c{idx:04d}")
                if _formal_inference_config(size)["cleanup_successful_chunk_temporaries"]:
                    try:
                        summary=_cleanup_orientation_temporaries(cdir)
                        audit("ORIENTATION_CHUNK_TEMPORARIES_CLEANED",run_id=run_id,chunk=idx,**summary)
                    except Exception as cleanup_exc:
                        audit("ORIENTATION_CHUNK_CLEANUP_WARNING",run_id=run_id,chunk=idx,error=str(cleanup_exc))
                status={
                    "status":"SUCCESS","run_id":run_id,"chunk":idx,"studies":len(sub),
                    "input_manifest_sha256":_frame_sha256(sub),
                    "resolved_manifest_sha256":_sha256_file(cdir/"resolved_manifest.csv"),
                    "version":"1.0.0","completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                write_json(cdir/"orientation_chunk_status.json",status)
                results.append(resolved); chunk_dirs.append((idx,cdir)); completed+=len(sub)
                audit("ORIENTATION_CHUNK_COMPLETED",run_id=run_id,chunk=idx,studies=len(sub))
            except Exception as exc:
                write_json(cdir/"orientation_chunk_status.json",{
                    "status":"FAILED","run_id":run_id,"chunk":idx,"studies":len(sub),
                    "input_manifest_sha256":_frame_sha256(sub),"error_type":type(exc).__name__,"error":str(exc),
                })
                write_json(output_dir/"orientation_chunk_progress.json",{
                    "run_id":run_id,"status":"INTERRUPTED_OR_FAILED","chunk_size":size,"total_studies":len(df),
                    "total_chunks":total_chunks,"completed_chunks":len(results),"completed_studies":completed,
                    "current_chunk":idx,"reused_chunks":reused,"error":str(exc),
                })
                raise
        write_json(output_dir/"orientation_chunk_progress.json",{
            "run_id":run_id,"status":"RUNNING","chunk_size":size,"total_studies":len(df),"total_chunks":total_chunks,
            "completed_chunks":len(results),"completed_studies":completed,"current_chunk":idx,"reused_chunks":reused,
        })
    resolved_all=pd.concat(results,ignore_index=True)
    if len(resolved_all)!=len(df) or resolved_all.study_id.astype(str).tolist()!=df.study_id.astype(str).tolist():
        raise RuntimeError("Orientation chunk aggregation changed formal study coverage/order")
    _aggregate_orientation_evidence(output_dir,chunk_dirs,resolved_all,run_id)
    write_json(output_dir/"orientation_chunk_progress.json",{
        "run_id":run_id,"status":"SUCCESS","chunk_size":size,"total_studies":len(df),"total_chunks":total_chunks,
        "completed_chunks":total_chunks,"completed_studies":len(df),"reused_chunks":reused,
    })
    audit("ORIENTATION_CHUNKED_COMPLETED",run_id=run_id,studies=len(df),chunks=total_chunks,reused_chunks=reused)
    return resolved_all

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

def _formal_exclusion_manifest_path(policy: dict) -> Path:
    raw=Path(str(policy["manifest"]))
    return raw if raw.is_absolute() else WORKSPACE_ROOT/raw


def _apply_formal_exclusions(df: pd.DataFrame, datasets: list[str]) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Exclude pre-observed diagnostic patients before any formal split.

    v0.30.2 preserves the formal experiment rule introduced in v0.30.1 and treats as 100% of the still-unobserved pool.
    For RSNA the previously frozen 10-study diagnostic set is therefore excluded
    from both Configuration Set and Final Test before the 30/70 patient split.
    """
    cfg=load_yaml("experiments.yaml").get("formal_exclusions",{})
    out=df.copy()
    excluded_parts=[]
    details=[]
    for dataset_key in datasets:
        policy=cfg.get(str(dataset_key))
        if not policy:
            continue
        manifest=_formal_exclusion_manifest_path(policy)
        required=bool(policy.get("required",False))
        if not manifest.exists():
            if required:
                raise FileNotFoundError(
                    f"Formal exclusion manifest required for {dataset_key}: {manifest}. "
                    "Create/freeze the diagnostic set before the formal experiment."
                )
            details.append({"dataset":dataset_key,"manifest":str(manifest),"status":"NOT_PRESENT_OPTIONAL","excluded_studies":0})
            continue
        ex=pd.read_csv(manifest,dtype={"study_id":str,"patient_id":str})
        needed={"study_id","patient_id","ground_truth"}
        missing=sorted(needed-set(ex.columns))
        if missing:
            raise ValueError(f"Formal exclusion manifest {manifest} missing columns: {missing}")
        if ex.study_id.astype(str).duplicated().any():
            raise ValueError(f"Formal exclusion manifest {manifest} contains duplicate study_id")
        if ex.patient_id.astype(str).duplicated().any():
            raise ValueError(f"Formal exclusion manifest {manifest} contains duplicate patient_id")

        dataset_mask=out.dataset_source.astype(str).eq(str(dataset_key))
        dataset_rows=out.loc[dataset_mask].copy()
        dataset_rows["study_id"]=dataset_rows.study_id.astype(str)
        dataset_rows["patient_id"]=dataset_rows.patient_id.astype(str)
        ex["study_id"]=ex.study_id.astype(str)
        ex["patient_id"]=ex.patient_id.astype(str)
        ex["ground_truth"]=pd.to_numeric(ex.ground_truth,errors="raise").astype(int)

        observed=dataset_rows[dataset_rows.patient_id.isin(set(ex.patient_id))].copy()
        if len(observed)!=len(ex):
            missing_patients=sorted(set(ex.patient_id)-set(observed.patient_id))
            raise ValueError(
                f"Formal exclusion manifest {manifest} does not map one-to-one to prepared {dataset_key} patients; "
                f"expected={len(ex)} observed={len(observed)} missing={missing_patients[:10]}"
            )
        check=observed[["study_id","patient_id","ground_truth"]].merge(
            ex[["study_id","patient_id","ground_truth"]],on=["study_id","patient_id"],how="outer",suffixes=("_prepared","_exclusion"),indicator=True
        )
        if not check._merge.eq("both").all():
            raise ValueError(f"Formal exclusion manifest {manifest} study/patient identities do not match the prepared dataset")
        if not (check.ground_truth_prepared.astype(int)==check.ground_truth_exclusion.astype(int)).all():
            raise ValueError(f"Formal exclusion manifest {manifest} ground_truth differs from the prepared dataset")

        observed["formal_exclusion_reason"]=str(policy.get("reason","PREVIOUSLY_OBSERVED_DIAGNOSTIC_SET"))
        observed["formal_exclusion_manifest"]=str(manifest)
        excluded_parts.append(observed)
        remove_patients=set(ex.patient_id)
        out=out.loc[~(dataset_mask & out.patient_id.astype(str).isin(remove_patients))].copy()
        details.append({
            "dataset":dataset_key,"manifest":str(manifest),"status":"APPLIED",
            "excluded_studies":int(len(observed)),"excluded_patients":int(observed.patient_id.nunique()),
            "excluded_class_distribution":_class_distribution(observed),
            "exclude_from_configuration":True,"exclude_from_final":True,
        })
    excluded=pd.concat(excluded_parts,ignore_index=True) if excluded_parts else df.head(0).copy()
    summary={
        "policies":details,
        "total_excluded_studies":int(len(excluded)),
        "remaining_formal_pool_studies":int(len(out)),
        "remaining_formal_pool_class_distribution":_class_distribution(out),
    }
    return out.reset_index(drop=True),excluded.reset_index(drop=True),summary


def _split_formal_pool(df: pd.DataFrame, configuration_ratio: float, seed: int) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Deterministic stratified split by dataset+patient with exhaustive coverage."""
    from sklearn.model_selection import train_test_split
    ratio=float(configuration_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("configuration_ratio must be between 0 and 1")
    if df.empty:
        raise ValueError("Formal experimental pool is empty")
    required={"dataset_source","patient_id","study_id","ground_truth"}
    missing=sorted(required-set(df.columns))
    if missing:
        raise ValueError(f"Formal pool missing columns required for patient split: {missing}")

    groups=df.groupby(["dataset_source","patient_id"],as_index=False).ground_truth.max()
    stratify=groups.ground_truth if groups.ground_truth.nunique()>1 else None
    config_groups,final_groups=train_test_split(
        groups,train_size=ratio,random_state=int(seed),stratify=stratify
    )
    config_keys=set(zip(config_groups.dataset_source.astype(str),config_groups.patient_id.astype(str)))
    final_keys=set(zip(final_groups.dataset_source.astype(str),final_groups.patient_id.astype(str)))
    if config_keys & final_keys:
        raise AssertionError("Patient leakage detected between Configuration Set and Final Test")

    keys=list(zip(df.dataset_source.astype(str),df.patient_id.astype(str)))
    config_mask=pd.Series([k in config_keys for k in keys],index=df.index)
    final_mask=pd.Series([k in final_keys for k in keys],index=df.index)
    if (config_mask & final_mask).any() or not (config_mask | final_mask).all():
        raise AssertionError("Formal split must assign every study to exactly one subset")
    config_df=df.loc[config_mask].copy().reset_index(drop=True)
    final_df=df.loc[final_mask].copy().reset_index(drop=True)
    if set(config_df.study_id.astype(str)) & set(final_df.study_id.astype(str)):
        raise AssertionError("Study leakage detected between Configuration Set and Final Test")
    if len(config_df)+len(final_df)!=len(df):
        raise AssertionError("Configuration + Final must cover 100% of the formal pool")

    summary={
        "split_by":"dataset_source+patient_id",
        "stratified_by":"patient_ground_truth",
        "seed":int(seed),
        "requested_configuration_ratio":ratio,
        "requested_final_ratio":float(1.0-ratio),
        "formal_pool_studies":int(len(df)),
        "formal_pool_patients":int(len(groups)),
        "formal_pool_class_distribution":_class_distribution(df),
        "configuration_studies":int(len(config_df)),
        "configuration_patients":int(len(config_groups)),
        "configuration_class_distribution":_class_distribution(config_df),
        "final_test_studies":int(len(final_df)),
        "final_test_patients":int(len(final_groups)),
        "final_test_class_distribution":_class_distribution(final_df),
        "study_overlap":0,
        "patient_overlap":0,
        "formal_pool_coverage_studies":int(len(config_df)+len(final_df)),
        "formal_pool_coverage_fraction":float((len(config_df)+len(final_df))/len(df)),
    }
    return config_df,final_df,summary


def _sha256_file(path: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def experimental_test(datasets, samples=None, configuration_ratio=0.30, seed=42, chunk_size=None, resume_experiment=None):
    """Configuration phase with immutable split plus resumable orientation/inference.

    The v0.30.1 methodology is unchanged: diagnostic exclusions occur before a
    deterministic patient-level stratified 30/70 split. v0.30.2 only changes the
    execution strategy by checkpointing orientation and model inference chunks.
    """
    inference_cfg=_formal_inference_config(chunk_size)
    if resume_experiment:
        run_id=str(resume_experiment)
        run_dir=WORKSPACE_ROOT/"output"/"experiments"/run_id
        plan_path=run_dir/"experiment_plan.json"
        if not plan_path.exists(): raise FileNotFoundError(f"experiment_plan.json not found for resume: {run_id}")
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        if list(plan.get("datasets",[])) != list(datasets):
            raise ValueError(f"Resume datasets differ from original plan: planned={plan.get('datasets')} requested={list(datasets)}")
        if int(plan.get("seed")) != int(seed): raise ValueError("Resume seed differs from original experiment plan")
        if abs(float(plan.get("configuration_ratio"))-float(configuration_ratio)) > 1e-12:
            raise ValueError("Resume configuration_ratio differs from original experiment plan")
        if plan.get("samples") != samples:
            raise ValueError(f"Resume samples differs from original experiment plan: planned={plan.get('samples')} requested={samples}")
        planned_chunk=int(plan.get("formal_inference",{}).get("chunk_size",inference_cfg["chunk_size"]))
        if int(inference_cfg["chunk_size"]) != planned_chunk:
            raise ValueError(f"Resume chunk_size differs from original plan: planned={planned_chunk} requested={inference_cfg['chunk_size']}")
        if (run_dir/"frozen_configuration.yaml").exists():
            raise RuntimeError("Configuration is already frozen; do not resume Configuration inference")
        before_path=run_dir/"configuration_set_manifest_before_orientation.csv"; final_path=run_dir/"final_test_manifest.csv"
        if not before_path.exists() or not final_path.exists(): raise RuntimeError("Resume experiment is missing frozen split manifests")
        if _sha256_file(before_path) != plan.get("configuration_manifest_before_orientation_sha256"):
            raise RuntimeError("Pre-orientation Configuration manifest changed after planning; refusing resume")
        if _sha256_file(final_path) != plan.get("final_test_manifest_sha256"):
            raise RuntimeError("Final Test manifest changed after planning; refusing resume")
        config_before=pd.read_csv(before_path); final_df=pd.read_csv(final_path)
        formal_df=pd.read_csv(run_dir/"formal_pool_manifest.csv"); excluded_df=pd.read_csv(run_dir/"formal_exclusions_applied.csv")
        prepared_studies=int(plan.get("prepared_studies_before_formal_exclusions",len(formal_df)+len(excluded_df)))
        exclusion_summary=plan.get("formal_exclusion_policy",{})
        audit("EXPERIMENT_CONFIGURATION_RESUMED",run_id=run_id,chunk_size=planned_chunk,configuration_studies=len(config_before))
    else:
        run_id=_id("experiment"); run_dir=WORKSPACE_ROOT/"output"/"experiments"/run_id; run_dir.mkdir(parents=True,exist_ok=True)
        audit("EXPERIMENT_CONFIGURATION_STARTED",run_id=run_id,datasets=datasets,samples=samples,seed=seed,chunk_size=inference_cfg["chunk_size"])
        prepared_df=load_datasets(datasets); prepared_studies=int(len(prepared_df))
        formal_df,excluded_df,exclusion_summary=_apply_formal_exclusions(prepared_df,list(datasets))
        if samples is not None:
            requested=int(samples)
            if requested<=0: raise ValueError("samples must be > 0")
            formal_df=formal_df.head(min(requested,len(formal_df))).copy().reset_index(drop=True)
        formal_df.to_csv(run_dir/"formal_pool_manifest.csv",index=False); excluded_df.to_csv(run_dir/"formal_exclusions_applied.csv",index=False)
        config_before,final_df,split_summary=_split_formal_pool(formal_df,configuration_ratio,seed)
        write_json(run_dir/"split_summary.json",{"exclusions":exclusion_summary,**split_summary})
        config_before.to_csv(run_dir/"configuration_set_manifest_before_orientation.csv",index=False)
        final_df.to_csv(run_dir/"final_test_manifest.csv",index=False)
        final_hash=_sha256_file(run_dir/"final_test_manifest.csv"); before_hash=_sha256_file(run_dir/"configuration_set_manifest_before_orientation.csv")
        plan={
            "run_id":run_id,"datasets":list(datasets),"seed":int(seed),"samples":samples,
            "prepared_studies_before_formal_exclusions":prepared_studies,"formal_excluded_studies":int(len(excluded_df)),
            "formal_pool_studies":int(len(formal_df)),"formal_pool_class_distribution":_class_distribution(formal_df),
            "configuration_ratio":float(configuration_ratio),"final_test_ratio":float(1.0-float(configuration_ratio)),
            "configuration_studies":int(len(config_before)),"configuration_class_distribution":_class_distribution(config_before),
            "final_test_reserved_studies":int(len(final_df)),"final_test_class_distribution":_class_distribution(final_df),
            "formal_pool_coverage_fraction":1.0,"split_by_patient":True,"stratified":True,"patient_overlap":0,"study_overlap":0,
            "formal_exclusion_policy":exclusion_summary,
            "configuration_manifest_before_orientation_sha256":before_hash,"configuration_manifest_sha256":None,
            "final_test_manifest_sha256":final_hash,
            "configuration_inference_before_freeze":True,"final_inference_before_freeze":False,
            "inference_policy":"Each formal study belongs to exactly one subset. Configuration is inferred before freeze; Final Test scores are created only after freeze.",
            "threshold_policy":"Five label-independent score quantiles are derived per weight combination from Configuration Set scores only.",
            "selection_policy":"Highest ROC-AUC by weights -> highest Balanced Accuracy by threshold -> Sensitivity -> Specificity/FP -> baseline distance. AUPRC/AP and F1 are reported but do not change this policy.",
            "orientation_policy":ORIENTATION_POLICY_ID,"orientation_policy_ground_truth_used":False,"final_orientation_resolution_before_freeze":False,
            "formal_inference":{"mode":"chunked_resumable","chunk_size":int(inference_cfg["chunk_size"]),"models_parallel":False,"model_order":MODELS,"resume_enabled":True,"cache_policy":inference_cfg["cache_policy"]},
            "reported_metrics_v0302":["roc_auc","auprc","sensitivity","specificity","precision_ppv","npv","f1","balanced_accuracy","accuracy","tn","fp","fn","tp"],
            "execution_status":"SPLIT_FROZEN_ORIENTATION_PENDING",
        }
        write_json(run_dir/"experiment_plan.json",plan)

    # Configuration orientation is part of formal execution and is checkpointed too.
    config_df=_resolve_orientation_chunked(config_before,run_dir/"configuration_orientation",f"{run_id}-configuration",inference_cfg["chunk_size"])
    config_df.to_csv(run_dir/"configuration_set_manifest.csv",index=False)
    plan_path=run_dir/"experiment_plan.json"; plan=json.loads(plan_path.read_text(encoding="utf-8"))
    config_hash=_sha256_file(run_dir/"configuration_set_manifest.csv")
    previous_hash=plan.get("configuration_manifest_sha256")
    if previous_hash and previous_hash != config_hash:
        raise RuntimeError("Resolved Configuration manifest differs from the previously completed orientation result")
    plan["configuration_manifest_sha256"]=config_hash; plan["execution_status"]="CONFIGURATION_INFERENCE_RUNNING"
    write_json(plan_path,plan)

    # Only Configuration Set can be inferred before freeze.
    scores=_infer_three_chunked(config_df,run_dir/"configuration_inference",run_id,inference_cfg["chunk_size"])
    scores.to_csv(run_dir/"configuration_set_predictions.csv",index=False)
    analyze_score_frame(scores,run_dir/"configuration_score_analysis",source=str(run_dir/"configuration_set_predictions.csv"))
    results=all_configurations(scores); results.to_csv(run_dir/"all_configurations.csv",index=False)
    rank=ranking(results); rank.to_csv(run_dir/"ranking.csv",index=False)
    sel=select_configuration(results); write_json(run_dir/"best_configuration.json",sel.to_dict())
    write_report(run_dir/"configuration_report.md","Experimental Configuration Report",{
        "run_id":run_id,"prepared_studies":prepared_studies,"formal_excluded_studies":len(excluded_df),
        "formal_pool_studies":len(formal_df),"configuration_studies":len(config_df),"final_test_reserved":len(final_df),
        "configurations":80,"threshold_strategy":threshold_strategy_config(),"formal_chunk_size":inference_cfg["chunk_size"],
        "selected_weight_id":sel.weight_id,"selected_threshold":sel.threshold,"selected_roc_auc":sel.roc_auc,
        "selected_auprc":sel.auprc,"selected_f1":sel.f1,"selected_balanced_accuracy":sel.balanced_accuracy,
        "selected_sensitivity":sel.sensitivity,"selected_specificity":sel.specificity,"selected_fp":int(sel.fp),"selected_fn":int(sel.fn),
        "next_step":f"python -m experiments.freeze --experiment {run_id}"})
    plan=json.loads(plan_path.read_text(encoding="utf-8")); plan["execution_status"]="CONFIGURATION_SELECTED"; write_json(plan_path,plan)
    save_run(run_id,"experimental_configuration","CONFIGURATION_SELECTED",str(run_dir))
    audit("EXPERIMENT_CONFIGURATION_COMPLETED",run_id=run_id,selected_weight_id=sel.weight_id,threshold=float(sel.threshold),configuration_studies=len(config_df),final_reserved=len(final_df))
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
    plan_path=run_dir/"experiment_plan.json"
    if plan_path.exists():
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        expected_hash=plan.get("final_test_manifest_sha256")
        if expected_hash and _sha256_file(final_manifest)!=expected_hash:
            raise RuntimeError("final_test_manifest.csv changed after experiment planning; refusing Final Test inference")
    frozen=yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    final_df=pd.read_csv(final_manifest)
    final_chunk_size=int((plan if plan_path.exists() else {}).get("formal_inference",{}).get("chunk_size",_formal_inference_config()["chunk_size"]))

    exclusions_path=run_dir/"formal_exclusions_applied.csv"
    if exclusions_path.exists() and exclusions_path.stat().st_size:
        ex=pd.read_csv(exclusions_path,dtype={"study_id":str,"patient_id":str})
        if not ex.empty:
            final_keys=set(zip(final_df.dataset_source.astype(str),final_df.patient_id.astype(str)))
            excluded_keys=set(zip(ex.dataset_source.astype(str),ex.patient_id.astype(str)))
            overlap=final_keys & excluded_keys
            if overlap:
                raise RuntimeError(f"Final Test contains formally excluded diagnostic patients: {sorted(overlap)[:10]}")

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
        resolved_final=_resolve_orientation_chunked(final_df,final_inference_dir/"orientation_resolution",f"{experiment_id}-final",final_chunk_size)
        resolved_final.to_csv(run_dir/"final_test_manifest_resolved.csv",index=False)
        scores=_infer_three_chunked(resolved_final,final_inference_dir,f"{experiment_id}-final",final_chunk_size)
    w=frozen["weights"]; t=float(frozen["threshold"])
    selected_score=scores.gmic_score*w["gmic"]+scores.nyu_score*w["nyu"]+scores.glam_score*w["glam"]
    selected_metrics=evaluate(scores.ground_truth,selected_score,t)
    baseline_score=scores.gmic_score*0.333333+scores.nyu_score*0.333333+scores.glam_score*0.333334
    baseline_metrics=evaluate(scores.ground_truth,baseline_score,0.50)
    individual_metrics={m:evaluate(scores.ground_truth,scores[f"{m}_score"],0.50) for m in MODELS}

    final=scores.copy(); final["selected_score"]=selected_score; final["baseline_score"]=baseline_score
    final.to_csv(run_dir/"final_predictions.csv",index=False)
    comparison=[]
    for model in MODELS:
        comparison.append({"system":model,"system_type":"individual_model","threshold_role":"reference_0.50",**individual_metrics[model]})
    comparison.append({"system":"uniform_ensemble","system_type":"baseline_ensemble","threshold_role":"baseline_0.50",**baseline_metrics})
    comparison.append({"system":"selected_ensemble","system_type":"frozen_ensemble","threshold_role":"configuration_selected",**selected_metrics})
    pd.DataFrame(comparison).to_csv(run_dir/"final_model_comparison.csv",index=False)

    analyze_score_frame(scores,run_dir/"final_score_analysis",source=str(run_dir/"final_inference"/"raw_model_predictions.csv"),include_candidate_thresholds=False)
    write_json(run_dir/"final_metrics.json",{
        "selected":selected_metrics,"baseline":baseline_metrics,
        "individual_models_reference_threshold_0_5":individual_metrics,
        "comparison_note":"ROC-AUC and AUPRC/AP are threshold-independent. Threshold-dependent individual-model metrics use 0.50 only as a transparent reference and are not a calibrated clinical comparison."
    })
    write_report(run_dir/"final_report.md","Final Evaluation Report",{
        "experiment_id":experiment_id,"final_studies":len(final),
        "selected":selected_metrics,"baseline":baseline_metrics,
        "individual_models_reference_threshold_0_5":individual_metrics,
        "post_freeze_reoptimization":False})
    save_run(experiment_id,"experimental_final","SUCCESS",str(run_dir)); audit("FINAL_TEST_COMPLETED",experiment_id=experiment_id)
    return run_dir

