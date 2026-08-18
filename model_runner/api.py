from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import threading
import time
import yaml

from .health_logging import install_healthcheck_access_filter

WORKSPACE = Path(os.getenv("WORKSPACE_ROOT", "/workspace"))
CONFIG = Path("/runner/config")
META = WORKSPACE / "runtime" / "mammography_metarepository"
ANCHOR = "mammography-workspace-anchor"
DEFAULT_MODEL_DEVICE = os.getenv("DEFAULT_MODEL_DEVICE", "cpu").strip().lower()
ALLOW_GPU = os.getenv("ALLOW_GPU", "false").lower() == "true"
GPU = os.getenv("GPU_NUMBER", "0")
BOOTSTRAP_MODE = os.getenv("MODEL_BOOTSTRAP_MODE", "lazy").lower()
RESOURCE_SAMPLE_SECONDS = float(os.getenv("RESOURCE_SAMPLE_SECONDS", "2"))

APP_VERSION = "0.30.2"

CONSOLE_LOG = logging.getLogger("mammography-model-runner")
if not CONSOLE_LOG.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    CONSOLE_LOG.addHandler(_console_handler)
CONSOLE_LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))

_CONSOLE_EVENT_EXACT = {
    "MODEL_RUNNER_READY", "GPU_LOCK_WAITING", "GPU_LOCK_ACQUIRED", "GPU_LOCK_RELEASED",
    "MODEL_RUN_STARTED", "MODEL_CHILD_CONTAINER_STARTED", "MODEL_COMMAND_STARTED",
    "MODEL_COMMAND_COMPLETED", "MODEL_RUN_SUCCESS", "MODEL_RUN_FAILED",
    "MODEL_IMAGE_BUILD_STARTED", "MODEL_IMAGE_BUILD_COMPLETED",
    "GPU_MODEL_IMAGE_BUILD_STARTED", "GPU_MODEL_IMAGE_BUILD_COMPLETED",
    "GPU_MODEL_PROBE_FAILED", "MODEL_SMOKE_FAILED",
    "MODEL_PREPROCESS_STARTED", "MODEL_PREPROCESS_COMPLETED", "MODEL_PREPROCESS_FAILED",
}

def _console_event(event: str) -> bool:
    return event in _CONSOLE_EVENT_EXACT or event.endswith("_FAILED")


def cfg() -> dict:
    return yaml.safe_load((CONFIG / "models.yaml").read_text(encoding="utf-8"))


def configured_models() -> dict:
    return cfg().get("models", {})


def spec_for(model: str) -> dict:
    model = model.strip().lower()
    models = configured_models()
    if model not in models:
        raise ValueError(f"Unknown model {model!r}; expected one of {sorted(models)}")
    return models[model]


def configured_device_for(model: str) -> str:
    """Resolve the deployment device for one model, independently of runtime profile.

    Runtime compatibility profiles are model characteristics stored in config/models.yaml.
    Device selection is a deployment choice and may be overridden per model through
    <MODEL>_DEVICE, falling back to DEFAULT_MODEL_DEVICE.
    """
    model = model.strip().lower()
    spec_for(model)
    value = os.getenv(f"{model.upper()}_DEVICE", DEFAULT_MODEL_DEVICE).strip().lower()
    if value not in {"cpu", "gpu"}:
        raise ValueError(f"{model.upper()}_DEVICE must be cpu or gpu; got {value!r}")
    return value


def configured_devices() -> dict[str, str]:
    return {model: configured_device_for(model) for model in sorted(configured_models())}


def configured_gpu_profiles() -> dict[str, str | None]:
    return {
        model: (str(gpu_compatibility_for(model).get("profile")) if gpu_compatibility_for(model).get("profile") else None)
        for model in sorted(configured_models())
    }


def log(event: str, model: str | None = None, **data):
    p = WORKSPACE / "logs"
    p.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "component": "model-runner",
        "event": event,
        **({"model": model} if model else {}),
        **data,
    }
    with (p / "model_runner.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    if _console_event(event):
        level=logging.ERROR if event.endswith("_FAILED") else logging.INFO
        compact={k:v for k,v in rec.items() if k not in {"timestamp","component","stdout","stderr","cmd"}}
        CONSOLE_LOG.log(level, "%s %s", event, compact)


@contextmanager
def shared_lock(name: str):
    """Process/container-safe lock backed by the host workspace."""
    lock_dir = WORKSPACE / "runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{name}.lock"
    with path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def sh(cmd: list[str], cwd: Path | None = None, timeout: int | float | None = None, model: str | None = None):
    log("COMMAND", model=model, cmd=cmd, cwd=str(cwd) if cwd else None)
    cp = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if cp.returncode:
        log("COMMAND_FAILED", model=model, cmd=cmd, stdout=cp.stdout[-8000:], stderr=cp.stderr[-8000:])
        raise RuntimeError(
            f"Command failed ({cp.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{cp.stdout[-4000:]}\nSTDERR:\n{cp.stderr[-4000:]}"
        )
    return cp.stdout.strip()


def ensure_git_safe_directory(path: Path) -> None:
    """Allow Git operations on a repository stored in the host-mounted workspace.

    WSL/Docker bind mounts can expose ownership that differs from the user inside the
    model-runner container. Git intentionally rejects such repositories unless the
    path is explicitly trusted. This changes only Git's local safety policy; it does
    not modify the upstream repository, model code, checkpoints or results.
    """
    target = str(path.resolve())
    probe = subprocess.run(
        ["git", "config", "--global", "--get-all", "safe.directory"],
        text=True,
        capture_output=True,
    )
    configured = {line.strip() for line in probe.stdout.splitlines() if line.strip()}
    if target not in configured:
        sh(["git", "config", "--global", "--add", "safe.directory", target])
        log(
            "GIT_SAFE_DIRECTORY_ADDED",
            path=target,
            reason="Repository is stored in the host-mounted workspace and ownership can differ inside Docker/WSL2.",
            model_code_changed=False,
            model_weights_changed=False,
        )


def compatibility_for(model: str) -> dict:
    return spec_for(model).get("build_compatibility", {}) or {}


def prepare_build_dockerfile(model: str) -> tuple[Path, dict]:
    """Create an auditable compatibility Dockerfile from the upstream Dockerfile.

    The patch is intentionally narrow: when configured, it replaces only the exact
    upstream base-image line. Any unexpected upstream drift fails explicitly rather
    than silently applying a broader modification.
    """
    spec = spec_for(model)
    upstream = spec["upstream_model_name"]
    source = META / "models" / upstream / "Dockerfile"
    if not source.is_file():
        raise FileNotFoundError(f"Upstream Dockerfile is missing: {source}")

    original = source.read_text(encoding="utf-8")
    compat = compatibility_for(model)
    if not compat or not compat.get("base_image_override"):
        return source, {
            "mode": "upstream",
            "source": str(source.relative_to(META)),
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "patched": False,
        }

    original_base = str(compat.get("original_base_image", "")).strip()
    replacement_base = str(compat["base_image_override"]).strip()
    expected = f"FROM {original_base}"
    lines = original.splitlines()
    if not lines or lines[0].strip() != expected:
        raise RuntimeError(
            "UPSTREAM_DOCKERFILE_DRIFT: compatibility patch refused because the first "
            f"Dockerfile line is {lines[0].strip() if lines else '<empty>'!r}, expected {expected!r}. "
            "Review the upstream Dockerfile before changing the thesis compatibility patch."
        )

    patched_lines = lines[:]
    patched_lines[0] = f"FROM {replacement_base}"

    # NVIDIA rotated the signing key used by CUDA APT repositories in 2022.
    # The historical GMIC/GLAM Dockerfiles predate that rotation, while the
    # NYU classifier Dockerfile already carries an upstream workaround.
    # In auto mode we preserve an upstream fix when present and otherwise
    # inject the same narrow key-refresh commands before the first apt update.
    key_mode = str(compat.get("nvidia_repository_key_rotation_fix", "disabled")).strip().lower()
    key_marker = "3bf863cc.pub"
    key_fix_status = "disabled"
    if key_mode == "auto":
        if key_marker in original:
            key_fix_status = "upstream_present"
        else:
            key_fix = [
                "",
                "# Thesis compatibility: NVIDIA CUDA repository signing-key rotation.",
                "# Runtime-only build repair; model source/checkpoints are unchanged.",
                "RUN apt-key del 7fa2af80 || true",
                "RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub",
                "RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub",
            ]
            patched_lines = [patched_lines[0], *key_fix, *patched_lines[1:]]
            key_fix_status = "injected"
            log(
                "NVIDIA_APT_KEY_ROTATION_COMPATIBILITY_APPLIED",
                model=model,
                reason="Historical CUDA Ubuntu 18.04 repository key is missing after NVIDIA signing-key rotation.",
                cuda_key="3bf863cc.pub",
                machine_learning_key="7fa2af80.pub",
                model_code_changed=False,
                model_weights_changed=False,
                training_performed=False,
            )
    elif key_mode not in {"disabled", "none", "false"}:
        raise RuntimeError(f"Unknown nvidia_repository_key_rotation_fix mode: {key_mode!r}")

    patched = "\n".join(patched_lines) + ("\n" if original.endswith("\n") else "")
    outdir = META / ".thesis_compat"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{model}.Dockerfile"
    out.write_text(patched, encoding="utf-8")

    metadata = {
        "mode": "base_image_override",
        "source": str(source.relative_to(META)),
        "generated": str(out.relative_to(META)),
        "original_base_image": original_base,
        "replacement_base_image": replacement_base,
        "nvidia_repository_key_rotation_fix": key_fix_status,
        "reason": compat.get("reason", "Compatibility replacement required to rebuild the historical environment."),
        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "model_code_changed": False,
        "model_weights_changed": False,
        "training_performed": False,
    }
    log("MODEL_COMPATIBILITY_DOCKERFILE_CREATED", model=model, **metadata)

    audit = WORKSPACE / "models" / "compatibility"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / f"{model}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, metadata


def ensure_meta() -> str:
    with shared_lock("metarepository"):
        META.parent.mkdir(parents=True, exist_ok=True)
        c = cfg()["meta_repository"]
        if META.exists() and not (META / ".git").exists():
            shutil.rmtree(META)
        if not (META / ".git").exists():
            sh(
                [
                    "git", "clone", "--depth", "1", "--branch", str(c.get("ref", "master")),
                    c["repository"], str(META),
                ],
                timeout=1800,
            )
        ensure_git_safe_directory(META)
        # Required by upstream Dockerfiles. Fixed research-only IDs keep the build reproducible.
        (META / "users.txt").write_text(
            "research,12000\nnyu_gmic_user,12001\nnyu_model_user,12002\nnyu_glam_user,12003\n",
            encoding="utf-8",
        )
        commit = sh(["git", "rev-parse", "HEAD"], cwd=META)
        registry = WORKSPACE / "models"
        registry.mkdir(parents=True, exist_ok=True)
        (registry / "metarepository.json").write_text(
            json.dumps(
                {
                    "repository": c["repository"],
                    "configured_ref": c.get("ref", "master"),
                    "resolved_commit": commit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return commit


def image_tag(model: str) -> str:
    spec = spec_for(model)
    return spec.get("local_image") or f"mammography-model-{model}:research"


def _probe(cmd: list[str], timeout: int = 10) -> dict:
    """Run a diagnostic command without raising; keep output bounded for health/doctor endpoints."""
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip()[-4000:],
            "stderr": cp.stderr.strip()[-4000:],
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(exc)}


def docker_diagnostics() -> dict:
    """Diagnose the runner -> host Docker Engine boundary.

    The runner uses the host Docker daemon through /var/run/docker.sock.
    A direct socket ping separates mount/permission problems from Docker CLI/API problems.
    """
    socket = Path("/var/run/docker.sock")
    direct = _probe(["curl", "--silent", "--show-error", "--unix-socket", str(socket), "http://localhost/_ping"]) if socket.exists() else {
        "ok": False, "returncode": None, "stdout": "", "stderr": "docker socket does not exist"
    }
    version = _probe(["docker", "version"])
    info = _probe(["docker", "info"])
    return {
        "docker_host": os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
        "socket_exists": socket.exists(),
        "socket_mode": oct(socket.stat().st_mode & 0o777) if socket.exists() else None,
        "direct_socket_ping": direct,
        "docker_version": version,
        "docker_info": info,
        "daemon_reachable": bool(direct.get("ok") and direct.get("stdout") == "OK" and info.get("ok")),
    }


def docker_daemon_ok() -> bool:
    return bool(docker_diagnostics()["daemon_reachable"])


def image_exists(model: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image_tag(model)], capture_output=True).returncode == 0


def gpu_compatibility_for(model: str) -> dict:
    return spec_for(model).get("gpu_compatibility", {}) or {}


def gpu_image_tag(model: str) -> str:
    compat = gpu_compatibility_for(model)
    tag = compat.get("image")
    if not tag:
        raise RuntimeError(f"GPU_PROFILE_NOT_CONFIGURED: {model} has no validated GPU compatibility image yet")
    return str(tag)


def gpu_image_exists(model: str) -> bool:
    try:
        tag = gpu_image_tag(model)
    except Exception:
        return False
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def gpu_probe_path(model: str) -> Path:
    return WORKSPACE / "models" / "gpu_compatibility" / f"{model}.probe.json"


def gpu_probe_passed(model: str) -> bool:
    p = gpu_probe_path(model)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("status") == "GPU_READY" and data.get("image") == gpu_image_tag(model)
    except Exception:
        return False


def ensure_gpu_image(model: str, force_rebuild: bool = False) -> dict:
    model = model.strip().lower()
    spec = spec_for(model)
    compat = gpu_compatibility_for(model)
    if not compat.get("enabled"):
        raise RuntimeError(f"GPU_PROFILE_NOT_CONFIGURED: {model} GPU compatibility is not enabled")
    configured_profile = str(compat.get("profile", "")).strip().lower()
    if not configured_profile:
        raise RuntimeError(f"GPU_PROFILE_NOT_CONFIGURED: {model} has no GPU compatibility profile")
    dockerfile = Path(str(compat.get("dockerfile", "")))
    if not dockerfile.is_file():
        raise FileNotFoundError(f"GPU compatibility Dockerfile is missing: {dockerfile}")
    commit = ensure_meta()
    tag = gpu_image_tag(model)
    build_revision = int(compat.get("build_revision", 1))
    dockerfile_sha256 = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    metadata = {
        "mode": "gpu_runtime_compatibility",
        "profile": configured_profile,
        "dockerfile": str(dockerfile),
        "dockerfile_sha256": dockerfile_sha256,
        "build_revision": build_revision,
        "python": str(compat.get("python")),
        "torch": str(compat.get("torch")),
        "torchvision": str(compat.get("torchvision")),
        "cuda_wheel": str(compat.get("cuda_wheel")),
        "reason": compat.get("reason"),
        "model_source_commit": spec.get("upstream_commit_from_metarepo"),
        "model_source_commit_changed": False,
        "compatibility_code_patches": list(compat.get("compatibility_code_patches", [])),
        "model_architecture_changed": False,
        "model_weights_changed": False,
        "training_performed": False,
        "runtime_dependencies_changed": True,
    }
    with shared_lock(f"gpu_image_build_{model}"):
        audit = WORKSPACE / "models" / "compatibility"
        audit.mkdir(parents=True, exist_ok=True)
        audit_file = audit / f"{model}-gpu.json"
        previous_revision = 1
        if audit_file.is_file():
            try:
                previous_revision = int(json.loads(audit_file.read_text(encoding="utf-8")).get("build_revision", 1))
            except Exception:
                previous_revision = 1
        image_present = gpu_image_exists(model)
        rebuild_required = bool(force_rebuild) or (not image_present) or previous_revision != build_revision
        if rebuild_required:
            log("GPU_MODEL_IMAGE_BUILD_STARTED", model=model, image=tag, previous_revision=previous_revision, force_rebuild=bool(force_rebuild), **metadata)
            sh(["docker", "build", "-t", tag, "-f", str(dockerfile), "."], cwd=META, timeout=7200, model=model)
            probe = gpu_probe_path(model)
            if probe.exists():
                probe.unlink()
            log("GPU_MODEL_IMAGE_BUILD_COMPLETED", model=model, image=tag, previous_revision=previous_revision, force_rebuild=bool(force_rebuild), probe_invalidated=True, **metadata)
        audit_file.write_text(
            json.dumps({"image": tag, "metarepo_commit": commit, **metadata}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return {
        "model": model,
        "image": tag,
        "metarepo_commit": commit,
        "gpu_compatibility": metadata,
        "status": "READY",
        "runner_service": "model-runner",
        "rebuild_performed": rebuild_required,
        "force_rebuild": bool(force_rebuild),
    }


def probe_gpu_runtime(model: str, timeout_seconds: int = 60) -> dict:
    model = model.strip().lower()
    info = ensure_gpu_image(model)
    tag = info["image"]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    container = f"mammography-gpu-probe-{safe}"
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)
    script = "\n".join([
        "import json, torch",
        "result = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}",
        "assert result['cuda_available'] and result['device_count'] >= 1, 'CUDA is not available inside the GPU compatibility image'",
        "result['gpu_name'] = torch.cuda.get_device_name(0)",
        "x = torch.tensor([1.0], device='cuda:0')",
        "y = x + 1.0",
        "torch.cuda.synchronize()",
        "result['allocation_ok'] = True",
        "result['kernel_ok'] = float(y.cpu().item()) == 2.0",
        "print(json.dumps(result))",
    ])
    try:
        sh([
            "docker", "run", "-d", "--name", container, "--gpus", f"device={GPU}",
            tag, "bash", "-lc", "trap : TERM INT; sleep infinity & wait"
        ], timeout=30, model=model)
        cp = subprocess.run(
            ["docker", "exec", container, "python", "-u", "-c", script],
            text=True, capture_output=True, timeout=timeout_seconds,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"GPU probe failed ({cp.returncode})\nSTDOUT:\n{cp.stdout[-4000:]}\nSTDERR:\n{cp.stderr[-4000:]}")
        payload = json.loads(cp.stdout.strip().splitlines()[-1])
        result = {
            "status": "GPU_READY", "model": model, "image": tag,
            "profile": str(gpu_compatibility_for(model).get("profile", "")), "gpu_number": GPU, **payload,
        }
        path = gpu_probe_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        # result already contains the model key; pass it once to avoid duplicate keyword binding.
        log("GPU_RUNTIME_PROBE_PASSED", **result)
        return result
    except subprocess.TimeoutExpired as exc:
        log("GPU_RUNTIME_PROBE_TIMEOUT", model=model, image=tag, timeout_seconds=timeout_seconds)
        raise RuntimeError(f"GPU_PROBE_TIMEOUT: {model} did not complete CUDA allocation/kernel test within {timeout_seconds}s") from exc
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(f.relative_to(path)).encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def write_model_registry(model: str, metarepo_commit: str, tag: str, build_metadata: dict | None = None):
    spec = spec_for(model)
    upstream = spec["upstream_model_name"]
    with shared_lock("model_registry"):
        reg_path = WORKSPACE / "models" / "registry.json"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
        image_id = sh(["docker", "image", "inspect", "--format={{.Id}}", tag], model=model) if image_exists(model) else None
        pred_dir = META / "models" / upstream / "predict"
        diagnostics = docker_diagnostics()
        reg[model] = {
            "academic_name": spec["academic_name"],
            "repository": spec["repository"],
            "upstream_commit": spec["upstream_commit_from_metarepo"],
            "license": spec["license"],
            "metarepository_commit": metarepo_commit,
            "docker_image": tag,
            "docker_image_id": image_id,
            "predict_package_sha256": tree_sha256(pred_dir) if pred_dir.exists() else None,
            "runner_service": "model-runner",
            "runner_docker_cli": diagnostics.get("docker_version", {}).get("stdout", "")[-2000:],
            "build_compatibility": build_metadata or {"mode": "upstream", "patched": False},
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_image(model: str) -> dict:
    model = model.strip().lower()
    spec = spec_for(model)
    upstream = spec["upstream_model_name"]
    commit = ensure_meta()
    tag = image_tag(model)
    build_dockerfile, build_metadata = prepare_build_dockerfile(model)
    dockerfile_arg = str(build_dockerfile.relative_to(META))
    with shared_lock(f"image_build_{model}"):
        if not image_exists(model):
            log(
                "MODEL_IMAGE_BUILD_STARTED",
                model=model,
                upstream=upstream,
                metarepo_commit=commit,
                image=tag,
                dockerfile=dockerfile_arg,
                build_compatibility=build_metadata,
            )
            sh(
                [
                    "docker", "build", "-t", tag,
                    "-f", dockerfile_arg,
                    "--build-arg", "GROUPNAME=research", ".",
                ],
                cwd=META,
                timeout=7200,
                model=model,
            )
            log("MODEL_IMAGE_BUILD_COMPLETED", model=model, image=tag, build_compatibility=build_metadata)
        write_model_registry(model, commit, tag, build_metadata=build_metadata)
    return {
        "model": model,
        "upstream": upstream,
        "image": tag,
        "metarepo_commit": commit,
        "build_compatibility": build_metadata,
        "status": "READY",
        "runner_service": "model-runner",
    }


def predict_path(model: str) -> str:
    upstream = spec_for(model)["upstream_model_name"]
    lines = (META / "models" / upstream / "config.txt").read_text(encoding="utf-8").splitlines()
    line = [x for x in lines if x.startswith("CONTAINER_PREDICT_PATH=")][0]
    return line.split("=", 1)[1].strip().strip('"')


def patched_predict_dir(model: str, run_id: str) -> Path:
    upstream = spec_for(model)["upstream_model_name"]
    src = META / "models" / upstream / "predict"
    dest = WORKSPACE / "runtime" / "predict_overrides" / run_id / model
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    # Only request visualization already implemented upstream. Never create synthetic XAI.
    if model in {"gmic", "glam"}:
        p = dest / "predict.sh"
        text = p.read_text(encoding="utf-8")
        if "--visualization-flag" not in text:
            text = re.sub(
                r'(--model-index\s+"\$\{(?:MODEL_INDEX|MODEL_NAME)\}")',
                r'\1 \\\n    --visualization-flag',
                text,
            )
            p.write_text(text, encoding="utf-8")
    return dest


def _parse_mem_mib(text: str) -> float | None:
    try:
        value, unit = text.strip().split()[:2]
        v = float(value)
        if unit.startswith("MiB"):
            return v
        if unit.startswith("GiB"):
            return v * 1024
        if unit.startswith("KiB"):
            return v / 1024
    except Exception:
        pass
    return None


def sample_container(container: str, device: str, stop: threading.Event, metric_samples: list[dict]):
    while not stop.is_set():
        rec: dict = {}
        cp = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}|{{.MemUsage}}", container],
            text=True,
            capture_output=True,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            try:
                cpu, mem = cp.stdout.strip().split("|", 1)
                rec["cpu_percent"] = float(cpu.strip().rstrip("%"))
                rec["memory_mib"] = _parse_mem_mib(mem.split("/", 1)[0])
            except Exception:
                pass
        if device == "gpu":
            # Query the child model container. The runner itself intentionally has no CUDA/PyTorch stack.
            gp = subprocess.run(
                [
                    "docker", "exec", container, "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
            )
            if gp.returncode == 0 and gp.stdout.strip():
                try:
                    util, mem = gp.stdout.strip().splitlines()[0].split(",")[:2]
                    rec["gpu_util_percent"] = float(util.strip())
                    rec["gpu_memory_mib"] = float(mem.strip())
                except Exception:
                    pass
        if rec:
            metric_samples.append(rec)
        stop.wait(RESOURCE_SAMPLE_SECONDS)


def exec_with_metrics(container: str, cmd: list[str], device: str):
    metric_samples: list[dict] = []
    stop = threading.Event()
    t = threading.Thread(target=sample_container, args=(container, device, stop, metric_samples), daemon=True)
    t.start()
    started = time.monotonic()
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=60 * 60 * 24)
    elapsed = time.monotonic() - started
    stop.set()
    t.join(timeout=RESOURCE_SAMPLE_SECONDS + 1)
    if cp.returncode:
        raise RuntimeError(
            f"Model command failed ({cp.returncode})\nSTDOUT:\n{cp.stdout[-4000:]}\nSTDERR:\n{cp.stderr[-4000:]}"
        )

    def avg(key: str):
        vals = [x[key] for x in metric_samples if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def mx(key: str):
        vals = [x[key] for x in metric_samples if x.get(key) is not None]
        return max(vals) if vals else None

    return cp.stdout.strip(), {
        "elapsed_seconds": elapsed,
        "monitoring_samples": len(metric_samples),
        "avg_cpu_percent": avg("cpu_percent"),
        "max_memory_mib": mx("memory_mib"),
        "avg_gpu_util_percent": avg("gpu_util_percent"),
        "max_gpu_memory_mib": mx("gpu_memory_mib"),
    }


class RunRequest(BaseModel):
    run_id: str
    image_dir: str
    data_pickle: str
    output_file: str
    preprocessed_dir: str
    device: str | None = None


class PreprocessRequest(BaseModel):
    run_id: str
    image_dir: str
    data_pickle: str
    preprocessed_dir: str



def validate_workspace_path(value: str) -> Path:
    p = Path(value).resolve()
    root = WORKSPACE.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path outside /workspace: {p}")
    return p


def _run_under_optional_gpu_lock(model: str, req: RunRequest):
    device = (req.device or configured_device_for(model)).lower()
    if device == "gpu":
        # One shared lock serializes all model GPU inference on the single research GPU.
        log("GPU_LOCK_WAITING", model=model, run_id=req.run_id, gpu=GPU)
        with shared_lock("gpu_inference"):
            log("GPU_LOCK_ACQUIRED", model=model, run_id=req.run_id, gpu=GPU)
            try:
                return _run_real_unlocked(model, req, device)
            finally:
                log("GPU_LOCK_RELEASED", model=model, run_id=req.run_id, gpu=GPU)
    return _run_real_unlocked(model, req, device)


def _run_success_payload(info: dict, output_file: str, xai: list[str], resource_metrics: dict, stdout: str) -> dict:
    """Return the contract for a completed inference run.

    ensure_image/ensure_gpu_image metadata intentionally uses status=READY to
    describe image readiness.  A completed /run operation has a different
    status contract: SUCCESS.  Merge model metadata first and operation status
    last so READY cannot overwrite SUCCESS.
    """
    return {
        **info,
        "status": "SUCCESS",
        "output_file": output_file,
        "xai_artifacts": xai,
        "resource_metrics": resource_metrics,
        "stdout_tail": stdout[-2000:],
    }


def _run_real_unlocked(model: str, req: RunRequest, device: str):
    model = model.strip().lower()
    spec = spec_for(model)
    upstream = spec["upstream_model_name"]
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if device == "gpu":
        if not ALLOW_GPU:
            raise RuntimeError("GPU_DISABLED: set ALLOW_GPU=true only after model_tools.gpu_probe returns GPU_READY")
        info = ensure_gpu_image(model)
        if not gpu_probe_passed(model):
            raise RuntimeError(f"GPU_PROBE_REQUIRED: run python -m model_tools.gpu_probe --models {model} before GPU inference")
        selected_image = gpu_image_tag(model)
    else:
        info = ensure_image(model)
        selected_image = image_tag(model)

    image_dir = validate_workspace_path(req.image_dir)
    data = validate_workspace_path(req.data_pickle)
    out = validate_workspace_path(req.output_file)
    pre = validate_workspace_path(req.preprocessed_dir)
    if not image_dir.is_dir() or not data.is_file():
        raise FileNotFoundError("Input images/data.pkl missing")

    out.parent.mkdir(parents=True, exist_ok=True)
    pre.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o777)
    os.chmod(pre, 0o777)
    if out.exists():
        os.chmod(out, 0o666)

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", req.run_id)[-72:]
    child_container = f"mammography-inference-{model}-{safe}".lower()
    pred = patched_predict_dir(model, safe)
    target = predict_path(model)
    args = [
        "docker", "run", "-d", "--name", child_container,
        "--volumes-from", ANCHOR,
        "--network", "none",
    ]
    if device == "gpu":
        args += ["--gpus", f"device={GPU}"]
    args += [selected_image, "bash", "-lc", "trap : TERM INT; sleep infinity & wait"]

    log(
        "MODEL_RUN_STARTED", model=model, run_id=req.run_id, device=device, image=selected_image,
        image_dir=str(image_dir), output_file=str(out), preprocessed_dir=str(pre),
        input_images=len(list(image_dir.glob("*.png"))),
    )
    try:
        sh(args, timeout=60, model=model)
        log("MODEL_CHILD_CONTAINER_STARTED", model=model, run_id=req.run_id, container=child_container, image=selected_image)
        sh(["docker", "cp", f"{pred}/.", f"{child_container}:{target}"], model=model)
        cmd = [
            "docker", "exec", "-w", target, child_container, "bash", "predict.sh",
            str(data), str(image_dir), str(out), safe, device, str(pre),
        ]
        log("MODEL_COMMAND_STARTED", model=model, run_id=req.run_id, container=child_container, device=device)
        stdout, resource_metrics = exec_with_metrics(child_container, cmd, device)
        log(
            "MODEL_COMMAND_COMPLETED", model=model, run_id=req.run_id,
            elapsed_seconds=resource_metrics.get("elapsed_seconds"),
            avg_gpu_util_percent=resource_metrics.get("avg_gpu_util_percent"),
            max_gpu_memory_mib=resource_metrics.get("max_gpu_memory_mib"),
        )
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Expected prediction CSV was not produced: {out}")
        xai: list[str] = []
        for candidate in pre.rglob("visualization"):
            if candidate.is_dir():
                xai += [str(p) for p in candidate.rglob("*") if p.is_file()]
        log(
            "MODEL_RUN_SUCCESS", model=model, run_id=req.run_id, output=str(out), xai_count=len(xai),
            elapsed_seconds=resource_metrics.get("elapsed_seconds"),
            max_gpu_memory_mib=resource_metrics.get("max_gpu_memory_mib"),
        )
        return _run_success_payload(
            info=info,
            output_file=str(out),
            xai=xai,
            resource_metrics=resource_metrics,
            stdout=stdout,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", child_container], capture_output=True, text=True)




def _run_glam_legacy_cpu_reference_unlocked(req: RunRequest):
    """Run the pinned upstream GLAM image on CPU with PyTorch 1.1.

    This is a reproduction differential, not a production path.  The only source
    edit performed inside the ephemeral child container is TkAgg -> Agg so the
    historical script can run headlessly.  Model source commit, architecture,
    checkpoint, preprocessing and numerical framework version remain upstream.
    """
    model = "glam"
    ensure_meta()
    info = ensure_image(model)
    selected_image = image_tag(model)
    image_dir = validate_workspace_path(req.image_dir)
    data = validate_workspace_path(req.data_pickle)
    out = validate_workspace_path(req.output_file)
    pre = validate_workspace_path(req.preprocessed_dir)
    if not image_dir.is_dir() or not data.is_file():
        raise FileNotFoundError("Input images/data.pkl missing")
    out.parent.mkdir(parents=True, exist_ok=True)
    pre.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o777)
    os.chmod(pre, 0o777)

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", req.run_id)[-72:]
    child_container = f"mammography-legacy-cpu-glam-{safe}".lower()
    upstream = spec_for(model)["upstream_model_name"]
    src_predict = META / "models" / upstream / "predict"
    pred = WORKSPACE / "runtime" / "predict_overrides" / safe / "glam_legacy_cpu"
    if pred.exists():
        shutil.rmtree(pred)
    shutil.copytree(src_predict, pred)
    target = predict_path(model)
    args = [
        "docker", "run", "-d", "--name", child_container,
        "--volumes-from", ANCHOR, "--network", "none",
        selected_image, "bash", "-lc", "trap : TERM INT; sleep infinity & wait",
    ]
    log("GLAM_LEGACY_CPU_STARTED", run_id=req.run_id, image=selected_image)
    try:
        sh(args, timeout=60, model=model)
        # Headless-only compatibility change.  Do not alter inference semantics.
        sh([
            "docker", "exec", "-u", "root", child_container, "bash", "-lc",
            "sed -i 's/matplotlib.use(\"TkAgg\")/matplotlib.use(\"Agg\")/' "
            "/home/glam/GLAM/src/scripts/run_model.py",
        ], model=model)
        sh(["docker", "cp", f"{pred}/.", f"{child_container}:{target}"], model=model)
        cmd = [
            "docker", "exec", "-w", target, child_container, "bash", "predict.sh",
            str(data), str(image_dir), str(out), safe, "cpu", str(pre),
        ]
        stdout, resource_metrics = exec_with_metrics(child_container, cmd, "cpu")
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Expected legacy GLAM prediction CSV was not produced: {out}")
        payload = _run_success_payload(
            info=info,
            output_file=str(out),
            xai=[],
            resource_metrics=resource_metrics,
            stdout=stdout,
        )
        payload.update({
            "diagnostic_runtime": "legacy_cpu_torch1.1",
            "device": "cpu",
            "headless_only_patch": "matplotlib TkAgg -> Agg",
            "model_source_changed": False,
            "model_weights_changed": False,
            "inference_semantics_intentionally_changed": False,
        })
        return payload
    finally:
        subprocess.run(["docker", "rm", "-f", child_container], capture_output=True, text=True)


def _preprocess_only_unlocked(model: str, req: PreprocessRequest):
    """Run only the upstream crop + optimal-center stages, without classifier inference.

    v0.27 uses this exact upstream preprocessing as a label-independent orientation
    preflight.  It intentionally runs inside the same pinned model image/source tree
    used for inference.  No checkpoint is loaded and no prediction is produced.
    """
    model = model.strip().lower()
    spec_for(model)
    image_dir = validate_workspace_path(req.image_dir)
    data = validate_workspace_path(req.data_pickle)
    pre = validate_workspace_path(req.preprocessed_dir)
    if not image_dir.is_dir() or not data.is_file():
        raise FileNotFoundError("Input images/data.pkl missing")
    pre.mkdir(parents=True, exist_ok=True)
    os.chmod(pre, 0o777)

    # Prefer the validated GPU-compatibility image when available because it is the
    # exact runtime used by the thesis inference.  Preprocessing itself remains CPU-only.
    if gpu_compatibility_for(model).get("enabled"):
        info = ensure_gpu_image(model)
        selected_image = gpu_image_tag(model)
    else:
        info = ensure_image(model)
        selected_image = image_tag(model)

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", req.run_id)[-72:]
    child_container = f"mammography-preprocess-{model}-{safe}".lower()
    cropped = pre / f"{safe}_cropped_images"
    cropped_list = cropped / "cropped_exam_list.pkl"
    center_data = pre / f"{safe}_center_data.pkl"
    if cropped.exists():
        shutil.rmtree(cropped)
    if center_data.exists():
        center_data.unlink()

    target = predict_path(model)
    args = [
        "docker", "run", "-d", "--name", child_container,
        "--volumes-from", ANCHOR, "--network", "none",
        selected_image, "bash", "-lc", "trap : TERM INT; sleep infinity & wait",
    ]
    log("MODEL_PREPROCESS_STARTED", model=model, run_id=req.run_id, image=selected_image,
        image_dir=str(image_dir), preprocessed_dir=str(pre), input_images=len(list(image_dir.glob("*.png"))))
    try:
        sh(args, timeout=60, model=model)
        # The upstream NYU repository explicitly requires its repository root in
        # PYTHONPATH when the individual preprocessing scripts are invoked directly.
        # Running ``python3 src/cropping/crop_mammogram.py`` sets sys.path[0] to the
        # script directory (src/cropping), so ``import src...`` otherwise fails even
        # when Docker's working directory is the repository root.  The normal upstream
        # run.sh/predict path masks this detail; PREPROCESS_ONLY must preserve it.
        repo_pythonpath = "/home/bcc/breast_cancer_classifier" if model == "nyu" else target
        command = " && ".join([
            f"export PYTHONPATH={shlex.quote(repo_pythonpath)}${{PYTHONPATH:+:$PYTHONPATH}}",
            "python3 src/cropping/crop_mammogram.py "
            f"--input-data-folder {shlex.quote(str(image_dir))} "
            f"--output-data-folder {shlex.quote(str(cropped))} "
            f"--exam-list-path {shlex.quote(str(data))} "
            f"--cropped-exam-list-path {shlex.quote(str(cropped_list))} --num-processes 10",
            "python3 src/optimal_centers/get_optimal_centers.py "
            f"--cropped-exam-list-path {shlex.quote(str(cropped_list))} "
            f"--data-prefix {shlex.quote(str(cropped))} "
            f"--output-exam-list-path {shlex.quote(str(center_data))} --num-processes 10",
        ])
        stdout, metrics = exec_with_metrics(
            child_container,
            ["docker", "exec", "-w", target, child_container, "bash", "-lc", command],
            "cpu",
        )
        if not center_data.is_file():
            raise RuntimeError(f"Expected preprocessing metadata was not produced: {center_data}")
        log("MODEL_PREPROCESS_COMPLETED", model=model, run_id=req.run_id,
            center_data=str(center_data), elapsed_seconds=metrics.get("elapsed_seconds"))
        return {
            **info,
            "status": "SUCCESS",
            "operation": "PREPROCESS_ONLY",
            "classifier_inference_performed": False,
            "center_data": str(center_data),
            "cropped_exam_list": str(cropped_list),
            "cropped_images": str(cropped),
            "resource_metrics": metrics,
            "stdout_tail": stdout[-2000:],
        }
    finally:
        subprocess.run(["docker", "rm", "-f", child_container], capture_output=True, text=True)


def model_info(model: str) -> dict:
    model = model.strip().lower()
    spec = spec_for(model)
    return {
        "model": model,
        "academic_name": spec["academic_name"],
        "upstream": spec["upstream_model_name"],
        "repository": spec["repository"],
        "license": spec["license"],
        "image": image_tag(model),
        "built": image_exists(model),
        "gpu_image": gpu_compatibility_for(model).get("image"),
        "gpu_built": gpu_image_exists(model),
        "gpu_probe_passed": gpu_probe_passed(model),
        "gpu_profile": gpu_compatibility_for(model).get("profile"),
        "runner_service": "model-runner",
        "device": configured_device_for(model),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_healthcheck_access_filter()
    diagnostics = docker_diagnostics()
    log(
        "MODEL_RUNNER_READY",
        bootstrap_mode=BOOTSTRAP_MODE,
        default_model_device=DEFAULT_MODEL_DEVICE,
        model_devices=configured_devices(),
        gpu_profiles=configured_gpu_profiles(),
        models=sorted(configured_models()),
        docker_daemon_reachable=diagnostics.get("daemon_reachable"),
        docker_host=diagnostics.get("docker_host"),
        docker_version=diagnostics.get("docker_version", {}).get("stdout", "")[-2000:],
        docker_error=diagnostics.get("docker_info", {}).get("stderr", "")[-2000:],
    )
    if BOOTSTRAP_MODE == "eager":
        for model in sorted(configured_models()):
            try:
                ensure_image(model)
            except Exception as exc:
                log("EAGER_MODEL_BUILD_FAILED", model=model, error=str(exc))
                raise
    yield

app = FastAPI(title="Mammography Model Runner", version=APP_VERSION, lifespan=lifespan)

@app.get("/doctor")
def doctor():
    """Always returns diagnostics, even when the Docker daemon is unavailable."""
    d = docker_diagnostics()
    return {
        "service": "model-runner",
        "version": APP_VERSION,
        "default_model_device": DEFAULT_MODEL_DEVICE,
        "model_devices": configured_devices(),
        "gpu_profiles": configured_gpu_profiles(),
        "gpu_allowed": ALLOW_GPU,
        "gpu_number": GPU,
        "runner_ml_frameworks": False,
        "docker": d,
    }


@app.get("/health")
def health():
    d = docker_diagnostics()
    if not d["daemon_reachable"]:
        log(
            "DOCKER_DAEMON_UNREACHABLE",
            docker_host=d.get("docker_host"),
            socket_exists=d.get("socket_exists"),
            direct_socket_ping=d.get("direct_socket_ping"),
            docker_version=d.get("docker_version"),
            docker_info=d.get("docker_info"),
        )
        raise HTTPException(
            503,
            detail={
                "message": "Docker daemon is not reachable from model-runner",
                "hint": "Open /doctor for detailed socket and Docker CLI diagnostics.",
                "docker": d,
            },
        )
    return {
        "status": "ok",
        "service": "model-runner",
        "version": APP_VERSION,
        "default_model_device": DEFAULT_MODEL_DEVICE,
        "model_devices": configured_devices(),
        "gpu_profiles": configured_gpu_profiles(),
        "gpu_allowed": ALLOW_GPU,
        "gpu_number": GPU,
        "docker_socket": d["socket_exists"],
        "docker_daemon": True,
        "docker_host": d["docker_host"],
        "models": {m: {"image": image_tag(m), "built": image_exists(m), "gpu_image": gpu_compatibility_for(m).get("image"), "gpu_built": gpu_image_exists(m), "gpu_probe_passed": gpu_probe_passed(m)} for m in sorted(configured_models())},
        "runner_ml_frameworks": False,
    }


@app.get("/models")
def models():
    return [model_info(m) for m in sorted(configured_models())]


@app.post("/meta/ensure")
def ensure_metarepository_endpoint():
    try:
        commit = ensure_meta()
        return {"status": "READY", "path": str(META), "resolved_commit": commit}
    except Exception as exc:
        log("METAREPOSITORY_ENSURE_FAILED", error=str(exc))
        raise HTTPException(500, str(exc))


@app.get("/models/{model}/info")
def info(model: str):
    try:
        return model_info(model)
    except Exception as exc:
        raise HTTPException(404, str(exc))


@app.post("/models/{model}/ensure")
def ensure(model: str):
    try:
        return ensure_image(model)
    except Exception as exc:
        log("MODEL_ENSURE_FAILED", model=model, error=str(exc))
        raise HTTPException(500, str(exc))


@app.post("/models/{model}/ensure-gpu")
def ensure_gpu(model: str, force_rebuild: bool = False):
    try:
        return ensure_gpu_image(model, force_rebuild=force_rebuild)
    except Exception as exc:
        log("GPU_MODEL_ENSURE_FAILED", model=model, error=str(exc))
        raise HTTPException(500, str(exc))


@app.post("/models/{model}/gpu-probe")
def gpu_probe(model: str):
    try:
        return probe_gpu_runtime(model)
    except Exception as exc:
        log("GPU_MODEL_PROBE_FAILED", model=model, error=str(exc))
        raise HTTPException(500, str(exc))


@app.post("/models/{model}/smoke-test")
def smoke_test(model: str):
    try:
        model = model.strip().lower()
        spec_for(model)
        ensure_meta()
        images = META / "sample_data" / "images"
        data = META / "sample_data" / "data.pkl"
        if not images.exists() or not data.exists():
            raise FileNotFoundError("Official metarepository sample_data is missing")
        outdir = WORKSPACE / "output" / "smoke_tests" / model
        outdir.mkdir(parents=True, exist_ok=True)
        os.chmod(outdir, 0o777)
        return _run_under_optional_gpu_lock(
            model,
            RunRequest(
                run_id=f"smoke-{model}",
                image_dir=str(images),
                data_pickle=str(data),
                output_file=str(outdir / "predictions.csv"),
                preprocessed_dir=str(outdir / "preprocessed"),
            ),
        )
    except Exception as exc:
        log("MODEL_SMOKE_FAILED", model=model, error=str(exc))
        raise HTTPException(500, str(exc))


@app.post("/diagnostics/glam-legacy-cpu")
def glam_legacy_cpu_reference(req: RunRequest):
    try:
        return _run_glam_legacy_cpu_reference_unlocked(req)
    except Exception as exc:
        log("GLAM_LEGACY_CPU_FAILED", run_id=req.run_id, error=str(exc))
        raise HTTPException(500, str(exc))


@app.post("/models/{model}/preprocess")
def preprocess(model: str, req: PreprocessRequest):
    try:
        spec_for(model)
        return _preprocess_only_unlocked(model, req)
    except Exception as exc:
        log("MODEL_PREPROCESS_FAILED", model=model, run_id=req.run_id, error=str(exc))
        raise HTTPException(500, str(exc))


@app.post("/models/{model}/run")
def run(model: str, req: RunRequest):
    try:
        spec_for(model)
        return _run_under_optional_gpu_lock(model, req)
    except Exception as exc:
        log("MODEL_RUN_FAILED", model=model, run_id=req.run_id, error=str(exc))
        raise HTTPException(500, str(exc))
