# Migration from v0.3 to v0.4

v0.4 keeps the same thesis architecture introduced in v0.3:

- one persistent `model-runner`;
- three isolated model images (`gmic`, `nyu`, `glam`);
- no ML/CUDA frameworks in the runner;
- the runner serializes GPU inference and starts temporary model containers.

The change is focused on the Docker Engine boundary used in Windows + WSL2 + Docker Desktop.

## Why this version exists

v0.3 installed Debian's `docker.io` package inside the Python-based Model Runner. On a newer Docker Desktop/Engine, that client can be too old for the daemon API and `/health` remains `503` even though Uvicorn itself is running.

v0.4 instead builds the Model Runner from Docker's official CLI image (`docker:29-cli` by default), adds an explicit `DOCKER_HOST=unix:///var/run/docker.sock`, and adds `/doctor` diagnostics.

## Upgrade

Do not delete volumes or the workspace:

```bash
docker compose down --remove-orphans
```

Replace the project files with v0.4, preserve `.env` and `workspace/`, then ensure `.env` contains:

```env
DOCKER_CLI_IMAGE=docker:29-cli
```

Rebuild the runner without stale cache and start detached:

```bash
docker compose build --no-cache model-runner
docker compose up -d
```

Check:

```bash
docker compose ps -a
curl http://localhost:8010/doctor
curl http://localhost:8010/health
```

The expected `/health` result is HTTP 200 with `"docker_daemon": true`. FastAPI and Streamlit will then start because their dependency conditions are satisfied.

## Important

Use Compose service names for diagnostics instead of a hard-coded `docker exec <container>` command:

```bash
docker compose exec model-runner docker version
docker compose exec model-runner docker info
```

If the service is not running, `docker compose ps -a` is the source of truth.
