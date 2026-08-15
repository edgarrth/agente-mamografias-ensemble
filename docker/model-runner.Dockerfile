# The Model Runner needs a Docker *client* only. It talks to Docker Desktop/Engine
# through the mounted /var/run/docker.sock and never runs a daemon of its own.
# Using Docker's official CLI image avoids the stale docker.io package/API mismatch
# that can occur when installing Debian's docker.io package in python:slim.
ARG DOCKER_CLI_IMAGE=docker:29-cli
FROM ${DOCKER_CLI_IMAGE}

WORKDIR /runner

# Keep the runner intentionally lightweight: no PyTorch/TensorFlow/CUDA/cuDNN.
# Model-specific ML dependencies live only in the isolated model images.
RUN apk add --no-cache \
      python3 \
      py3-pip \
      git \
      curl \
      bash \
      ca-certificates \
    && python3 -m venv /opt/runner-venv \
    && /opt/runner-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/runner-venv/bin/pip install --no-cache-dir fastapi uvicorn pydantic PyYAML

ENV PATH="/opt/runner-venv/bin:${PATH}" \
    PYTHONPATH=/runner \
    DOCKER_HOST=unix:///var/run/docker.sock

COPY model_runner /runner/model_runner
COPY config /runner/config
COPY docker/model-compat /runner/model_compat

# Deliberately no PyTorch, TensorFlow, CUDA Toolkit or cuDNN here.
CMD ["uvicorn", "model_runner.api:app", "--host", "0.0.0.0", "--port", "8010"]
