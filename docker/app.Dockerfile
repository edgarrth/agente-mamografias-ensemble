FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml /app/
COPY VERSION /app/VERSION
COPY mammography_agent /app/mammography_agent
COPY dataset_pipeline /app/dataset_pipeline
COPY tests_flow /app/tests_flow
COPY experiments /app/experiments
COPY model_tools /app/model_tools
COPY model_runner /app/model_runner
COPY ui /app/ui
COPY tests /app/tests
COPY docker /app/docker
COPY docker-compose.yml /app/docker-compose.yml
COPY .env.example /app/.env.example
RUN pip install --no-cache-dir ".[test]"
COPY config /app/config
CMD ["uvicorn", "mammography_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
