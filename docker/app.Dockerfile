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
COPY ui /app/ui
RUN pip install --no-cache-dir .
COPY config /app/config
CMD ["uvicorn", "mammography_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
