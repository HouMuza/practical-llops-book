FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest
# Shared by: local `make docker-build`, Azure ML managed endpoints (environment build),
# and LoRA training jobs (command job). Azure ML calls serve/score.py, not uvicorn CMD.
 
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=Qwen/Qwen3-0.6B \
    DEVICE=cuda \
    DTYPE=bf16
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt
 
COPY engine /app/engine
COPY serve /app/serve
COPY obs /app/obs
 
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
