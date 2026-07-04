# Local GPU, Docker & streaming

## GPU / CUDA local serving

Requirements: NVIDIA driver, CUDA-capable PyTorch (installed via `requirements.txt` on Linux/WSL; on macOS use CPU or Docker).

```bash
make install
make serve-gpu
```

Environment variables:

```text
MODEL_NAME=Qwen/Qwen3-0.6B
DEVICE=cuda
DTYPE=bf16
```

In another terminal:

```bash
make health
make smoke          # non-streaming JSON
make smoke-stream   # SSE token stream
```

### What `smoke-stream` returns

OpenAI-style SSE lines:

```text
data: {"token":"Lo", "token_id":..., "request_id":"..."}

data: {"token":"RA", ...}

event: done
data: {"finish_reason":"length", "request_id":"..."}
```

Implementation: `serve/streaming.py` checks `request.is_disconnected()` so KV memory is released when clients drop.

## Docker build

The same `serve/Dockerfile` is used by **Azure ML** (environment build) and **local Docker**.

```bash
make docker-build
```

### CPU container

```bash
make docker-run
curl -s http://localhost:8000/health
```

### GPU container (requires NVIDIA Container Toolkit)

```bash
make docker-run-gpu
make smoke-stream
```

Dockerfile defaults:

```dockerfile
ENV DEVICE=cuda DTYPE=bf16
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

> **Azure ML note:** Managed endpoints ignore `CMD` and call `serve/score.py` via `code_configuration.scoring_script`. The Dockerfile mainly defines the Python environment and CUDA base image.

### Manual Docker commands

```bash
docker build -f serve/Dockerfile -t practical-llops-book:latest .
docker run --rm -p 8000:8000 --gpus all \
  -e MODEL_NAME=Qwen/Qwen3-0.6B \
  -e DEVICE=cuda -e DTYPE=bf16 \
  practical-llops-book:latest
```

## Offline eval (requires running API)

Terminal 1:

```bash
make serve        # or make serve-gpu
```

Terminal 2:

```bash
make eval
```

`eval/generate_predictions.py` calls `POST /v1/completions` for each row in `eval/eval_set.jsonl`, then `eval/run_eval.py` logs metrics to MLflow.

## Verify model without the custom engine

```bash
python test_model.py
```

Uses Hugging Face `model.generate()` directly. Useful to confirm CUDA and model download before debugging the serving stack.
