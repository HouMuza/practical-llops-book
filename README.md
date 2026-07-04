# Practical LLMOps Deep Dive: Companion Code

**Runnable reference implementation for Appendix E of *[Practical LLMOps Deep Dive](https://houmuza.gumroad.com/l/practical-llmops-deep-dive)*.**

> **Get the book:** [houmuza.gumroad.com/l/practical-llmops-deep-dive](https://houmuza.gumroad.com/l/practical-llmops-deep-dive)

This repository is the companion code for **Practical LLMOps Deep Dive: Building Production LLM Systems from First Principles** (First Edition, 2026) by Houston Muzamhindo.

The book explains *what happens between* `model.generate()` and a production endpoint. This repo is where those ideas become a complete, runnable system with code that is typed, instrumented, and structured like a real service.

---

## About the book

Most LLM tutorials teach you to call an API or run `vllm serve`. Few explain the machinery in between: tokenisation, prefill, decode, KV cache layout, batching, scheduling, quantisation, parallelism, deployment, monitoring, evaluation, and cost.

**Practical LLMOps Deep Dive** is a serving-focused guide for engineers and data scientists who want to understand those mechanics from first principles, then translate that understanding into production systems with deployment examples anchored on **Azure ML**.

The book is **not** mainly about prompt engineering, chatbot UX, or wrapping a closed-source API. It is about operating a model as a shared production service.

**Examples are anchored on [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** because it is small enough to run on accessible hardware, large enough to surface the same serving problems that appear at 7B and 70B scale.

### What the book covers (Parts 1–4)


| Part                               | Topics                                                                                                |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Part 1: Foundations**                | Why naive inference fails in production, the five layers of LLMOps, Qwen3-0.6B as the reference model |
| **Part 2: The Serving Layer**          | Attention & KV cache, the inference loop, batching & scheduling, quantisation, speculative decoding   |
| **Part 3: Deployment at Scale**        | Multi-GPU inference, LoRA/QLoRA fine-tuning, structured output, Azure ML deployment                   |
| **Part 4: Observability & Operations** | OpenTelemetry, MLflow, Azure Monitor, cost optimisation, evaluation & quality monitoring              |


Part 5 contains hands-on projects for readers who want to apply the material end to end. **This repository implements Appendix E**, the production-shaped reference scaffold, not the project homework.

### How this repo relates to the book


| In the book                              | In this repo                                                                                                          |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Short PyTorch snippets in chapter bodies | Pedagogical only that is optimised for clarity                                                                        |
| Appendix E scaffold (printed)            | Full source tree below                                                                                                |
| Companion GitHub repository (Preface)    | **You are here**. Runnable code with typing, structured logging, OpenTelemetry, deployment assets, and eval utilities |


---

## About this repository

This repo is intentionally **small enough to read end to end**. It is **not** a replacement for vLLM, TGI, SGLang, or Ray Serve.

It *is* the production-shaped version of the book's teaching code:

- **Typed** request/engine configuration
- **Instrumented** with OpenTelemetry spans and structured JSON logs
- **Testable** offline eval pipeline with deterministic scorers
- **Deployable** to Azure ML managed online endpoints
- **Readable**: the goal is understanding, not winning throughput benchmarks out of the box

The attention path delegates to Hugging Face / PyTorch for the runnable teaching implementation. In a high-throughput service you would replace `engine/attention.py` with a vLLM, Triton, or FlashAttention kernel wrapper. The operational shape (scheduling, KV ownership, streaming cancellation, metrics, and eval gates) stays the same.

### Repository layout

```text
practical-llops-book/
├── engine/          # Inference engine, scheduler, KV cache, prefix cache, sampling, quantisation
├── serve/           # FastAPI API, Azure ML score.py, SSE streaming, Dockerfile
├── train/           # LoRA / QLoRA training entry points and data formatting
├── obs/             # OpenTelemetry setup, Azure Monitor / MLflow wiring, KQL queries, alerts
├── deploy/          # Azure ML endpoint & deployment YAML, autoscale Bicep, deploy helpers
├── eval/            # Offline eval runner, scorers, sample eval set
├── Makefile         # Local dev, deploy, observability, and eval targets
└── requirements.txt
```

### Key patterns implemented


| Pattern                           | What the code demonstrates                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------- |
| Paged KV cache with refcount      | Shared prefix blocks are safe; blocks return to the free list only when refcount reaches zero  |
| Chunked prefill with fairness     | Decodes first; long prompts progress over multiple scheduler iterations without destroying ITL |
| Prefix cache with content hashing | Token ids + LoRA name are hashed; LRU eviction when the warm pool grows too large              |
| Multi-adapter routing             | Requests carry `lora_name`; the serving layer keeps adapter selection explicit                 |
| Speculative decoding boundary     | Draft-model path is behind a replaceable interface; acceptance rate is logged                  |
| Streaming SSE                     | Tokens emitted as SSE `data:` lines; client disconnect checked so KV memory is freed promptly  |
| OpenTelemetry instrumentation     | Public methods create spans with `gen_ai.*` attributes aligned with the observability chapters |


---

## Prerequisites

- **Python 3.11+** (3.13 tested locally)
- **PyTorch** with CUDA if you want GPU inference locally
- **Hugging Face account** (optional): `Qwen/Qwen3-0.6B` downloads on first run
- **Azure subscription** (optional): only needed for the Azure ML deployment path
- **Azure CLI + `ml` extension** (optional): for `make deploy-`* targets

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/HouMuza/practical-llops-book.git
cd practical-llops-book

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify the model loads

```bash
python test_model.py
```

This downloads `Qwen/Qwen3-0.6B` (cached under `~/.cache/huggingface/hub/`) and runs a short generation.

### 3. Run the local API

```bash
export MODEL_NAME=Qwen/Qwen3-0.6B
export DEVICE=cpu          # use cuda when a GPU is available
export DTYPE=fp32          # bf16 on CUDA-capable GPUs

make serve
# or: uvicorn serve.api:app --host 0.0.0.0 --port 8000
```

### 4. Smoke test

```bash
make health
make smoke
```

Non-streaming request:

```bash
curl -s -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV cache in one paragraph.","max_new_tokens":80,"stream":false}' \
  | python -m json.tool
```

Streaming request:

```bash
make smoke-stream
```

### 5. Run offline evaluation

Requires the API in another terminal (`make serve`).

```bash
make eval
```

---

## Operations guides

Step-by-step guides for every path readers need:

| Guide | Commands |
|-------|----------|
| [**docs/README.md**](docs/README.md) | Index & decision tree |
| [Local GPU, Docker & streaming](docs/local-gpu-docker-streaming.md) | `make serve-gpu`, `make smoke-stream`, `make docker-run-gpu` |
| [Azure ML deployment](docs/azure-deployment.md) | `make deploy-all`, `make deploy-all-gpu`, GitHub Actions |
| [Training & evaluation](docs/training-and-eval.md) | `make train-lora-local`, `make train-lora-azure`, `make eval-judge` |
| [YAML reference](docs/yaml-reference.md) | Field-by-field deploy file reference |

### Common commands

```bash
# Local GPU + streaming
make serve-gpu && make smoke-stream

# Docker (matches Azure image)
make docker-build && make docker-run-gpu

# Azure CPU (no GPU quota)
make deploy-all && make endpoint-test

# Azure GPU (T4 quota required)
make deploy-all-gpu && make endpoint-test

# LoRA on Azure
make compute-gpu-create && make train-lora-azure

# LLM-as-judge (needs JUDGE_API_KEY)
make eval && make eval-judge
```

---

## Make targets


| Target               | Purpose                                                   |
| -------------------- | --------------------------------------------------------- |
| `make install`       | Create venv and install dependencies                      |
| `make serve`         | Start local FastAPI server                                |
| `make health`        | Hit `/health`                                             |
| `make smoke`         | Non-streaming completion smoke test                       |
| `make smoke-stream`  | Streaming completion smoke test                           |
| `make eval`          | Generate predictions and run offline eval                 |
| `make obs-show`      | Show resolved Azure Monitor / MLflow config               |
| `make obs-report`    | Generate observability report markdown                    |
| `make deploy-all`    | Full Azure ML provisioning + blue deployment (idempotent) |
| `make endpoint-test` | Call managed endpoint `/score`                            |


See the `Makefile` for the full list, including blue/green traffic split and SDK fallbacks.

---

## Azure ML deployment

See **[docs/azure-deployment.md](docs/azure-deployment.md)** for the full YAML pipeline walkthrough.

The verified default path in this repository uses a **CPU-backed** configuration so readers with fresh Azure subscriptions are not blocked by GPU quota:


| Setting        | Default           |
| -------------- | ----------------- |
| Region         | `swedencentral`   |
| Endpoint auth  | key               |
| Deployment SKU | `Standard_E4s_v3` |
| Runtime device | `cpu`             |
| Runtime dtype  | `fp32`            |


GPU quota for `Standard_NC4as_T4_v3` is often `0` until a quota request is approved. Once you have quota:

```bash
make deploy-all-gpu
```

Or use the GitHub Actions workflow with **profile: gpu**. CPU defaults remain in `deployment-blue.yaml`; GPU uses `deployment-blue-gpu.yaml`.

### CLI deployment

```bash
export SUBSCRIPTION_ID=<azure-subscription-id>
export RESOURCE_GROUP=<azure-resource-group>
export WORKSPACE_NAME=<azure-ml-workspace-name>

make deploy-all
make endpoint-test
```

Targets are safe to rerun: existing resources are updated or skipped rather than failing on duplicates.

### GitHub Actions

A manual workflow is available at `.github/workflows/azureml-deploy.yml`.

**Repository secrets:**

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
AZURE_ACTION_GROUP_ID   # optional, alerts only
```

Use Azure federated credentials (OIDC) for the workflow identity. Trigger from the Actions tab and provide resource group, workspace name, location, endpoint name, instance type, instance count, and model asset version.

---

## Observability

Three layers are wired end to end.

### Azure Monitor

Managed deployments resolve the Azure ML workspace's linked Application Insights component and inject:

```text
APPLICATIONINSIGHTS_CONNECTION_STRING
LOG_FORMAT=json
MLFLOW_TRACKING_URI
```

OpenTelemetry traces export to Azure Monitor; structured request logs land in Application Insights and the linked Log Analytics workspace.

### MLflow

Training and eval scripts auto-discover the Azure ML workspace MLflow tracking URI when `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` are set. Override with `MLFLOW_TRACKING_URI` if needed.

### Serving telemetry

The serving layer emits request count, active requests, latency histograms, prompt/completion token histograms, and error counts. Local runs fall back to console exporters when Azure Monitor is not configured.

**Correlation headers (local API):**

```text
X-LLMOps-Request-Id
X-LLMOps-Trace-Id
```

**Azure ML `/score` responses** include `request_id` and `trace_id` fields.

Starter KQL queries live in `obs/kql/` (`system.kql`, `quality.kql`, `cost.kql`).

```bash
make obs-show
make obs-report    # writes obs/reports/observability_report.md by default
```

---

## Training (LoRA / QLoRA)

```bash
python -m train.lora_train --train-jsonl /path/to/train.jsonl --output-dir ./outputs/lora
python -m train.qlora_train --train-jsonl /path/to/train.jsonl --output-dir ./outputs/qlora
```

Prepare data in ChatML-style JSONL as described in Chapter 9. MLflow logging is enabled when a tracking URI is configured.

---

## Evaluation

```bash
make eval
```

The `eval/` package includes:

- Deterministic scorers (exact match, token F1, JSON schema validity)
- LLM-as-judge with **position-bias debiasing** (pairwise judging in both orders)
- Sample `eval_set.jsonl` to run the pipeline without curating your own set first

---

## Who this is for

- **ML engineers** building or operating LLM serving infrastructure
- **Data scientists** moving from notebooks to production services
- **Platform / infrastructure engineers** deploying models on Azure ML
- **Technical leads** who need to reason about vLLM defaults, KV memory, and cost, not just configure them

---

## What this repo is not

- Not a drop-in vLLM / Ray Serve replacement
- Not guaranteed to hit production throughput SLOs without tuning and kernel upgrades
- Not a substitute for independent security review before production use

Code examples are provided for **educational purposes**. Production use requires your own testing, security review, and operational validation.

---

## License

See [LICENSE](LICENSE) in this repository.

---

## Author

**Houston Muzamhindo**

- **Book:** [Practical LLMOps Deep Dive on Gumroad](https://houmuza.gumroad.com/l/practical-llmops-deep-dive)
- **Code:** [github.com/HouMuza/practical-llops-book](https://github.com/HouMuza/practical-llops-book)

---

## Trademarks

Qwen, Azure, PyTorch, Hugging Face, vLLM, TGI, and other named products are trademarks of their respective owners. Their use here is for explanation and does not imply endorsement.