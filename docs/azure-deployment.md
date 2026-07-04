# Azure ML deployment

This guide walks through the **YAML pipelines** in `deploy/` and how they connect to `make` targets and GitHub Actions.

## Architecture

```text
deploy/endpoint.yaml              → managed online endpoint (auth, traffic map)
deploy/deployment-blue.yaml       → CPU scoring deployment (default)
deploy/deployment-blue-gpu.yaml   → T4 GPU scoring deployment
deploy/deployment-green*.yaml     → canary slot for blue/green
deploy/compute-gpu.yaml           → training cluster (not serving)
deploy/job-lora-train.yaml        → command job for LoRA fine-tuning
serve/score.py                    → Azure ML entry point (init + run)
serve/Dockerfile                  → container image built by Azure ML
```

Azure ML builds the environment from `serve/Dockerfile`, uploads the repo as `code`, and calls `serve/score.py` for each `/score` request.

## Prerequisites

```bash
export SUBSCRIPTION_ID=<your-subscription>
export RESOURCE_GROUP=llmops-book-rg
export WORKSPACE_NAME=llmops-book-mlw

az login
az account set --subscription $SUBSCRIPTION_ID
az extension add -n ml -y
```

Install local Python deps (for `render_deployment.py`):

```bash
make install
```

## Profile 1: CPU (no GPU quota required)

Best for first-time Azure subscribers. Uses `Standard_E4s_v3` and `DEVICE=cpu`.

| File | Purpose |
|------|---------|
| `deploy/deployment-blue.yaml` | Blue slot, CPU, fp32 |
| `Makefile` target | `make deploy-all` |

```bash
make deploy-all
make endpoint-test
```

### What `make deploy-all` does

1. `group-create`: resource group in `westeurope` (override with `AML_LOCATION`)
2. `workspace-create`: Azure ML workspace
3. `model-register`: placeholder asset at `deploy/model_artifact` (model loads from Hugging Face at runtime)
4. `endpoint-create`: applies `deploy/endpoint.yaml`
5. `deploy-blue`: renders `deployment-blue.yaml` + observability env, creates deployment
6. `deploy-observability`: diagnostic settings → Log Analytics
7. `deploy-autoscale`: `deploy/autoscale.bicep`
8. `deploy-alerts`: optional if `ACTION_GROUP_ID` is set

## Profile 2: GPU (T4, book default)

Requires **quota** for `Standard_NC4as_T4_v3` in your region.

| File | Purpose |
|------|---------|
| `deploy/deployment-blue-gpu.yaml` | Blue slot, CUDA, bf16 |
| `Makefile` target | `make deploy-all-gpu` |

```bash
# Request quota in Azure Portal → Quotas → Machine Learning → NCasT4_v3
make deploy-all-gpu
make endpoint-test
```

Key YAML differences vs CPU:

```yaml
instance_type: Standard_NC4as_T4_v3
environment_variables:
  DEVICE: cuda
  DTYPE: bf16
```

## How `render_deployment.py` works

Before `az ml online-deployment create`, the Makefile runs:

```bash
python -m deploy.render_deployment \
  --input deploy/deployment-blue-gpu.yaml \
  --output /tmp/rendered.yaml
```

This injects workspace-linked values when available:

- `APPLICATIONINSIGHTS_CONNECTION_STRING`
- `MLFLOW_TRACKING_URI`
- `LOG_FORMAT=json`

So observability works without hard-coding secrets in YAML.

## Blue / green canary

```bash
# Deploy candidate to green (CPU or GPU spec)
make deploy-green
# or
make deploy-green-gpu

# 90/10 traffic split
make traffic-split

# Full cutover
make traffic-cutover
```

Green CPU spec: `deploy/deployment-green.yaml`  
Green GPU spec: `deploy/deployment-green-gpu.yaml`

## GitHub Actions deployment

Workflow: `.github/workflows/azureml-deploy.yml`

### Required repository secrets

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
AZURE_ACTION_GROUP_ID   # optional
```

Use **federated credentials (OIDC)** on an Azure app registration for the workflow identity.

### Manual trigger inputs

| Input | `cpu` | `gpu` |
|-------|-------|-------|
| Instance SKU | `Standard_E4s_v3` | `Standard_NC4as_T4_v3` |
| Deployment YAML | `deployment-blue.yaml` | `deployment-blue-gpu.yaml` |

Actions → **Deploy Azure ML Endpoint** → choose `profile: cpu` or `gpu`.

## Testing the deployed endpoint

```bash
make endpoint-test
```

Under the hood this:

1. Reads `scoring_uri` from the endpoint
2. Gets the endpoint key (or AAD token)
3. POSTs to `/score`:

```json
{"prompt":"Explain KV cache.","max_new_tokens":40}
```

Response includes `text`, `request_id`, and `trace_id`.

### Streaming on Azure

Azure ML managed online endpoints use **`/score`** with a JSON body and return the **full completion** in one response. Server-sent streaming (`make smoke-stream`) is supported on the **local FastAPI** path (`serve/api.py`), not on the managed `/score` route.

For streaming in production on Azure, you would deploy the same container to AKS or Container Apps with `serve/api.py`. That is out of scope for this appendix, but the Docker image is shared.

## SDK fallback

If `az ml` has extension issues:

```bash
make deploy-sdk DEVICE=cuda DTYPE=bf16 INSTANCE_TYPE=Standard_NC4as_T4_v3
make endpoint-show-sdk
```

## Troubleshooting

| Sympt problem | Fix |
|---------------|-----|
| `Quota exceeded` for T4 | Use `make deploy-all` (CPU) or request quota |
| Deployment stuck on `Creating` | Check build logs in Azure ML studio → Endpoints → blue → Logs |
| Model download slow on cold start | First request downloads Qwen from Hugging Face; set `HF_TOKEN` as workspace secret |
| 401 on endpoint-test | Re-run `az ml online-endpoint get-credentials` |

## Related files

- `deploy/autoscale.bicep`: GPU utilisation-based scaling
- `obs/alerts.bicep`: alert rules (needs `ACTION_GROUP_ID`)
- `obs/kql/*.kql`: dashboards after deployment
