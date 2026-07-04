# YAML reference

Field-by-field guide for deployment and training files in `deploy/`.

## `endpoint.yaml`

| Field | Value | Meaning |
|-------|-------|---------|
| `name` | `qwen3-yourname-prod` | Managed online endpoint name |
| `auth_mode` | `key` | Endpoint key auth (`aad_token` for Entra ID) |
| `traffic.blue` | `100` | 100% traffic to blue deployment |

## `deployment-blue.yaml` (CPU)

| Field | Value | Meaning |
|-------|-------|---------|
| `model` | `azureml:qwen3-0-6b:1` | Registered placeholder asset |
| `code_configuration.code` | `../` | Repo root uploaded as scoring code |
| `code_configuration.scoring_script` | `serve/score.py` | Azure ML calls `init()` then `run()` |
| `environment.build.dockerfile_path` | `serve/Dockerfile` | Image build spec |
| `instance_type` | `Standard_E4s_v3` | 4 vCPU CPU VM |
| `environment_variables.DEVICE` | `cpu` | Passed to `InferenceEngine` |
| `environment_variables.DTYPE` | `fp32` | Weight/activation dtype |

## `deployment-blue-gpu.yaml` (GPU)

Same structure; changes:

| Field | GPU value |
|-------|-----------|
| `instance_type` | `Standard_NC4as_T4_v3` |
| `DEVICE` | `cuda` |
| `DTYPE` | `bf16` |
| `max_concurrent_requests_per_instance` | `16` |

## `compute-gpu.yaml`

| Field | Meaning |
|-------|---------|
| `name` | `llmops-gpu-cluster`, referenced by training job |
| `size` | `Standard_NC4as_T4_v3` |
| `min_instances` / `max_instances` | Autoscale bounds for training cluster |

## `job-lora-train.yaml`

| Field | Meaning |
|-------|---------|
| `code` | `..`, repo root as job code |
| `command` | Shell pipeline: pip install + `train.lora_train` |
| `compute` | `azureml:llmops-gpu-cluster` |
| `inputs.train_jsonl` | Registered data asset `qwen3-train-sample:1` |
| `outputs.adapter` | Writable folder for adapter weights |

## Makefile mapping

| Target | YAML / action |
|--------|----------------|
| `make deploy-blue` | `DEPLOY_BLUE_SPEC` → render → `az ml online-deployment create` |
| `make deploy-blue-gpu` | `deployment-blue-gpu.yaml` + T4 SKU |
| `make deploy-all` | Full CPU pipeline |
| `make deploy-all-gpu` | Full GPU pipeline |
| `make compute-gpu-create` | `compute-gpu.yaml` |
| `make train-lora-azure` | `job-lora-train.yaml` |

## GitHub Actions mapping

| Workflow | Trigger | Effect |
|----------|---------|--------|
| `azureml-deploy.yml` | `profile: cpu\|gpu` | `make deploy-all` or GPU variant |
| `azureml-train-lora.yml` | manual | `make train-lora-azure` |
