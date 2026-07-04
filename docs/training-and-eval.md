# Training & evaluation

## LoRA / QLoRA on real data

### Data format

JSONL with either ChatML `messages` or `input` / `output` pairs. See `train/data/sample.jsonl`:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### Local training (GPU recommended)

```bash
make install
# Use your own file: make train-lora-local TRAIN_JSONL=/path/to/train.jsonl
make train-lora-local
make train-qlora-local
```

Outputs:

```text
outputs/lora/     # LoRA adapter weights
outputs/qlora/    # QLoRA adapter weights
```

MLflow tracking auto-configures when `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` are set, or set `MLFLOW_TRACKING_URI` explicitly.

### Azure ML training job

YAML: `deploy/job-lora-train.yaml`

```bash
export SUBSCRIPTION_ID=...
export RESOURCE_GROUP=...
export WORKSPACE_NAME=...

make compute-gpu-create      # Standard_NC4as_T4_v3 cluster
make train-lora-azure      # uploads sample data + submits job
```

Monitor in Azure ML Studio → Jobs.

#### GitHub Actions

Workflow: `.github/workflows/azureml-train-lora.yml`

Requires the same OIDC secrets as deployment. Triggers `compute-gpu-create` and `train-lora-azure`.

#### Use your own training file on Azure

```bash
az ml data create --name my-train-set --version 1 \
  --path /path/to/train.jsonl --type uri_file \
  -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

Edit `deploy/job-lora-train.yaml`:

```yaml
inputs:
  train_jsonl:
    type: uri_file
    path: azureml:my-train-set:1
```

## Offline evaluation

```bash
# Terminal 1
make serve-gpu

# Terminal 2
make eval
```

Metrics logged: `exact_match`, `token_f1`, `schema_validity`.

## LLM-as-judge (position-bias debiased)

Requires an **OpenAI-compatible** chat completions API (OpenAI, Azure OpenAI, etc.).

### Setup

```bash
export JUDGE_API_KEY=sk-...
export JUDGE_ENDPOINT=https://api.openai.com/v1/chat/completions   # optional
export JUDGE_MODEL=gpt-4o-mini                                      # optional
```

### Azure OpenAI example

```bash
export JUDGE_ENDPOINT=https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-02-15-preview
export JUDGE_API_KEY=<azure-openai-key>
export JUDGE_MODEL=gpt-4o-mini
```

### Run

```bash
make eval          # generates eval/predictions.jsonl from live API
make eval-judge    # compares against eval/baseline_predictions.jsonl
```

`eval/run_judge_eval.py` calls `debiased_win_rate` in `eval/scorers.py`. Each example is judged **twice** with candidate/baseline order swapped.

Results: `judge_eval_report.json` + MLflow metric `candidate_win_rate`.

### Provide your own baseline

Replace `eval/baseline_predictions.jsonl` with outputs from a previous model version:

```json
{"id":"ex-001","input":"...","prediction":"...","reference":"..."}
```

Or generate baseline predictions by pointing `generate_predictions` at a different endpoint (fork the script or run twice with different servers).

## Evaluation against Azure ML `/score`

For deployed endpoints, adapt the payload:

```bash
SCORING_URI=$(az ml online-endpoint show --name qwen3-prod -g $RG -w $WS --query scoring_uri -o tsv)
KEY=$(az ml online-endpoint get-credentials --name qwen3-prod -g $RG -w $WS --query primaryKey -o tsv)

curl -s -X POST "$SCORING_URI" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Extract ticket T-55","max_new_tokens":20}'
```

The response is JSON with `text`, not OpenAI format. Use `make endpoint-test` as the template.
