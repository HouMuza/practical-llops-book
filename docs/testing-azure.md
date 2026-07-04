# Azure end-to-end test (companion repo)

Use a local `.env` for subscription settings. It is gitignored; copy from `.env.example`.

## Prerequisites

```bash
az login
az account set --subscription "$SUBSCRIPTION_ID"
az extension add -n ml -y   # if needed
make install
```

Load env:

```bash
set -a && source .env && set +a
```

Set `ENDPOINT_NAME` in `.env` to something unique to you. Azure ML endpoint names
must be unique across the whole region, not just within your workspace.

## 1. CPU deployment (default, no GPU quota required)

Create or reuse a workspace named by `RESOURCE_GROUP` and `WORKSPACE_NAME` in
`.env`.

```bash
make deploy-all
make endpoint-test
```

`deploy-all` is idempotent: reruns update or skip existing resources.

Expected `endpoint-test` response fields: `text`, `request_id`, `trace_id`.

## 2. GPU deployment (after quota approval)

Check quota:

```bash
az vm list-usage --location westeurope -o table | grep -i T4
```

If `Standard NCASv3_T4 Family` limit is greater than 0:

```bash
make deploy-all-gpu
make endpoint-test
```

## 3. Local API + eval (before or after Azure)

Terminal A:

```bash
make serve
```

Terminal B:

```bash
make eval
```

## 4. LLM-as-judge (optional, needs API key in `.env`)

```bash
export JUDGE_API_KEY=...   # or add to .env
make eval
make eval-judge
```

## 5. LoRA training on Azure

```bash
make compute-gpu-create    # fails without GPU quota
make train-lora-azure
```

Monitor jobs in Azure ML Studio.

## 6. GitHub Actions (optional)

Repository secrets (Settings → Secrets):

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

Workflow: **Deploy Azure ML Endpoint** → profile `cpu` or `gpu`.

## Troubleshooting

| Issue | Action |
|-------|--------|
| `ReadOnlyDisabledSubscription` | Re-enable subscription in Azure Portal |
| Endpoint stuck in **Failed** after suspension | `bash scripts/clean_failed_endpoints.sh qwen3-yourname-prod` then `make endpoint-create` |
| `InternalServerError` on endpoint create | Often subscription/workspace propagation after reactivation. Wait 1-2 hours and retry. If it persists, open an [Azure support ticket](https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade) with the Correlation ID from the CLI error. |
| Endpoint YAML had `traffic: blue` before deployment exists | Fixed in `deploy/endpoint.yaml`; traffic is set by `make deploy-blue --all-traffic` |
| Workspace tag `SubscriptionState: Suspended` | Verify subscription is active in Azure Portal |
| Deployment stuck on Creating | Studio → Endpoints → blue → Environment build logs |
| GPU deploy fails | Use `make deploy-all` (CPU) until T4 quota is approved |

## What not to commit

`.env`, `HF_TOKEN`, `JUDGE_API_KEY`, endpoint keys, and `mlruns/` stay local. See `.gitignore`.
