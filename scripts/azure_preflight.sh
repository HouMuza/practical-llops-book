#!/usr/bin/env bash
# Verify Azure CLI login, subscription state, and optional GPU quota before deploy.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi

: "${SUBSCRIPTION_ID:?Set SUBSCRIPTION_ID in .env}"
: "${RESOURCE_GROUP:?Set RESOURCE_GROUP in .env}"
: "${WORKSPACE_NAME:?Set WORKSPACE_NAME in .env}"

echo "== Account =="
az account show --subscription "$SUBSCRIPTION_ID" --query '{name:name, id:id, state:state}' -o table

STATE=$(az account show --subscription "$SUBSCRIPTION_ID" --query state -o tsv)
if [[ "$STATE" != "Enabled" ]]; then
  echo "ERROR: Subscription state is '$STATE'. Re-enable it in Azure Portal before deploying."
  exit 1
fi

echo "== ML workspace =="
az ml workspace show -g "$RESOURCE_GROUP" -w "$WORKSPACE_NAME" \
  --query '{name:name, location:location, rg:resource_group}' -o table

echo "== T4 quota (westeurope) =="
az vm list-usage --location "${AML_LOCATION:-westeurope}" \
  --query "[?name.value=='standardNCASv3T4Family'].{family:name.localizedValue,used:currentValue,limit:limit}" -o table

echo "== Endpoints =="
az ml online-endpoint list -g "$RESOURCE_GROUP" -w "$WORKSPACE_NAME" -o table || true

echo "Preflight OK. Run: make deploy-all  (CPU)  or  make deploy-all-gpu  (if T4 limit > 0)"
