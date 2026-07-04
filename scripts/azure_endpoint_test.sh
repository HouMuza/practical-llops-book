#!/usr/bin/env bash
# Load .env and run Azure ML endpoint smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${SUBSCRIPTION_ID:?Set SUBSCRIPTION_ID}"
: "${RESOURCE_GROUP:?Set RESOURCE_GROUP}"
: "${WORKSPACE_NAME:?Set WORKSPACE_NAME}"
az account set --subscription "$SUBSCRIPTION_ID"
make endpoint-test
