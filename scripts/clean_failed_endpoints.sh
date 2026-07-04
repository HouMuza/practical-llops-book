#!/usr/bin/env bash
# Remove failed managed online endpoints so you can recreate cleanly.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${RESOURCE_GROUP:?}"
: "${WORKSPACE_NAME:?}"

echo "Endpoints in $WORKSPACE_NAME:"
az ml online-endpoint list -g "$RESOURCE_GROUP" -w "$WORKSPACE_NAME" -o table

for name in "$@"; do
  echo "Deleting $name ..."
  az ml online-endpoint delete --name "$name" -g "$RESOURCE_GROUP" -w "$WORKSPACE_NAME" --yes || true
done

echo "Done. Wait ~30s, then: make endpoint-create"
