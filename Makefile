# ==============================================================================
# LLMOps Deep Dive — Makefile
# Usage: make <target>
# Requires: az cli, azure-ai-ml, active Azure ML workspace configured via env vars
# ==============================================================================

# ── Local settings ──────────────────────────────────────────────────────────────
PYTHON        ?= .venv/bin/python
AZ            ?= az
MODEL_NAME    ?= Qwen/Qwen3-0.6B
PORT          ?= 8000
TRAIN_JSONL   ?= train/data/sample.jsonl
DOCKER_IMAGE  ?= practical-llops-book:latest
DEPLOY_BLUE_SPEC ?= deploy/deployment-blue.yaml
DEPLOY_GREEN_SPEC ?= deploy/deployment-green.yaml

# ── Azure ML settings (override via env or CLI) ──────────────────────────────────
AML_RESOURCE_GROUP   ?= $(RESOURCE_GROUP)
AML_WORKSPACE        ?= $(WORKSPACE_NAME)
AML_SUBSCRIPTION     ?= $(SUBSCRIPTION_ID)
AML_LOCATION         ?= swedencentral
ENDPOINT_NAME        ?= qwen3-prod
MODEL_ASSET_NAME     ?= qwen3-0-6b
MODEL_ASSET_VERSION  ?= 1
MODEL_ASSET_PATH     ?= deploy/model_artifact
INSTANCE_TYPE        ?= Standard_E4s_v3
INSTANCE_COUNT       ?= 1

# ── Local development ────────────────────────────────────────────────────────────

.PHONY: install
install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

.PHONY: serve
serve:
	MODEL_NAME=$(MODEL_NAME) DEVICE=cpu DTYPE=fp32 \
	  .venv/bin/uvicorn serve.api:app --host 0.0.0.0 --port $(PORT) --reload

.PHONY: serve-gpu
serve-gpu:
	MODEL_NAME=$(MODEL_NAME) DEVICE=cuda DTYPE=bf16 \
	  .venv/bin/uvicorn serve.api:app --host 0.0.0.0 --port $(PORT) --reload

.PHONY: docker-build
docker-build:
	docker build -f serve/Dockerfile -t $(DOCKER_IMAGE) .

.PHONY: docker-run
docker-run: docker-build
	docker run --rm -p $(PORT):8000 \
	  -e MODEL_NAME=$(MODEL_NAME) -e DEVICE=cpu -e DTYPE=fp32 \
	  $(DOCKER_IMAGE)

.PHONY: docker-run-gpu
docker-run-gpu: docker-build
	docker run --rm -p $(PORT):8000 --gpus all \
	  -e MODEL_NAME=$(MODEL_NAME) -e DEVICE=cuda -e DTYPE=bf16 \
	  $(DOCKER_IMAGE)

.PHONY: health
health:
	curl -s http://localhost:$(PORT)/health | python -m json.tool

.PHONY: smoke
smoke:
	curl -s -X POST http://localhost:$(PORT)/v1/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"prompt":"Explain KV cache in one paragraph.","max_new_tokens":80,"stream":false}' \
	  | python -m json.tool

.PHONY: smoke-stream
smoke-stream:
	curl -s -X POST http://localhost:$(PORT)/v1/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"prompt":"What is LoRA?","max_new_tokens":30,"stream":true}'

# ── Evaluation ───────────────────────────────────────────────────────────────────

.PHONY: eval
eval:
	@echo "Requires API running: make serve (in another terminal)"
	$(PYTHON) -m eval.generate_predictions
	$(PYTHON) -m eval.run_eval \
	  --predictions eval/predictions.jsonl \
	  --model-version $(MODEL_ASSET_VERSION)

.PHONY: eval-judge
eval-judge:
	@echo "Requires JUDGE_API_KEY and predictions JSONL (run make eval first)"
	$(PYTHON) -m eval.run_judge_eval \
	  --predictions eval/predictions.jsonl \
	  --baseline-predictions eval/baseline_predictions.jsonl

.PHONY: train-lora-local
train-lora-local:
	$(PYTHON) -m train.lora_train --train-jsonl $(TRAIN_JSONL) --output-dir outputs/lora --epochs 1

.PHONY: train-qlora-local
train-qlora-local:
	$(PYTHON) -m train.qlora_train --train-jsonl $(TRAIN_JSONL) --output-dir outputs/qlora --epochs 1

.PHONY: compute-gpu-create
compute-gpu-create:
	$(AZ) ml compute create \
	  --file deploy/compute-gpu.yaml \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: upload-train-data
upload-train-data:
	@if $(AZ) ml data show \
	  --name qwen3-train-sample \
	  --version 1 \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) >/dev/null 2>&1; then \
	  echo "Data asset qwen3-train-sample:1 already exists; skipping upload."; \
	else \
	  $(AZ) ml data create \
	    --name qwen3-train-sample \
	    --version 1 \
	    --path train/data/sample.jsonl \
	    --type uri_file \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	fi

.PHONY: train-lora-azure
train-lora-azure: upload-train-data
	$(AZ) ml job create \
	  --file deploy/job-lora-train.yaml \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: obs-show
obs-show:
	$(PYTHON) -m deploy.obs_env show

.PHONY: obs-report
obs-report:
	$(PYTHON) -m obs.live_report --hours $${HOURS:-24} --output $${OUT:-obs/reports/observability_report.md}

.PHONY: deploy-observability
deploy-observability:
	@eval "$$( $(PYTHON) -m deploy.obs_env shell )"; \
	if [ -z "$$LOG_ANALYTICS_WORKSPACE_ID" ] || [ -z "$$AML_WORKSPACE_ID" ]; then \
	  echo "Skipping deploy-observability because workspace observability resources could not be resolved."; \
	else \
	  $(AZ) monitor diagnostic-settings create \
	    --name aml-workspace-observability \
	    --resource "$$AML_WORKSPACE_ID" \
	    --workspace "$$LOG_ANALYTICS_WORKSPACE_ID" \
	    --logs '[{"categoryGroup":"allLogs","enabled":true}]' \
	    --metrics '[{"category":"AllMetrics","enabled":true}]' >/dev/null; \
	  ENDPOINT_ID=$$($(AZ) ml online-endpoint show \
	    --name $(ENDPOINT_NAME) \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE) \
	    --query id -o tsv 2>/dev/null || true); \
	  if [ -n "$$ENDPOINT_ID" ]; then \
	    $(AZ) monitor diagnostic-settings create \
	      --name aml-endpoint-observability \
	      --resource "$$ENDPOINT_ID" \
	      --workspace "$$LOG_ANALYTICS_WORKSPACE_ID" \
	      --logs '[{"categoryGroup":"allLogs","enabled":true}]' \
	      --metrics '[{"category":"AllMetrics","enabled":true}]' >/dev/null; \
	  fi; \
	  echo "Azure Monitor diagnostics wired to $$LOG_ANALYTICS_WORKSPACE_ID"; \
	fi

# ── Azure ML deployment ──────────────────────────────────────────────────────────

.PHONY: group-create
group-create:
	$(AZ) group create \
	  --name $(AML_RESOURCE_GROUP) \
	  --location $(AML_LOCATION)

.PHONY: workspace-create
workspace-create:
	$(AZ) ml workspace create \
	  --name $(AML_WORKSPACE) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --location $(AML_LOCATION)

.PHONY: deploy-sdk
deploy-sdk:
	AZURE_SUBSCRIPTION_ID=$(AML_SUBSCRIPTION) \
	AZURE_RESOURCE_GROUP=$(AML_RESOURCE_GROUP) \
	AZURE_ML_WORKSPACE=$(AML_WORKSPACE) \
	$(PYTHON) -m deploy.azureml_deploy deploy-all \
	  --endpoint $(ENDPOINT_NAME) \
	  --deployment blue \
	  --model-name $(MODEL_ASSET_NAME) \
	  --model-version $(MODEL_ASSET_VERSION) \
	  --model-path $(MODEL_ASSET_PATH) \
	  --instance-type $(INSTANCE_TYPE) \
	  --instance-count $(INSTANCE_COUNT)

.PHONY: traffic-split-sdk
traffic-split-sdk:
	AZURE_SUBSCRIPTION_ID=$(AML_SUBSCRIPTION) \
	AZURE_RESOURCE_GROUP=$(AML_RESOURCE_GROUP) \
	AZURE_ML_WORKSPACE=$(AML_WORKSPACE) \
	$(PYTHON) -m deploy.azureml_deploy set-traffic \
	  --endpoint $(ENDPOINT_NAME) \
	  --blue 90 \
	  --green 10

.PHONY: traffic-cutover-sdk
traffic-cutover-sdk:
	AZURE_SUBSCRIPTION_ID=$(AML_SUBSCRIPTION) \
	AZURE_RESOURCE_GROUP=$(AML_RESOURCE_GROUP) \
	AZURE_ML_WORKSPACE=$(AML_WORKSPACE) \
	$(PYTHON) -m deploy.azureml_deploy set-traffic \
	  --endpoint $(ENDPOINT_NAME) \
	  --blue 0 \
	  --green 100

.PHONY: endpoint-show-sdk
endpoint-show-sdk:
	AZURE_SUBSCRIPTION_ID=$(AML_SUBSCRIPTION) \
	AZURE_RESOURCE_GROUP=$(AML_RESOURCE_GROUP) \
	AZURE_ML_WORKSPACE=$(AML_WORKSPACE) \
	$(PYTHON) -m deploy.azureml_deploy show --endpoint $(ENDPOINT_NAME)

.PHONY: az-login
az-login:
	$(AZ) login
	$(AZ) account set --subscription $(AML_SUBSCRIPTION)

.PHONY: model-register
model-register:
	@if $(AZ) ml model show \
	  --name $(MODEL_ASSET_NAME) \
	  --version $(MODEL_ASSET_VERSION) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) >/dev/null 2>&1; then \
	  echo "Model $(MODEL_ASSET_NAME):$(MODEL_ASSET_VERSION) already exists; skipping registration."; \
	else \
	  $(AZ) ml model create \
	    --name $(MODEL_ASSET_NAME) \
	    --version $(MODEL_ASSET_VERSION) \
	    --type custom_model \
	    --path $(MODEL_ASSET_PATH) \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	fi

.PHONY: endpoint-create
endpoint-create:
	@if $(AZ) ml online-endpoint show \
	  --name $(ENDPOINT_NAME) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) >/dev/null 2>&1; then \
	  echo "Endpoint $(ENDPOINT_NAME) already exists; applying update."; \
	  $(AZ) ml online-endpoint update \
	    --name $(ENDPOINT_NAME) \
	    --file deploy/endpoint.yaml \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	else \
	  $(AZ) ml online-endpoint create \
	    --file deploy/endpoint.yaml \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	fi

.PHONY: endpoint-update
endpoint-update:
	$(AZ) ml online-endpoint update \
	  --name $(ENDPOINT_NAME) \
	  --file deploy/endpoint.yaml \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: deploy-blue
deploy-blue:
	@SPEC=$$(mktemp /tmp/deploy-blue.XXXXXX.yaml); \
	$(PYTHON) -m deploy.render_deployment --input $(DEPLOY_BLUE_SPEC) --output "$$SPEC"; \
	if $(AZ) ml online-deployment show \
	  --name blue \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) >/dev/null 2>&1; then \
	  echo "Deployment blue already exists; applying update."; \
	  $(AZ) ml online-deployment update \
	    --name blue \
	    --endpoint-name $(ENDPOINT_NAME) \
	    --file "$$SPEC" \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	  $(AZ) ml online-endpoint update \
	    --name $(ENDPOINT_NAME) \
	    --traffic "blue=100" \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	else \
	  $(AZ) ml online-deployment create \
	    --file "$$SPEC" \
	    --all-traffic \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	fi; \
	rm -f "$$SPEC"

.PHONY: deploy-blue-gpu
deploy-blue-gpu:
	$(MAKE) deploy-blue DEPLOY_BLUE_SPEC=deploy/deployment-blue-gpu.yaml INSTANCE_TYPE=Standard_NC4as_T4_v3

.PHONY: deploy-blue-update
deploy-blue-update:
	@SPEC=$$(mktemp /tmp/deploy-blue.XXXXXX.yaml); \
	$(PYTHON) -m deploy.render_deployment --input $(DEPLOY_BLUE_SPEC) --output "$$SPEC"; \
	$(AZ) ml online-deployment update \
	  --name blue \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --file "$$SPEC" \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE); \
	rm -f "$$SPEC"
	$(AZ) ml online-endpoint update \
	  --name $(ENDPOINT_NAME) \
	  --traffic "blue=100" \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: deploy-green
deploy-green:
	@SPEC=$$(mktemp /tmp/deploy-green.XXXXXX.yaml); \
	$(PYTHON) -m deploy.render_deployment --input $(DEPLOY_GREEN_SPEC) --output "$$SPEC"; \
	if $(AZ) ml online-deployment show \
	  --name green \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) >/dev/null 2>&1; then \
	  echo "Deployment green already exists; applying update."; \
	  $(AZ) ml online-deployment update \
	    --name green \
	    --endpoint-name $(ENDPOINT_NAME) \
	    --file "$$SPEC" \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	else \
	  $(AZ) ml online-deployment create \
	    --file "$$SPEC" \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE); \
	fi; \
	rm -f "$$SPEC"

.PHONY: deploy-green-gpu
deploy-green-gpu:
	$(MAKE) deploy-green DEPLOY_GREEN_SPEC=deploy/deployment-green-gpu.yaml INSTANCE_TYPE=Standard_NC4as_T4_v3

.PHONY: deploy-green-update
deploy-green-update:
	@SPEC=$$(mktemp /tmp/deploy-green.XXXXXX.yaml); \
	$(PYTHON) -m deploy.render_deployment --input $(DEPLOY_GREEN_SPEC) --output "$$SPEC"; \
	$(AZ) ml online-deployment update \
	  --name green \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --file "$$SPEC" \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE); \
	rm -f "$$SPEC"

.PHONY: traffic-split
traffic-split:
	# Route 10% to green for canary validation, 90% stays on blue
	$(AZ) ml online-endpoint update \
	  --name $(ENDPOINT_NAME) \
	  --traffic "blue=90 green=10" \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: traffic-cutover
traffic-cutover:
	# Full cutover to green after validation passes
	$(AZ) ml online-endpoint update \
	  --name $(ENDPOINT_NAME) \
	  --traffic "blue=0 green=100" \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: delete-blue
delete-blue:
	$(AZ) ml online-deployment delete \
	  --name blue \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --yes \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE)

.PHONY: deploy-autoscale
deploy-autoscale:
	$(eval DEPLOYMENT_ID=$(shell $(AZ) ml online-deployment show \
	  --name blue \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) \
	  --query id -o tsv))
	$(AZ) deployment group create \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --template-file deploy/autoscale.bicep \
	  --parameters deploymentResourceId=$(DEPLOYMENT_ID)

.PHONY: deploy-alerts
deploy-alerts:
	@if [ -z "$(ACTION_GROUP_ID)" ]; then \
	  echo "Skipping deploy-alerts because ACTION_GROUP_ID is not set."; \
	else \
	  ENDPOINT_ID=$$($(AZ) ml online-endpoint show \
	    --name $(ENDPOINT_NAME) \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --workspace-name $(AML_WORKSPACE) \
	    --query id -o tsv) && \
	  $(AZ) deployment group create \
	    --resource-group $(AML_RESOURCE_GROUP) \
	    --template-file obs/alerts.bicep \
	    --parameters \
	      targetResourceId=$$ENDPOINT_ID \
	      actionGroupId=$(ACTION_GROUP_ID); \
	fi

.PHONY: endpoint-test
endpoint-test:
	$(eval SCORING_URI=$(shell $(AZ) ml online-endpoint show \
	  --name $(ENDPOINT_NAME) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) \
	  --query scoring_uri -o tsv))
	$(eval TOKEN=$(shell $(AZ) ml online-endpoint get-credentials \
	  --name $(ENDPOINT_NAME) \
	  --resource-group $(AML_RESOURCE_GROUP) \
	  --workspace-name $(AML_WORKSPACE) \
	  --query primaryKey -o tsv 2>/dev/null || $(AZ) account get-access-token --query accessToken -o tsv))
	curl -s -X POST $(SCORING_URI) \
	  -H "Authorization: Bearer $(TOKEN)" \
	  -H 'Content-Type: application/json' \
	  -d '{"prompt":"Explain KV cache.","max_new_tokens":40}' \
	  | $(PYTHON) -c 'import json, sys; payload = sys.stdin.read(); obj = json.loads(payload); obj = json.loads(obj) if isinstance(obj, str) else obj; print(json.dumps(obj, indent=2))'

# ── Full deploy pipeline (first time) ───────────────────────────────────────────

.PHONY: deploy-all-gpu
deploy-all-gpu: group-create workspace-create model-register endpoint-create deploy-blue-gpu deploy-observability deploy-autoscale deploy-alerts
	@echo "✓ GPU endpoint deployed: $(ENDPOINT_NAME)"
	@echo "Run 'make endpoint-test' to validate."

.PHONY: deploy-all
deploy-all: group-create workspace-create model-register endpoint-create deploy-blue deploy-observability deploy-autoscale deploy-alerts
	@echo "✓ Endpoint deployed: $(ENDPOINT_NAME)"
	@echo "Run 'make endpoint-test' to validate."
