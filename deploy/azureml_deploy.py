from __future__ import annotations

import argparse
import os
from typing import Any

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    BuildContext,
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    OnlineRequestSettings,
)
from azure.identity import DefaultAzureCredential

from deploy.obs_env import resolve_observability_env


def get_ml_client() -> MLClient:
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    workspace_name = os.environ["AZURE_ML_WORKSPACE"]
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return MLClient(credential, subscription_id, resource_group, workspace_name)


def ensure_model(ml_client: MLClient, model_name: str, model_version: str, path: str) -> Model:
    try:
        return ml_client.models.get(name=model_name, version=model_version)
    except Exception:
        model = Model(name=model_name, version=model_version, path=path, type="custom_model")
        return ml_client.models.create_or_update(model)


def create_or_update_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    instance_type: str,
    instance_count: int,
    *,
    device: str | None = None,
    dtype: str | None = None,
) -> ManagedOnlineDeployment:
    obs_env = resolve_observability_env()
    device = device or os.getenv("DEVICE", "cpu")
    dtype = dtype or os.getenv("DTYPE", "fp32")
    environment = Environment(
        name=f"{endpoint_name}-env",
        version="1",
        build=BuildContext(path=".", dockerfile_path="serve/Dockerfile"),
    )
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{model_version}",
        environment=environment,
        code_configuration=CodeConfiguration(code=".", scoring_script="serve/score.py"),
        instance_type=instance_type,
        instance_count=instance_count,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=90000,
            max_concurrent_requests_per_instance=8,
        ),
        app_insights_enabled=True,
        environment_variables={
            "MODEL_NAME": "Qwen/Qwen3-0.6B",
            "DEVICE": device,
            "DTYPE": dtype,
            "OTEL_SERVICE_NAME": f"llmops-qwen3-{deployment_name}",
            **{k: v for k, v in obs_env.items() if k in {"APPLICATIONINSIGHTS_CONNECTION_STRING", "MLFLOW_TRACKING_URI", "LOG_FORMAT"}},
        },
    )
    return ml_client.online_deployments.begin_create_or_update(deployment).result()


def set_traffic(ml_client: MLClient, endpoint_name: str, traffic: dict[str, int]) -> ManagedOnlineEndpoint:
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = traffic
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def show_endpoint(ml_client: MLClient, endpoint_name: str) -> dict[str, Any]:
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    return {
        "name": endpoint.name,
        "auth_mode": endpoint.auth_mode,
        "scoring_uri": endpoint.scoring_uri,
        "provisioning_state": endpoint.provisioning_state,
        "traffic": endpoint.traffic,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Azure ML endpoint deployment helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    full = sub.add_parser("deploy-all", help="Register model, create endpoint, create blue deployment, route 100% traffic")
    full.add_argument("--endpoint", default="qwen3-yourname-prod")
    full.add_argument("--deployment", default="blue")
    full.add_argument("--model-name", default="qwen3-0.6b")
    full.add_argument("--model-version", default="1")
    full.add_argument("--model-path", default="deploy/model_artifact")
    full.add_argument("--instance-type", default=os.getenv("INSTANCE_TYPE", "Standard_E4s_v3"))
    full.add_argument("--instance-count", type=int, default=1)
    full.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    full.add_argument("--dtype", default=os.getenv("DTYPE", "fp32"))

    traffic = sub.add_parser("set-traffic", help="Update endpoint traffic split")
    traffic.add_argument("--endpoint", default="qwen3-yourname-prod")
    traffic.add_argument("--blue", type=int, default=100)
    traffic.add_argument("--green", type=int, default=0)

    show = sub.add_parser("show", help="Show endpoint details")
    show.add_argument("--endpoint", default="qwen3-yourname-prod")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ml_client = get_ml_client()

    if args.cmd == "deploy-all":
        print("Ensuring model is registered...")
        ensure_model(ml_client, args.model_name, args.model_version, args.model_path)

        print("Creating/updating endpoint...")
        create_or_update_endpoint(ml_client, args.endpoint)

        print("Creating/updating deployment...")
        create_or_update_deployment(
            ml_client,
            endpoint_name=args.endpoint,
            deployment_name=args.deployment,
            model_name=args.model_name,
            model_version=args.model_version,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            device=args.device,
            dtype=args.dtype,
        )

        print("Routing traffic...")
        set_traffic(ml_client, args.endpoint, {args.deployment: 100})

        print("Deployment complete")
        print(show_endpoint(ml_client, args.endpoint))

    elif args.cmd == "set-traffic":
        traffic = {"blue": args.blue, "green": args.green}
        # Remove zero-value slots to keep endpoint traffic map clean.
        traffic = {k: v for k, v in traffic.items() if v > 0}
        print("Updating traffic:", traffic)
        set_traffic(ml_client, args.endpoint, traffic)
        print(show_endpoint(ml_client, args.endpoint))

    elif args.cmd == "show":
        print(show_endpoint(ml_client, args.endpoint))


if __name__ == "__main__":
    main()
