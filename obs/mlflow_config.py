from __future__ import annotations

import logging
import os

import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

log = logging.getLogger(__name__)


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def resolve_tracking_uri() -> str | None:
    tracking_uri = _env("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    subscription_id = _env("AZURE_SUBSCRIPTION_ID", "SUBSCRIPTION_ID")
    resource_group = _env("AZURE_RESOURCE_GROUP", "RESOURCE_GROUP")
    workspace_name = _env("AZURE_ML_WORKSPACE", "WORKSPACE_NAME")
    if not subscription_id or not resource_group or not workspace_name:
        return None

    try:
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        client = MLClient(credential, subscription_id, resource_group, workspace_name)
        workspace = client.workspaces.get(workspace_name)
        return getattr(workspace, "mlflow_tracking_uri", None)
    except Exception as exc:
        log.warning("mlflow_tracking_uri_resolution_failed", extra={"error": str(exc)})
        return None


def configure_mlflow(experiment_name: str) -> str | None:
    tracking_uri = resolve_tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        log.info("mlflow_tracking_configured", extra={"tracking_uri": tracking_uri})
    else:
        log.info("mlflow_tracking_default_local")
    mlflow.set_experiment(experiment_name)
    return tracking_uri