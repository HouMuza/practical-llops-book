from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _az_base_command() -> list[str]:
    override = os.getenv("AZ_CLI_CMD")
    if override:
        return shlex.split(override)

    workspace_cli = Path(__file__).resolve().parents[1] / ".azcli-venv" / "bin" / "python"
    if workspace_cli.exists():
        return [str(workspace_cli), "-m", "azure.cli"]

    return ["az"]


def _run_az(args: list[str]) -> str:
    proc = subprocess.run(
        _az_base_command() + args,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _run_az_json(args: list[str]) -> dict[str, Any]:
    output = _run_az(args + ["-o", "json"])
    return json.loads(output) if output else {}


def resolve_observability_env() -> dict[str, str]:
    env: dict[str, str] = {"LOG_FORMAT": os.getenv("LOG_FORMAT", "json")}
    resource_group = _env("RESOURCE_GROUP", "AZURE_RESOURCE_GROUP")
    workspace_name = _env("WORKSPACE_NAME", "AZURE_ML_WORKSPACE")

    if not resource_group or not workspace_name:
        return env

    try:
        workspace = _run_az_json(
            [
                "ml",
                "workspace",
                "show",
                "--name",
                workspace_name,
                "--resource-group",
                resource_group,
                "--query",
                "{id:id,application_insights:application_insights,mlflow_tracking_uri:mlflow_tracking_uri}",
            ]
        )
        if workspace.get("id"):
            env["AML_WORKSPACE_ID"] = str(workspace["id"])
        if workspace.get("mlflow_tracking_uri"):
            env["MLFLOW_TRACKING_URI"] = str(workspace["mlflow_tracking_uri"])

        app_insights_id = workspace.get("application_insights")
        if app_insights_id:
            env["APPLICATIONINSIGHTS_RESOURCE_ID"] = str(app_insights_id)
            component = _run_az_json(
                [
                    "resource",
                    "show",
                    "--ids",
                    str(app_insights_id),
                    "--query",
                    "{connection_string:properties.ConnectionString,workspace_id:properties.WorkspaceResourceId}",
                ]
            )
            if component.get("connection_string"):
                env["APPLICATIONINSIGHTS_CONNECTION_STRING"] = str(component["connection_string"])
            if component.get("workspace_id"):
                env["LOG_ANALYTICS_WORKSPACE_ID"] = str(component["workspace_id"])
    except Exception as exc:
        print(f"warning: could not resolve workspace observability settings: {exc}", file=sys.stderr)

    return env


def build_az_set_args(env: dict[str, str]) -> str:
    pairs: list[str] = []
    for key in ["LOG_FORMAT", "APPLICATIONINSIGHTS_CONNECTION_STRING", "MLFLOW_TRACKING_URI"]:
        value = env.get(key)
        if value:
            pairs.append(f"environment_variables.{key}={value}")
    if not pairs:
        return ""
    return "--set " + " ".join(shlex.quote(pair) for pair in pairs)


def build_shell_assignments(env: dict[str, str]) -> str:
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()) if value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve Azure Monitor and MLflow settings for the AML workspace")
    parser.add_argument("command", choices=["show", "az-set", "shell"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = resolve_observability_env()
    if args.command == "show":
        print(json.dumps(env, indent=2, sort_keys=True))
    elif args.command == "az-set":
        print(build_az_set_args(env))
    else:
        print(build_shell_assignments(env))


if __name__ == "__main__":
    main()