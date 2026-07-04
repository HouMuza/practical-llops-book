from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from deploy.obs_env import resolve_observability_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a managed online deployment YAML with observability env vars")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    spec = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    env = resolve_observability_env()

    base_dir = input_path.parent

    endpoint_name = os.getenv("ENDPOINT_NAME")
    if endpoint_name:
        spec["endpoint_name"] = endpoint_name

    def resolve_local_path(value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    code_config = spec.get("code_configuration") or {}
    if code_config.get("code"):
        code_config["code"] = resolve_local_path(code_config["code"])
        spec["code_configuration"] = code_config

    build = (spec.get("environment") or {}).get("build") or {}
    if build.get("path"):
        build["path"] = resolve_local_path(build["path"])
        spec.setdefault("environment", {})["build"] = build

    env_vars = dict(spec.get("environment_variables") or {})
    for key in ["LOG_FORMAT", "APPLICATIONINSIGHTS_CONNECTION_STRING", "MLFLOW_TRACKING_URI"]:
        value = env.get(key)
        if value:
            env_vars[key] = value
    spec["environment_variables"] = env_vars

    Path(args.output_path).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()