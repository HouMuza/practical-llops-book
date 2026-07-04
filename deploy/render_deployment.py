from __future__ import annotations

import argparse
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
    spec = yaml.safe_load(Path(args.input_path).read_text(encoding="utf-8"))
    env = resolve_observability_env()

    env_vars = dict(spec.get("environment_variables") or {})
    for key in ["LOG_FORMAT", "APPLICATIONINSIGHTS_CONNECTION_STRING", "MLFLOW_TRACKING_URI"]:
        value = env.get(key)
        if value:
            env_vars[key] = value
    spec["environment_variables"] = env_vars

    Path(args.output_path).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()