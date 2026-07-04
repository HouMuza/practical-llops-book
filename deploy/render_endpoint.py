from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a managed online endpoint YAML")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(Path(args.input_path).read_text(encoding="utf-8"))
    endpoint_name = os.getenv("ENDPOINT_NAME")
    if endpoint_name:
        spec["name"] = endpoint_name
    Path(args.output_path).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
