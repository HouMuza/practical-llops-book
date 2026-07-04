"""Run LLM-as-judge evaluation with position-bias debiasing.

Requires an OpenAI-compatible judge endpoint and API key, for example Azure OpenAI
or OpenAI. Configure via environment variables:

  JUDGE_ENDPOINT   default https://api.openai.com/v1/chat/completions
  JUDGE_API_KEY    required
  JUDGE_MODEL      default gpt-4o-mini
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import mlflow

from eval.scorers import JudgeConfig, debiased_win_rate
from obs.mlflow_config import configure_mlflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-judge eval with debiasing")
    p.add_argument("--predictions", default="eval/predictions.jsonl")
    p.add_argument("--baseline-predictions", default=None, help="Optional baseline JSONL for pairwise judging")
    p.add_argument("--experiment", default="qwen3-judge-eval")
    p.add_argument("--endpoint", default=os.getenv("JUDGE_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
    p.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY"))
    p.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    return p.parse_args()


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


async def run(args: argparse.Namespace) -> list[dict]:
    if not args.api_key:
        raise SystemExit("Set JUDGE_API_KEY or pass --api-key")

    preds = read_jsonl(args.predictions)
    baseline_rows = read_jsonl(args.baseline_predictions) if args.baseline_predictions else None
    baseline_by_id = {row["id"]: row.get("prediction", "") for row in baseline_rows} if baseline_rows else {}

    config = JudgeConfig(endpoint=args.endpoint, api_key=args.api_key, model=args.model)
    results: list[dict] = []

    for row in preds:
        question = row.get("input") or row.get("question") or row["id"]
        reference = row.get("reference", "")
        candidate = row.get("prediction", "")
        baseline = baseline_by_id.get(row["id"], row.get("baseline", ""))
        if not baseline:
            print(f"[skip] {row['id']}: no baseline prediction for pairwise judge")
            continue
        verdict = await debiased_win_rate(
            config,
            question=question,
            reference=reference,
            candidate=candidate,
            baseline=baseline,
        )
        results.append({"id": row["id"], **verdict})
        print(f"[{row['id']}] candidate_win_rate={verdict['candidate_win_rate']:.2f}")
    return results


def main() -> None:
    args = parse_args()
    results = asyncio.run(run(args))
    if not results:
        raise SystemExit("No judge results produced")

    avg_win = sum(r["candidate_win_rate"] for r in results) / len(results)
    configure_mlflow(args.experiment)
    with mlflow.start_run(run_name="judge-eval"):
        mlflow.log_metric("candidate_win_rate", avg_win)
        out = Path("judge_eval_report.json")
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))
    print(f"\nAverage candidate_win_rate: {avg_win:.4f}")


if __name__ == "__main__":
    main()
