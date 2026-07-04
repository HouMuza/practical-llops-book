from __future__ import annotations
 
import argparse
import asyncio
import json
from pathlib import Path
 
import mlflow
import pandas as pd
 
from eval.scorers import exact_match, schema_validity, token_f1
from obs.mlflow_config import configure_mlflow
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="JSONL with id, prediction, reference, optional schema")
    p.add_argument("--model-version", required=True)
    p.add_argument("--experiment", default="qwen3-offline-eval")
    return p.parse_args()
 
 
def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
 
 
def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    scored = []
    for row in rows:
        pred = row.get("prediction", "")
        ref = row.get("reference", "")
        scored.append(
            {
                "id": row.get("id"),
                "exact_match": exact_match(pred, ref),
                "token_f1": token_f1(pred, ref),
                "schema_validity": schema_validity(pred, row.get("schema")),
            }
        )
    df = pd.DataFrame(scored)
    configure_mlflow(args.experiment)
    with mlflow.start_run(run_name=f"eval-{args.model_version}"):
        mlflow.log_param("model_version", args.model_version)
        for col in ["exact_match", "token_f1", "schema_validity"]:
            mlflow.log_metric(col, float(df[col].mean()))
        report = Path("offline_eval_report.csv")
        df.to_csv(report, index=False)
        mlflow.log_artifact(str(report))
        print(df.mean(numeric_only=True).to_string())
 
 
if __name__ == "__main__":
    main()
