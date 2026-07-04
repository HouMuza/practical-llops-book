from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import mlflow
import pandas as pd
 
from eval.scorers import schema_validity, token_f1
from obs.mlflow_config import configure_mlflow
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--run-name", default="eval")
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    rows = [json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip()]
    scores = []
    for row in rows:
        scores.append(
            {
                "id": row.get("id"),
                "token_f1": token_f1(row.get("prediction", ""), row.get("reference", "")),
                "schema_valid": schema_validity(row.get("prediction", ""), row.get("schema")),
            }
        )
    df = pd.DataFrame(scores)
    configure_mlflow("qwen3-train-eval")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_metric("eval_token_f1", float(df["token_f1"].mean()))
        mlflow.log_metric("eval_schema_validity", float(df["schema_valid"].mean()))
        out = Path("eval_report.csv")
        df.to_csv(out, index=False)
        mlflow.log_artifact(str(out))
 
 
if __name__ == "__main__":
    main()
