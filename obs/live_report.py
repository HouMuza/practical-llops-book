from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from mlflow.tracking import MlflowClient

from deploy.obs_env import resolve_observability_env
from obs.mlflow_config import configure_mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a chapter-friendly live observability report")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", default="obs/reports/observability_report.md")
    return parser.parse_args()


def _arm_to_customer_id(arm_id: str) -> str:
    from deploy.obs_env import _run_az  # type: ignore[attr-defined]

    return _run_az(["resource", "show", "--ids", arm_id, "--query", "properties.customerId", "-o", "tsv"])


def _query_logs(workspace_customer_id: str, query: str, hours: int) -> list[dict[str, object]]:
    client = LogsQueryClient(DefaultAzureCredential(exclude_interactive_browser_credential=False))
    result = client.query_workspace(
        workspace_id=workspace_customer_id,
        query=query,
        timespan=timedelta(hours=hours),
    )
    if result.status != LogsQueryStatus.SUCCESS:
        return []

    rows: list[dict[str, object]] = []
    for table in result.tables:
        columns = [column.name if hasattr(column, "name") else str(column) for column in table.columns]
        for row in table.rows:
            rows.append(dict(zip(columns, row)))
    return rows


def _fmt_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "No rows returned."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, datetime):
                value = value.astimezone(timezone.utc).isoformat()
            value = str(value).replace("\n", " ")
            if len(value) > 120:
                value = value[:117] + "..."
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _recent_mlflow_runs(limit: int = 5) -> list[dict[str, object]]:
    configure_mlflow("llmops-observability-smoke")
    client = MlflowClient()
    experiments = client.search_experiments()
    rows: list[dict[str, object]] = []
    for experiment in experiments[:10]:
        runs = client.search_runs([experiment.experiment_id], max_results=limit, order_by=["attribute.start_time DESC"])
        for run in runs[:limit]:
            rows.append(
                {
                    "experiment": experiment.name,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "run_id": run.info.run_id,
                }
            )
    return rows[:limit]


def main() -> None:
    args = parse_args()
    env = resolve_observability_env()
    law_arm_id = env.get("LOG_ANALYTICS_WORKSPACE_ID")
    if not law_arm_id:
        raise SystemExit("LOG_ANALYTICS_WORKSPACE_ID could not be resolved")

    workspace_customer_id = _arm_to_customer_id(law_arm_id)
    request_summary = _query_logs(
        workspace_customer_id,
        """
AppRequests
| where TimeGenerated > ago(24h)
| summarize requests=count(), failures=countif(Success == false), p95_duration_ms=percentile(DurationMs, 95) by Name
| order by requests desc
        """,
        args.hours,
    )
    recent_requests = _query_logs(
        workspace_customer_id,
        """
AppRequests
| where TimeGenerated > ago(24h)
| project TimeGenerated, Name, ResultCode, Success, DurationMs, OperationId
| order by TimeGenerated desc
| take 10
        """,
        args.hours,
    )
    recent_traces = _query_logs(
        workspace_customer_id,
        """
AppTraces
| where TimeGenerated > ago(24h)
| project TimeGenerated, SeverityLevel, Message, OperationId
| order by TimeGenerated desc
| take 10
        """,
        args.hours,
    )
    mlflow_runs = _recent_mlflow_runs(limit=5)

    report = f"""# Observability Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Workspace

- AML workspace ID: {env.get('AML_WORKSPACE_ID', '')}
- Application Insights resource: {env.get('APPLICATIONINSIGHTS_RESOURCE_ID', '')}
- Log Analytics workspace: {law_arm_id}
- MLflow tracking URI: {env.get('MLFLOW_TRACKING_URI', '')}

## Request Summary

{_fmt_table(request_summary, ['Name', 'requests', 'failures', 'p95_duration_ms'])}

## Recent Requests

{_fmt_table(recent_requests, ['TimeGenerated', 'Name', 'ResultCode', 'Success', 'DurationMs', 'OperationId'])}

## Recent Traces

{_fmt_table(recent_traces, ['TimeGenerated', 'SeverityLevel', 'Message', 'OperationId'])}

## Recent MLflow Runs

{_fmt_table(mlflow_runs, ['experiment', 'run_name', 'status', 'run_id'])}
"""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()