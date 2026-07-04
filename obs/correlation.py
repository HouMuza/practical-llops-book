from __future__ import annotations

from opentelemetry import trace


def current_trace_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context or not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


def correlation_headers(request_id: str, trace_id: str | None) -> dict[str, str]:
    headers = {"X-LLMOps-Request-Id": request_id}
    if trace_id:
        headers["X-LLMOps-Trace-Id"] = trace_id
    return headers