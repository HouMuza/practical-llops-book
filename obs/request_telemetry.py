from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

from opentelemetry import metrics

log = logging.getLogger("llmops.request")

_instruments: dict[str, object] | None = None


def _get_instruments() -> dict[str, object]:
    global _instruments
    if _instruments is None:
        meter = metrics.get_meter("llmops.request")
        _instruments = {
            "active_requests": meter.create_up_down_counter(
                "llm.active_requests",
                description="Number of active inference requests",
                unit="1",
            ),
            "requests": meter.create_counter(
                "llm.requests",
                description="Completed inference requests",
                unit="1",
            ),
            "errors": meter.create_counter(
                "llm.request.errors",
                description="Inference request failures",
                unit="1",
            ),
            "duration": meter.create_histogram(
                "llm.request.duration",
                description="End-to-end inference latency",
                unit="ms",
            ),
            "prompt_tokens": meter.create_histogram(
                "llm.request.prompt_tokens",
                description="Prompt token counts",
                unit="token",
            ),
            "completion_tokens": meter.create_histogram(
                "llm.request.completion_tokens",
                description="Completion token counts",
                unit="token",
            ),
        }
    return _instruments


@dataclass(slots=True)
class RequestTelemetry:
    route: str
    transport: str
    source: str
    model_name: str
    stream: bool
    prompt_tokens: int
    tenant_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    completion_tokens: int = 0
    start_time: float = field(default_factory=perf_counter)
    closed: bool = False


def _active_attributes(ctx: RequestTelemetry) -> dict[str, str | bool]:
    attrs: dict[str, str | bool] = {
        "route": ctx.route,
        "transport": ctx.transport,
        "source": ctx.source,
        "model": ctx.model_name,
        "stream": ctx.stream,
    }
    if ctx.tenant_id:
        attrs["tenant.id"] = ctx.tenant_id
    return attrs


def _final_attributes(ctx: RequestTelemetry, *, status: str, finish_reason: str | None) -> dict[str, str | bool]:
    attrs = _active_attributes(ctx)
    attrs["status"] = status
    attrs["finish_reason"] = finish_reason or "unknown"
    return attrs


def begin_request(
    *,
    route: str,
    transport: str,
    source: str,
    model_name: str,
    stream: bool,
    prompt_tokens: int,
    tenant_id: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> RequestTelemetry:
    ctx = RequestTelemetry(
        route=route,
        transport=transport,
        source=source,
        model_name=model_name,
        stream=stream,
        prompt_tokens=prompt_tokens,
        tenant_id=tenant_id,
        request_id=request_id,
        trace_id=trace_id,
    )
    _get_instruments()["active_requests"].add(1, _active_attributes(ctx))
    return ctx


def observe_step(ctx: RequestTelemetry, *, request_id: str | None, emitted_token: bool) -> None:
    if request_id and not ctx.request_id:
        ctx.request_id = request_id
    if emitted_token:
        ctx.completion_tokens += 1


def finish_request(
    ctx: RequestTelemetry,
    *,
    status: str,
    finish_reason: str | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    if ctx.closed:
        return
    ctx.closed = True

    instruments = _get_instruments()
    active_attrs = _active_attributes(ctx)
    final_attrs = _final_attributes(ctx, status=status, finish_reason=finish_reason)
    duration_ms = (perf_counter() - ctx.start_time) * 1000.0

    instruments["active_requests"].add(-1, active_attrs)
    instruments["requests"].add(1, final_attrs)
    instruments["duration"].record(duration_ms, final_attrs)
    instruments["prompt_tokens"].record(ctx.prompt_tokens, final_attrs)
    instruments["completion_tokens"].record(ctx.completion_tokens, final_attrs)
    if status != "ok":
        error_attrs = dict(final_attrs)
        error_attrs["error.type"] = error_type or "unknown"
        instruments["errors"].add(1, error_attrs)

    log_payload: dict[str, object] = {
        "route": ctx.route,
        "transport": ctx.transport,
        "source": ctx.source,
        "model": ctx.model_name,
        "stream": ctx.stream,
        "status": status,
        "finish_reason": finish_reason or "unknown",
        "request_id": ctx.request_id,
        "trace_id": ctx.trace_id,
        "tenant_id": ctx.tenant_id,
        "prompt_tokens": ctx.prompt_tokens,
        "completion_tokens": ctx.completion_tokens,
        "duration_ms": round(duration_ms, 2),
    }
    if error_type:
        log_payload["error_type"] = error_type
    if error_message:
        log_payload["error_message"] = error_message

    if status == "ok":
        log.info("llm_request_complete", extra=log_payload)
    else:
        log.warning("llm_request_complete", extra=log_payload)