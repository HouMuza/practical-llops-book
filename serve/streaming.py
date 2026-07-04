from __future__ import annotations
 
import asyncio
import json
from typing import AsyncIterator
 
from fastapi import Request
 
from engine.types import StepOutput
from obs.request_telemetry import RequestTelemetry, finish_request, observe_step
 
 
def sse_event(data: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return prefix + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
 
 
async def stream_tokens(
    request: Request,
    outputs: AsyncIterator[StepOutput],
    telemetry: RequestTelemetry | None = None,
) -> AsyncIterator[str]:
    """Server-sent events stream with cancellation and heartbeats."""
    last_heartbeat = asyncio.get_running_loop().time()
    disconnected = False
    try:
        async for step in outputs:
            if telemetry is not None:
                observe_step(telemetry, request_id=step.request_id, emitted_token=step.token_text is not None)
            if await request.is_disconnected():
                disconnected = True
                break
            if step.error:
                if telemetry is not None:
                    finish_request(
                        telemetry,
                        status="error",
                        finish_reason="error",
                        error_type="generation_error",
                        error_message=step.error,
                    )
                yield sse_event({"error": step.error, "request_id": step.request_id}, event="error")
                return
            if step.token_text is not None:
                yield sse_event({"token": step.token_text, "token_id": step.token_id, "request_id": step.request_id})
            if step.finished:
                if telemetry is not None:
                    finish_request(telemetry, status="ok", finish_reason=step.finish_reason)
                yield sse_event({"finish_reason": step.finish_reason, "request_id": step.request_id}, event="done")
                return
            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= 15:
                yield ": heartbeat\n\n"
                last_heartbeat = now
    except Exception as exc:
        if telemetry is not None:
            finish_request(
                telemetry,
                status="error",
                finish_reason="exception",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise

    if telemetry is not None:
        finish_request(
            telemetry,
            status="cancelled" if disconnected else "ok",
            finish_reason="client_disconnect" if disconnected else "stream_ended",
        )
