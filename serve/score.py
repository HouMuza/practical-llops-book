from __future__ import annotations
 
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from opentelemetry import trace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
 
from engine.engine import InferenceEngine
from engine.types import EngineConfig, SamplingParams
from obs.correlation import current_trace_id
from obs.otel_setup import configure_observability
from obs.request_telemetry import begin_request, finish_request, observe_step
 
log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
engine: InferenceEngine | None = None
 
 
def init() -> None:
    """Azure ML entry point. Called once per worker process."""
    global engine
    configure_observability(service_name=os.getenv("OTEL_SERVICE_NAME", "azureml-qwen3"))
    cfg = EngineConfig(
        model_name=os.getenv("MODEL_NAME", os.getenv("AZUREML_MODEL_DIR", "Qwen/Qwen3-0.6B")),
        device=os.getenv("DEVICE", "cuda"),
        dtype=os.getenv("DTYPE", "bf16"),
    )
    engine = InferenceEngine(cfg)
    log.info("azureml_score_init_complete", extra={"model": cfg.model_name})
 
 
async def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    if engine is None:
        raise RuntimeError("engine not initialized")
    prompt = str(payload["prompt"])
    request_id = str(payload.get("request_id") or uuid.uuid4())
    with tracer.start_as_current_span("azureml.score") as span:
        span.set_attribute("request.id", request_id)
        trace_id = current_trace_id()
        telemetry = begin_request(
            route="/score",
            transport="azureml",
            source="azureml",
            model_name=engine.config.model_name,
            stream=False,
            prompt_tokens=engine.count_tokens(prompt),
            tenant_id=payload.get("tenant_id"),
            request_id=request_id,
            trace_id=trace_id,
        )
        sampling = SamplingParams(
            max_new_tokens=int(payload.get("max_new_tokens", 256)),
            temperature=float(payload.get("temperature", 0.7)),
            top_k=int(payload.get("top_k", 20)),
            top_p=float(payload.get("top_p", 0.8)),
            min_p=float(payload.get("min_p", 0.0)),
            presence_penalty=float(payload.get("presence_penalty", 0.0)),
        )
        parts: list[str] = []
        finish_reason = "unknown"
        try:
            async for step in engine.generate_stream(
                prompt,
                sampling,
                request_id=request_id,
                lora_name=payload.get("lora_name"),
                tenant_id=payload.get("tenant_id"),
            ):
                observe_step(telemetry, request_id=step.request_id, emitted_token=step.token_text is not None)
                if step.error:
                    finish_request(
                        telemetry,
                        status="error",
                        finish_reason="error",
                        error_type="generation_error",
                        error_message=step.error,
                    )
                    raise RuntimeError(step.error)
                if step.token_text:
                    parts.append(step.token_text)
                if step.finished:
                    finish_reason = step.finish_reason or finish_reason
                    finish_request(telemetry, status="ok", finish_reason=finish_reason)
                    return {
                        "text": "".join(parts),
                        "finish_reason": finish_reason,
                        "request_id": request_id,
                        "trace_id": trace_id,
                    }
        except Exception as exc:
            finish_request(
                telemetry,
                status="error",
                finish_reason="exception",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        return {
            "text": "".join(parts),
            "finish_reason": finish_reason,
            "request_id": request_id,
            "trace_id": trace_id,
        }
 
 
def run(raw_data: str | bytes) -> str:
    """Azure ML may call run concurrently; avoid mutable request-level globals."""
    import asyncio
 
    try:
        payload = json.loads(raw_data.decode("utf-8") if isinstance(raw_data, bytes) else raw_data)
        result = asyncio.run(_generate(payload))
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        log.exception("azureml_score_failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
