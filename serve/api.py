from __future__ import annotations
 
import logging
import os
import uuid
from typing import Any
 
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from pydantic import BaseModel, Field
 
from engine.engine import InferenceEngine
from engine.types import EngineConfig, SamplingParams
from obs.correlation import correlation_headers, current_trace_id
from obs.otel_setup import configure_observability
from obs.request_telemetry import begin_request, finish_request, observe_step
from serve.streaming import stream_tokens
 
log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
configure_observability(service_name=os.getenv("OTEL_SERVICE_NAME", "llmops-deepdive-api"))
 
app = FastAPI(title="LLMOps Deep Dive API", version="1.0.0")
engine: InferenceEngine | None = None
 
 
class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0)
    top_k: int = Field(default=20, ge=0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = 0.0
    stream: bool = True
    lora_name: str | None = None
    tenant_id: str | None = None
 
 
@app.on_event("startup")
def startup() -> None:
    global engine
    cfg = EngineConfig(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B"),
        device=os.getenv("DEVICE", "cuda"),
        dtype=os.getenv("DTYPE", "bf16"),
    )
    engine = InferenceEngine(cfg)
 
 
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": engine is not None}
 
 
@app.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    request_id = str(uuid.uuid4())
    with tracer.start_as_current_span("http.completions") as span:
        span.set_attribute("request.id", request_id)
        trace_id = current_trace_id()
        prompt_tokens = engine.count_tokens(body.prompt)
        telemetry = begin_request(
            route="/v1/completions",
            transport="http",
            source="fastapi",
            model_name=engine.config.model_name,
            stream=body.stream,
            prompt_tokens=prompt_tokens,
            tenant_id=body.tenant_id,
            request_id=request_id,
            trace_id=trace_id,
        )
        sampling = SamplingParams(
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            min_p=body.min_p,
            presence_penalty=body.presence_penalty,
        )
        outputs = engine.generate_stream(
            body.prompt,
            sampling,
            request_id=request_id,
            lora_name=body.lora_name,
            tenant_id=body.tenant_id,
        )
        if body.stream:
            return StreamingResponse(
                stream_tokens(request, outputs, telemetry=telemetry),
                media_type="text/event-stream",
                headers=correlation_headers(request_id, trace_id),
            )

        text_parts: list[str] = []
        finish_reason = "stream_ended"
        try:
            async for step in outputs:
                observe_step(telemetry, request_id=step.request_id, emitted_token=step.token_text is not None)
                if step.error:
                    finish_request(
                        telemetry,
                        status="error",
                        finish_reason="error",
                        error_type="generation_error",
                        error_message=step.error,
                    )
                    raise HTTPException(status_code=500, detail=step.error)
                if step.token_text:
                    text_parts.append(step.token_text)
                if step.finished:
                    finish_reason = step.finish_reason or finish_reason
                    break
        except HTTPException:
            raise
        except Exception as exc:
            finish_request(
                telemetry,
                status="error",
                finish_reason="exception",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        finish_request(telemetry, status="ok", finish_reason=finish_reason)
        return JSONResponse(
            {"text": "".join(text_parts)},
            headers=correlation_headers(request_id, trace_id),
        )
