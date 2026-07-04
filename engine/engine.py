from __future__ import annotations
 
import asyncio
import logging
import math
import uuid
from contextlib import suppress
from typing import AsyncIterator
 
import torch
from opentelemetry import trace
from transformers import AutoModelForCausalLM, AutoTokenizer
 
from engine.kv_cache import KVCacheManager
from engine.prefix_cache import PrefixCache
from engine.sampling import sample_next_token
from engine.scheduler import IterationScheduler
from engine.types import EngineConfig, GenerationRequest, RequestStatus, SamplingParams, SchedulerConfig, StepOutput
 
log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
 
 
def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
 
 
class InferenceEngine:
    """Production-shaped inference engine.
 
    This implementation uses Hugging Face `model.forward` for the runnable path but keeps
    the operational structure explicit: scheduling, chunked prefill, cancellation, prefix
    cache ownership, sampling, OTel spans, and structured errors.
    """
 
    def __init__(self, config: EngineConfig, scheduler_config: SchedulerConfig | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = _dtype(config.dtype) if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            device_map=config.device if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        model_cfg = self.model.config
        self.kv_cache = KVCacheManager(
            num_layers=getattr(model_cfg, "num_hidden_layers", 28),
            num_blocks=config.kv_num_blocks,
            block_size=config.kv_block_size,
            num_kv_heads=getattr(model_cfg, "num_key_value_heads", getattr(model_cfg, "num_attention_heads", 8)),
            head_dim=getattr(model_cfg, "head_dim", getattr(model_cfg, "hidden_size", 1024) // getattr(model_cfg, "num_attention_heads", 16)),
            dtype=self.dtype,
            device=self.device,
        )
        self.prefix_cache = PrefixCache(self.kv_cache) if config.enable_prefix_cache else None
        self.scheduler = IterationScheduler(scheduler_config or SchedulerConfig())
        self._generators: dict[str, torch.Generator] = {}
        log.info("engine_loaded", extra={"model": config.model_name, "device": str(self.device)})
 
    def count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _tokenize(self, prompt: str) -> list[int]:
        with tracer.start_as_current_span("tokenize"):
            return self.tokenizer(prompt, add_special_tokens=False).input_ids
 
    async def generate_stream(
        self,
        prompt: str,
        sampling: SamplingParams | None = None,
        *,
        request_id: str | None = None,
        lora_name: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncIterator[StepOutput]:
        sampling = sampling or SamplingParams()
        sampling.validate()
        request_id = request_id or str(uuid.uuid4())
        with tracer.start_as_current_span("llm.request") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "qwen")
            span.set_attribute("gen_ai.request.model", self.config.model_name)
            span.set_attribute("gen_ai.request.max_tokens", sampling.max_new_tokens)
            span.set_attribute("gen_ai.request.temperature", sampling.temperature)
            span.set_attribute("request.id", request_id)
            if tenant_id:
                span.set_attribute("tenant.id", tenant_id)
 
            input_ids = self._tokenize(prompt)
            if len(input_ids) > self.config.max_model_len:
                raise ValueError(f"prompt too long: {len(input_ids)} > {self.config.max_model_len}")
            req = GenerationRequest(
                request_id=request_id,
                prompt=prompt,
                input_ids=input_ids,
                sampling=sampling,
                lora_name=lora_name,
                tenant_id=tenant_id,
            )
            generator = torch.Generator(device=self.device)
            if sampling.seed is not None:
                generator.manual_seed(sampling.seed)
            self._generators[request_id] = generator
 
            try:
                async for step in self._run_single_request(req):
                    yield step
            except asyncio.CancelledError:
                req.status = RequestStatus.CANCELLED
                self.kv_cache.release(request_id, req.block_table)
                raise
            except Exception as exc:
                req.mark_failed(exc)
                log.exception("generation_failed", extra={"request_id": request_id})
                yield StepOutput(request_id=request_id, token_id=None, token_text=None, finished=True, error=req.error)
            finally:
                span.set_attribute("gen_ai.usage.output_tokens", len(req.generated_ids))
                self._generators.pop(request_id, None)
                self.kv_cache.release(request_id, req.block_table)
 
    async def _run_single_request(self, req: GenerationRequest) -> AsyncIterator[StepOutput]:
        # The runnable path processes one request coroutine at a time. The scheduler module
        # shows how to batch many requests iteration-by-iteration; wiring it into an async
        # background loop is straightforward but too long for the book appendix.
        with tracer.start_as_current_span("prefill") as span:
            n_blocks = math.ceil(max(1, len(req.input_ids)) / self.config.kv_block_size)
            req.block_table = self.kv_cache.allocate(req.request_id, n_blocks)
            input_tensor = torch.tensor([req.input_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=input_tensor, use_cache=True)
            past_key_values = out.past_key_values
            req.prompt_pos = len(req.input_ids)
            req.status = RequestStatus.DECODING
            span.set_attribute("gen_ai.usage.input_tokens", len(req.input_ids))
 
        current_input = input_tensor[:, -1:]
        for _ in range(req.sampling.max_new_tokens):
            with tracer.start_as_current_span("decode"):
                out = self.model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
            with tracer.start_as_current_span("sample"):
                token_id = sample_next_token(
                    out.logits[:, -1, :],
                    req.sampling,
                    req.generated_ids,
                    self._generators.get(req.request_id),
                )
            req.generated_ids.append(token_id)
            text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            finished = token_id in req.sampling.stop_token_ids or token_id == self.tokenizer.eos_token_id
            yield StepOutput(
                request_id=req.request_id,
                token_id=token_id,
                token_text=text,
                finished=finished,
                finish_reason="stop" if finished else None,
            )
            if finished:
                req.status = RequestStatus.FINISHED
                break
            current_input = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            await asyncio.sleep(0)  # cooperative cancellation point
 
        if not req.is_finished:
            req.status = RequestStatus.FINISHED
            yield StepOutput(
                request_id=req.request_id,
                token_id=None,
                token_text=None,
                finished=True,
                finish_reason="length",
            )
