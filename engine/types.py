from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, AsyncIterator, Iterable, Literal
 
import torch
 
 
class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
 
 
@dataclass(slots=True)
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    presence_penalty: float = 0.0
    stop_token_ids: set[int] = field(default_factory=set)
    seed: int | None = None
 
    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in [0, 1]")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
 
 
@dataclass(slots=True)
class GenerationRequest:
    request_id: str
    prompt: str
    input_ids: list[int]
    sampling: SamplingParams
    priority: int = 0
    lora_name: str | None = None
    tenant_id: str | None = None
    created_at: float = field(default_factory=monotonic)
    status: RequestStatus = RequestStatus.WAITING
    prompt_pos: int = 0
    generated_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    prefix_block_count: int = 0
    error: str | None = None
 
    @property
    def total_tokens(self) -> int:
        return len(self.input_ids) + len(self.generated_ids)
 
    @property
    def remaining_prefill(self) -> int:
        return max(0, len(self.input_ids) - self.prompt_pos)
 
    @property
    def is_prefill_done(self) -> bool:
        return self.prompt_pos >= len(self.input_ids)
 
    @property
    def is_finished(self) -> bool:
        return self.status in {RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED}
 
    def mark_failed(self, exc: Exception) -> None:
        self.status = RequestStatus.FAILED
        self.error = f"{type(exc).__name__}: {exc}"
 
 
@dataclass(slots=True)
class SchedulerConfig:
    max_num_seqs: int = 64
    max_tokens_per_iteration: int = 2048
    prefill_chunk_size: int = 512
    aging_after_seconds: float = 30.0
    decode_first: bool = True
 
 
@dataclass(slots=True)
class EngineConfig:
    model_name: str
    device: str = "cuda"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    max_model_len: int = 32768
    kv_block_size: int = 16
    kv_num_blocks: int = 4096
    enable_prefix_cache: bool = True
    enable_speculative: bool = False
    draft_model_name: str | None = None
    otel_service_name: str = "llmops-deepdive"
 
 
@dataclass(slots=True)
class StepOutput:
    request_id: str
    token_id: int | None
    token_text: str | None
    finished: bool
    finish_reason: str | None = None
    error: str | None = None
