from __future__ import annotations
 
import heapq
from dataclasses import dataclass, field
from time import monotonic
from typing import Iterable
 
from opentelemetry import trace
 
from engine.types import GenerationRequest, RequestStatus, SchedulerConfig
 
tracer = trace.get_tracer(__name__)
 
 
@dataclass(order=True)
class _QueueItem:
    sort_key: tuple[float, int, float]
    request: GenerationRequest = field(compare=False)
 
 
@dataclass(slots=True)
class ScheduledWork:
    decodes: list[GenerationRequest]
    prefills: list[tuple[GenerationRequest, int]]  # request, token_count_to_prefill
 
 
class IterationScheduler:
    """Continuous batching scheduler with priority overlay and aging.
 
    Decodes go first to keep inter-token latency smooth. Remaining token budget is used
    for chunked prefill. Requests that wait longer than `aging_after_seconds` are promoted
    by lowering their effective priority value.
    """
 
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._waiting: list[_QueueItem] = []
        self._active: dict[str, GenerationRequest] = {}
        self._finished: dict[str, GenerationRequest] = {}
 
    def add(self, request: GenerationRequest) -> None:
        with tracer.start_as_current_span("scheduler.add") as span:
            request.status = RequestStatus.WAITING
            heapq.heappush(self._waiting, _QueueItem(self._sort_key(request), request))
            span.set_attribute("request.id", request.request_id)
            span.set_attribute("scheduler.waiting", len(self._waiting))
 
    def cancel(self, request_id: str) -> None:
        req = self._active.get(request_id)
        if req:
            req.status = RequestStatus.CANCELLED
            self._finished[request_id] = req
            self._active.pop(request_id, None)
 
    def finish(self, request: GenerationRequest) -> None:
        request.status = RequestStatus.FINISHED
        self._active.pop(request.request_id, None)
        self._finished[request.request_id] = request
 
    def fail(self, request: GenerationRequest, exc: Exception) -> None:
        request.mark_failed(exc)
        self._active.pop(request.request_id, None)
        self._finished[request.request_id] = request
 
    def next_iteration(self) -> ScheduledWork:
        with tracer.start_as_current_span("scheduler.next_iteration") as span:
            self._admit_waiting()
            budget = self.config.max_tokens_per_iteration
            decodes: list[GenerationRequest] = []
            prefills: list[tuple[GenerationRequest, int]] = []
 
            if self.config.decode_first:
                for req in self._active.values():
                    if req.status == RequestStatus.DECODING and not req.is_finished and budget > 0:
                        decodes.append(req)
                        budget -= 1
 
            # Prefill chunks fill the rest of the iteration. Long prompts therefore make
            # progress without blocking all decodes behind one huge prefill.
            for req in list(self._active.values()):
                if budget <= 0:
                    break
                if req.status in {RequestStatus.WAITING, RequestStatus.PREFILLING} and req.remaining_prefill > 0:
                    req.status = RequestStatus.PREFILLING
                    n = min(req.remaining_prefill, self.config.prefill_chunk_size, budget)
                    prefills.append((req, n))
                    budget -= n
 
            span.set_attribute("scheduler.active", len(self._active))
            span.set_attribute("scheduler.decode_count", len(decodes))
            span.set_attribute("scheduler.prefill_count", len(prefills))
            span.set_attribute("scheduler.remaining_budget", budget)
            return ScheduledWork(decodes=decodes, prefills=prefills)
 
    def _admit_waiting(self) -> None:
        while self._waiting and len(self._active) < self.config.max_num_seqs:
            item = heapq.heappop(self._waiting)
            req = item.request
            if req.is_finished:
                continue
            self._active[req.request_id] = req
 
    def _sort_key(self, request: GenerationRequest) -> tuple[float, int, float]:
        waited = monotonic() - request.created_at
        aged_boost = 1 if waited >= self.config.aging_after_seconds else 0
        effective_priority = request.priority - aged_boost
        # Lower tuple sorts first. created_at preserves FCFS inside the same priority.
        return (effective_priority, len(request.input_ids), request.created_at)
 
    @property
    def active_requests(self) -> list[GenerationRequest]:
        return list(self._active.values())
 
    @property
    def queue_depth(self) -> int:
        return len(self._waiting)
