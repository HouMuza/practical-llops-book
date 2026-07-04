from __future__ import annotations
 
from collections import Counter
from typing import Iterable
 
import torch
from opentelemetry import trace
 
from engine.types import SamplingParams
 
tracer = trace.get_tracer(__name__)
 
 
def apply_presence_penalty(logits: torch.Tensor, generated_ids: Iterable[int], penalty: float) -> torch.Tensor:
    if penalty == 0:
        return logits
    out = logits.clone()
    counts = Counter(generated_ids)
    for token_id in counts:
        out[token_id] -= penalty
    return out
 
 
def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, k=top_k)
    threshold = values[-1]
    return torch.where(logits < threshold, torch.full_like(logits, -torch.inf), logits)
 
 
def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0 or top_p >= 1:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = cum > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    out = torch.full_like(logits, -torch.inf)
    out.scatter_(0, sorted_idx, sorted_logits)
    return out
 
 
def filter_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    max_prob = torch.max(probs)
    keep = probs >= (min_p * max_prob)
    return torch.where(keep, logits, torch.full_like(logits, -torch.inf))
 
 
def sample_next_token(
    logits: torch.Tensor,
    sampling: SamplingParams,
    generated_ids: list[int],
    generator: torch.Generator | None = None,
) -> int:
    """Fused sampling pipeline: temperature -> top-k -> top-p -> min-p -> sample."""
    with tracer.start_as_current_span("sample_next_token") as span:
        span.set_attribute("gen_ai.request.max_new_tokens", sampling.max_new_tokens)
        span.set_attribute("gen_ai.request.temperature", sampling.temperature)
        span.set_attribute("gen_ai.request.top_k", sampling.top_k)
        span.set_attribute("gen_ai.request.top_p", sampling.top_p)
 
        x = logits.float().squeeze(0)
        x = apply_presence_penalty(x, generated_ids, sampling.presence_penalty)
 
        if sampling.temperature == 0:
            token_id = int(torch.argmax(x).item())
            span.set_attribute("sampling.greedy", True)
            return token_id
 
        x = x / max(sampling.temperature, 1e-6)
        x = filter_top_k(x, sampling.top_k)
        x = filter_top_p(x, sampling.top_p)
        x = filter_min_p(x, sampling.min_p)
        probs = torch.softmax(x, dim=-1)
        if torch.isnan(probs).any() or probs.sum() == 0:
            token_id = int(torch.argmax(logits.float()).item())
            span.set_attribute("sampling.fallback_argmax", True)
            return token_id
        token_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        span.set_attribute("gen_ai.response.token_id", token_id)
        return token_id
