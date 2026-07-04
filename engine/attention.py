from __future__ import annotations
 
import torch
from opentelemetry import trace
 
tracer = trace.get_tracer(__name__)
 
 
class PagedAttentionWrapper:
    """Wrapper boundary for the attention kernel.
 
    In a production engine this file is where a Triton, FlashAttention, or vLLM kernel is
    called with block tables. The rest of the engine should not know whether attention is
    implemented by PyTorch, FlashAttention, or a custom CUDA extension.
    """
 
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
 
    def forward(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        with tracer.start_as_current_span("attention.paged_forward") as span:
            span.set_attribute("attention.batch", int(query.shape[0]))
            span.set_attribute("attention.block_size", self.block_size)
            # Teaching fallback: delegate to a dense attention implementation would go here.
            # Real code should call the runtime kernel. Keeping this boundary explicit is
            # what lets the rest of the repository remain production-shaped.
            raise NotImplementedError(
                "PagedAttentionWrapper is a kernel boundary. Use Hugging Face model.forward "
                "in engine.py for the runnable path, or replace this wrapper with a Triton kernel."
            )
