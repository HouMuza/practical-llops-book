from __future__ import annotations
 
from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable
 
import torch
from opentelemetry import trace
 
tracer = trace.get_tracer(__name__)
 
 
@dataclass(slots=True)
class KVBlock:
    block_id: int
    refcount: int = 0
    owner_request_ids: set[str] = field(default_factory=set)
    swapped: bool = False
 
 
class KVCacheError(RuntimeError):
    pass
 
 
class KVCacheManager:
    """Paged KV cache manager with refcounts and a CPU swap placeholder.
 
    The physical K/V tensors are represented as two dense tensors:
        key_cache:   [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
 
    Blocks can be shared between requests through prefix caching. Refcounting makes
    that safe: a block is returned to the free list only when the last request releases it.
    """
 
    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self._lock = RLock()
 
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.key_cache = torch.empty(shape, dtype=dtype, device=self.device)
        self.value_cache = torch.empty(shape, dtype=dtype, device=self.device)
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self.cpu_swap: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
 
    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self.free_blocks)
 
    @property
    def used_count(self) -> int:
        return self.num_blocks - self.free_count
 
    def allocate(self, request_id: str, n_blocks: int) -> list[int]:
        with tracer.start_as_current_span("kv_cache.allocate") as span:
            span.set_attribute("kv.blocks_requested", n_blocks)
            with self._lock:
                if n_blocks > len(self.free_blocks):
                    raise KVCacheError(
                        f"KV cache exhausted: requested={n_blocks}, free={len(self.free_blocks)}"
                    )
                out: list[int] = []
                for _ in range(n_blocks):
                    block_id = self.free_blocks.pop()
                    block = self.blocks[block_id]
                    block.refcount = 1
                    block.owner_request_ids = {request_id}
                    block.swapped = False
                    out.append(block_id)
                span.set_attribute("kv.blocks_allocated", len(out))
                span.set_attribute("kv.free_after", len(self.free_blocks))
                return out
 
    def retain(self, request_id: str, block_ids: Iterable[int]) -> None:
        with tracer.start_as_current_span("kv_cache.retain") as span:
            count = 0
            with self._lock:
                for block_id in block_ids:
                    block = self.blocks[block_id]
                    if block.refcount <= 0:
                        raise KVCacheError(f"Cannot retain free block {block_id}")
                    block.refcount += 1
                    block.owner_request_ids.add(request_id)
                    count += 1
            span.set_attribute("kv.blocks_retained", count)
 
    def release(self, request_id: str, block_ids: Iterable[int]) -> None:
        with tracer.start_as_current_span("kv_cache.release") as span:
            released = 0
            with self._lock:
                for block_id in block_ids:
                    block = self.blocks[block_id]
                    if block.refcount <= 0:
                        continue
                    block.owner_request_ids.discard(request_id)
                    block.refcount -= 1
                    if block.refcount == 0:
                        block.owner_request_ids.clear()
                        block.swapped = False
                        self.cpu_swap.pop(block_id, None)
                        self.free_blocks.append(block_id)
                        released += 1
            span.set_attribute("kv.blocks_freed", released)
            span.set_attribute("kv.free_after", self.free_count)
 
    def ensure_capacity_or_preempt(self, required_blocks: int, victim_tables: list[tuple[str, list[int]]]) -> None:
        """Free enough blocks by preempting victims.
 
        In a real serving engine this method would coordinate with the scheduler and mark
        victims as swapped/preempted. Here we keep the mechanism explicit: release victim
        block tables until the required free capacity exists.
        """
        with tracer.start_as_current_span("kv_cache.ensure_capacity") as span:
            span.set_attribute("kv.required_blocks", required_blocks)
            for request_id, block_table in victim_tables:
                if self.free_count >= required_blocks:
                    return
                self.swap_out(request_id, block_table)
            if self.free_count < required_blocks:
                raise KVCacheError(
                    f"Unable to free KV blocks: required={required_blocks}, free={self.free_count}"
                )
 
    def swap_out(self, request_id: str, block_ids: Iterable[int]) -> None:
        """Move blocks to CPU and release GPU ownership.
 
        This is deliberately conservative: a block with refcount > 1 is not swapped because
        another request still needs it on GPU. Production systems use async DMA and pinned
        CPU buffers; this version is synchronous and easy to reason about.
        """
        with tracer.start_as_current_span("kv_cache.swap_out") as span:
            swapped = 0
            with self._lock:
                for block_id in list(block_ids):
                    block = self.blocks[block_id]
                    if block.refcount != 1 or request_id not in block.owner_request_ids:
                        continue
                    k = self.key_cache[:, block_id].detach().cpu().clone()
                    v = self.value_cache[:, block_id].detach().cpu().clone()
                    self.cpu_swap[block_id] = (k, v)
                    block.swapped = True
                    block.refcount = 0
                    block.owner_request_ids.clear()
                    self.free_blocks.append(block_id)
                    swapped += 1
            span.set_attribute("kv.blocks_swapped", swapped)
 
    def restore(self, request_id: str, block_ids: Iterable[int]) -> None:
        with tracer.start_as_current_span("kv_cache.restore") as span:
            restored = 0
            with self._lock:
                for block_id in block_ids:
                    pair = self.cpu_swap.pop(block_id, None)
                    if pair is None:
                        continue
                    k, v = pair
                    self.key_cache[:, block_id].copy_(k.to(self.device, non_blocking=True))
                    self.value_cache[:, block_id].copy_(v.to(self.device, non_blocking=True))
                    block = self.blocks[block_id]
                    block.refcount = 1
                    block.owner_request_ids = {request_id}
                    block.swapped = False
                    restored += 1
            span.set_attribute("kv.blocks_restored", restored)
