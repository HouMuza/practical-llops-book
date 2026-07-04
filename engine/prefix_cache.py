from __future__ import annotations
 
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Iterable
 
from opentelemetry import trace
 
from engine.kv_cache import KVCacheManager
 
tracer = trace.get_tracer(__name__)
 
 
@dataclass(slots=True)
class PrefixCacheEntry:
    key: str
    block_ids: tuple[int, ...]
    token_count: int
    lora_name: str | None
 
 
class PrefixCache:
    """Content-addressed prefix cache with LRU eviction.
 
    Each prefix is keyed by token ids plus adapter name. Adapter id is part of the hash
    because the same tokens under different LoRA adapters do not necessarily produce the
    same hidden states.
    """
 
    def __init__(self, kv_cache: KVCacheManager, max_entries: int = 2048) -> None:
        self.kv_cache = kv_cache
        self.max_entries = max_entries
        self._lock = RLock()
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
 
    @staticmethod
    def hash_tokens(token_ids: Iterable[int], lora_name: str | None) -> str:
        h = hashlib.blake2b(digest_size=24)
        h.update((lora_name or "base").encode("utf-8"))
        h.update(b"\0")
        for token_id in token_ids:
            h.update(int(token_id).to_bytes(4, "little", signed=False))
        return h.hexdigest()
 
    def lookup(self, request_id: str, token_ids: list[int], lora_name: str | None) -> PrefixCacheEntry | None:
        key = self.hash_tokens(token_ids, lora_name)
        with tracer.start_as_current_span("prefix_cache.lookup") as span:
            span.set_attribute("prefix.hash", key)
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    span.set_attribute("prefix.hit", False)
                    return None
                self._entries.move_to_end(key)
                self.kv_cache.retain(request_id, entry.block_ids)
                span.set_attribute("prefix.hit", True)
                span.set_attribute("prefix.blocks", len(entry.block_ids))
                return entry
 
    def insert(
        self,
        *,
        token_ids: list[int],
        block_ids: list[int],
        lora_name: str | None,
    ) -> str:
        key = self.hash_tokens(token_ids, lora_name)
        with tracer.start_as_current_span("prefix_cache.insert") as span:
            span.set_attribute("prefix.hash", key)
            with self._lock:
                self._entries[key] = PrefixCacheEntry(
                    key=key,
                    block_ids=tuple(block_ids),
                    token_count=len(token_ids),
                    lora_name=lora_name,
                )
                self._entries.move_to_end(key)
                self._evict_if_needed()
            return key
 
    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            _, entry = self._entries.popitem(last=False)
            # The cache itself holds one reference. Release under a synthetic owner.
            self.kv_cache.release("prefix-cache", entry.block_ids)
 
    def clear(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                self.kv_cache.release("prefix-cache", entry.block_ids)
            self._entries.clear()
