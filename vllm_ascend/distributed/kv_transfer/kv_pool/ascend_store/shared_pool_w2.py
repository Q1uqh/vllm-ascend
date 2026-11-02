"""Opt-in ordinary-path shared-pool safety and compact observation helpers.

No NPU dependencies: ownership and observation can be tested on a CPU host.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable

SUMMARY_INTERVAL_SECONDS = 60.0
SUMMARY_MARKER = "SHARED_POOL_W2 "


def validate_shared_pool_w2(extra: dict, role: str, groups: int | None = None) -> bool:
    enabled = extra.get("enable_shared_pool_w2", False)
    if not isinstance(enabled, bool):
        raise ValueError("enable_shared_pool_w2 must be a JSON boolean")
    if not enabled:
        return False
    if extra.get("backend", "mooncake").lower() != "mooncake" or extra.get("use_layerwise", False):
        raise ValueError("Shared-pool W2 requires backend=mooncake and use_layerwise=false")
    if groups is not None and groups != 1:
        raise ValueError("Shared-pool W2 requires exactly one KV cache group")
    if role not in ("kv_producer", "kv_consumer"):
        raise ValueError("Shared-pool W2 requires a PD producer or consumer")
    if role == "kv_consumer" and (
        not extra.get("consumer_is_to_load", False)
        or extra.get("consumer_is_to_put", False)
        or extra.get("save_decode_cache", False)
    ):
        raise ValueError("Shared-pool W2 consumer must load KV and retain decode KV locally without saving it")
    return True


class SourceBlockTracker:
    """One independent pin per save generation/block; all ranks acknowledge.

    The caller owns BlockPool mutations. Late/duplicate acknowledgements never
    remove another generation's reference, including after block-ID reuse.
    """

    def __init__(self, worker_count: int):
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        self.workers = frozenset(range(worker_count))
        self.next_id = 0
        self.active_blocks: Counter[int] = Counter()
        self.pending: dict[tuple[int, int], set[int]] = {}

    def register(self, blocks: Iterable[int]) -> tuple[int, tuple[int, ...]]:
        save_id = self.next_id
        self.next_id += 1
        block_ids = tuple(sorted(set(blocks)))
        for block_id in block_ids:
            self.pending[save_id, block_id] = set()
            self.active_blocks[block_id] += 1
        return save_id, block_ids

    def acknowledge(self, completions: Iterable[tuple[int, int, int]]) -> list[int]:
        released = []
        for save_id, block_id, worker in completions:
            seen = self.pending.get((save_id, block_id))
            if seen is None or worker not in self.workers:
                continue
            seen.add(worker)
            if seen == self.workers:
                del self.pending[save_id, block_id]
                self.active_blocks[block_id] -= 1
                if not self.active_blocks[block_id]:
                    del self.active_blocks[block_id]
                released.append(block_id)
        return released


class SharedPoolObservation:
    """Cumulative counters, one bounded snapshot per minute and at shutdown.

    Source safety counters are not proof of physical allocator reuse. Consumers
    of these snapshots must not infer PASS for unobserved core/executor events.
    """

    def __init__(self, side: str, role: str, rank: int, workers: int):
        self.identity = dict(instance=uuid.uuid4().hex, side=side, role=role, rank=rank, workers=workers)
        self.counts: Counter = Counter()
        self.lock = threading.Lock()
        self.last_emit = 0.0
        self.sequence = 0
        self.loads: dict[str, set[str]] = {}

    def add(self, **values: int | float) -> None:
        with self.lock:
            self.counts.update(values)

    def record_load(self, req_id: str, keys: list[str], sizes: list[list[int]], success: bool) -> None:
        with self.lock:
            loaded = self.loads.setdefault(req_id, set())
            self.counts["load_calls"] += 1
            self.counts["load_failed"] += int(not success)
            self.counts["load_bytes"] += sum(sum(row) for row in sizes)
            self.counts["repeated_load_keys"] += sum(key in loaded for key in keys)
            if success:
                loaded.update(keys)

    def retire(self, req_ids: Iterable[str], recovery: bool = False) -> None:
        with self.lock:
            for req_id in req_ids:
                self.loads.pop(req_id, None)
            if recovery:
                self.counts["recovery_events"] += 1

    def emit(self, logger: logging.Logger, final: bool = False, **gauges: int) -> None:
        now = time.monotonic()
        with self.lock:
            if not final and now - self.last_emit < SUMMARY_INTERVAL_SECONDS:
                return
            self.last_emit = now
            self.sequence += 1
            snapshot = dict(self.identity, sequence=self.sequence, final=final, counters=dict(self.counts), **gauges)
        logger.info("%s%s", SUMMARY_MARKER, json.dumps(snapshot, sort_keys=True))


class BlockPoolObservation:
    """Observe only one opted-in engine's allocator, without global patches.

    Core request references and deferred frees remain the computation-lifetime
    authority. We check real free-list transitions against outstanding saves;
    reference release alone is never counted as physical reuse.
    """

    def __init__(self, pool, sources: SourceBlockTracker, observation: SharedPoolObservation):
        self.pool, self.sources, self.observation = pool, sources, observation
        self.tracked: set[int] = set()
        self.returned: set[int] = set()
        self.original_free = pool.free_blocks
        self.original_get = pool.get_new_blocks

        def free_blocks(blocks):
            blocks = list(blocks)
            self.original_free(blocks)
            for block in {block.block_id: block for block in blocks}.values():
                if block.block_id not in self.tracked:
                    continue
                observation.add(allocator_free_checked=1)
                if block.ref_cnt == 0 and not block.is_null:
                    self._check(block.block_id)
                    self.returned.add(block.block_id)
                    observation.add(allocator_returned=1)

        def get_new_blocks(num_blocks):
            blocks = self.original_get(num_blocks)
            for block in blocks:
                if block.block_id in self.tracked:
                    self._check(block.block_id)
                if block.block_id in self.returned:
                    self.returned.remove(block.block_id)
                    observation.add(allocator_reused=1)
            return blocks

        self.free_wrapper, self.get_wrapper = free_blocks, get_new_blocks
        pool.free_blocks, pool.get_new_blocks = free_blocks, get_new_blocks

    def _check(self, block_id: int) -> None:
        if self.sources.active_blocks.get(block_id, 0):
            self.observation.add(early_reuse=1)
            raise RuntimeError("Shared-pool W2 allocator reused a source with an outstanding save")

    def detach(self) -> None:
        if self.pool.free_blocks is self.free_wrapper:
            self.pool.free_blocks = self.original_free
        if self.pool.get_new_blocks is self.get_wrapper:
            self.pool.get_new_blocks = self.original_get
