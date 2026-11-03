"""Opt-in ordinary-path shared-pool safety and compact observation helpers.

No NPU dependencies: ownership and observation can be tested on a CPU host.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import socket
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


MAX_NOTIFICATION_DRAIN = 4096
MAX_NOTIFICATION_BYTES = 65536
MAX_TRACE_EVENTS = 800
MAX_COMPUTE_EVENTS = 128
COMPUTE_POLL_SECONDS = 0.001

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

    def __init__(self, worker_count: int, initial_id: int = 0):
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        self.workers = frozenset(range(worker_count))
        self.next_id = initial_id
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
        self.events: list[dict] = []

    def trace(self, event: str, **values) -> dict:
        record = dict(event=event, t=time.monotonic_ns(), rank=self.identity["rank"], **values)
        with self.lock:
            if len(self.events) < MAX_TRACE_EVENTS:
                self.events.append(record)
            else:
                self.counts["trace_dropped"] += 1
        return record

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
            snapshot = dict(self.identity, sequence=self.sequence, final=final, counters=dict(self.counts),
                            events=list(self.events), **gauges)
        logger.info("%s%s", SUMMARY_MARKER, json.dumps(snapshot, sort_keys=True))


class BlockPoolObservation:
    """Observe only one opted-in engine's allocator, without global patches.

    Core request references and deferred frees remain the computation-lifetime
    authority. We check real free-list transitions against outstanding saves;
    reference release alone is never counted as physical reuse.
    """

    def __init__(self, pool, sources: SourceBlockTracker, observation: SharedPoolObservation, poll=None):
        self.pool, self.sources, self.observation = pool, sources, observation
        self.tracked: set[int] = set()
        self.returned: set[int] = set()
        self.original_free = pool.free_blocks
        self.original_get = pool.get_new_blocks
        self.original_count = getattr(pool, "get_num_free_blocks", None)
        self.poll = poll
        self.provenance: dict[int, int] = {}

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
                    observation.trace("reusable", block=block.block_id, save=self.provenance.get(block.block_id))

        def get_new_blocks(num_blocks):
            if self.poll is not None:
                self.poll()
            blocks = self.original_get(num_blocks)
            for block in blocks:
                if block.block_id in self.tracked:
                    self._check(block.block_id)
                if block.block_id in self.returned:
                    self.returned.remove(block.block_id)
                    observation.add(allocator_reused=1)
                    observation.trace("reuse", block=block.block_id, save=self.provenance.get(block.block_id))
            return blocks

        def get_num_free_blocks():
            if self.poll is not None:
                self.poll()
            return self.original_count()

        self.count_wrapper = get_num_free_blocks
        if self.original_count is not None and poll is not None:
            pool.get_num_free_blocks = get_num_free_blocks
        self.free_wrapper, self.get_wrapper = free_blocks, get_new_blocks
        pool.free_blocks, pool.get_new_blocks = free_blocks, get_new_blocks

    def _check(self, block_id: int) -> None:
        if self.sources.active_blocks.get(block_id, 0):
            self.observation.add(early_reuse=1)
            raise RuntimeError("Shared-pool W2 allocator reused a source with an outstanding save")

    def detach(self) -> None:
        if self.original_count is not None and self.pool.get_num_free_blocks is self.count_wrapper:
            self.pool.get_num_free_blocks = self.original_count
        if self.pool.free_blocks is self.free_wrapper:
            self.pool.free_blocks = self.original_free
        if self.pool.get_new_blocks is self.get_wrapper:
            self.pool.get_new_blocks = self.original_get


class SourceCompletionChannel:
    """Local-engine datagrams; allocator changes stay on the scheduler thread.

    A full/unavailable socket drops only the fast notification, never the
    worker-output fallback. Pins remain held until either path is consumed.
    """

    def __init__(self, path: str, receiver: bool):
        self.path = path
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.receiver = receiver
        self.closed = False
        if receiver:
            try:
                try:
                    self.socket.bind(path)
                except OSError as error:
                    if error.errno != errno.EADDRINUSE:
                        raise
                    # Never unlink a live engine's endpoint. A harmless probe
                    # distinguishes a stale pathname after ungraceful shutdown.
                    try:
                        self.socket.sendto(b'{"event":"probe"}', path)
                    except ConnectionRefusedError:
                        Path(path).unlink(missing_ok=True)
                        self.socket.bind(path)
                    else:
                        raise RuntimeError(f"Shared-pool W2 endpoint already active: {path}") from error
            except Exception:
                self.socket.close()
                raise

    def send(self, event: dict) -> bool:
        try:
            self.socket.sendto(json.dumps(event).encode(), self.path)
            return True
        except OSError:
            return False

    def drain(self) -> list[dict]:
        events = []
        for _ in range(MAX_NOTIFICATION_DRAIN):
            try:
                events.append(json.loads(self.socket.recv(MAX_NOTIFICATION_BYTES)))
            except BlockingIOError:
                break
        return events

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.socket.close()
        if self.receiver:
            Path(self.path).unlink(missing_ok=True)


def completion_channel_path(config, rpc_base_path: str) -> str:
    # Use the existing per-engine lookup namespace, including serving DP index.
    if getattr(config.parallel_config, "nnodes_within_dp", 1) != 1:
        raise ValueError("Shared-pool W2 early release currently requires a single-node P engine")
    extra = config.kv_transfer_config.kv_connector_extra_config
    port = extra.get("lookup_rpc_port", extra.get("mooncake_rpc_port", 0))
    namespace = f"{rpc_base_path}:{port}:{config.parallel_config.data_parallel_index}"
    digest = hashlib.sha256(namespace.encode()).hexdigest()[:24]
    return str(Path(tempfile.gettempdir()) / f"ascend-w2-{digest}.sock")


class ComputeCompletionObserver:
    """Poll already-recorded device events without synchronizing the device.

    Observation only: unavailable/overflowed events reduce evidence coverage,
    never block execution or change cache ownership.
    """

    def __init__(self, observation, enter_device):
        self.observation = observation
        self.enter_device = enter_device
        self.pending = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="w2-compute-observer")
        self.thread.start()

    def submit(self, event, blocks):
        with self.lock:
            if len(self.pending) >= MAX_COMPUTE_EVENTS:
                self.observation.add(compute_observation_dropped=1)
                return
            self.pending.append((event, tuple(blocks)))

    def _run(self):
        try:
            self.enter_device()
            while not self.stop_event.wait(COMPUTE_POLL_SECONDS):
                with self.lock:
                    pending, self.pending = self.pending, []
                remaining = []
                for event, blocks in pending:
                    if event.query():
                        for block in blocks:
                            self.observation.trace("compute_done", block=block)
                    else:
                        remaining.append((event, blocks))
                with self.lock:
                    self.pending = remaining + self.pending
        except Exception:
            self.observation.add(compute_observation_errors=1)

    def close(self):
        self.stop_event.set()
        self.thread.join()
        with self.lock:
            self.observation.add(compute_observation_pending=len(self.pending))
