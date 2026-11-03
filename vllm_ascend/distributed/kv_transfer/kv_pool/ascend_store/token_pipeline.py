# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Ordinary block-key range reads with per-token readiness (no layerwise puts)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field


TOKEN_READY_POLL_SECONDS = 0.01
STALL_REPORT_SECONDS = 10.0
MAX_STALL_REPORTS = 6


@dataclass
class TokenRangeRead:
    key: str
    block: int
    start: int
    end: int
    addresses: list[int]
    sizes: list[int]
    offsets: list[int]
    request_id: str = ""
    ready: threading.Event = field(default_factory=threading.Event)


def dense_token_ranges(key, block, start, end, batch_tokens, entries, block_tokens, request_id=""):
    """Split a whole-block object into exact token ranges across all KV entries.

    Entries: base address, physical block stride, heads, head dim, element bytes,
    layout (head-major or token-major). Object bytes concatenate whole entries.
    """
    if not 0 <= start < end <= block_tokens or batch_tokens <= 0:
        raise ValueError("Invalid token range or batch size")
    for first in range(start, end, batch_tokens):
        last = min(first + batch_tokens, end)
        addresses, sizes, offsets = [], [], []
        object_offset = 0
        for base, stride, heads, dim, element_bytes, head_major in entries:
            token_bytes = dim * element_bytes
            entry_bytes = block_tokens * heads * token_bytes
            if head_major:
                for head in range(heads):
                    offset = (head * block_tokens + first) * token_bytes
                    addresses.append(base + block * stride + offset)
                    sizes.append((last - first) * token_bytes)
                    offsets.append(object_offset + offset)
            else:
                offset = first * heads * token_bytes
                addresses.append(base + block * stride + offset)
                sizes.append((last - first) * heads * token_bytes)
                offsets.append(object_offset + offset)
            object_offset += entry_bytes
        yield TokenRangeRead(key, block, first, last, addresses, sizes, offsets, request_id)


class TokenLoadPipeline:
    """One background reader; overlapping consumers wait only on their ranges.

    A request cannot release its destination cache until finish() joins the
    reader. Failure wakes consumers, but never marks incomplete data readable.
    """

    def __init__(self, backend, reads, observation, enter_device, generation=0):
        self.backend, self.reads, self.observation = backend, list(reads), observation
        self.enter_device = enter_device
        self.generation = generation
        self.error = None
        self.progress = "starting"
        self.completed_reads = 0
        self.observation.add(pending_load_batches=len(self.reads))
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.by_block = {}
        for read in self.reads:
            self.by_block.setdefault(read.block, []).append(read)
        self.thread = threading.Thread(target=self._run, name="w2-token-load", daemon=True)
        self.thread.start()
        self.watchdog = threading.Thread(target=self._watch_progress, name="w2-token-progress", daemon=True)
        self.watchdog.start()

    def _run(self):
        keys = list(dict.fromkeys(read.key for read in self.reads))
        started = False
        completed = 0
        try:
            self.enter_device()
            # The session is only a read lease; no layerwise publication or put.
            started = True
            self.progress = "acquiring_read_sessions"
            results = self.backend.batch_get_start(keys)
            if len(results) != len(keys) or any(code != 0 for code in results):
                raise RuntimeError("W2-b token read session could not acquire every key")
            for index, read in enumerate(self.reads):
                self.observation.trace("d_transfer_submit", generation=self.generation, request=read.request_id,
                                       batch=index, block=read.block,
                                       first=read.start, last=read.end)
                self.progress = f"reading_block={read.block} tokens=[{read.start},{read.end})"
                results = self.backend.batch_copy_get(
                    [read.key], [read.addresses], [read.sizes], [read.offsets]
                )
                expected_bytes = sum(read.sizes)
                # Unlike session start/end status codes, a range read returns
                # transferred bytes on success and a negative error on failure.
                if len(results) != 1 or results[0] != expected_bytes:
                    raise RuntimeError(
                        f"W2-b range read failed for block {read.block}, "
                        f"tokens=[{read.start},{read.end}), "
                        f"expected_bytes={expected_bytes}, results={results}"
                    )
                # The native range call completes the transfer before returning.
                self.observation.add(token_range_batches=1, remote_read_bytes=sum(read.sizes))
                self.observation.record_load(read.request_id, [f"{read.key}@{read.start}:{read.end}"],
                                             [read.sizes], True)
                self.observation.trace("d_transfer_ready", generation=self.generation, request=read.request_id,
                                       batch=index, block=read.block,
                                       first=read.start, last=read.end)
                read.ready.set()
                completed += 1
                self.completed_reads = completed
                self.observation.add(pending_load_batches=-1)
        except Exception as error:
            self.error = error
            self.observation.add(load_failed=1)
        finally:
            try:
                self.progress = "closing_read_sessions"
                if started and self.backend.batch_get_end(keys) != 0:
                    raise RuntimeError("W2-b failed to close token read session")
            except Exception as error:
                self.error = self.error or error
                self.observation.add(load_failed=1)
            if completed < len(self.reads):
                self.observation.add(pending_load_batches=-(len(self.reads) - completed),
                                     aborted_load_batches=len(self.reads) - completed)
            self.finished.set()

    def _watch_progress(self):
        started = time.monotonic()
        while not self.closed.wait(STALL_REPORT_SECONDS):
            # Bound diagnostics across all pipelines in this worker process.
            with self.observation.lock:
                count = self.observation.counts["token_wait_reports"]
                if count >= MAX_STALL_REPORTS:
                    return
                self.observation.counts["token_wait_reports"] += 1
                attention = self.observation.counts["partial_attention_batches"]
            logging.getLogger("vllm").warning(
                "W2_TOKEN_PROGRESS generation=%s elapsed_s=%.1f stage=%s reads=%s/%s partial_submits=%s",
                self.generation, time.monotonic() - started,
                "loads_complete_waiting_for_attention" if self.finished.is_set() else self.progress,
                self.completed_reads, len(self.reads), attention,
            )

    def wait(self, block, start, end):
        for read in self.by_block.get(block, ()):
            if read.start < end and start < read.end:
                while not read.ready.wait(timeout=TOKEN_READY_POLL_SECONDS):
                    if self.error is not None:
                        raise RuntimeError("W2-b cannot consume incomplete KV") from self.error
                if self.error is not None:
                    raise RuntimeError("W2-b token loading failed") from self.error

    def finish(self):
        self.thread.join()
        self.closed.set()
        self.watchdog.join()
        if self.error is not None:
            raise RuntimeError("W2-b token loading failed; no partial result may be published") from self.error
