# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_placement import build_kvpp_layer_layout, create_kvpp_cache_allocation_plan
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    is_layerwise_transfer_enabled,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.timeline import emit_timeline_event
from vllm_ascend.distributed.kvpp import BroadcastKVPPTransport
from vllm_ascend.distributed.parallel_state import get_kvpp_group
from vllm_ascend.worker.kvpp_cache import get_kvpp_cache_specs


class KVPPRuntime:
    """Bind contiguous cache storage to the shared layer prefetch scheduler."""

    def __init__(self, scheduler: KVPPScheduler | None = None) -> None:
        self.scheduler = scheduler

    @classmethod
    def create_from_kv_cache(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any] | None = None,
    ) -> KVPPRuntime:
        config = KVPPConfig.from_vllm_config(vllm_config)
        if config.size <= 1:
            return cls()
        if kv_caches is None:
            kv_caches = {
                name: static_forward_context[name].kv_cache
                for group in kv_cache_config.kv_cache_groups
                for name in group.layer_names
            }
        group = get_kvpp_group()
        plan = create_kvpp_cache_allocation_plan(
            vllm_config, get_kvpp_cache_specs(kv_cache_config), group.rank_in_group
        )
        if not plan.layer_owner_ranks:
            return cls()
        layer_buffers = {}
        for name, bundle in plan.layer_bundles.items():
            if name not in plan.layer_owner_ranks:
                continue
            _, size = build_kvpp_layer_layout(bundle, plan.tensor_sizes, kv_cache_config.num_blocks)
            first = kv_caches[name][0]
            storage = first.untyped_storage()
            base = first.storage_offset() * first.element_size()
            # The allocator binds every component to this contiguous layer span.
            raw = torch.empty(0, dtype=torch.int8, device=first.device).set_(storage, base, (size,), (1,))
            layer_buffers[name] = raw
        scheduler = KVPPScheduler(
            transport=BroadcastKVPPTransport(group, plan.layer_owner_ranks, layer_buffers),
            attention_layer_names=tuple(layer_buffers),
            layerwise=is_layerwise_transfer_enabled(vllm_config.kv_transfer_config),
            prefetch_depth=plan.scratch_buffer_count - 1,
        )
        for name in layer_buffers:
            static_forward_context[name].impl.layerwise_kv_cache_hook = scheduler
        return cls(scheduler)

    def prepare_forward(self, has_history: bool) -> None:
        if self.scheduler is not None:
            self.scheduler.schedule_forward(has_history)

    def complete_forward(self) -> None:
        if self.scheduler is not None:
            self.scheduler.complete_forward()


class KVPPScheduler:
    """Queue ordered KVPP broadcasts in a rolling per-layer window."""

    def __init__(
        self,
        transport: BroadcastKVPPTransport,
        attention_layer_names: tuple[str, ...],
        *,
        layerwise: bool = False,
        prefetch_depth: int = 1,
    ) -> None:
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be positive")
        self.layerwise = layerwise
        self.prefetch_depth = prefetch_depth
        self._wait_for_load: Callable[[str], None] | None = None
        self.transport = transport
        self.attention_layer_names = attention_layer_names
        self._has_history = False
        self._next_attention_layer_index = 0
        self._next_layer_to_schedule = 0
        self._prefetch_futures: dict[int, Future[None]] = {}
        self._last_submitted_future: Future[None] | None = None
        self._npu_device_id = torch.npu.current_device()
        self._kv_transfer_stream = torch.npu.Stream()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpp-prefetch")
        self._traced_layers = frozenset(
            name
            for index, name in enumerate(attention_layer_names)
            if index % 10 == 0 or index == len(attention_layer_names) - 1
        )
        self._remaining_trace_forwards = 2
        self._trace_this_forward = False

    def schedule_forward(self, has_history: bool) -> None:
        self._trace_this_forward = has_history and self._remaining_trace_forwards > 0
        if self._trace_this_forward:
            self._remaining_trace_forwards -= 1
            emit_timeline_event(
                "kvpp_schedule_forward",
                has_history=has_history,
                layerwise=self.layerwise,
                prefetch_depth=self.prefetch_depth,
                num_layers=len(self.attention_layer_names),
            )
        self._has_history = has_history
        self._next_attention_layer_index = 0
        self._next_layer_to_schedule = 0
        self._prefetch_futures.clear()
        self._last_submitted_future = None
        if has_history and not self.layerwise:
            self._schedule_through(0)

    def _schedule_through(self, end_index: int) -> None:
        last_index = min(end_index, len(self.attention_layer_names) - 1)
        while self._next_layer_to_schedule <= last_index:
            self.start_layer_prefetch(self._next_layer_to_schedule)

    def start_layer_prefetch(self, layer_index: int) -> None:
        layer_name = self.attention_layer_names[layer_index]
        cache_ready = torch.npu.Event()
        cache_ready.record(torch.npu.current_stream())
        previous = self._last_submitted_future
        future = self._prefetch_executor.submit(self._run_ordered_layer_prefetch, layer_name, cache_ready, previous)
        self._prefetch_futures[layer_index] = future
        self._last_submitted_future = future
        self._next_layer_to_schedule = layer_index + 1
        if self._trace_this_forward and layer_name in self._traced_layers:
            emit_timeline_event(
                "kvpp_broadcast_queued",
                layer_name=layer_name,
                layer_index=layer_index,
                scratch_index=layer_index % (self.prefetch_depth + 1),
            )

    def _run_ordered_layer_prefetch(
        self, layer_name: str, cache_ready: Any, previous: Future[None] | None
    ) -> None:
        # ThreadPoolExecutor keeps submission order. The explicit dependency
        # additionally prevents later collectives from running after a failed
        # layer, which would otherwise desynchronize KVPP ranks.
        if previous is not None:
            previous.result()
        self.run_layer_prefetch(layer_name, cache_ready)

    def run_layer_prefetch(self, layer_name: str, cache_ready: Any) -> None:
        torch.npu.set_device(self._npu_device_id)
        if self._wait_for_load is not None:
            if self._trace_this_forward and layer_name in self._traced_layers:
                emit_timeline_event("kvpp_wait_load_ready_begin", layer_name=layer_name)
            self._wait_for_load(layer_name)
            if self._trace_this_forward and layer_name in self._traced_layers:
                emit_timeline_event("kvpp_wait_load_ready_end", layer_name=layer_name)
        if self.layerwise:
            # A transfer-stream wait orders device execution, but it does not
            # delay this executor thread's host-side HCCL collective launch.
            # Ensure every rank has completed the main-stream work preceding
            # this layer before submitting the KVPP broadcast, otherwise ranks
            # can enqueue collectives in different communicator orders.
            # Later-layer events are recorded before their attention compute,
            # so the intended prefetch/compute overlap remains intact.
            cache_ready.synchronize()
            if self._trace_this_forward and layer_name in self._traced_layers:
                emit_timeline_event("kvpp_collective_ready", layer_name=layer_name)
        if self._trace_this_forward and layer_name in self._traced_layers:
            emit_timeline_event("kvpp_broadcast_begin", layer_name=layer_name)
        self.transport.prefetch(layer_name, cache_ready, self._kv_transfer_stream)
        if self._trace_this_forward and layer_name in self._traced_layers:
            emit_timeline_event("kvpp_broadcast_end", layer_name=layer_name)

    def wait_for_layer(self, layer_name: str) -> None:
        if not self._has_history:
            return
        layer_index = self._next_attention_layer_index
        expected_name = self.attention_layer_names[layer_index]
        if layer_name != expected_name:
            raise RuntimeError(f"KVPP layer order mismatch: expected {expected_name}, got {layer_name}")
        if self.layerwise:
            # The connector's attention hook has now submitted this step's loads.
            if self._wait_for_load is None:
                self._wait_for_load = get_kv_transfer_group().wait_for_layer_load_ready
            # At layer N, attention for N-1 is complete. Its scratch slot can
            # therefore be reused safely for N+prefetch_depth.
            self._schedule_through(layer_index + self.prefetch_depth)
        else:
            self._schedule_through(layer_index)
        future = self._prefetch_futures[layer_index]
        if self._trace_this_forward and layer_name in self._traced_layers:
            emit_timeline_event("kvpp_attention_wait_begin", layer_name=layer_name, layer_index=layer_index)
        try:
            future.result()
        except BaseException:
            for pending in self._prefetch_futures.values():
                pending.cancel()
            self._prefetch_futures.clear()
            raise
        if self._trace_this_forward and layer_name in self._traced_layers:
            emit_timeline_event("kvpp_attention_wait_end", layer_name=layer_name, layer_index=layer_index)
        del self._prefetch_futures[layer_index]
        self._next_attention_layer_index += 1
        if not self.layerwise and self._next_attention_layer_index < len(self.attention_layer_names):
            self._schedule_through(self._next_attention_layer_index)

    def complete_forward(self) -> None:
        self._has_history = False
        self._next_attention_layer_index = 0
        self._next_layer_to_schedule = 0
        self._prefetch_futures.clear()
        self._last_submitted_future = None
