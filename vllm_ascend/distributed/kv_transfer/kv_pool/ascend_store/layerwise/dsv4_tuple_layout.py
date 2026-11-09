# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Placement for temporal sharing within DeepSeek V4's existing cache groups."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, KVCacheTensor


DSV4_CACHE_ALIGNMENT = 2 * 1024 * 1024

_LAYER_COMPONENT = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.(.+)$")
_SWA = frozenset({"swa_cache"})
_HCA = _SWA | {"attn", "compressor.state_cache"}
_CSA = _HCA | {"indexer.k_cache", "indexer.compressor.state_cache"}
_TARGET_COMPONENTS = (_SWA, _HCA, _CSA)


@dataclass(frozen=True)
class TupleRegion:
    group: int
    slot: int
    page_size: int
    layers: tuple[str, ...]


@dataclass(frozen=True)
class DSV4TuplePlan:
    page_sizes: tuple[int, ...]
    regions: tuple[TupleRegion, ...]
    predecessors: tuple[tuple[int, int], ...]
    num_slots: int

    @property
    def bytes_per_block(self) -> int:
        return self.num_slots * sum(self.page_sizes)

    def region_offset(self, region: TupleRegion, num_blocks: int) -> int:
        return num_blocks * (
            region.slot * sum(self.page_sizes) + sum(size for size in self.page_sizes if size < region.page_size)
        )


def plan_dsv4_tuples(
    groups: list[dict[str, KVCacheSpec]],
    num_target_layers: int,
    num_buffers: int,
    independent_layers: set[int],
    resident_components: set[str],
) -> DSV4TuplePlan:
    """Compact each group/page bucket without changing logical layer indices.

    Spec equality covers the full storage format. A cohort includes group and
    component identity, so every alias in that cohort has the same predecessor.
    Across groups, the original global block ID exclusion still governs overlay.
    """
    if not groups or not groups[0] or num_buffers < 1 or num_target_layers < 1:
        raise ValueError("DSV4 temporal layout requires cache groups, target layers and positive buffer count")
    if not independent_layers <= set(range(num_target_layers)):
        raise ValueError("DSV4 independent layer is outside the target layer range")

    page_sizes = tuple(sorted({spec.page_size_bytes for spec in groups[0].values()}))
    components: dict[int, dict[str, tuple[int, str, KVCacheSpec]]] = defaultdict(dict)
    seen: set[str] = set()
    resident: list[tuple[int, str, int]] = []
    for group, specs in enumerate(groups):
        for name, spec in specs.items():
            page_size = spec.page_size_bytes
            if name in seen or page_size <= 0 or page_size not in page_sizes:
                raise ValueError(f"DSV4 temporal layout has duplicate or incompatible component {name}")
            seen.add(name)
            if name in resident_components:
                resident.append((group, name, page_size))
                continue
            match = _LAYER_COMPONENT.search(name)
            if match is None:
                raise ValueError(f"Unknown DSV4 cache component {name}")
            layer = int(match[1])
            role = match[2]
            if layer >= num_target_layers or role in components[layer]:
                raise ValueError(f"Unexpected DSV4 target cache component {name}")
            components[layer][role] = (group, name, spec)

    if resident_components - seen:
        raise ValueError("DSV4 resident components must belong to a cache group")

    cohorts: list[tuple[tuple[tuple[str, int, KVCacheSpec], ...], list[int]]] = []
    slots: list[tuple[int, ...]] = []
    for layer in range(num_target_layers):
        roles = components[layer]
        if frozenset(roles) not in _TARGET_COMPONENTS:
            raise ValueError(f"Unsupported DSV4 component manifest for layer {layer}: {sorted(roles)}")
        if layer in independent_layers:
            slots.append((layer,))
            continue
        signature = tuple((role, group, spec) for role, (group, _, spec) in sorted(roles.items()))
        for candidate, members in cohorts:
            if candidate == signature:
                members.append(layer)
                break
        else:
            cohorts.append((signature, [layer]))

    for _, members in cohorts:
        count = min(len(members), num_buffers)
        slots.extend(tuple(members[index::count]) for index in range(count))

    predecessors = []
    buckets: dict[tuple[int, int], list[tuple[str, ...]]] = defaultdict(list)
    for members in slots:
        predecessors.extend(zip(members[1:], members))
        for role, (group, _, spec) in sorted(components[members[0]].items()):
            aliases = tuple(components[layer][role][1] for layer in members)
            buckets[group, spec.page_size_bytes].append(aliases)

    regions = [
        TupleRegion(group, index, size, aliases)
        for (group, size), entries in sorted(buckets.items())
        for index, aliases in enumerate(entries)
    ]
    num_slots = max((region.slot for region in regions), default=-1) + 1
    for group, name, size in resident:
        regions.append(TupleRegion(group, num_slots, size, (name,)))
        num_slots += 1
    return DSV4TuplePlan(page_sizes, tuple(regions), tuple(sorted(predecessors)), num_slots)


def dsv4_temporal_config(config: VllmConfig) -> dict | None:
    """Select the producer's block-key configuration before cache sizing."""
    transfer = getattr(config, "kv_transfer_config", None)
    if transfer is None:
        return None
    extra = transfer.kv_connector_extra_config
    if transfer.kv_connector in ("MultiConnector", "AscendMultiConnector"):
        selected = []
        for child in transfer.kv_connector_extra_config.get("connectors", ()):
            if child.get("kv_connector") not in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
                continue
            extra = child.get("kv_connector_extra_config", {})
            if str(extra.get("backend", "mooncake")).lower() == "mooncake" and (
                extra.get("layerwise_num_shared_buffers") is not None or extra.get("layerwise_dsv4_handoff")
            ):
                selected.append(child)
        if not selected:
            return None
        children = transfer.kv_connector_extra_config["connectors"]
        if (
            len(selected) != 1
            or len(children) != 2
            or children[0].get("kv_connector") != "MooncakePushConnector"
            or selected[0]["kv_connector_extra_config"].get("layerwise_dsv4_pd_transport") != "d2d"
            or not any(
                child.get("kv_connector") == "MooncakePushConnector"
                and child.get("kv_role") == "kv_producer"
                and child.get("kv_connector_extra_config", {}).get("dsv4_temporal_d2d") is True
                for child in children
            )
        ):
            raise ValueError("DSV4 temporal MultiConnector requires one store and one D2D MooncakePushConnector")
        extra = selected[0]["kv_connector_extra_config"]
        if selected[0].get("kv_role") != "kv_producer":
            raise ValueError("DSV4 temporal store requires a kv_producer")
    elif transfer.kv_connector not in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
        return None
    if extra.get("layerwise_dsv4_pd_transport", "snapshot") not in ("snapshot", "d2d"):
        raise ValueError("Unsupported DSV4 PD transport")
    if extra.get("layerwise_dsv4_pd_transport") == "d2d" and extra.get("layerwise_dsv4_handoff"):
        raise ValueError("DSV4 D2D cannot enable the bulk H2D handoff")
    if str(extra.get("backend", "mooncake")).lower() != "mooncake" or extra.get("layerwise_num_shared_buffers") is None:
        return None
    if not extra.get("use_layerwise"):
        raise ValueError("DSV4 temporal cache sharing requires use_layerwise=true")
    if transfer.kv_role != "kv_producer":
        raise ValueError("DSV4 temporal cache sharing requires a kv_producer")
    _validate_dsv4_snapshot_topology(config)
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError("DSV4 temporal producer does not support active speculative prefill")
    if config.scheduler_config.async_scheduling or config.cache_config.enable_prefix_caching:
        raise ValueError("DSV4 temporal producer requires synchronous scheduling and disabled local prefix caching")
    return extra


def validate_dsv4_d2d_config(config: VllmConfig) -> None:
    """Keep temporal D2D on the endpoint/topology validated by the store."""
    _validate_dsv4_snapshot_topology(config)
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError("DSV4 temporal D2D does not support speculative decoding")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("DSV4 temporal D2D requires disabled local prefix caching on P and D")


def build_dsv4_temporal_plan(config: VllmConfig, groups: list[KVCacheGroupSpec]) -> DSV4TuplePlan:
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        build_layerwise_cache_layout,
    )

    extra = dsv4_temporal_config(config)
    if extra is None:
        raise ValueError("DSV4 temporal plan requires temporal producer configuration")
    num_layers = config.model_config.get_num_layers(config.parallel_config)
    layout = build_layerwise_cache_layout(num_layers, extra)
    components: list[dict[str, KVCacheSpec]] = []
    for group in groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, UniformTypeKVCacheSpecs):
            raise ValueError("DSV4 temporal cache sharing requires per-component cache specs")
        components.append({name: spec.kv_cache_specs[name] for name in group.layer_names})
    resident = {name for group in components for name in group if "mtp" in name.split(".")}
    return plan_dsv4_tuples(components, num_layers, layout.num_shared_buffers, set(layout.independent_layers), resident)


def dsv4_mutable_groups(groups: list[KVCacheGroupSpec]) -> set[int]:
    return {
        index
        for index, group in enumerate(groups)
        if any(name.endswith((".swa_cache", ".state_cache")) for name in group.layer_names)
    }


def dsv4_mutable_windows(groups: list[KVCacheGroupSpec], mutable_groups: set[int]) -> dict[int, int]:
    from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs

    windows = {}
    for group_id in mutable_groups:
        spec = groups[group_id].kv_cache_spec
        components = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if components and all(isinstance(component, SlidingWindowSpec) for component in components):
            windows[group_id] = max(
                component.sliding_window + (getattr(component, "extra_retained_tokens", 0) or 0)
                for component in components
                if isinstance(component, SlidingWindowSpec)
            )
    return windows


def dsv4_history_ratios(groups: list[KVCacheGroupSpec], mutable_groups: set[int]) -> dict[int, int]:
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    ratios: dict[int, int] = {}
    for group_id, group in enumerate(groups):
        if group_id in mutable_groups:
            continue
        spec = group.kv_cache_spec
        components = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        group_ratios = {int(component.tokens_per_state) for component in components}
        if len(group_ratios) != 1 or min(group_ratios) <= 0:
            raise ValueError(f"DSV4 history group {group_id} has incompatible compressed row counts")
        ratios[group_id] = group_ratios.pop()
    return ratios


def dsv4_descriptor_predecessors(descriptors: list[KVCacheTensor]) -> dict[int, int]:
    """Recover the compute dependencies from the same descriptors allocating storage."""
    predecessors: dict[int, int] = {}
    for descriptor in descriptors:
        if descriptor.layer_stride or len(descriptor.layers) < 2:
            continue
        matches = [_LAYER_COMPONENT.search(name) for name in descriptor.layers]
        if not all(matches):
            raise ValueError("Unknown component in DSV4 alias descriptor")
        layers = [int(match[1]) for match in matches if match is not None]
        if layers != sorted(set(layers)):
            raise ValueError("DSV4 alias members must follow model execution order")
        for layer, predecessor in zip(layers[1:], layers):
            if predecessors.setdefault(layer, predecessor) != predecessor:
                raise ValueError(f"DSV4 component predecessors disagree for layer {layer}")
    return predecessors


def _validate_dsv4_snapshot_topology(config: VllmConfig) -> None:
    parallel = config.parallel_config
    if (
        not config.use_v2_model_runner
        or not config.model_config.enforce_eager
        or any(
            size != 1
            for size in (
                parallel.pipeline_parallel_size,
                parallel.prefill_context_parallel_size,
                parallel.decode_context_parallel_size,
            )
        )
    ):
        raise ValueError("DSV4 snapshot protocol requires eager MRV2 with PP, PCP and DCP equal to one")


def dsv4_producer_replica(config: VllmConfig) -> int | None:
    parallel = config.parallel_config
    if parallel.data_parallel_size == 1:
        return None
    replica = parallel.data_parallel_index
    if not 0 <= replica < parallel.data_parallel_size:
        raise ValueError("DSV4 producer has an invalid DP replica index")
    return replica


def uses_dsv4_snapshot_protocol(config: VllmConfig) -> bool:
    transfer = getattr(config, "kv_transfer_config", None)
    if transfer is None:
        return False
    if dsv4_temporal_config(config) is not None:
        return True
    extra = transfer.kv_connector_extra_config
    if extra.get("layerwise_dsv4_handoff") is not True:
        return False
    if transfer.kv_role != "kv_consumer" or str(extra.get("backend", "mooncake")).lower() != "mooncake":
        raise ValueError("layerwise_dsv4_handoff requires a resident Mooncake consumer")
    _validate_dsv4_snapshot_topology(config)
    return True


def dsv4_manifest_digest(groups: list[KVCacheGroupSpec]) -> str:
    """Hash logical page formats independently of temporal placement and N."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    manifest = []
    for group in groups:
        wrapper = group.kv_cache_spec
        if not isinstance(wrapper, UniformTypeKVCacheSpecs):
            raise ValueError("DSV4 snapshot manifest requires the original per-component specs")
        manifest.append([(name, repr(wrapper.kv_cache_specs[name])) for name in sorted(group.layer_names)])
    return hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
