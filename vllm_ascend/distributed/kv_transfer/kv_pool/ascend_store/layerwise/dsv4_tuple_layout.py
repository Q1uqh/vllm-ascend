# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Placement for temporal sharing within DeepSeek V4's existing cache groups."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpec

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
