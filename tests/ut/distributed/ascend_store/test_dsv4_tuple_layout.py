# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.dsv4_tuple_layout import plan_dsv4_tuples


@dataclass(frozen=True)
class PageSpec:
    page_size_bytes: int
    representation: str


def _groups():
    groups = [{} for _ in range(7)]
    for layer in range(43):
        prefix = f"model.layers.{layer}.self_attn"
        groups[5 + layer % 2][f"{prefix}.swa_cache"] = PageSpec(19456, "swa")
        if layer < 2:
            continue
        csa = layer % 2 == 0
        groups[0 if csa else 1][f"{prefix}.attn"] = PageSpec(19456, "c4" if csa else "c128")
        groups[2 if csa else 3][f"{prefix}.compressor.state_cache"] = PageSpec(19456, "state")
        if csa:
            groups[0][f"{prefix}.indexer.k_cache"] = PageSpec(4224, "fp8-and-scale")
            groups[4][f"{prefix}.indexer.compressor.state_cache"] = PageSpec(4224, "indexer-state")
    return groups


@pytest.mark.parametrize("buffers", [1, 2, 3, 4])
def test_dsv4_tuple_cost_and_component_coverage(buffers):
    groups = _groups()
    plan = plan_dsv4_tuples(groups, 43, buffers, {0, 1}, set())

    assert plan.num_slots == buffers + 1
    assert plan.bytes_per_block == (buffers + 1) * (19456 + 4224)
    assert dict(plan.predecessors)[2 + 2 * buffers] == 2
    assert dict(plan.predecessors)[3 + 2 * buffers] == 3
    assert {name for region in plan.regions for name in region.layers} == set().union(*groups)
    assert sum(len(region.layers) for region in plan.regions) == 167
    for region in plan.regions:
        assert set(region.layers) <= groups[region.group].keys()
        assert plan.region_offset(region, 13) + region.page_size * 13 <= plan.bytes_per_block * 13


def test_dsv4_different_indexer_scale_isolates_the_entire_layer():
    groups = _groups()
    name = "model.layers.6.self_attn.indexer.k_cache"
    groups[0][name] = PageSpec(4224, "different-scale")
    plan = plan_dsv4_tuples(groups, 43, 2, {0, 1}, set())
    assert 6 not in dict(plan.predecessors)
    assert all(
        len(region.layers) == 1
        for region in plan.regions
        if any(name.startswith("model.layers.6.") for name in region.layers)
    )


def test_dsv4_missing_state_rejected():
    groups = _groups()
    del groups[4]["model.layers.6.self_attn.indexer.compressor.state_cache"]
    with pytest.raises(ValueError, match="manifest for layer 6"):
        plan_dsv4_tuples(groups, 43, 2, {0, 1}, set())


def test_dsv4_resident_component_has_separate_slot():
    groups = _groups()
    name = "model.mtp.0.self_attn.swa_cache"
    groups[5][name] = PageSpec(19456, "swa")
    plan = plan_dsv4_tuples(groups, 43, 2, {0, 1}, {name})
    assert plan.num_slots == 4
    assert plan.regions[-1].layers == (name,)
    assert plan.regions[-1].slot == 3
