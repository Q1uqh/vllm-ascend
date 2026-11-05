# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from uuid import uuid4

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.dsv4_snapshot import (
    DSV4Snapshot,
    select_private_snapshot_pages,
)


def test_dsv4_snapshot_uses_live_pages_and_group_spans():
    ids = ([4, 5, 6], [0, 0, 22, 23, 24, 25], [0, 32, 33, 34])
    assert select_private_snapshot_pages(ids, [128, 32, 64], 129, {1, 2}) == ((1,), (2, 3, 4), (1, 2))
    assert select_private_snapshot_pages(ids, [128, 32, 64], 128, {1, 2}) == ((), (2, 3), (1,))


@pytest.mark.parametrize(("end", "expected"), [(3, ()), (4, (0,)), (5, (0,)), (128, ())])
def test_dsv4_history_tail_has_a_key_only_after_a_complete_compressed_row(end, expected):
    assert select_private_snapshot_pages(([5],), [128], end, set(), {0: 4}) == (expected,)


def test_dsv4_snapshot_wire_roundtrip_preserves_generation_and_extent():
    snapshot = DSV4Snapshot(uuid4().hex, 129, ((1,), (2, 3, 4)))
    recovered = DSV4Snapshot.from_wire(snapshot.to_wire(), 2)
    assert recovered == snapshot
    assert recovered.suffix(1, 3) == snapshot.suffix(1, 3)
    assert DSV4Snapshot(uuid4().hex, 129, snapshot.private_pages).suffix(1, 3) != snapshot.suffix(1, 3)
    assert DSV4Snapshot(snapshot.generation, 128, snapshot.private_pages).suffix(1, 3) != snapshot.suffix(1, 3)


@pytest.mark.parametrize("pages", [[[1, 1]], [[2, 1]], [[-1]], [[True]], [None]])
def test_dsv4_snapshot_rejects_malformed_page_manifest(pages):
    payload = DSV4Snapshot(uuid4().hex, 1, ((0,),)).to_wire()
    payload["private_pages"] = pages
    with pytest.raises(ValueError):
        DSV4Snapshot.from_wire(payload, 1)


def test_dp_producers_keep_public_history_separate():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.dsv4_snapshot import public_history_suffix

    assert public_history_suffix("hash", None) == "hash"
    assert public_history_suffix("hash", 0) != public_history_suffix("hash", 3)
    snapshot = DSV4Snapshot(uuid4().hex, 129, ((1,),), producer_replica=3)
    assert DSV4Snapshot.from_wire(snapshot.to_wire(), 1).producer_replica == 3


@pytest.mark.parametrize(
    ("pp", "pcp", "dcp", "accepted"), [(1, 1, 1, True), (1, 2, 1, True), (2, 1, 1, False), (1, 1, 2, False)]
)
def test_temporal_topology_for_dp_expert_parallel(pp, pcp, dcp, accepted):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.dsv4_tuple_layout import (
        dsv4_temporal_config,
    )

    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            data_parallel_size=4,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"use_layerwise": True, "layerwise_num_shared_buffers": 2},
        ),
    )
    if accepted:
        assert dsv4_temporal_config(config) == config.kv_transfer_config.kv_connector_extra_config
    else:
        with pytest.raises(ValueError, match="PP and DCP"):
            dsv4_temporal_config(config)
