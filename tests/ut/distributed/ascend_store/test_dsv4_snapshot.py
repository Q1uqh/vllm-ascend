# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

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
