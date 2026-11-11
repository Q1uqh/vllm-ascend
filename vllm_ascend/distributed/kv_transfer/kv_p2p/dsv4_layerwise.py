# SPDX-License-Identifier: Apache-2.0
"""Chunk-wise DSV4 component copies between different physical layouts."""


def component_transfer_ranges(
    local, remote, local_ids, remote_ids, spec, endpoint, *, transferred_tokens=0, chunk_finish=True
):
    """Copy live pages only, using each peer's stride and each segment's payload.

    Block zero is the allocator's null block. Never read it or write it. A
    missing required source is an error rather than an incomplete handoff.
    Both the compressed-history and state groups express spans in raw tokens.
    """
    size = spec.block_size
    window = getattr(spec, "sliding_window", None)
    # The receiver reserves the final sliding window. Earlier windows can
    # contain null destination blocks and mutable state that D cannot use.
    if window and not chunk_finish:
        return [], [], []
    if window:
        start = max(0, (endpoint - window + 1) // size)
    else:
        start = transferred_tokens // size
    end = (endpoint + size - 1) // size if chunk_finish else endpoint // size
    if endpoint <= 0 or end > len(local_ids) or end > len(remote_ids):
        raise ValueError("DSV4 handoff endpoint exceeds allocated component tables")
    if not local.block_stride or not remote.block_stride:
        raise ValueError("DSV4 handoff requires page strides on both peers")
    count = len(local.block_len)
    if any(
        len(values) != count
        for values in (
            local.kv_caches_base_addr,
            local.block_stride,
            remote.block_len,
            remote.kv_caches_base_addr,
            remote.block_stride,
        )
    ):
        raise ValueError("DSV4 handoff component segment count differs")
    if local.block_len != remote.block_len:
        raise ValueError("DSV4 handoff component payload differs")
    src, dst, lengths = [], [], []
    for index in range(start, end):
        source, target = local_ids[index], remote_ids[index]
        if not source or not target:
            raise ValueError("DSV4 handoff selected a required null block")
        for segment, length in enumerate(local.block_len):
            if length > min(local.block_stride[segment], remote.block_stride[segment]):
                raise ValueError("DSV4 handoff payload exceeds physical page stride")
            src.append(local.kv_caches_base_addr[segment] + source * local.block_stride[segment])
            dst.append(remote.kv_caches_base_addr[segment] + target * remote.block_stride[segment])
            lengths.append(length)
    return src, dst, lengths
