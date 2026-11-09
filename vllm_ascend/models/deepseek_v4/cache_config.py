# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 cache geometry shared by ordinary and offload planning."""

import hashlib
import json
from collections import defaultdict
from dataclasses import replace

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseCacheTopology,
    apply_layerwise_kv_cache_plan,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
    get_physical_layer_index,
)


def _page_sizes(spec):
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and rank normalization.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            padded = getattr(specs[name], "page_size_padded", None)
            page_size = padded if padded is not None else specs[name].page_size_bytes
            if "mtp" not in name:
                b[page_size].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = page_size
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def build_deepseek_v4_cache_tensors(groups, num_blocks):
    """Ordinary tuple descriptors, shared by both capacity entry points."""
    pages, buckets, drafts, draft_page, tuples = _get_deepseek_v4_cache_layout(groups)
    stride = sum(pages) * num_blocks
    offsets = {}
    offset = 0
    for page in pages:
        offsets[page] = offset
        offset += page * num_blocks
    tensors = [
        KVCacheTensor(
            size=stride * tuples, layers=list(names), offset=offsets[page], layer_stride=stride, block_stride=page
        )
        for bucket in buckets
        for page in pages
        for names in [bucket.get(page)]
        if names
    ]
    for index, name in enumerate(drafts):
        tensors.append(
            KVCacheTensor(
                size=stride * tuples,
                layers=[name],
                block_stride=draft_page,
                layer_stride=0,
                offset=(tuples - len(drafts) + index) * stride + offsets[draft_page],
            )
        )
    return tensors


def build_deepseek_v4_physical_plan(config, groups, num_blocks):
    """Supply complete component declarations to the common reuse planner."""
    extra = get_layerwise_reuse_config(getattr(config, "kv_transfer_config", None))
    if extra is None or not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for group in groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    ):
        return None
    if not getattr(config.model_config, "enforce_eager", False) or not getattr(config, "use_v2_model_runner", False):
        raise ValueError("DeepSeek V4 shared KV offload requires eager execution with the V2 model runner")
    if getattr(getattr(config, "scheduler_config", None), "async_scheduling", False):
        raise ValueError("DeepSeek V4 shared KV offload currently requires synchronous scheduling")
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError("DeepSeek V4 shared KV offload currently requires speculative decoding disabled")
    parallel = config.parallel_config
    if any(
        getattr(parallel, field, 1) != 1
        for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("DeepSeek V4 shared KV offload requires PP=PCP=DCP=1")
    _, _, drafts, _, _ = _get_deepseek_v4_cache_layout(groups)
    tensors = build_deepseek_v4_cache_tensors(groups, num_blocks)
    plan = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)
    specs = get_layerwise_kv_cache_specs(plan)
    base = config.model_config.get_num_layers(parallel)
    group_ids = {name: gid for gid, group in enumerate(groups) for name in group.layer_names}
    by_layer = {}
    for name in specs:
        if name not in drafts:
            by_layer.setdefault(get_physical_layer_index(name, base), []).append(name)
    # Preserve model registration order inside each component vector. Group
    # and full specs include dtype, compression and physical padded page size.
    domains = []
    signatures = []
    for layer, names in sorted(by_layer.items()):
        signature = [(group_ids[name], specs[name]) for name in names]
        for index, existing in enumerate(signatures):
            if signature == existing:
                domains[index].append(layer)
                break
        else:
            signatures.append(signature)
            domains.append([layer])
    plan.layerwise_cache_topology = LayerwiseCacheTopology(
        main_packs=tuple((name,) for name in specs if name not in drafts),
        group_ids=tuple(group_ids.items()),
        draft_layer_names=tuple(drafts),
        component_domains=tuple(tuple(domain) for domain in domains),
        shared_backing=True,
    )
    geometry = [
        (
            name,
            group_ids[name],
            type(spec).__name__,
            spec.block_size,
            spec.page_size_bytes,
            str(getattr(spec, "dtype", None)),
            getattr(spec, "head_size", None),
            str(getattr(spec, "scale_dtype", None)),
            getattr(spec, "scale_dim", 0),
            getattr(spec, "sliding_window", None),
            getattr(spec, "offload_mutable", False),
            getattr(spec, "compress_ratio", None),
            getattr(spec, "tokens_per_state", None),
            getattr(spec, "extra_retained_tokens", None),
        )
        for name, spec in specs.items()
    ]
    plan.layerwise_component_namespace = (
        "components_v1:" + hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    )
    # Pure SWA layers must survive every layer rotation, even when omitted in
    # the user's independent-layer list.
    plan.layerwise_cache_topology = replace(
        plan.layerwise_cache_topology,
        resident_layer_names=tuple(name for layer, names in by_layer.items() if layer < 2 for name in names),
    )
    plan.layerwise_requires_live_block_table = True
    plan.kv_cache_shared_backing = True
    plan.kv_cache_allocation_alignment = 2 * 1024 * 1024
    apply_layerwise_kv_cache_plan(plan, config)
    return plan
