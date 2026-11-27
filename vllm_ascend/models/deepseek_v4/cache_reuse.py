# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 layerwise cache reuse and transfer helpers."""

from types import MappingProxyType

from vllm.distributed.kv_transfer import get_kv_transfer_group, is_v1_kv_transfer_group
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_layout,
)


DEEPSEEK_V4_SPEC_REUSE_RULES = MappingProxyType(
    {
        AscendMLAAttentionSpec: lambda spec: spec.model_version == "deepseek_v4",
        AscendSlidingWindowMLASpec: True,
    }
)


def get_swa_save_layer_name(attn_layer_name: str, swa_cache_name: str) -> str:
    """Use the registered SWA cache name when it shares a rotating slot."""
    if not is_v1_kv_transfer_group():
        return attn_layer_name
    connector = get_kv_transfer_group()
    layout = get_layerwise_reuse_layout(getattr(connector, "_kv_cache_config", None))
    if layout is not None and swa_cache_name in layout.reused_cache_names:
        return swa_cache_name
    return attn_layer_name


def is_deepseek_v4_cache(groups_or_specs) -> bool:
    specs = (
        groups_or_specs.values()
        if isinstance(groups_or_specs, dict)
        else (
            spec
            for group in groups_or_specs
            for spec in (
                group.kv_cache_spec.kv_cache_specs.values()
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else (group.kv_cache_spec,)
            )
        )
    )
    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs)


def validate_dsv4_transfer_speculation(config) -> None:
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None and speculative.method not in ("mtp", "dspark"):
        raise ValueError("DSV4 Mooncake transfer supports native MTP and DSpark speculation")
