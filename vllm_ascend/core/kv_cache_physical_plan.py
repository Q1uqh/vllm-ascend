# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical placement probes shared by workers and version-specific builders."""

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec

from vllm_ascend.models.glm5next.cache_config import (
    _build_glm5_next_kv_cache_config,
    _get_glm5_next_cache_layout,
    _is_glm5_next_spec,
    get_glm5_next_kv_cache_groups,
)


def get_physical_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec], num_blocks: int
) -> KVCacheConfig | None:
    """Resolve model-owned physical placement at the existing builder seam."""
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return _build_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, num_blocks)
    from vllm_ascend.models.deepseek_v4.cache_config import build_deepseek_v4_physical_plan

    return build_deepseek_v4_physical_plan(vllm_config, kv_cache_groups, num_blocks)


def get_kv_cache_allocation_overhead(kv_cache_config: KVCacheConfig) -> int:
    """Fixed cost for plans allocating one aligned raw tensor per descriptor."""
    allocations = (
        1 if getattr(kv_cache_config, "kv_cache_shared_backing", False) else len(kv_cache_config.kv_cache_tensors)
    )
    return allocations * getattr(kv_cache_config, "kv_cache_allocation_alignment", 0)


def get_physical_kv_cache_config_from_specs(
    vllm_config: VllmConfig, kv_cache_specs: dict[str, KVCacheSpec]
) -> KVCacheConfig | None:
    """Probe physical builders without regrouping ordinary model layouts."""
    if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in kv_cache_specs.values()):
        from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

        groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(vllm_config, kv_cache_specs)
        return get_physical_kv_cache_config(vllm_config, groups, 1)
    if not any(_is_glm5_next_spec(spec) for spec in kv_cache_specs.values()):
        return None
    groups = get_glm5_next_kv_cache_groups(vllm_config, kv_cache_specs)
    return get_physical_kv_cache_config(vllm_config, groups, 1)
