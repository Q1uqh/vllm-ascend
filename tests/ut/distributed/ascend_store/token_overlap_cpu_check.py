"""Run in a fresh interpreter so the host-mock suite cannot replace torch."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    sys.exit(77)

for name in ("vllm", "vllm.distributed", "vllm.distributed.kv_transfer"):
    sys.modules[name] = types.ModuleType(name)
transfer = sys.modules["vllm.distributed.kv_transfer"]
transfer.has_kv_transfer_group = lambda: True
transfer.get_kv_transfer_group = lambda: connector
source = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/token_overlap.py"
spec = importlib.util.spec_from_file_location("token_overlap_check", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(7)
for dtype in (torch.float16, torch.bfloat16):
    for head_major in (False, True):
        for window in (None, 3):
            keys = torch.randn(4, 4, 2, 8).to(dtype)
            values = torch.randn(4, 4, 2, 8).to(dtype)
            query = torch.randn(3, 4, 8).to(dtype)
            output = torch.empty_like(query)
            impl = SimpleNamespace(
                sinks=None, alibi_slopes=None, num_heads=4, head_size=8,
                num_queries_per_kv=2, use_bnsd_kv_cache=head_major,
                key_cache=keys.transpose(1, 2).contiguous() if head_major else keys,
                value_cache=values.transpose(1, 2).contiguous() if head_major else values,
                scale=8**-0.5, sliding_window=window, _layer_name="layer.0",
            )
            calls = []
            pipeline = SimpleNamespace(wait=lambda *args: calls.append(args), generation=1)
            observer = SimpleNamespace(add=lambda **kwargs: None, trace=lambda *args, **kwargs: None)
            worker = SimpleNamespace(_w2_token_pipeline=pipeline, w2_token_batch_size=2, _w2_observation=observer)
            connector = SimpleNamespace(connector_worker=worker)
            metadata = SimpleNamespace(
                model_runner_type="generate", causal=True, uses_pcp_sharded_layout=False,
                seq_lens_list=[7, 4], actual_seq_lengths_q=[2, 3],
                block_tables=torch.tensor([[1, 2], [3, 0]]),
            )
            module.maybe_token_overlap(impl, query, metadata, output)
            expected = []
            for first, last, blocks, length in ((0, 2, [1, 2], 7), (2, 3, [3], 4)):
                k = torch.cat([keys[b] for b in blocks])[:length].repeat_interleave(2, dim=1).float().transpose(0, 1)
                v = torch.cat([values[b] for b in blocks])[:length].repeat_interleave(2, dim=1).float().transpose(0, 1)
                q = query[first:last].float().transpose(0, 1)
                qpos = torch.arange(length - last + first, length)
                kpos = torch.arange(length)
                mask = kpos[None, :] <= qpos[:, None]
                if window is not None:
                    mask &= kpos[None, :] >= qpos[:, None] - window + 1
                scores = (q @ k.transpose(-1, -2) * impl.scale).masked_fill(~mask, -torch.inf)
                expected.append((scores.softmax(-1) @ v).transpose(0, 1).to(dtype))
            torch.testing.assert_close(output, torch.cat(expected), atol=0.01, rtol=0.01)
            assert calls and all(block > 0 for block, _, _ in calls)
print("dense GQA causal/windowed token-overlap reference checks passed")
