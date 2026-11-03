# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Eager dense MHA/GQA reference path for ordinary W2 token-range loading.

This path favors explicit dependencies and correctness over fused-kernel speed.
No full-cache staging tensor is allocated; each gathered range is temporary.
"""

import torch

from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

MAX_QUERY_CHUNK = 32


def merge_attention_states(output, lse, partial_output, partial_lse):
    """Merge normalized partial outputs using log-sum-exp, including empty masks."""
    combined = torch.logaddexp(lse, partial_lse)
    finite = torch.isfinite(combined)
    safe = torch.where(finite, combined, torch.zeros_like(combined))
    weight = torch.where(torch.isfinite(lse), torch.exp(lse - safe), torch.zeros_like(lse))
    partial_weight = torch.where(torch.isfinite(partial_lse), torch.exp(partial_lse - safe),
                                 torch.zeros_like(partial_lse))
    return output * weight + partial_output * partial_weight, combined


def _active_worker(connector):
    worker = getattr(connector, "connector_worker", None)
    if worker is not None and getattr(worker, "_w2_token_pipeline", None) is not None:
        return worker
    workers = [result for child in getattr(connector, "_connectors", ())
               if (result := _active_worker(child)) is not None]
    if len(workers) > 1:
        raise RuntimeError("W2-b cannot compose multiple token-load producers")
    return workers[0] if workers else None


def maybe_token_overlap(impl, query, metadata, output):
    if not has_kv_transfer_group():
        return None
    worker = _active_worker(get_kv_transfer_group())
    if worker is None:
        return None
    pipeline = worker._w2_token_pipeline
    if pipeline.finished.is_set():
        # All destination ranges are now resident. Later layers should use
        # the existing fused attention rather than replaying tiny partitions.
        # Check the result before allowing that path to read the full cache.
        pipeline.finish()
        return None
    if (impl.sinks is not None or impl.alibi_slopes is not None
            or metadata.model_runner_type == "pooling" or not metadata.causal
            or metadata.uses_pcp_sharded_layout):
        raise NotImplementedError("W2-b supports ordinary causal dense attention without sinks or ALiBi")
    key_cache, value_cache = impl.key_cache, impl.value_cache
    if (key_cache is None or value_cache is None or key_cache.ndim != 4
            or key_cache.dtype != query.dtype or value_cache.dtype != query.dtype
            or query.dtype not in (torch.float16, torch.bfloat16)):
        raise NotImplementedError("W2-b requires unquantized dense K/V and FP16/BF16 queries")
    query = query.reshape(-1, impl.num_heads, impl.head_size)
    # Eager metadata only. No device-to-host query-length synchronization.
    lengths, boundaries = metadata.seq_lens_list, metadata.actual_seq_lengths_q
    if lengths is None or boundaries is None:
        raise NotImplementedError("W2-b requires eager CPU sequence metadata")
    # The page table is small; this eager reference path copies it once per
    # layer. This overhead is explicit and must be included in performance QA.
    tables = metadata.block_tables.cpu().tolist()
    block_tokens = key_cache.shape[2 if impl.use_bnsd_kv_cache else 1]
    qfirst = 0
    for row, qlast in enumerate(boundaries):
        seq_len = lengths[row]
        qlen = qlast - qfirst
        if qlen <= 0:
            qfirst = qlast
            continue
        for qoffset in range(0, qlen, MAX_QUERY_CHUNK):
            qend = min(qoffset + MAX_QUERY_CHUNK, qlen)
            q = query[qfirst + qoffset:qfirst + qend].float().transpose(0, 1)
            qpositions = torch.arange(seq_len - qlen + qoffset, seq_len - qlen + qend, device=query.device)
            accumulated = torch.zeros_like(q)
            lse = torch.full((*q.shape[:2], 1), -torch.inf, device=query.device, dtype=torch.float32)
            kv_last = seq_len - qlen + qend
            kv_first = (max(0, seq_len - qlen + qoffset - impl.sliding_window + 1)
                        if impl.sliding_window is not None else 0)
            for first in range(kv_first, kv_last, worker.w2_token_batch_size):
                # Split once more at physical page boundaries.
                last = min(first + worker.w2_token_batch_size, kv_last)
                keys, values = [], []
                position = first
                while position < last:
                    page, slot = divmod(position, block_tokens)
                    stop = min(last - position + slot, block_tokens)
                    block = tables[row][page]
                    if block <= 0:
                        raise RuntimeError("W2-b missing cache page inside the live attention window")
                    pipeline.wait(block, slot, stop)
                    if impl.use_bnsd_kv_cache:
                        keys.append(key_cache[block, :, slot:stop, :].transpose(0, 1))
                        values.append(value_cache[block, :, slot:stop, :].transpose(0, 1))
                    else:
                        keys.append(key_cache[block, slot:stop])
                        values.append(value_cache[block, slot:stop])
                    position += stop - slot
                k = keys[0] if len(keys) == 1 else torch.cat(keys)
                v = values[0] if len(values) == 1 else torch.cat(values)
                k = k.repeat_interleave(impl.num_queries_per_kv, dim=1).float().transpose(0, 1)
                v = v.repeat_interleave(impl.num_queries_per_kv, dim=1).float().transpose(0, 1)
                scores = torch.matmul(q, k.transpose(-1, -2)) * impl.scale
                kpositions = torch.arange(first, last, device=query.device)
                allowed = kpositions[None, :] <= qpositions[:, None]
                if impl.sliding_window is not None:
                    allowed = allowed & (kpositions[None, :] >= qpositions[:, None] - impl.sliding_window + 1)
                scores = scores.masked_fill(~allowed[None, :, :], -torch.inf)
                partial_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                safe_lse = torch.where(torch.isfinite(partial_lse), partial_lse, torch.zeros_like(partial_lse))
                partial = torch.matmul(torch.exp(scores - safe_lse), v)
                accumulated, lse = merge_attention_states(accumulated, lse, partial, partial_lse)
                worker._w2_observation.add(partial_attention_batches=1)
                worker._w2_observation.trace("d_partial_attention_submit", generation=pipeline.generation, layer=impl._layer_name,
                                             first=first, last=last, row=row,
                                             blocks=tables[row][first // block_tokens:(last - 1) // block_tokens + 1])
            result = accumulated.transpose(0, 1).to(query.dtype)
            destination = output[qfirst + qoffset:qfirst + qend]
            destination.copy_(result.reshape(destination.shape))
        worker._w2_observation.trace("d_attention_merge_submit", generation=pipeline.generation, layer=impl._layer_name, row=row)
        qfirst = qlast
    return output
