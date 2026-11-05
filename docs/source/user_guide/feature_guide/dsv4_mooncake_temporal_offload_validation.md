# DeepSeek V4 Flash Mooncake temporal offload validation

In eager Model Runner V2, a Mooncake producer can reuse DeepSeek V4 Flash
device cache slots across compatible layers. The resident decoder restores the
producer's final checkpoint. Chunk continuation restores immutable history
from public keys and active SWA and compressor state from request snapshots.

## Server setup

Start with a known-good DeepSeek V4 Flash deployment on Ascend A5. Use the
same model path and revision, cache dtype, and TP size on P and D. Use eager
MRV2 (`VLLM_USE_V2_MODEL_RUNNER=1`, `--enforce-eager`) with DP, PP, PCP and DCP equal to 1. Use the vLLM revision recorded in
`.github/vllm-main-verified.commit` and a Mooncake DRAM build with session/range
APIs and `batch_remove`. Record the resolved revisions and per-rank KV budget.
Use a standalone `AscendStoreConnector`. Run P without speculative decoding,
with `--no-async-scheduling` and `--no-enable-prefix-caching`. External public
prefix lookup remains active with local prefix caching disabled.

P configuration:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 2,
    "layerwise_independent_layers": [0, 1],
    "layerwise_prefetch_layers": 2
  }
}
```

D configuration:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true,
    "layerwise_dsv4_handoff": true
  }
}
```

Use `max_tokens=1` on P. Forward its returned `kv_transfer_params` to D
together with the original prompt token IDs and the first token sampled by P.
The decoder starts at the exact prefill extent and acknowledges the handoff by
removing private objects after reads finish. Removal respects backend leases.
Use sufficient Mooncake DRAM for active snapshots and history. Missing required
objects fail the transfer, including eviction between chunks or before handoff.
GET sessions protect active reads. Size the pool to retain data between
sessions. If a handoff is abandoned before D can acknowledge it, the
unacknowledged snapshot remains eligible for Mooncake eviction.

## Compare with the resident deployment

For capacity comparisons, rerun P at the same budget with
`layerwise_num_shared_buffers` omitted. For numerical comparison, use a
known-good resident generation endpoint with the same model and runtime.
Use a fresh Mooncake namespace or pool for cold runs. Set
`--max-num-batched-tokens` below the length of at least one prompt to force
multiple chunks. Supply a JSON array of prompt token-ID arrays.

```bash
python3 tests/e2e/cases/features/kv_pool/dsv4_mooncake_temporal_smoke.py \
  --prefill http://prefill:8000 --decode http://decode:8000 \
  --resident http://resident:8000 --model dsv4 \
  --prompt-tokens prompts.json --output-tokens 8 --repeat 2 | tee smoke.log
```

## Acceptance matrix

- Compare token IDs and finite logprobs for cold, partial-hit, and warm
  requests. Include interleaving, 32K and 128K inputs, and chunk endpoints
  around 3/4/5, 127/128/129 and each actual group boundary.
- Compare per-rank block counts, unique backing bytes, and HBM peaks.
  Exercise buffer counts 1 through 4 and independent layers. Verify the
  planner uses each group's actual page format and logical block table.
- Delay PUT, GET, and commit independently. Check that no-load successors
  and non-upload ranks wait for previous-layer device accesses and source
  copy. Check that the next chunk and D read committed data only.
- Test aligned and unaligned P/D handoff and delay D submission. Confirm
  that final private objects survive until read completion and are removed
  after active read leases permit removal.
- Inject missing objects, short transfer responses, cancellation, and
  preemption. Verify fail-stop and ownership-aware cleanup. Run 1,000
  interleaved or multichunk requests or one hour of traffic.
- Collect TTFT P50/P95, throughput, and HBM peak across at least three warmed
  runs. Report actual copied bytes separately from requested bytes. The
  smoke script reports request duration for diagnostics.

Enable `kv_pool_range_debug` in Ascend additional config during diagnostics.
Collect all P/D logs and the smoke report together. The script emits at most
300 lines and preserves recent errors and failed transfer events:

```bash
python3 tests/e2e/cases/features/kv_pool/summarize_mooncake_range_logs.py \
  prefill.log decode.log smoke.log > mooncake-range-summary.txt
```
