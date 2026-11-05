# DeepSeek V4 Flash Mooncake temporal offload validation

The tuple planner and Mooncake PUT ownership handling can be tested on CPU.
The temporal plan is currently exercised by unit tests. Shared buffers remain
disabled at runtime pending continuation snapshots, overwrite fences, and
commit-aware prefill/decode handoff. Explicitly supplying
`layerwise_num_shared_buffers` with Mooncake layerwise pooling fails at startup.

## Server baseline

Start from a known-good DeepSeek V4 Flash P/D deployment on Ascend A5 with
Model Runner V2, eager execution, identical TP on P and D, and DP, PP, PCP
and DCP set to 1. Keep the checkpoint revision, vLLM revision, Ascend revision,
Mooncake build, KV dtype, per-rank KV budget and input corpus in the run record.
Use a Mooncake build with the session/range methods required by the connector.

Keep the P connector role `kv_producer` and the D connector role `kv_consumer`.
Set `backend` to `mooncake` and `use_layerwise` to `true` on both, and omit
`layerwise_num_shared_buffers`. Use a fresh pool for cold runs and repeat the
same prompts for warm runs. Enable `kv_pool_range_debug` in the existing Ascend
additional config for diagnostic runs.

## Acceptance after runtime integration

- Compare resident and temporal runs with the same budget, input tokens,
  output length, seed and sampling parameters. Check exact token IDs for
  deterministic requests and finite logprobs. Include cold, partial-hit and
  warm requests, interleaved requests, and 32K and 128K inputs.
- Force multiple chunks and check endpoints around 3/4/5, 127/128/129 and the
  actual group boundaries. Confirm SWA, attention compressor state and
  indexer compressor state restore from the same endpoint as the history.
- Compare the common per-rank block count, unique backing bytes and HBM peaks.
  Exercise buffer counts 1 through 4 and resident layers. Verify each group
  uses its own block table and each component lands in its planned view.
- Delay PUT, GET and commit independently. Check no-load successors and
  non-upload ranks wait for the previous layer's device access and source
  copy. Check the next chunk and D read only after publication.
- Test final P/D handoff at aligned and unaligned endpoints. Delay D's read
  acknowledgment and confirm P keeps the objects until the handoff ends.
- Inject missing objects, short transfer responses, cancellation and
  preemption. Verify failures propagate before cache consumption and that
  the cleanup targets only owned PUT sessions. Run 1,000 interleaved or
  multichunk requests or one hour of traffic.
- Collect TTFT P50/P95, throughput and HBM peak across at least three warmed
  runs. Report actual copied bytes and time separately from requested bytes.

Collect the P and D logs together. The script caps the combined output at
300 lines, preserving recent errors and failed transfer events:

```bash
python3 tests/e2e/cases/features/kv_pool/summarize_mooncake_range_logs.py \
  prefill.log decode.log > mooncake-range-summary.txt
```

The summary reports transfer diagnostics. Compare tokens, capacity, cleanup
and performance using the server test results.
