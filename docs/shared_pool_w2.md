# Ordinary-path shared-pool W2

This opt-in mode uses `AscendStoreConnector` with the Mooncake backend and one
KV cache group. It does not enable layerwise transfer or alter attention math.
Qwen3-32B is the first hardware validation target; other single-group layouts
and parallel configurations require separate validation. The early-release channel
currently requires a single-node P engine (TP/PP ranks on the same host);
initialization rejects a multi-node P engine instead of silently losing notifications.

Add `"enable_shared_pool_w2": true` to **both** P and D connector extra configs.
The option defaults to false. Existing pooling, P2P and other backends retain
their existing behavior when it is absent. An incompatible enabled config fails
initialization rather than silently choosing a different route.

P example:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "enable_shared_pool_w2": true,
    "use_layerwise": false,
    "lookup_rpc_port": "60000"
  }
}
```

D example:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "enable_shared_pool_w2": true,
    "use_layerwise": false,
    "consumer_is_to_load": true,
    "lookup_rpc_port": "60001"
  }
}
```

## Behavior

P pins physical source blocks independently of the core request references.
Each ordinary put submits one logical block, including its relevant key shards.
After a synchronous put returns, the worker reports that it no longer reads the
source. Completion from every participating worker releases that save's pin.
Non-owner or skipped blocks are acknowledged when the save job finishes. The
request, shared-prefix references and core execution fences still protect blocks
needed by computation. Speculative align-state scratch retains its existing
exclusive-ownership and original reuse protection rather than receiving extra pins.

The ordinary save callback only enqueues the background save, following the
original path. It does not wait for publication before returning the step or
add a new P-to-D handoff barrier. Existing background error handling and
source-reuse protection remains in place. Source-reading completion is separate
from publication success. An independent nonblocking UNIX datagram channel sends
per-block completion to the scheduler. The opted-in BlockPool instance polls it
before free-count admission checks and allocation, and pending-push checks also
drain it. Only the scheduler thread changes references. Every rank must finish
reading a block before its transfer pin is released. Duplicate worker-output
completions remain a safe fallback if a datagram cannot be delivered.

For ordinary W2 sources, independent pins replace the old whole-request reuse
wait, allowing a new forward to use a safely returned block while other blocks
of the old save remain in flight. Non-W2 paths and unpinned speculative scratch
retain the original wait. Computation/shared references are not bypassed.
Freeing a pin does not guarantee immediate reuse when other references remain.

D loads missing historical KV on admission and retains it across decode steps.
New KV stays on D. No per-step host writeback or history reload is introduced.
The mode does not reduce D's single-request KV peak or destroy preallocated KV
tensors. Recovery after preemption uses the existing load/recompute path.

One-block puts can affect TTFT. Measure against
an otherwise identical run with the option absent before making performance
claims. The code does not change Mooncake or the vLLM core allocator.

## Validation report (at most 1000 lines)

Cumulative `SHARED_POOL_W2` JSON records are emitted at most once per minute
per process, plus a final record during orderly shutdown. They contain cumulative counters and up to 800 bounded event records per process,
including source block IDs, save generations and same-host monotonic timestamps.
Prompts and full storage keys are not recorded. Existing application logs are unchanged.
The summary tool is maintained separately in `notes/D2RH池化方案/w2/`.
Use fresh logs from one run; include scheduler and worker logs for both roles.
Snapshots refresh on connector activity, not on a separate timer. After
completing requests, stop the test servers gracefully for final snapshots;
merely waiting while an engine is idle may leave an older snapshot. Avoid forced termination
when collecting final records.

```bash
python /path/to/notes/D2RH池化方案/w2/summarize_shared_pool_w2.py \
  prefill.log decode.log > w2-summary.txt
```

The report starts with compact summaries followed by a complete selected event
chain when available; total output is capped at 1000 lines. For multi-rank logs,
pass every rank's log file; the script keeps the latest cumulative snapshot for
each role/side/rank rather than summing snapshots over time. A missing rank,
unreturned pin, malformed record or transfer error is reported explicitly.

`OBSERVED_CHECKS_OK` covers only the counters actually measured:

- `returned` counts source blocks actually reaching zero references through the
  opted-in BlockPool instance. `reused` counts those blocks later obtained via
  `get_new_blocks`, rather than merely a transfer reference being dropped.
- `early_reuse` checks both transitions against active source-save generations.
  Core request references and deferred frees remain responsible for computation
  lifetime; this observation does not independently audit device execution.
- Saves remain asynchronous. The former save-before-return checks and pending
  return counters are removed; `executor_return=NOT_REQUIRED` records this choice.
- `save_done` / `save_failed` observe background save results, without adding a
  step-return barrier or changing the original retry/error handling.
- `notify_before_batch_done`, `reusable_before_batch_done`,
  `reuse_before_batch_done`, and `compute_before_batch_done` must all be positive
  to label the stronger W2-a behavior observed. The tool requires all source
  completions from every rank to establish the old save's end time, and compares
  each event against that same save, not an unrelated transfer.
- Device computation is observed by querying the already-recorded post-forward
  event from a bounded observer queue; no additional device synchronization is
  introduced. A completed new computation before the old save ends is a
  conservative proof of overlap. Missing/late observations yield INCONCLUSIVE;
  allocation alone never counts as device computation.
- The trace captures only a bounded prefix. Missing complete evidence, a dropped
  observation, or absence of reuse opportunities is not a passing result.

These hooks apply only to the bound W2 producer pool. They do not globally patch
vLLM classes. Instance wrappers are removed on shutdown/rebind. They are temporary
until upstream BlockPool offers callbacks for allocation/free observation, at
which point the wrappers should be replaced by those callbacks.

The report labels allocator checks `CHECKED_SOURCE_REFS`, reports W2-a evidence
separately, and uses `executor_return=NOT_REQUIRED`. Output accuracy remains
UNVERIFIED and HTTP timing is outside scope. Older logs without the new counters are not retroactively validated. A
finished request may retain shared prefix blocks legitimately. Pin/unpin balance
does not prove zero HBM allocation, and missing final snapshots do not prove
cleanup completed.

## Experimental W2-b token-range overlap

This first implementation is separately enabled on D; it does not change
existing W2-D deployments by default. Add these extra config fields on D:

```json
{
  "enable_shared_pool_w2": true,
  "consumer_is_to_load": true,
  "shared_pool_w2_token_batch_size": 32,
  "use_layerwise": false,
  "load_async": false
}
```

Also launch D with `--enforce-eager`. Keep `shared_pool_w2_token_batch_size`
absent or zero on P. Zero is the default and preserves D's existing whole-key
load/attention path. Positive values (1–4096) opt into the experimental path.

The installed Mooncake client must provide `batch_get_session_start`,
`batch_get_into_multi_buffer_ranges`, and `batch_get_session_end`. These are
read-lease/range APIs; the feature does not enable layerwise puts or commits.
The existing local Mooncake source has these APIs, but an older installed wheel
may need rebuilding before this mode can initialize.

Currently supported: eager, unquantized FP16/BF16 dense MHA/GQA with matching
P/D TP, ordinary unsegmented token-major or head-major cache pages, and the
standard Ascend, FA or FA3 backend using the base forward hook. Cache padding
between pages is supported. MLA/sparse attention, CP, KVPP, speculation,
quantized caches, ALiBi/sinks, KV sharing, padded head dimensions, cross/encoder
attention, and other custom backends are rejected. This is not yet support for
all single-group model families or all parallel combinations.

A background reader transfers only the requested token ranges directly into
persistent cache pages. Each batch includes the ordinary object's local KV
entries across layers. The current attention layer waits on only the ranges it
consumes, while later ranges can still be transferring. Recomputed current-token
slots are excluded from DMA, preventing a late read from overwriting the current
reshape-and-cache write. Subsequent decode steps return to normal resident-cache
attention. No history eviction after each partial computation is introduced.

The initial attention implementation uses bounded-query FP32 tensor operations
and stable log-sum-exp merging, with causal/sliding-window masks. It allocates
small range intermediates rather than staging the entire cache. It is a
correctness-first eager implementation, not a performance-optimized fused
kernel: range-call overhead, temporary tensors and a page-table CPU copy per
layer may outweigh overlap gains. Benchmarking and further kernel optimization
are required before any speed claim.

The reader is joined before publishing model-runner output, starting another
load, or closing the store. Read failures wake range consumers and drain before
cleanup; incomplete results are never treated as valid. In this experimental
path, a failed range raises an execution error rather than transparently
recovering inside an already-running partial attention. The original zero-batch
path keeps its existing error/recompute behavior.

The report exposes `token_range_batches`, `remote_read_bytes`,
`partial_attention_batches` and `pending_load_batches`, plus bounded D events.
Attention events explicitly say **submit**, not device completion. CPU overlap
is not proof of DMA/compute overlap: `w2b=DEVICE_TRACE_REQUIRED` remains
INCONCLUSIVE until device profiling verifies a same-request, same-layer
intersection and numerical/end-to-end validation passes. The tool does not
fabricate device overlap milliseconds from CPU enqueue timestamps.

Start with a controlled Qwen3-32B run and compare first-layer outputs and greedy
generation against batch size zero. Sweep batch sizes, local/external prefix
mixes, block boundaries and lengths; verify no resident-history reloads over
multiple decode steps, and verify failure/shutdown draining. Capture a short
NPU profiling window to verify actual later-range DMA intersects partial
attention, then compare TTFT/TPOT/throughput without test delays.

## Controlled early-reuse validation

Run concurrent requests with enough allocation pressure that a returned block
can actually be reassigned. For one controlled run only, add
`"shared_pool_w2_test_delay_ms": 5000` to P's extra config. This test hook pauses
once before the second block put of the first multi-block save, after its first
block has completed; it defaults to zero and accepts 0–30000 ms. It is interruptible
at shutdown. Use up to 30000 ms if the next prefill takes longer than the window.
Do not use the delay for performance measurements.

After all requests finish, stop P/D normally and run the summary command above.
Look for all four positive early-reuse fields, zero `early_reuse`/transfer
failures, balanced pins and zero pending refs. The selected event chain should
show the first block's source completion, scheduler notification, free-list
return, allocation and all-rank new computation before the old save's final
source completion. Shared prefix references can legitimately prevent reuse;
use a workload that can release at least one nonshared block.

Remove the test delay for greedy-output comparison and TTFT/TPOT/throughput
measurements. No hardware result is claimed by this implementation. Local tests
exercise the actual sender/scheduler/allocator methods with a fake store and
in-memory datagram transport because the managed host forbids socket binding;
real IPC and NPU event querying still require validation on the deployment host.

Run in the supported project test environment:

```bash
pytest tests/ut/distributed/ascend_store/test_shared_pool_w2.py \
       tests/ut/distributed/ascend_store/test_kv_transfer.py \
       tests/ut/distributed/ascend_store/test_token_pipeline.py \
       tests/ut/distributed/ascend_store/test_pool_worker.py \
       tests/ut/distributed/ascend_store/test_pool_scheduler.py
```
