# Ordinary-path shared-pool W2

This opt-in mode uses `AscendStoreConnector` with the Mooncake backend and one
KV cache group. It does not enable layerwise transfer or alter attention math.
Qwen3-32B is the first hardware validation target; other single-group layouts
and parallel configurations require separate validation.

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
exclusive-ownership and step-fence protection rather than receiving extra pins.

The ordinary save callback waits for the queued batch fence and checks each
job's result. Failed jobs abort the handoff with an error; they cannot advance
successful save progress. Source-reading completion is separate from publication
success. This callback barrier may make all prefill blocks available together
at request completion; no staggered release timing is promised. A put returning
is the granularity of the observation, not a device profiling timestamp.

D loads missing historical KV on admission and retains it across decode steps.
New KV stays on D. No per-step host writeback or history reload is introduced.
The mode does not reduce D's single-request KV peak or destroy preallocated KV
tensors. Recovery after preemption uses the existing load/recompute path.

One-block puts and the publication barrier can increase TTFT. Measure against
an otherwise identical run with the option absent before making performance
claims. The code does not change Mooncake or the vLLM core allocator.

## Five-line validation report

Cumulative `SHARED_POOL_W2` JSON records are emitted at most once per minute
per process, plus a final record during orderly shutdown. They do not print
request keys, prompts or block lists. Existing application logs are unchanged.
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

Only send `w2-summary.txt` (five lines) for initial diagnosis. For multi-rank logs,
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
- `return_checked` counts save jobs whose completion was checked at worker output
  construction and acknowledged by every worker in the executor output received
  by the scheduler. `return_before_save` records a premature/failed save result
  at that boundary. This checks save ordering at the result boundary, not HTTP
  response timing or independent network readback.
- `return_pending` must drain to zero, and every scheduled save job must have a
  checked return. Zero checks or no actual allocator reuse is INCONCLUSIVE.

These hooks apply only to the bound W2 producer pool. They do not globally patch
vLLM classes. Instance wrappers are removed on shutdown/rebind. They are temporary
until upstream BlockPool offers callbacks for allocation/free observation, at
which point the wrappers should be replaced by those callbacks.

The report labels covered allocator and result checks `CHECKED_SOURCE_REFS` and
`CHECKED_SAVE_JOBS`; output accuracy remains UNVERIFIED and HTTP timing is outside
scope. Older logs without the new counters are not retroactively validated. A
finished request may retain shared prefix blocks legitimately. Pin/unpin balance
does not prove zero HBM allocation, and missing final snapshots do not prove
cleanup completed.

Run the ownership/failure tests, compare greedy outputs on Ascend, verify the
P-to-D handoff and measure TTFT/TPOT/throughput before accepting a deployment.

## Local validation status (2026-11-02)

The shared-pool and existing transfer test modules passed 76 CPU mock tests;
the standalone summary-tool passed two tests.
Additional targeted ordinary-worker/TP-mismatch regressions passed. Tests exercise
real connector methods against mocked backend/runtime dependencies; they are not
NPU or network-transfer validation. On the development Mac, a temporary host-only
runner supplied missing Prometheus, Ascend configuration/hardware and wire-protocol
imports. No test shim was added to production code.

The broader worker suite could not be completed with the host mocks (including a
missing mock `torch.bfloat16`). `bash format.sh ci` could not run because
`pre-commit` is unavailable. Python syntax and `git diff --check` passed.

Before deployment, run these modules in the supported project test environment:

```bash
pytest tests/ut/distributed/ascend_store/test_shared_pool_w2.py \
       tests/ut/distributed/ascend_store/test_kv_transfer.py \
       tests/ut/distributed/ascend_store/test_pool_worker.py \
       tests/ut/distributed/ascend_store/test_pool_scheduler.py
```

Then run Ascend greedy-output and performance comparisons. No hardware result is
claimed by this implementation.
