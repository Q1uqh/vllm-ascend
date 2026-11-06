# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Summarize Mooncake KV-pool logs for remote diagnosis."""

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

_MARKER = "[KVPOOL_RANGE_DEBUG]"
_ERROR = re.compile(r"\b(ERROR|CRITICAL|Traceback|failed|rejected|short transfer)\b", re.IGNORECASE)
_CONFIG = re.compile(
    r"layerwise config|layerwise group|DSV4 temporal|DSV4_ACCEPT|num_gpu_blocks|KV cache size", re.I
)
_SLOT_TRACE = re.compile(r"DSV4_SLOT_TRACE .*expected_edges=(\d+)")
_SLOT_EDGE = re.compile(r"DSV4_SLOT_REUSE path=(\w+) predecessor=(\d+) layer_id=(\d+)")
_SLOT_PLAN = re.compile(r"DSV4 temporal slots .*predecessors=(\{[^}]*\})")
_SLOT_PAIR = re.compile(r"(\d+):\s*(\d+)")
_WORKER = re.compile(r"\(([^()]*\bpid=\d+)\)")
_EVIDENCE_EVENTS = {
    "snapshot_prepare",
    "snapshot_committed",
    "handoff_bulk_h2d",
    "handoff_read_complete",
    "snapshot_cleanup",
}


def summarize(paths: list[Path]) -> list[str]:
    events: Counter[tuple[str, str]] = Counter()
    byte_counts: Counter[str] = Counter()
    copied_counts: Counter[str] = Counter()
    max_fence_wait_ms = 0.0
    max_bulk_h2d_ms = 0.0
    errors: deque[str] = deque(maxlen=100)
    config: deque[str] = deque(maxlen=50)
    failures: deque[str] = deque(maxlen=70)
    evidence: deque[str] = deque(maxlen=30)
    slot_expected: dict[str, int] = {}
    slot_plans: dict[str, set[tuple[int, int]]] = {}
    slot_edges: dict[str, set[tuple[int, int]]] = {}
    slot_paths: dict[str, set[str]] = {}
    malformed = 0
    error_count = 0
    failure_count = 0
    for path in paths:
        with path.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if _MARKER not in line:
                    if "DSV4_SLOT_" in line and (worker := _WORKER.search(line)):
                        identity = f"{path.name} {worker[1]}"
                        if match := _SLOT_TRACE.search(line):
                            slot_expected[identity] = int(match[1])
                        elif match := _SLOT_EDGE.search(line):
                            slot_edges.setdefault(identity, set()).add((int(match[2]), int(match[3])))
                            slot_paths.setdefault(identity, set()).add(match[1])
                    elif (plan := _SLOT_PLAN.search(line)) and (worker := _WORKER.search(line)):
                        identity = f"{path.name} {worker[1]}"
                        slot_plans[identity] = {(int(prev), int(layer)) for layer, prev in _SLOT_PAIR.findall(plan[1])}
                        config.append(f"{path.name}:{number} {line.strip()[:600]}")
                    elif _ERROR.search(line):
                        error_count += 1
                        errors.append(f"{path.name}:{number} {line.strip()[:600]}")
                    elif _CONFIG.search(line):
                        config.append(f"{path.name}:{number} {line.strip()[:600]}")
                    continue
                payload = line.split(_MARKER, 1)[1].strip()
                try:
                    record, _ = json.JSONDecoder().raw_decode(payload)
                    event = str(record.get("event", "unknown"))
                    direction = str(record.get("direction", "-"))
                    results = record.get("results", [])
                    requested = record.get("requested_bytes", [])
                    if not isinstance(requested, list):
                        requested = [requested]
                    requested_bytes = sum(requested)
                    if event == "range":
                        copied_counts[direction] += sum(max(0, result) for result in results)
                        failed = len(results) != len(requested) or any(
                            result != expected for result, expected in zip(results, requested)
                        )
                    elif event == "handoff_bulk_h2d":
                        copied = int(record["copied_bytes"])
                        requested_bytes = copied
                        copied_counts[direction] += copied
                        max_bulk_h2d_ms = max(max_bulk_h2d_ms, float(record["elapsed_ms"]))
                        failed = False
                    else:
                        if event == "overwrite_fence":
                            max_fence_wait_ms = max(max_fence_wait_ms, float(record.get("wait_ms", 0)))
                        failed = any(
                            result != 0 if event in ("commit", "snapshot_remove") else result < 0 for result in results
                        )
                except (AttributeError, KeyError, TypeError, ValueError):
                    malformed += 1
                    continue
                events[event, direction] += 1
                if event in _EVIDENCE_EVENTS:
                    evidence.append(f"{path.name}:{number} {payload[:600]}")
                byte_counts[direction] += requested_bytes
                if failed:
                    failure_count += 1
                    failures.append(f"{path.name}:{number} {payload[:600]}")

    slot_summary = []
    for worker in sorted(slot_expected.keys() | slot_edges.keys())[-18:]:
        edges = slot_edges.get(worker, set())
        plan = slot_plans.get(worker)
        comparison = ""
        if plan is not None:
            comparison = f" missing={len(plan - edges)} unexpected={len(edges - plan)}"
        slot_summary.append(
            f"slot_reuse {worker} observed={len(edges)} expected={slot_expected.get(worker, '?')}{comparison} "
            f"paths={','.join(sorted(slot_paths.get(worker, set())))} "
            f"edges={','.join(f'{prev}->{layer}' for prev, layer in sorted(edges))}"
        )
    return [
        f"files={len(paths)} events={sum(events.values())} malformed={malformed} "
        f"transfer_failures={failure_count} error_lines={error_count}",
        f"retained_config={len(config)} retained_errors={len(errors)} retained_transfer_failures={len(failures)} "
        f"max_overwrite_wait_ms={max_fence_wait_ms:.3f} max_bulk_h2d_ms={max_bulk_h2d_ms:.3f}",
        *(f"event={event} direction={direction} count={count}" for (event, direction), count in events.most_common(20)),
        *(
            f"direction={direction} requested_bytes={total} copied_bytes={copied_counts[direction]}"
            for direction, total in byte_counts.most_common(10)
        ),
        *slot_summary,
        *evidence,
        *config,
        *errors,
        *failures,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    for line in summarize(args.logs):
        print(line)


if __name__ == "__main__":
    main()
