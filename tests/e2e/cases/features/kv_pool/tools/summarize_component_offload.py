# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce P/D lifecycle logs to at most 280 lines; never infer device D2D from logs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MARKER = "[KVPOOL_RANGE_DEBUG]"
MAX_LINES = 280


def summarize(paths):
    counts = Counter()
    samples = defaultdict(list)
    errors = []
    pending = defaultdict(set)
    slots_by_layer = {}
    violations = 0
    malformed = 0
    for path in paths:
        with Path(path).open(errors="replace") as stream:
            for line in stream:
                if MARKER not in line:
                    if (
                        any(word in line for word in ("Traceback", "ERROR", "RuntimeError", "ValueError"))
                        and len(errors) < 40
                    ):
                        errors.append(f"ERROR {Path(path).name}: {line.strip()[:800]}")
                    continue
                try:
                    event = json.loads(line.split(MARKER, 1)[1].strip())
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                kind = event.get("event", "unknown")
                counts[kind] += 1
                # PIDs can repeat on different hosts. Paths distinguish workers
                # here; concatenate ranks from the same host into one P log.
                identity = (str(path), event.get("pid"))
                layer = event.get("layer_id")
                if kind == "pd_layout":
                    for index in event.get("layers", []):
                        slots_by_layer[identity, index] = set(event.get("slots", []))
                elif kind == "pd_reader_start":
                    pending[identity].update(event.get("slots", []))
                elif kind == "pd_reader_done":
                    if not event.get("error"):
                        pending[identity].difference_update(event.get("slots", []))
                elif kind == "reuse_ready":
                    if pending[identity] & slots_by_layer.get((identity, layer), set()):
                        violations += 1
                failed = bool(event.get("error")) or any(
                    result != 0 if kind == "commit" else result < 0 for result in event.get("results", [])
                )
                failed |= kind == "pd_write" and event.get("result", 0) < 0
                if failed and len(errors) < 40:
                    errors.append("FAILED " + json.dumps(event, ensure_ascii=False)[:1400])
                limit = 30 if kind in ("layout", "pd_layout", "snapshot_plan") else 12
                if len(samples[kind]) < limit:
                    samples[kind].append(event)
    lines = [
        "COUNTS " + json.dumps(dict(counts), sort_keys=True),
        f"CHECK malformed={malformed} observed_pd_early_overwrite={violations}",
    ]
    lines.extend(errors)
    for kind in sorted(samples):
        for event in samples[kind]:
            lines.append("SAMPLE " + json.dumps(event, ensure_ascii=False, separators=(",", ":"))[:1800])
    lines = lines[: MAX_LINES - 3]
    lines.extend(
        [
            "OUTPUT_ACCURACY UNVERIFIED: attach deterministic baseline comparison result",
            "DEVICE_D2D UNVERIFIED: attach device profiling evidence; WRITE logs alone are insufficient",
            "RESULT "
            + ("FAILED" if errors or malformed or violations else "EVIDENCE_ONLY")
            + ": counts and sampled events do not establish complete all-rank coverage",
        ]
    )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    print("\n".join(summarize(args.logs)))


if __name__ == "__main__":
    main()
