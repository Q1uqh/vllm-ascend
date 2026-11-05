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
_CONFIG = re.compile(r"layerwise config|layerwise group|shared buffers require|num_gpu_blocks|KV cache size", re.I)


def summarize(paths: list[Path]) -> list[str]:
    events: Counter[tuple[str, str]] = Counter()
    byte_counts: Counter[str] = Counter()
    errors: deque[str] = deque(maxlen=120)
    config: deque[str] = deque(maxlen=60)
    failures: deque[str] = deque(maxlen=80)
    malformed = 0
    error_count = 0
    failure_count = 0
    for path in paths:
        with path.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if _MARKER not in line:
                    if _ERROR.search(line):
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
                        failed = len(results) != len(requested) or any(
                            result != expected for result, expected in zip(results, requested)
                        )
                    else:
                        failed = any(result != 0 if event == "commit" else result < 0 for result in results)
                except (AttributeError, TypeError, ValueError):
                    malformed += 1
                    continue
                events[event, direction] += 1
                byte_counts[direction] += requested_bytes
                if failed:
                    failure_count += 1
                    failures.append(f"{path.name}:{number} {payload[:600]}")

    return [
        f"files={len(paths)} events={sum(events.values())} malformed={malformed} "
        f"transfer_failures={failure_count} error_lines={error_count}",
        f"retained_config={len(config)} retained_errors={len(errors)} retained_transfer_failures={len(failures)}",
        *(f"event={event} direction={direction} count={count}" for (event, direction), count in events.most_common(20)),
        *(f"direction={direction} requested_bytes={total}" for direction, total in byte_counts.most_common(10)),
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
