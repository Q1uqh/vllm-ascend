# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlate complete P/D traces; output <=280 lines, device profiling is separate."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MARKER = "[KVPOOL_RANGE_DEBUG]"
MAX_LINES = 280


def summarize(paths):
    counts = Counter()
    range_totals = defaultdict(Counter)
    events, errors, warnings = [], [], []
    malformed = 0
    legacy = 0
    for path in paths:
        with Path(path).open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if MARKER not in line:
                    # A warning quoting the word 'error' is not a runtime failure.
                    if "WARNING" in line and len(warnings) < 8:
                        warnings.append(f"WARNING {Path(path).name}:{number} {line.strip()[:500]}")
                    elif (" ERROR " in line or "Traceback (most recent call last)" in line) and len(errors) < 25:
                        errors.append(f"ERROR {Path(path).name}:{number} {line.strip()[:700]}")
                    continue
                try:
                    event = json.loads(line.split(MARKER, 1)[1])
                    if not isinstance(event, dict):
                        raise ValueError("Expected an event object")
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                kind = event.get("event", "unknown")
                counts[kind] += 1
                if kind == "range":
                    totals = range_totals[event.get("direction", "unknown")]
                    totals["count"] += 1
                    totals["requested_bytes"] += sum(event.get("requested_bytes", []))
                    totals["copied_bytes"] += sum(max(0, result) for result in event.get("results", []))
                identity = (event.get("host"), event.get("pid"), event.get("trace_session"))
                if not all(value is not None for value in identity) or "monotonic_ns" not in event:
                    legacy += 1
                    identity = (str(path), event.get("pid"), "legacy")
                events.append((identity, event, f"{Path(path).name}:{number}"))
                failed = (
                    kind == "trace_error"
                    or bool(event.get("error"))
                    or any(result != 0 if kind == "commit" else result < 0 for result in event.get("results", []))
                )
                failed |= kind == "pd_write" and event.get("result", 0) < 0
                if failed and len(errors) < 25:
                    errors.append("FAILED " + json.dumps(event, ensure_ascii=False)[:1000])

    # Pre-index D registrations: file order and clocks on different hosts do
    # not establish causality. Address matching does not compare timestamps.
    local_layouts, remote_layouts = {}, {}
    for identity, event, _ in events:
        if event.get("event") == "pd_layout":
            local_layouts[identity, event.get("component")] = event
            host, port = event.get("te_host"), event.get("te_rpc_port")
            if host is not None and port is not None:
                remote_layouts[f"{host}:{port}", event.get("component")] = event

    pending = defaultdict(dict)
    load_done = set()
    device_done = set()
    gaps, violations, samples = [], [], []
    checked_overwrites = checked_computes = matched_addresses = address_plans = 0
    max_wait_ns = 0
    identities = set()
    cross_chunk = set()
    observed_slots = defaultdict(set)
    # Namespace every comparison by host/process/session; never compare clocks
    # from different machines to infer a transfer dependency.
    events.sort(key=lambda row: (str(row[0]), row[1].get("monotonic_ns", 0)))
    for identity, event, location in events:
        kind, layer = event.get("event"), event.get("layer_id")
        context = event.get("trace_context") or event
        requests = context.get("requests", [])
        requests = requests if isinstance(requests, list) and all(isinstance(r, dict) for r in requests) else []
        slots = context.get("slots", [])
        slots = list(dict.fromkeys(s["slot"] for s in slots if isinstance(s, dict) and "slot" in s))
        token = (identity, context.get("step"), layer)
        for request in requests:
            identities.add((identity, request.get("request"), request.get("incarnation"), request.get("endpoint")))
            if (request.get("restore_endpoint") or 0) > 0:
                cross_chunk.add((identity, request.get("request"), request.get("endpoint")))
        if kind in ("store_source_start", "pd_reader_start"):
            reader = "store" if kind == "store_source_start" else "pd"
            if not slots or not requests:
                gaps.append(f"{location} {kind}: missing request/slot context")
            for slot in slots:
                key = (identity, slot)
                # Preserve errors and overlapping readers rather than replacing
                # them with a later start event for the same address.
                operation = (
                    reader,
                    layer,
                    context.get("step"),
                    tuple((r.get("request"), r.get("endpoint")) for r in requests),
                )
                pending[key][operation] = location
        elif kind in ("store_source_done", "pd_reader_done"):
            reader = "store" if kind == "store_source_done" else "pd"
            if not event.get("error"):
                for slot in slots:
                    operation = (
                        reader,
                        layer,
                        context.get("step"),
                        tuple((r.get("request"), r.get("endpoint")) for r in requests),
                    )
                    if operation not in pending[identity, slot]:
                        gaps.append(f"{location} {kind}: missing matching start")
                    pending[identity, slot].pop(operation, None)
        elif kind == "predecessor_device_done":
            device_done.add(token)
        elif kind == "overwrite_begin":
            if not slots or not requests or context.get("step") is None:
                gaps.append(f"{location}: overwrite missing step/request/slot context")
                continue
            checked_overwrites += 1
            if event.get("predecessor") is not None and token not in device_done:
                violations.append(f"{location}: overwrite before predecessor device completion, layer={layer}")
            for slot in slots:
                observed_slots[identity, slot].add(layer)
                if pending[identity, slot]:
                    violations.append(
                        f"{location}: overwrite with active readers slot={slot}: "
                        + str(list(pending[identity, slot].values())[:4])
                    )
            if len(samples) < 24:
                samples.append(
                    dict(
                        location=location,
                        event=kind,
                        layer=layer,
                        step=context.get("step"),
                        requests=requests,
                        slots=slots,
                        predecessor=event.get("predecessor"),
                    )
                )
        elif kind == "load_done":
            load_done.add(token)
        elif kind == "compute_ready":
            if context.get("has_load"):
                checked_computes += 1
                if token not in load_done:
                    violations.append(f"{location}: compute before load completion, layer={layer}")
        elif kind == "reuse_ready":
            max_wait_ns = max(max_wait_ns, event.get("wait_ns", 0))
        elif kind == "pd_component_plan":
            address_plans += 1
            source = local_layouts.get((identity, event.get("component")))
            destination = remote_layouts.get((event.get("destination_session"), event.get("component")))
            if source is None or destination is None:
                gaps.append(f"{location}: missing P/D component registration")
                continue
            if (
                source.get("address") != event.get("source_bases")
                or destination.get("address") != event.get("destination_bases")
                or event.get("group") not in destination.get("groups", [])
            ):
                violations.append(f"{location}: transfer address/group differs from cache registration")
            elif (
                not source.get("devices")
                or not destination.get("devices")
                or not all(
                    str(device).startswith(("npu", "privateuseone"))
                    for device in source["devices"] + destination["devices"]
                )
            ):
                gaps.append(f"{location}: device memory registration not established")
            else:
                matched_addresses += 1

    outstanding = sum(len(readers) for readers in pending.values())
    if outstanding:
        gaps.append(f"{outstanding} reader operations have no successful completion (incomplete log or failure)")
    if legacy:
        gaps.append(f"{legacy} legacy events lack host/process/session/time correlation")
    if not checked_overwrites or not checked_computes or not cross_chunk:
        gaps.append("missing overwrite/load-compute/cross-chunk coverage")
    if not counts["pd_reader_start"] or not counts["store_source_start"]:
        gaps.append("missing PD or Store reader coverage")
    status = "FAILED" if errors or malformed or violations else "INCONCLUSIVE" if gaps else "OBSERVED_ORDER_OK"
    lines = [
        "COUNTS " + json.dumps(dict(counts), sort_keys=True),
        f"TIMING status={status} checked_overwrites={checked_overwrites} checked_load_compute={checked_computes} "
        f"cross_chunk_endpoints={len(cross_chunk)} outstanding_readers={outstanding} "
        f"max_wait_ms={max_wait_ns / 1_000_000:.3f}",
        f"ADDRESSES matched_component_plans={matched_addresses}/{address_plans} "
        "(cache registration evidence, not proof of physical D2D)",
    ]
    lines.extend(
        f"RANGE direction={direction} count={totals['count']} "
        f"requested_bytes={totals['requested_bytes']} copied_bytes={totals['copied_bytes']}"
        for direction, totals in sorted(range_totals.items())
    )
    lines.append(f"PARSE files={len(paths)} events={len(events)} malformed={malformed} legacy={legacy}")
    lines.extend(errors)
    lines.extend("VIOLATION " + row for row in violations[:30])
    lines.extend("GAP " + row for row in gaps[:20])
    lines.extend(warnings)
    lines.extend(
        "SLOT " + json.dumps(dict(host=key[0][0], pid=key[0][1], slot=key[1], layers=sorted(value)))
        for key, value in list(observed_slots.items())[:30]
        if len(value) > 1
    )
    lines.extend("SAMPLE " + json.dumps(event, ensure_ascii=False, separators=(",", ":"))[:1800] for event in samples)
    lines = lines[: MAX_LINES - 4]
    lines.extend(
        [
            "OUTPUT_ACCURACY UNVERIFIED: attach deterministic baseline comparison result",
            "DELAY_INJECTION UNVERIFIED: normal waits do not prove independently delayed PUT/WRITE tests",
            "DEVICE_D2D UNVERIFIED: attach device profiling evidence; WRITE logs alone are insufficient",
            f"RESULT {status}: applies to observed trace only, not all-rank or device acceptance",
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
