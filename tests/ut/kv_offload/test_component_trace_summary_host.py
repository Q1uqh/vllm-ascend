# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU acceptance checks for the log evidence evaluator, without device imports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "component_summary", ROOT / "tests/e2e/cases/features/kv_pool/tools/summarize_component_offload.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class TraceSummaryTests(unittest.TestCase):
    def event(self, kind, **fields):
        return dict(event=kind, host="P", pid=1, trace_session="run", monotonic_ns=0, **fields)

    def context(self, layer=2):
        return dict(
            layer_id=layer,
            step=2,
            requests=[dict(request="R", incarnation="R:1", endpoint=8192, restore_endpoint=4096)],
            slots=[dict(slot="0:100")],
        )

    def trace(self):
        old = self.context()
        new = self.context(6)
        return [
            self.event("store_source_start", **old),
            self.event("pd_reader_start", **old),
            self.event("store_source_done", **old),
            self.event("pd_reader_done", **old),
            self.event("predecessor_device_done", predecessor=2, **new),
            self.event("overwrite_begin", predecessor=2, **new),
            self.event("load_done", **new),
            self.event("compute_ready", has_load=True, **new),
        ]

    def summarize(self, events, prefix=""):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            path.write_text(
                prefix
                + "".join(
                    SUMMARY.MARKER + " " + json.dumps(dict(event, monotonic_ns=index + 1)) + "\n"
                    for index, event in enumerate(events)
                )
            )
            return "\n".join(SUMMARY.summarize([path]))

    def test_observed_order_with_duplicate_alias_views(self):
        events = self.trace()
        for event in events:
            event["slots"].append(dict(slot="0:100"))
        result = self.summarize(events)
        self.assertIn("RESULT OBSERVED_ORDER_OK", result)
        self.assertIn("DEVICE_D2D UNVERIFIED", result)

    def test_overwrite_while_pd_reader_is_active(self):
        events = self.trace()
        events.insert(6, events.pop(3))
        self.assertIn("overwrite with active readers", self.summarize(events))
        self.assertIn("RESULT FAILED", self.summarize(events))

    def test_compute_before_load(self):
        events = self.trace()
        events[-2], events[-1] = events[-1], events[-2]
        self.assertIn("compute before load completion", self.summarize(events))

    def test_truncated_reader_completion_is_inconclusive(self):
        result = self.summarize(self.trace()[:3])
        self.assertIn("RESULT INCONCLUSIVE", result)
        self.assertIn("no successful completion", result)

    def test_failed_reader_is_not_released(self):
        events = self.trace()
        events[3]["error"] = "WRITE failed"
        result = self.summarize(events)
        self.assertIn("RESULT FAILED", result)
        self.assertIn("overwrite with active readers", result)

    def test_other_host_same_pid_does_not_release_reader(self):
        events = self.trace()
        events[3]["host"] = "other"
        self.assertIn("overwrite with active readers", self.summarize(events))

    def test_warning_does_not_become_runtime_error(self):
        result = self.summarize(self.trace(), "worker WARNING failed ulimit with error current limit\n")
        self.assertIn("RESULT OBSERVED_ORDER_OK", result)
        self.assertIn("WARNING worker.log", result)

    def test_legacy_trace_is_inconclusive_and_bounded(self):
        events = self.trace()
        for event in events:
            del event["trace_session"]
        result = self.summarize(events)
        self.assertIn("RESULT INCONCLUSIVE", result)
        self.assertLessEqual(len(result.splitlines()), 280)

    def test_registered_device_addresses_match_but_d2d_stays_unverified(self):
        events = self.trace()
        source = self.event(
            "pd_layout", component="attn", address=[100], groups=[0], devices=["npu:0"], te_host="p", te_rpc_port=1
        )
        destination = dict(
            source, host="D", trace_session="decoder", address=[200], devices=["npu:4"], te_host="d", te_rpc_port=2
        )
        plan = self.event(
            "pd_component_plan",
            component="attn",
            source_bases=[100],
            destination_bases=[200],
            destination_session="d:2",
            group=0,
        )
        events.extend([source, destination, plan])
        result = self.summarize(events)
        self.assertIn("matched_component_plans=1/1", result)
        self.assertIn("DEVICE_D2D UNVERIFIED", result)
        plan["destination_bases"] = [300]
        self.assertIn("RESULT FAILED", self.summarize(events))

    def test_diagnostic_capture_failure_is_reported(self):
        self.assertIn(
            "RESULT FAILED", self.summarize(self.trace() + [self.event("trace_error", detail="capture failed")])
        )


if __name__ == "__main__":
    unittest.main()
