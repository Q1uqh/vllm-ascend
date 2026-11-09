# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Failure-only PUT diagnostics exercise actual session and validation functions."""

import json
import threading
import unittest
from collections import Counter
from numbers import Integral
from types import SimpleNamespace
from unittest.mock import MagicMock

from test_dsv4_shared_components_host import STORE, definitions


class PutFailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        validation = definitions(
            STORE + "backend/base.py",
            dict(Integral=Integral, BatchResultShapeError=ValueError),
            {"require_aligned_batch_results"},
        )
        self.ns = definitions(
            STORE + "backend/mooncake_offload.py",
            dict(
                logger=self.logger,
                json=json,
                Counter=Counter,
                require_aligned_batch_results=validation["require_aligned_batch_results"],
            ),
            {"_open_puts", "_log_put_failure"},
        )
        self.view = SimpleNamespace(
            req_id="R", layerwise_request_id="generation", target_token_len=8192, save_block_keys=[None, None]
        )
        self.slots = [(self.view, "ok", 0, 100), (self.view, "bad", 1, 200)]
        self.worker = SimpleNamespace(
            _filter_pool_existing_keys=MagicMock(side_effect=[["ok", "bad"], ["bad"]]),
            _layerwise_key_batches=lambda keys: [keys],
            m_store=SimpleNamespace(batch_put_start=MagicMock(return_value=[0, -7])),
            _put_started_keys_lock=threading.Lock(),
            _put_started_keys=set(),
            _queue_layerwise_revoke_keys=MagicMock(),
        )

    def open(self):
        return self.ns["_open_puts"](self.worker, self.slots, {3: [self.view]})

    def payload(self):
        return json.loads(self.logger.error.call_args.args[1])

    def test_failure_has_code_owner_group_and_cleanup_without_debug(self):
        with self.assertRaisesRegex(RuntimeError, "required PUT session"):
            self.open()
        payload = self.payload()
        self.assertEqual(payload["code_counts"], {"0": 1, "-7": 1})
        self.assertEqual(payload["unreadable_keys"], 1)
        self.assertEqual(payload["opened_to_revoke"], 1)
        sample = payload["samples"][0]
        self.assertEqual((sample["key"], sample["size"], sample["code"]), ("bad", 200, -7))
        self.assertEqual(sample["owners"][0]["group"], 3)
        self.assertEqual(sample["owners"][0]["endpoint"], 8192)
        self.worker._queue_layerwise_revoke_keys.assert_called_once_with(["ok"])

    def test_success_is_silent(self):
        self.worker.m_store.batch_put_start.return_value = [0, 0]
        self.open()
        self.logger.error.assert_not_called()
        self.assertEqual(self.view.save_block_keys, ["ok", "bad"])

    def test_competing_writer_committed_is_silent_and_not_owned(self):
        self.worker._filter_pool_existing_keys.side_effect = [["ok", "bad"], []]
        self.open()
        self.logger.error.assert_not_called()
        self.assertEqual(self.view.save_block_keys, ["ok", None])

    def test_logger_failure_does_not_replace_original_or_skip_cleanup(self):
        self.logger.error.side_effect = RuntimeError("logger broken")
        with self.assertRaisesRegex(RuntimeError, "required PUT session"):
            self.open()
        self.worker._queue_layerwise_revoke_keys.assert_called_once_with(["ok"])

    def test_rpc_exception_is_reported(self):
        self.worker.m_store.batch_put_start.side_effect = TimeoutError("rpc timed out")
        with self.assertRaises(TimeoutError):
            self.open()
        self.assertEqual(self.payload()["phase"], "batch_put_start")
        self.assertIn("rpc timed out", self.payload()["error"])

    def test_malformed_return_is_reported(self):
        self.worker.m_store.batch_put_start.return_value = None
        with self.assertRaises(ValueError):
            self.open()
        self.assertEqual(self.payload()["result_type"], "NoneType")


if __name__ == "__main__":
    unittest.main()
