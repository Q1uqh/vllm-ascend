# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""CPU checks for range offsets, asynchronous readiness and stable state merge."""

import ctypes
import threading
import unittest
from unittest.mock import patch

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.token_pipeline import (
    TokenLoadPipeline,
    dense_token_ranges,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_pool_w2 import SharedPoolObservation
from vllm_ascend.attention import token_overlap


class TestTokenPipeline(unittest.TestCase):
    def observation(self):
        return SharedPoolObservation("worker", "kv_consumer", 0, 1)

    def test_partial_object_reads_preserve_unselected_tokens_and_page_padding(self):
        for head_major in (False, True):
            # One host object concatenates K/V for two layers. Device blocks
            # have padding, which is not part of the host object.
            objects, destinations, entries = [], [], []
            for layer in range(4):
                shape = (2, 8, 3) if head_major else (8, 2, 3)
                source = np.arange(48, dtype=np.int16).reshape(shape) + layer * 100
                objects.append(source.tobytes())
                destination = np.full((3, 64), -1, dtype=np.int16)
                destinations.append(destination)
                entries.append((destination.ctypes.data, destination.strides[0], 2, 3, 2, head_major))
            obj = b"".join(objects)
            reads = list(dense_token_ranges("key", 1, 1, 7, 3, entries, 8))
            self.assertEqual([(r.start, r.end) for r in reads], [(1, 4), (4, 7)])
            for read in reads:
                for address, size, offset in zip(read.addresses, read.sizes, read.offsets):
                    ctypes.memmove(address, obj[offset:offset+size], size)
            for layer, destination in enumerate(destinations):
                expected = np.frombuffer(objects[layer], dtype=np.int16).reshape(shape)
                actual = destination[1, :48].reshape(shape)
                if head_major:
                    np.testing.assert_array_equal(actual[:, 1:7], expected[:, 1:7])
                    self.assertTrue(np.all(actual[:, [0, 7]] == -1))
                else:
                    np.testing.assert_array_equal(actual[1:7], expected[1:7])
                    self.assertTrue(np.all(actual[[0, 7]] == -1))
                self.assertTrue(np.all(destination[0] == -1))
                self.assertTrue(np.all(destination[1, 48:] == -1))

    def test_ready_first_batch_is_usable_while_second_read_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        reads = list(dense_token_ranges("key", 1, 0, 8, 4, [(1000, 128, 2, 4, 2, False)], 8, "r"))
        class Backend:
            calls = 0
            closed = False
            def batch_get_start(self, keys):
                return [0]
            def batch_copy_get(self, keys, addresses, sizes, offsets):
                self.calls += 1
                if self.calls == 2:
                    entered.set()
                    if not release.wait(timeout=3):
                        raise RuntimeError("Test reader timed out")
                return [sum(row) for row in sizes]
            def batch_get_end(self, keys):
                self.closed = True
                return 0
        backend = Backend()
        pipeline = TokenLoadPipeline(backend, reads, self.observation(), lambda: None)
        try:
            self.assertTrue(entered.wait(timeout=1))
            pipeline.wait(1, 0, 4)
            self.assertFalse(reads[1].ready.is_set())
            self.assertFalse(pipeline.finished.is_set())
        finally:
            release.set()
            pipeline.finish()
        self.assertTrue(backend.closed)

    def test_failure_wakes_consumer_and_closes_session_without_readying_bad_range(self):
        reads = list(dense_token_ranges("key", 1, 0, 8, 4, [(1000, 128, 2, 4, 2, False)], 8))
        from unittest.mock import MagicMock
        backend = MagicMock()
        backend.batch_get_start.return_value = [0]
        backend.batch_copy_get.return_value = [-1]
        backend.batch_get_end.return_value = 0
        pipeline = TokenLoadPipeline(backend, reads, self.observation(), lambda: None)
        with self.assertRaisesRegex(RuntimeError, "incomplete KV|loading failed"):
            pipeline.wait(1, 0, 4)
        with self.assertRaisesRegex(RuntimeError, "no partial result"):
            pipeline.finish()
        self.assertFalse(reads[0].ready.is_set())
        backend.batch_get_end.assert_called_once_with(["key"])

    def test_normalized_partial_outputs_merge_like_full_attention_even_with_empty_masks(self):
        rng = np.random.default_rng(7)
        scores = rng.normal(size=(4, 3, 11)) * 15
        values = rng.normal(size=(4, 11, 5))
        scores[:, 0, :4] = -np.inf  # first partition is fully masked for a query
        maximum = scores.max(-1, keepdims=True)
        weights = np.exp(scores - maximum)
        expected = weights @ values / weights.sum(-1, keepdims=True)
        for size in (1, 2, 4, 9):
            output = np.zeros((4, 3, 5))
            lse = np.full((4, 3, 1), -np.inf)
            with np.errstate(invalid="ignore"), patch.object(token_overlap, "torch", np):
                for first in range(0, 11, size):
                    part_scores = scores[..., first:first+size]
                    part_lse = np.logaddexp.reduce(part_scores, axis=-1, keepdims=True)
                    safe = np.where(np.isfinite(part_lse), part_lse, 0)
                    part_output = np.exp(part_scores - safe) @ values[:, first:first+size]
                    output, lse = token_overlap.merge_attention_states(output, lse, part_output, part_lse)
            np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_worker_launches_token_pipeline_instead_of_whole_key_get(self):
        from unittest.mock import MagicMock
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            AscendConnectorMetadata, ChunkedTokenDatabase, KeyMetadata, LoadSpec, ReqMeta,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker._w2_token_pipeline = None
        worker._w2_token_generation = 0
        worker.w2_token_batch_size = 4
        worker._w2_range_entries = [(1000, 16, 1, 1, 1, False)]
        worker._w2_observation = self.observation()
        worker.group_uses_align_state = [False]
        worker.cache_transfer_granularity = 16
        worker.grouped_block_size = [16]
        worker.block_size = 16
        worker.load_async = worker.tp_mismatch = False
        worker.token_database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None)
        worker.token_database.set_group_buffers({0: [1000]}, {0: [16]}, {0: [1]}, group_num_layers={0: 1})
        worker.m_store = MagicMock()
        worker.m_store.batch_get_start.side_effect = lambda keys: [0] * len(keys)
        worker.m_store.batch_copy_get.side_effect = lambda keys, addresses, sizes, offsets: [sum(row) for row in sizes]
        worker.m_store.batch_get_end.return_value = 0
        request = ReqMeta("r", token_len_chunk=32, block_ids=[7, 8], block_hashes=["a", "b"],
                          can_save=False, load_spec=LoadSpec(0, 31, True))
        metadata = AscendConnectorMetadata(set(), set())
        metadata.add_request(request)
        worker.start_load_kv(metadata)
        worker._finish_token_load()
        worker.m_store.get.assert_not_called()
        self.assertEqual(worker.m_store.batch_copy_get.call_count, 8)
        self.assertEqual(worker._w2_observation.counts["remote_read_bytes"], 31)
        last_sizes = worker.m_store.batch_copy_get.call_args.args[2]
        self.assertEqual(last_sizes, [[3]])  # token 31 is computed locally, never DMA-written
        self.assertIsNone(worker._w2_token_pipeline)

    def test_full_dense_attention_with_real_torch_when_available(self):
        import subprocess
        import sys
        from pathlib import Path
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("token_overlap_cpu_check.py"))],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 77:
            self.skipTest("CPU torch is not installed; native attention check requires the project runtime")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shutdown_closes_backend_even_if_token_reader_failed(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker._closed = False
        worker.shared_pool_w2 = False
        worker.kv_send_thread = worker.kv_recv_thread = None
        worker.m_store = MagicMock()
        worker._w2_token_pipeline = SimpleNamespace(finish=MagicMock(side_effect=RuntimeError("read failed")))
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            worker.shutdown()
        worker.m_store.close.assert_called_once()
        self.assertTrue(worker._closed)
        self.assertIsNone(worker._w2_token_pipeline)

    def test_range_results_require_exact_bytes_not_status_zero_or_short_reads(self):
        from unittest.mock import MagicMock
        for result in (0, 63, 65, -1):
            with self.subTest(result=result):
                reads = list(dense_token_ranges("key", 1, 0, 4, 4, [(1000, 128, 2, 4, 2, False)], 8))
                backend = MagicMock()
                backend.batch_get_start.return_value = [0]
                backend.batch_copy_get.return_value = [result]
                backend.batch_get_end.return_value = 0
                pipeline = TokenLoadPipeline(backend, reads, self.observation(), lambda: None)
                with self.assertRaises(RuntimeError) as error:
                    pipeline.finish()
                self.assertIn(f"expected_bytes=64, results=[{result}]", str(error.exception.__cause__))
                self.assertFalse(reads[0].ready.is_set())
                backend.batch_get_end.assert_called_once_with(["key"])
