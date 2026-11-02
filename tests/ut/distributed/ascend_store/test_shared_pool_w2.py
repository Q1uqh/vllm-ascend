"""Ordinary shared-pool ownership, publication fences and opt-in isolation."""

import json
import logging
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_pool_w2 import (
    BlockPoolObservation,
    SharedPoolObservation,
    SourceBlockTracker,
    validate_shared_pool_w2,
)


class TestSharedPoolW2(unittest.TestCase):
    def observation(self):
        return SharedPoolObservation("worker", "kv_producer", 0, 1)

    def make_thread(self, enabled=True):
        database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None)
        database.set_group_buffers({0: [1000]}, {0: [16]}, {0: [1]}, group_num_layers={0: 1})
        store = MagicMock()
        store.requires_exists_before_put = True
        store.exists.side_effect = lambda keys: [0] * len(keys)
        store.put.side_effect = lambda keys, *args: [True] * len(keys)
        worker = SimpleNamespace(
            shared_pool_w2=enabled, tp_mismatch=False, w2_worker_rank=0, _w2_observation=self.observation()
        )
        thread = KVCacheStoreSendingThread(store, database, 16, 0, worker=worker)
        request = ReqMeta(
            "r", token_len_chunk=32, block_ids=[7, 8], block_hashes=["a", "b"],
            can_save=True, source_save_id=0, source_block_ids=(7, 8),
        )
        return thread, store, request

    def drain(self, thread):
        while not thread.request_queue.empty():
            thread._handle_request(thread.request_queue.get_nowait())

    def test_opt_in_and_validation(self):
        self.assertFalse(validate_shared_pool_w2({}, "kv_both", 2))
        self.assertTrue(validate_shared_pool_w2({"enable_shared_pool_w2": True}, "kv_producer", 1))
        for extra, role, groups in [
            ({"enable_shared_pool_w2": "false"}, "kv_producer", 1),
            ({"enable_shared_pool_w2": True, "backend": "memcache"}, "kv_producer", 1),
            ({"enable_shared_pool_w2": True, "use_layerwise": True}, "kv_producer", 1),
            ({"enable_shared_pool_w2": True}, "kv_producer", 2),
            ({"enable_shared_pool_w2": True}, "kv_consumer", 1),
            ({"enable_shared_pool_w2": True}, "kv_both", 1),
            ({"enable_shared_pool_w2": True, "consumer_is_to_load": True, "save_decode_cache": True},
             "kv_consumer", 1),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_shared_pool_w2(extra, role, groups)

    def test_rank_dedup_and_old_generation_cannot_free_new_pin(self):
        tracker = SourceBlockTracker(2)
        old, _ = tracker.register([7, 7, 8])
        self.assertEqual(tracker.acknowledge([(old, 7, 0)] * 2), [])
        self.assertEqual(tracker.acknowledge([(old, 7, 1)]), [7])
        new, _ = tracker.register([7])
        self.assertEqual(tracker.acknowledge([(old, 7, 1), (new, 7, 99)]), [])
        self.assertEqual(tracker.acknowledge([(new, 7, 0), (new, 7, 1)]), [7])
        self.assertIn((old, 8), tracker.pending)

    def test_metadata_union_deduplicates_workers(self):
        a = AscendStoreKVConnectorWorkerMetadata({1: 1}, {(5, 7, 0)})
        b = AscendStoreKVConnectorWorkerMetadata({1: 1}, {(5, 7, 0), (5, 7, 1)})
        merged = a.aggregate(b)
        self.assertEqual(merged.completed_events, {1: 2})
        self.assertEqual(merged.source_block_completions, {(5, 7, 0), (5, 7, 1)})

    def test_put_is_one_block_and_completion_precedes_next_put(self):
        thread, store, request = self.make_thread()
        batch = thread.add_save_batch([request])
        seen = []
        def put(keys, *args):
            if seen:
                self.assertIn((0, 7, 0), thread._source_completions)
                self.assertNotIn((0, 8, 0), thread._source_completions)
            seen.extend(keys)
            self.assertFalse(batch.done.is_set())
            return [True] * len(keys)
        store.put.side_effect = put
        self.drain(thread)
        thread.wait_for_batch(batch)
        self.assertEqual([len(call.args[0]) for call in store.put.call_args_list], [1, 1])
        self.assertEqual(thread.get_source_block_completions(), {(0, 7, 0), (0, 8, 0)})
        self.assertFalse(thread._source_remaining)

    def test_disabled_keeps_original_batch_and_no_new_observation(self):
        thread, store, request = self.make_thread(enabled=False)
        batch = thread.add_save_batch([request])
        self.drain(thread)
        self.assertTrue(batch.done.is_set())
        self.assertEqual([len(call.args[0]) for call in store.put.call_args_list], [2])
        self.assertIsNone(request.w2_save_succeeded)
        self.assertFalse(hasattr(thread, "_source_remaining"))
        self.assertFalse(thread.worker._w2_observation.counts)

    def test_failed_put_releases_safe_source_but_aborts_handoff(self):
        thread, store, request = self.make_thread()
        store.put.side_effect = lambda keys, *args: [False] * len(keys)
        batch = thread.add_save_batch([request])
        self.drain(thread)
        with self.assertRaisesRegex(RuntimeError, "handoff aborted"):
            thread.wait_for_batch(batch)
        self.assertEqual(thread.get_source_block_completions(), {(0, 7, 0), (0, 8, 0)})
        self.assertEqual(thread.get_saved_offset("r"), 0)

    def test_backend_exception_and_existing_keys_do_not_leak_pins(self):
        for raises in (False, True):
            thread, store, request = self.make_thread()
            if raises:
                store.put.side_effect = RuntimeError("backend stopped reading sources")
            else:
                store.exists.side_effect = lambda keys: [1] * len(keys)
            batch = thread.add_save_batch([request])
            self.drain(thread)
            if raises:
                with self.assertRaises(RuntimeError):
                    thread.wait_for_batch(batch)
            else:
                thread.wait_for_batch(batch)
                store.put.assert_not_called()
            self.assertFalse(thread._source_remaining)
            self.assertEqual(len(thread.get_source_block_completions()), 2)

    def test_rejected_submission_cannot_look_successful(self):
        thread, _, request = self.make_thread()
        thread._stopping.set()
        with self.assertRaises(RuntimeError):
            thread.add_save_batch([request])
        thread._stopping.clear()
        thread._fatal_error = RuntimeError("failed consumer")
        with self.assertRaises(RuntimeError):
            thread.add_save_batch([request])

    def test_wait_blocks_until_backend_finishes(self):
        thread, store, request = self.make_thread()
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        def put(keys, *args):
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test did not release the backend")
            return [True] * len(keys)
        store.put.side_effect = put
        batch = thread.add_save_batch([request])
        consumer = threading.Thread(target=self.drain, args=(thread,))
        waiter = threading.Thread(target=lambda: (thread.wait_for_batch(batch), returned.set()))
        consumer.start()
        waiter.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            self.assertFalse(returned.is_set())
            self.assertFalse(batch.done.is_set())
        finally:
            release.set()
            consumer.join(timeout=3)
            waiter.join(timeout=3)
        self.assertTrue(returned.is_set())

    def test_worker_wait_for_save_really_waits_only_when_enabled(self):
        for enabled in (False, True):
            thread, _, request = self.make_thread(enabled)
            # Exercise the actual worker entry with a synchronous fake queue.
            original_submit = thread.add_save_batch
            def submit(requests):
                batch = original_submit(requests)
                self.drain(thread)
                return batch
            thread.add_save_batch = submit
            thread.wait_for_batch = MagicMock(wraps=thread.wait_for_batch)
            worker = KVPoolWorker.__new__(KVPoolWorker)
            worker.shared_pool_w2 = enabled
            worker.kv_send_thread = thread
            worker.group_uses_align_state = [False]
            worker._w2_observation = thread.worker._w2_observation
            metadata = AscendConnectorMetadata(set(), set())
            metadata.add_request(request)
            worker.wait_for_save(metadata)
            self.assertEqual(thread.wait_for_batch.call_count, int(enabled))

    def test_scheduler_returns_only_transfer_reference(self):
        # The core owns the request reference. Simulate a shared prefix: this
        # task's pin plus two requests, one request ending before completion.
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.shared_pool_w2 = True
        scheduler._source_blocks = SourceBlockTracker(2)
        scheduler._w2_expected_checks = {}
        scheduler._w2_observation = self.observation()
        save_id, _ = scheduler._source_blocks.register([7])
        refs = {7: 2}  # one request ended: other request + this save pin remain
        def free(blocks):
            for block in blocks:
                refs[block] -= 1
        scheduler._block_pool = SimpleNamespace(blocks={7: 7}, free_blocks=free)
        scheduler.sending_events, scheduler.sending_blocks = {}, {}
        scheduler._loading_req_ids = set()
        for rank in (0, 0, 1, 1):
            scheduler.update_connector_output(SimpleNamespace(
                finished_recving=set(), kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata(
                    {}, {(save_id, 7, rank)}
                ),
            ))
        self.assertEqual(refs[7], 1)  # other request still owns it
        self.assertFalse(scheduler.has_pending_push_work())

    def test_load_observation_recovery_and_compact_snapshot(self):
        obs = self.observation()
        obs.record_load("r", ["a"], [[32]], True)
        obs.record_load("r", ["b"], [[32]], True)
        obs.record_load("r", ["a"], [[32]], True)
        self.assertEqual(obs.counts["repeated_load_keys"], 1)
        obs.retire(["r"], recovery=True)
        obs.record_load("r", ["a"], [[32]], True)
        self.assertEqual(obs.counts["repeated_load_keys"], 1)
        logger = MagicMock(spec=logging.Logger)
        obs.emit(logger)
        obs.emit(logger)
        obs.emit(logger, final=True)
        self.assertEqual(logger.info.call_count, 2)
        payload = json.loads(logger.info.call_args.args[2])
        self.assertNotIn("r", payload)
        self.assertTrue(payload["final"])

    def test_d_normal_decode_does_not_create_transfer_metadata(self):
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler._unfinished_requests = {"r": (object(), [7])}
        scheduler._request_trackers = {"r": SimpleNamespace(prefill_end_tokens=16, num_prompt_tokens=16)}
        scheduler.save_decode_cache = False
        scheduler.layerwise_offload = False
        result = scheduler._process_running_cached_request(
            [8], "r", 0, SimpleNamespace(num_computed_tokens=[16]), object(), True
        )
        self.assertIsNone(result)

    def test_all_shards_of_one_source_block_finish_before_ack(self):
        thread, store, request = self.make_thread()
        thread._source_remaining[0] = {7, 8}
        calls = []
        def put(keys, *args):
            if not calls:
                self.assertNotIn((0, 7, 0), thread._source_completions)
            calls.append(list(keys))
            return [True] * len(keys)
        store.put.side_effect = put
        result = thread._put_shared_pool_blocks(
            request, ["7a", "7b", "8a", "8b"], [[1]] * 4, [[4]] * 4, [7, 7, 8, 8]
        )
        self.assertEqual(calls, [["7a", "7b"], ["8a", "8b"]])
        self.assertEqual(result, [True] * 4)
        self.assertEqual(len(thread.get_source_block_completions()), 2)

    def test_scheduler_pins_before_dispatch_without_changing_request_ownership(self):
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.shared_pool_w2 = True
        scheduler.kv_role = "kv_producer"
        scheduler.can_put = True
        scheduler._request_trackers = {}
        scheduler._unfinished_requests = {}
        scheduler._preempted_req_ids = set()
        scheduler._loading_req_ids = set()
        scheduler.num_speculative_blocks_by_group = {}
        scheduler._source_blocks = SourceBlockTracker(1)
        scheduler._w2_expected_checks = {}
        scheduler._w2_observation = self.observation()
        pool = SimpleNamespace(blocks={7: 7, 8: 8}, touch=MagicMock())
        scheduler._block_pool = pool
        request = ReqMeta("r", block_ids=[7, 7, 8, 0], can_save=True)
        scheduler._process_new_request = MagicMock(return_value=request)
        metadata = scheduler.build_connector_meta(SimpleNamespace(
            finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[SimpleNamespace(req_id="r")],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        ))
        pool.touch.assert_called_once_with([7, 8])
        self.assertEqual(metadata.requests[0].source_block_ids, (7, 8))
        self.assertEqual(request.block_ids_by_group, [[7, 7, 8, 0]])
        self.assertEqual(len(scheduler._source_blocks.pending), 2)

    def test_save_disabled_after_dispatch_still_returns_source_reference(self):
        thread, _, request = self.make_thread()
        request.can_save = False
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.shared_pool_w2 = True
        worker.kv_send_thread = thread
        worker._w2_observation = thread.worker._w2_observation
        metadata = AscendConnectorMetadata(set(), set())
        metadata.add_request(request)
        worker.wait_for_save(metadata)
        self.assertEqual(thread.get_source_block_completions(), {(0, 7, 0), (0, 8, 0)})
        self.assertFalse(thread._source_remaining)
        self.assertTrue(thread.request_queue.empty())

    def test_real_thread_factory_propagates_w2_without_tp_mismatch(self):
        # Construct real transfer objects through the production factory. Mock
        # only start(), so this catches missing constructor context on TP1.
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreRecvingThread

        for enabled in (False, True):
            template, store, _ = self.make_thread(enabled)
            worker = KVPoolWorker.__new__(KVPoolWorker)
            worker.shared_pool_w2 = enabled
            worker.tp_mismatch = False
            worker._transfer_threads_started = False
            worker.can_put = True
            worker.load_async = True
            worker.m_store = store
            worker.token_database = template.token_database
            worker.grouped_block_size = [16]
            worker.tp_rank = 0
            worker.tp_size = 1
            worker.pcp_rank = 0
            worker.pcp_size = 1
            worker.dcp_size = 1
            worker.put_step = 1
            worker.kv_role = "kv_producer"
            worker.use_kvpp = False
            worker.group_uses_align_state = [False]
            worker.enable_kv_events = False
            worker._invalid_block_ids = set()
            worker._invalid_block_ids_lock = threading.Lock()
            worker.w2_worker_rank = 0
            worker._w2_observation = self.observation()

            def construct(cls, *args, **kwargs):
                thread = cls(*args, **kwargs)
                thread.start = thread.ready_event.set
                return thread

            prefix = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker."
            with patch(prefix + "KVCacheStoreSendingThread", side_effect=lambda *a, **kw: construct(
                KVCacheStoreSendingThread, *a, **kw
            )), patch(prefix + "KVCacheStoreRecvingThread", side_effect=lambda *a, **kw: construct(
                KVCacheStoreRecvingThread, *a, **kw
            )):
                worker._start_kv_transfer_threads()
            self.assertEqual(worker.kv_send_thread.shared_pool_w2, enabled)
            self.assertIs(worker.kv_send_thread.worker, worker if enabled else None)
            self.assertIs(worker.kv_recv_thread.worker, worker if enabled else None)
            if enabled:
                self.assertEqual(worker.kv_send_thread.get_source_block_completions(), set())

    def test_real_worker_metadata_with_no_legacy_events_reaches_scheduler(self):
        thread, _, request = self.make_thread()
        batch = thread.add_save_batch([request])
        self.drain(thread)
        thread.wait_for_batch(batch)
        # Real getter returns None when there are no old Mamba events.
        self.assertIsNone(thread.get_completed_events())
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.shared_pool_w2 = True
        worker.kv_send_thread = thread
        meta = worker.build_connector_worker_meta()
        self.assertEqual(meta.completed_events, {})
        self.assertEqual(meta.source_block_completions, {(0, 7, 0), (0, 8, 0)})
        self.assertIsNone(worker.build_connector_worker_meta())

        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.shared_pool_w2 = True
        scheduler._source_blocks = SourceBlockTracker(1)
        scheduler._w2_expected_checks = {}
        scheduler._source_blocks.register([7, 8])
        scheduler._w2_observation = self.observation()
        scheduler._loading_req_ids = set()
        scheduler.sending_events, scheduler.sending_blocks = {}, {}
        scheduler._block_pool = SimpleNamespace(blocks={7: 7, 8: 8}, free_blocks=MagicMock())
        # Simulate old/serialized metadata explicitly containing None as well.
        meta.completed_events = None
        scheduler.update_connector_output(SimpleNamespace(
            finished_recving=set(), kv_connector_worker_meta=meta
        ))
        self.assertFalse(scheduler._source_blocks.pending)
        scheduler._block_pool.free_blocks.assert_called_once()

    def test_aggregation_accepts_legacy_none_with_w2_only_notifications(self):
        a = AscendStoreKVConnectorWorkerMetadata(None, {(0, 7, 0)})
        b = AscendStoreKVConnectorWorkerMetadata(None, {(0, 7, 1)})
        merged = a.aggregate(b)
        self.assertEqual(merged.completed_events, {})
        self.assertEqual(merged.source_block_completions, {(0, 7, 0), (0, 7, 1)})
        legacy = AscendStoreKVConnectorWorkerMetadata({5: 2})
        self.assertEqual(a.aggregate(legacy).completed_events, {5: 2})
        self.assertEqual(legacy.aggregate(a).completed_events, {5: 2})

    def test_allocator_observes_return_and_actual_reuse_with_shared_owner(self):
        tracker = SourceBlockTracker(1)
        obs = self.observation()
        block = SimpleNamespace(block_id=7, ref_cnt=3, is_null=False)
        pool = SimpleNamespace()
        def free(blocks):
            for item in blocks:
                item.ref_cnt -= 1
        def allocate(count):
            self.assertEqual(block.ref_cnt, 0)
            block.ref_cnt = 1
            return [block]
        pool.free_blocks, pool.get_new_blocks = free, allocate
        monitor = BlockPoolObservation(pool, tracker, obs)
        monitor.tracked.add(7)
        save, _ = tracker.register([7])
        pool.free_blocks([block])  # request finishes, shared request remains
        self.assertEqual(obs.counts["allocator_returned"], 0)
        tracker.acknowledge([(save, 7, 0)])
        pool.free_blocks([block])  # release transfer reference
        self.assertEqual(obs.counts["allocator_returned"], 0)
        pool.free_blocks([block])  # last shared owner finishes
        self.assertEqual(obs.counts["allocator_returned"], 1)
        pool.get_new_blocks(1)
        self.assertEqual(obs.counts["allocator_reused"], 1)
        self.assertEqual(obs.counts["early_reuse"], 0)
        monitor.detach()
        self.assertIs(pool.free_blocks, free)
        self.assertIs(pool.get_new_blocks, allocate)

    def test_allocator_detects_return_while_source_is_still_inflight(self):
        tracker = SourceBlockTracker(1)
        obs = self.observation()
        block = SimpleNamespace(block_id=7, ref_cnt=1, is_null=False)
        pool = SimpleNamespace(
            free_blocks=lambda blocks: setattr(block, "ref_cnt", 0),
            get_new_blocks=lambda count: [block],
        )
        monitor = BlockPoolObservation(pool, tracker, obs)
        monitor.tracked.add(7)
        tracker.register([7])
        with self.assertRaisesRegex(RuntimeError, "outstanding save"):
            pool.free_blocks([block])
        self.assertEqual(obs.counts["early_reuse"], 1)
        monitor.detach()

    def test_worker_output_boundary_reports_missing_save_not_as_success(self):
        thread, _, request = self.make_thread()
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.shared_pool_w2 = True
        worker.w2_worker_rank = 0
        worker.kv_send_thread = thread
        worker._w2_observation = self.observation()
        metadata = AscendConnectorMetadata(set(), set())
        metadata.add_request(request)
        output = worker.build_connector_worker_meta(metadata)
        self.assertEqual(output.w2_save_checks, {(0, 0, False)})
        request.can_save = False  # cancellation is not proof of publication
        self.assertEqual(worker.build_connector_worker_meta(metadata).w2_save_checks, {(0, 0, False)})
        request.can_save = True
        request.w2_save_succeeded = True
        output = worker.build_connector_worker_meta(metadata)
        self.assertEqual(output.w2_save_checks, {(0, 0, True)})
        self.assertEqual(output.aggregate(output).w2_save_checks, {(0, 0, True)})

    def test_scheduler_checks_executor_output_before_unpinning(self):
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.shared_pool_w2 = True
        scheduler._source_blocks = SourceBlockTracker(1)
        scheduler._source_blocks.register([7])
        scheduler._w2_expected_checks = {0: set()}
        scheduler._w2_observation = self.observation()
        scheduler._loading_req_ids = set()
        scheduler.sending_events, scheduler.sending_blocks = {}, {}
        scheduler._block_pool = SimpleNamespace(blocks={7: 7}, free_blocks=MagicMock())
        invalid = AscendStoreKVConnectorWorkerMetadata({}, {(0, 7, 0)}, {(0, 0, False)})
        with self.assertRaisesRegex(RuntimeError, "before a successful save"):
            scheduler.update_connector_output(SimpleNamespace(
                finished_recving=set(), kv_connector_worker_meta=invalid
            ))
        scheduler._block_pool.free_blocks.assert_not_called()
        valid = AscendStoreKVConnectorWorkerMetadata({}, {(0, 7, 0)}, {(0, 0, True)})
        scheduler.update_connector_output(SimpleNamespace(finished_recving=set(), kv_connector_worker_meta=valid))
        self.assertEqual(scheduler._w2_observation.counts["return_checked_jobs"], 1)
        self.assertFalse(scheduler._w2_expected_checks)
