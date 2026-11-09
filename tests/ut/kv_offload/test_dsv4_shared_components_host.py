# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow tests of production definitions without torch/NPU imports.

Run directly: python tests/ut/kv_offload/test_dsv4_shared_components_host.py.
AST loading isolates framework imports, not the tested planner/transfer code.
It cannot validate framework integration, kernels, or actual device copies.
"""

import __future__

import ast
import ctypes
import dataclasses
import hashlib
import json
import logging
import math
import re
import sys
import threading
import time
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
STORE = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/"
PD = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py"


@dataclasses.dataclass(frozen=True)
class Spec:
    page_size_bytes: int = 32
    block_size: int = 4
    sliding_window: int | None = None
    offload_mutable: bool = False
    model_version: str | None = "deepseek_v4"


@dataclasses.dataclass
class Tensor:
    size: int
    layers: list[str]
    block_stride: int
    layer_stride: int = 0
    offset: int = 0


@dataclasses.dataclass
class Cache:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list


class Uniform:
    def __init__(self, specs):
        self.kv_cache_specs = specs


class Recurrent:
    pass


def definitions(relative, namespace, names=None):
    namespace.setdefault("emit_kv_lifecycle", lambda *args, **kwargs: None)
    namespace.setdefault("kv_trace_enabled", lambda: False)
    namespace.setdefault("trace_requests", lambda requests: [])
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)
    ]
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def planner():
    namespace = dict(
        dataclass=dataclasses.dataclass,
        replace=dataclasses.replace,
        Any=object,
        re=re,
        torch=MagicMock(),
        KVCacheTensor=Tensor,
        AttentionSpec=Spec,
        KVCacheConfig=Cache,
        MambaSpec=Recurrent,
        UniformTypeKVCacheSpecs=Uniform,
        AscendSFAIndexerCacheSpec=type("Indexer", (), {}),
        is_prefix_cacheable=lambda spec: True,
        get_kv_cache_tensor_layers=lambda tensor: tensor.layers,
        logger=logging.getLogger("host"),
        get_backend_class=lambda _: types.SimpleNamespace(layerwise_data_plane="block_key"),
        _NUM_SHARED_BUFFERS="layerwise_num_shared_buffers",
        _PREFETCH_LAYERS="layerwise_prefetch_layers",
        _INDEPENDENT_LAYERS="layerwise_independent_layers",
        _DEFAULT_MAX_PREFETCH_LAYERS=8,
        _INDEXER_CACHE_SUFFIX=".indexer.k_cache",
    )
    definitions(STORE + "layerwise/layerwise_cache_layout.py", namespace)
    return definitions(
        "vllm_ascend/models/deepseek_v4/cache_config.py",
        namespace | dict(defaultdict=defaultdict, hashlib=hashlib, json=json),
    )


def config(buffers=2):
    return types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config=dict(
                backend="mooncake",
                use_layerwise=True,
                layerwise_num_shared_buffers=buffers,
                layerwise_independent_layers=[0, 1],
            ),
        ),
        model_config=types.SimpleNamespace(get_num_layers=lambda _: 14, enforce_eager=True),
        parallel_config=types.SimpleNamespace(),
        speculative_config=None,
        use_v2_model_runner=True,
    )


def groups():
    result = []
    # Complete CSA and HCA vectors, including pure SWA layers. Match a
    # canonical pair of padded page buckets and cross-group overlay.
    components = [
        (range(2, 14), "attn", 32),
        (range(14), "swa", 32),
        (range(2, 14, 2), "state", 64),
        (range(3, 14, 2), "state", 64),
        (range(2, 14, 2), "indexer", 64),
        (range(2, 14, 2), "index_state", 64),
    ]
    # Canonical full-MLA bucket must include both page sizes.
    components[0] = (range(2, 14), "attn", None)
    for layers, component, page in components:
        specs = {f"model.layers.{layer}.{component}": Spec(page or (32 if layer % 2 == 0 else 64)) for layer in layers}
        result.append(types.SimpleNamespace(layer_names=list(specs), kv_cache_spec=Uniform(specs)))
    return result


class TestSharedComponents(unittest.TestCase):
    def test_complete_components_rotation_and_budget(self):
        ns = planner()
        for blocks in (1, 7, 101):
            plan = ns["build_deepseek_v4_physical_plan"](config(), groups(), blocks)
            self.assertEqual(
                plan.layerwise_reuse_layout.prefetch_layer_map, {6: 2, 10: 6, 8: 4, 12: 8, 7: 3, 11: 7, 9: 5, 13: 9}
            )
            self.assertEqual(plan.layerwise_reuse_layout.independent_layers, [0, 1])
            names = [name for tensor in plan.kv_cache_tensors for name in tensor.layers]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(set(names), {name for group in groups() for name in group.layer_names})
            self.assertEqual(len({tensor.size for tensor in plan.kv_cache_tensors}), 1)
            one = ns["build_deepseek_v4_physical_plan"](config(), groups(), 1)
            self.assertEqual(plan.kv_cache_tensors[0].size, one.kv_cache_tensors[0].size * blocks)
            before = plan.kv_cache_tensors
            ns["apply_layerwise_kv_cache_plan"](plan, config())
            self.assertIs(before, plan.kv_cache_tensors)
            for tensor in before:
                self.assertLessEqual(tensor.offset + blocks * tensor.block_stride, tensor.size)

    def test_same_group_regions_do_not_overlap_except_temporal_alias(self):
        plan = planner()["build_deepseek_v4_physical_plan"](config(), groups(), 7)
        for group in groups():
            regions = {
                name: (tensor.offset, tensor.offset + 7 * tensor.block_stride)
                for tensor in plan.kv_cache_tensors
                for name in tensor.layers
                if name in group.layer_names
            }
            for left, (start, end) in regions.items():
                for right, (other, limit) in regions.items():
                    self.assertTrue(left == right or (start, end) == (other, limit) or end <= other or limit <= start)

    def test_glm_main_packs_retain_existing_predecessors(self):
        ns = planner()
        names = [f"model.layers.{index}.attn" for index in range(6)]
        topology = ns["LayerwiseCacheTopology"](
            main_packs=tuple(tuple(names[i : i + 2]) for i in (0, 2, 4)),
            group_ids=tuple((name, index % 2) for index, name in enumerate(names)),
        )
        layout = ns["build_layerwise_reuse_layout"](
            dict.fromkeys(names, Spec()),
            6,
            dict(layerwise_num_shared_buffers=1, layerwise_independent_layers=[]),
            topology=topology,
        )
        self.assertEqual(layout.prefetch_layer_map, {2: 0, 4: 2, 3: 1, 5: 3})

    def test_glm_auxiliary_descriptors_remain_resident(self):
        ns = planner()
        names = [f"model.layers.{index}.attn" for index in range(6)]
        auxiliary = [f"model.layers.{index}.indexer.k_cache" for index in (0, 2, 4)]
        groups = [
            types.SimpleNamespace(layer_names=names[::2], kv_cache_spec=Spec()),
            types.SimpleNamespace(layer_names=names[1::2], kv_cache_spec=Spec()),
            types.SimpleNamespace(layer_names=auxiliary, kv_cache_spec=Spec()),
        ]
        packs = tuple(tuple(names[index : index + 2]) for index in (0, 2, 4))
        aux_tensors = [Tensor(128, [name], 32) for name in auxiliary]
        cache = Cache(4, [Tensor(128, list(pack), 32) for pack in packs] + aux_tensors, groups)
        cache.layerwise_cache_topology = ns["LayerwiseCacheTopology"](
            main_packs=packs,
            group_ids=tuple((name, group) for group, item in enumerate(groups) for name in item.layer_names),
        )
        cfg = config(1)
        cfg.model_config.get_num_layers = lambda _: 6
        cfg.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []
        ns["apply_layerwise_kv_cache_plan"](cache, cfg)
        self.assertEqual(sum(tensor.size for tensor in cache.kv_cache_tensors), 512)
        for tensor in aux_tensors:
            self.assertIn(tensor, cache.kv_cache_tensors)

    def test_pd_sibling_validation(self):
        ns = planner()
        store = dict(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config=config().kv_transfer_config.kv_connector_extra_config,
        )
        multi = types.SimpleNamespace(
            kv_connector="MultiConnector",
            kv_role="kv_producer",
            kv_connector_extra_config=dict(connectors=[store, dict(kv_connector="MooncakeLayerwiseConnector")]),
        )
        self.assertIs(ns["get_layerwise_reuse_config"](multi), store["kv_connector_extra_config"])
        multi.kv_connector_extra_config["connectors"][1]["kv_connector"] = "Unknown"
        with self.assertRaises(ValueError):
            ns["get_layerwise_reuse_config"](multi)

    def test_unsupported_speculation_rejected(self):
        cfg = config()
        cfg.speculative_config = object()
        with self.assertRaisesRegex(ValueError, "speculative"):
            planner()["build_deepseek_v4_physical_plan"](cfg, groups(), 1)

    def test_offload_disabled_keeps_ordinary_path(self):
        cfg = config()
        cfg.kv_transfer_config = None
        self.assertIsNone(planner()["build_deepseek_v4_physical_plan"](cfg, groups(), 1))


class TestSnapshots(unittest.TestCase):
    def setUp(self):
        self.ns = definitions(
            STORE + "backend/mooncake_offload.py",
            dict(
                block_hash_to_str=str,
                get_partial_block_index=lambda end, size, count, enabled: (
                    end // size if enabled and end % size else None
                ),
            ),
        )
        self.worker = types.SimpleNamespace(
            kv_cache_config=types.SimpleNamespace(
                kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=Spec(sliding_window=8, offload_mutable=True))]
            ),
            grouped_block_size=[4],
            group_uses_align_state=[False],
            use_eagle=False,
            cache_transfer_granularity=4,
            _make_layerwise_full_key=lambda group, key: f"g{group}:{key}",
            _layerwise_load_snapshot_start=lambda *args: 0,
            _is_layerwise_save_owner=lambda group: True,
            _layerwise_save_full_end=lambda req, group, size, partial: req.target_token_len // size,
        )
        self.req = types.SimpleNamespace(
            layerwise_request_id="request-generation",
            target_token_len=17,
            save_start_token=16,
            can_save=True,
            store_masks=None,
            block_ids_by_group=[[1, 2, 3, 4, 5]],
            load_spec=types.SimpleNamespace(
                can_load=True,
                layerwise_previous_step=True,
                kvpool_cached_tokens=17,
                kvpool_store_skip_tokens=None,
                layerwise_snapshot_end_token=17,
            ),
        )

    def test_mutable_window_has_exact_endpoint_keys(self):
        entries = self.ns["_load_entries"](self.worker, self.req, 0, ["hash"] * 5, None)
        self.assertEqual([index for _, index in entries], [2, 3, 4])
        self.assertTrue(all(":17:" in key for key, _ in entries))
        view = types.SimpleNamespace()
        saved = self.ns["_save_entries"](self.worker, self.req, view, 0, ["hash"] * 5)
        self.assertEqual([slot for _, slot in saved], [0, 1, None])
        self.assertEqual(view.save_start_token, 8)
        self.assertEqual({key for key, _ in saved}, {key for key, _ in entries})

    def test_wrong_endpoint_is_rejected(self):
        self.req.load_spec.layerwise_snapshot_end_token = 16
        with self.assertRaisesRegex(ValueError, "exact previous endpoint"):
            self.ns["_load_entries"](self.worker, self.req, 0, [], None)

    def test_historical_kv_still_uses_public_full_pages(self):
        self.worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec = Spec()
        self.worker._layerwise_load_snapshot_start = lambda *args: 4
        entries = self.ns["_load_entries"](self.worker, self.req, 0, list(range(4)), None)
        self.assertEqual([key for key, _ in entries[:4]], ["g0:0", "g0:1", "g0:2", "g0:3"])
        self.assertIn("offload_v1:request-generation:17:4", entries[-1][0])


class TestSchedulerContracts(unittest.TestCase):
    def test_aligned_mutable_state_keeps_exact_endpoint(self):
        class Base:
            def _build_req_meta(self, tracker):
                tracker.last_layerwise_snapshot_end_token = None
                return types.SimpleNamespace(can_save=True, target_token_len=4096)

        ns = definitions(
            STORE + "layerwise/pool_scheduler.py", dict(BaseKVPoolScheduler=Base), {"LayerwiseKVPoolScheduler"}
        )
        cls = ns["LayerwiseKVPoolScheduler"]
        worker = object.__new__(cls)
        worker.layerwise_offload = True
        worker.kv_cache_config = types.SimpleNamespace(layerwise_component_namespace="geometry")
        tracker = types.SimpleNamespace(layerwise_request_id="generation")
        cls._build_req_meta(worker, tracker)
        self.assertEqual(tracker.last_layerwise_snapshot_end_token, 4096)
        worker.kv_cache_config.layerwise_component_namespace = ""
        cls._build_req_meta(worker, tracker)
        self.assertIsNone(tracker.last_layerwise_snapshot_end_token)

    def test_live_tables_are_copied_and_missing_required_state_rejected(self):
        function = definitions(STORE + "metadata.py", {}, {"get_live_grouped_block_ids"})["get_live_grouped_block_ids"]
        groups = [[0, 4], [8]]
        output = types.SimpleNamespace(kv_connector_block_state=types.SimpleNamespace(get_block_ids=lambda _: groups))
        resolved = function(output, "request", required=True)
        groups[0][1] = 99
        self.assertEqual(resolved, [[0, 4], [8]])
        with self.assertRaisesRegex(RuntimeError, "KVConnectorBlockState"):
            function(types.SimpleNamespace(), "request", required=True)

    def test_summary_is_bounded_and_does_not_claim_d2d(self):
        import tempfile

        ns = definitions(
            "tests/e2e/cases/features/kv_pool/tools/summarize_component_offload.py",
            dict(
                Counter=__import__("collections").Counter,
                defaultdict=defaultdict,
                Path=Path,
                json=json,
                MARKER="[KVPOOL_RANGE_DEBUG]",
                MAX_LINES=280,
            ),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as stream:
            for index in range(1000):
                stream.write("[KVPOOL_RANGE_DEBUG] " + json.dumps(dict(event="snapshot_plan", endpoint=index)) + "\n")
            stream.flush()
            lines = ns["summarize"]([stream.name])
        self.assertLessEqual(len(lines), 280)
        self.assertTrue(any("DEVICE_D2D UNVERIFIED" in line for line in lines))


class TestOverwriteFence(unittest.TestCase):
    def test_range_load_waits_for_store_device_and_pd_even_without_get(self):
        ns = definitions(
            STORE + "layerwise/kv_transfer.py",
            dict(KVTransferThread=object, time=time),
            {"KVCacheStoreLayerRecvingThread"},
        )
        copied = threading.Event()
        pd_finished = threading.Event()
        device_synchronized = threading.Event()
        complete = threading.Event()
        layer_finished = [threading.Event() for _ in range(7)]
        lane = types.SimpleNamespace(
            layer_save_finished_events={2: copied},
            save_failure_checker=None,
            _wait_for_event=lambda event, *args: event.wait(1),
            sync_save_events={2: types.SimpleNamespace(synchronize=device_synchronized.set)},
            external_slot_release_waiter=lambda layer: pd_finished.wait(1),
            layer_load_finished_events=layer_finished,
            request_queue=MagicMock(),
            get_event=complete,
        )
        task = types.SimpleNamespace(layer_id=6, wait_for_save_layer=2, attention_start_gate=None, transfer_tasks=[])
        thread = threading.Thread(
            target=ns["KVCacheStoreLayerRecvingThread"]._handle_range_layer_task, args=(lane, task)
        )
        thread.start()
        self.assertFalse(complete.wait(0.02))
        self.assertFalse(device_synchronized.is_set())
        copied.set()
        self.assertTrue(device_synchronized.wait(1))
        self.assertFalse(complete.wait(0.02))
        pd_finished.set()
        thread.join(1)
        self.assertTrue(complete.is_set())
        self.assertTrue(layer_finished[6].is_set())


class TestPD(unittest.TestCase):
    def setUp(self):
        self.ns = definitions(
            PD,
            dict(
                dataclass=dataclasses.dataclass,
                field=dataclasses.field,
                KVConnectorMetadata=object,
                torch=MagicMock(),
                round_down=lambda count, size: count // size * size,
                threading=threading,
                MambaSpec=Recurrent,
                math=math,
                logging=logging,
                group_concurrent_contiguous=lambda remote, local: ([remote], [local]),
                LAYERWISE_REUSE_WAIT_LOG_INTERVAL_SECONDS=0.01,
                logger=logging.getLogger("host"),
            ),
            {
                "LayerMetadata",
                "KVCacheSendingLayerThread",
                "MooncakeLayerwiseConnectorWorker",
                "MooncakeLayerwiseConnectorScheduler",
                "SendReqInfo",
                "SendTask",
                "ReqMeta",
                "MooncakeLayerwiseConnectorMetadata",
            },
        )

    def test_strided_components_copy_bytes_using_own_group(self):
        meta = self.ns["LayerMetadata"]
        source = ctypes.create_string_buffer(bytes(range(128)))
        target = ctypes.create_string_buffer(128)
        send = types.SimpleNamespace(layer_name="component", k_cache=None, k_quant_cache=None)
        worker = types.SimpleNamespace(
            kv_cache_specs=[Spec(), Spec()],
            pd_head_ratio=1,
            layer_metadata={"component": meta([1], [ctypes.addressof(source)], [4], [1], [16])},
        )
        req = types.SimpleNamespace(
            local_block_ids=[[7], [1, 3]],
            remote_block_ids=[[7], [2, 4]],
            remote_layer_metadata={"component": meta([1], [ctypes.addressof(target)], [4], [1], [20])},
        )
        src, dst, lengths = self.ns["KVCacheSendingLayerThread"].get_transfer_meta(
            worker, send, "req", req, "component", 1
        )
        for start, end, size in zip(src, dst, lengths):
            ctypes.memmove(end, start, size)
        self.assertEqual(target.raw[40:44], source.raw[16:20])
        self.assertEqual(target.raw[80:84], source.raw[48:52])
        self.assertEqual(target.raw[44:80], bytes(36))

    def test_only_final_chunk_sends_full_context_and_live_tables(self):
        cls = self.ns["MooncakeLayerwiseConnectorScheduler"]
        scheduler = object.__new__(cls)
        scheduler.layerwise_offload = True
        scheduler.block_size = [4096]
        scheduler.kv_cache_config = types.SimpleNamespace(layerwise_requires_live_block_table=True)
        scheduler.vllm_config = types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(is_kv_consumer=False))
        request = types.SimpleNamespace(all_token_ids=[1] * 8192, kv_transfer_params=dict(remote_cached_tokens=0))
        scheduler._reqs_need_send_layerwise = {"req": self.ns["SendReqInfo"]([[99]], 0, 0, request)}
        output = types.SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_spec_decode_tokens={},
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["req"], new_block_ids=[None], num_computed_tokens=[0]),
            num_scheduled_tokens={"req": 4096},
        )
        module = types.ModuleType("metadata_stub")
        module.get_live_grouped_block_ids = lambda *args, **kwargs: [[1, 2]]
        path = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata"
        with patch.dict(sys.modules, {path: module}):
            first = cls.build_connector_meta(scheduler, output)
            self.assertEqual(first.requests, {})
            output.scheduled_cached_reqs.num_computed_tokens = [4096]
            final = cls.build_connector_meta(scheduler, output).requests["req"]
        self.assertEqual(final.local_block_ids, [[1, 2]])
        self.assertEqual(final.local_transed_tokens, 0)
        self.assertEqual(final.local_computed_tokens, 8192)
        self.assertTrue(final.chunk_finish)

    def test_optional_kv_quantization_flags_during_worker_initialization(self):
        tree = ast.parse((ROOT / PD).read_text())
        worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MooncakeLayerwiseConnectorWorker"
        )
        initializer = next(
            node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        flags = {"enable_kv_quant", "enable_c8_quant"}
        assignments = [
            node
            for node in initializer.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr in flags for target in node.targets)
        ]
        self.assertEqual(len(assignments), 2)
        code = compile(ast.Module(body=assignments, type_ignores=[]), str(ROOT / PD), "exec")
        for quant, expected in [
            (None, (False, False)),
            (types.SimpleNamespace(), (False, False)),
            (types.SimpleNamespace(enable_fa_quant=True, enable_c8_quant=False), (True, False)),
            (types.SimpleNamespace(enable_fa_quant=False, enable_c8_quant=True), (False, True)),
        ]:
            with self.subTest(quant=quant):
                instance = types.SimpleNamespace()
                exec(code, {"self": instance, "vllm_config": types.SimpleNamespace(quant_config=quant)})
                self.assertEqual((instance.enable_kv_quant, instance.enable_c8_quant), expected)

    def test_endpoint_block_trimming_preserves_already_truncated_tables(self):
        method = self.ns["MooncakeLayerwiseConnectorScheduler"]._trim_hybrid_remote_block_ids
        worker = types.SimpleNamespace(need_truncate=True, block_size=[4096, 128])
        already = ([1, 2], list(range(1, 65)))
        self.assertEqual(method(worker, already, 8193), already)
        extra = ([1, 2, 3], list(range(1, 66)))
        self.assertEqual(method(worker, extra, 8193), already)
        self.assertEqual(method(worker, already, 8192), already)

    def test_transfer_mapping_skips_null_sources_and_reports_short_destinations(self):
        ns = definitions(
            "vllm_ascend/distributed/kv_transfer/utils/utils.py",
            dict(logger=logging.getLogger("host")),
            {"get_transfer_mappings"},
        )
        mapping = ns["get_transfer_mappings"]
        req = types.SimpleNamespace(local_block_ids=[[0, 4]], remote_block_ids=[[9]], local_computed_tokens=8192)
        arguments = (
            [[[[0, 1]]]],
            {0: {0: dict(host="d", port=31000, block_idx=0)}, 1: {0: dict(host="d", port=31000, block_idx=0)}},
            {0: [0]},
            {("d", 31000): 1},
            req,
            0,
            types.SimpleNamespace(dcp_size=1),
            "req",
            0,
            2,
            0,
            0,
            0,
        )
        result = mapping(*arguments)
        self.assertEqual(result[("d", 31000)]["local_block_ids"], [4])
        req.remote_block_ids = [[]]
        with self.assertRaisesRegex(ValueError, "destination.*group=0.*destination_blocks=0"):
            mapping(*arguments)

    def test_single_token_needs_no_external_prefill(self):
        method = self.ns["MooncakeLayerwiseConnectorScheduler"]._hybrid_prefill_token_count
        worker = types.SimpleNamespace(need_truncate=True)
        self.assertEqual(method(worker, 1), 0)
        self.assertEqual(method(worker, 8193), 8192)

    def test_pure_swa_parent_hook_maps_to_registered_layer(self):
        cls = self.ns["MooncakeLayerwiseConnectorWorker"]
        worker = types.SimpleNamespace(
            current_layer=0,
            total_layers=2,
            index_to_name={0: ["model.layers.0.self_attn.swa_cache"]},
            _cache_write_events=[None, None],
        )
        for layer in (0, 1):
            worker.current_layer = layer
            worker.index_to_name[layer] = [f"model.layers.{layer}.self_attn.swa_cache"]
            for hook in (f"model.layers.{layer}.self_attn", f"model.layers.{layer}.self_attn.attn"):
                cls.on_kv_cache_written(worker, hook)
                self.assertIsNotNone(worker._cache_write_events[layer])
        worker.current_layer = 0
        with self.assertRaises(RuntimeError):
            cls.on_kv_cache_written(worker, "model.layers.1.self_attn")

    def test_hook_owner_resolution_preserves_exact_main_and_rejects_other_layers(self):
        resolve = self.ns["MooncakeLayerwiseConnectorWorker"]._resolve_layer_cache_owner
        swa = "model.layers.0.self_attn.swa_cache"
        self.assertEqual(resolve("model.layers.0.self_attn.attn", [swa], 0), swa)
        main = "model.layers.2.self_attn.attn"
        self.assertEqual(resolve(main, ["model.layers.2.self_attn.swa_cache", main], 2), main)
        with self.assertRaises(RuntimeError):
            resolve("model.layers.1.self_attn.attn", [swa], 0)
        with self.assertRaises(RuntimeError):
            resolve("model.layers.0.unrelated.attn", [swa], 0)

    def test_null_scheduler_blocks_remain_null_after_kernel_expansion(self):
        cls = self.ns["MooncakeLayerwiseConnectorWorker"]
        worker = types.SimpleNamespace(num_kv_cache_groups=1, kv_cache_specs=[Spec()], kernel_block_size_scale=[2])
        self.assertEqual(cls._get_kernel_block_ids(worker, [[0, 3]]), [[0, 0, 6, 7]])

    def test_delayed_reader_blocks_successor_until_completion(self):
        cls = self.ns["MooncakeLayerwiseConnectorWorker"]
        meta = self.ns["LayerMetadata"]
        worker = types.SimpleNamespace(
            index_to_name={2: ["a"], 6: ["b"]},
            layer_metadata={"a": meta([0], [100], [8], [1]), "b": meta([0], [100], [8], [1])},
            _pending_reuse_layers=set(),
            _storage_send_errors={},
            kv_send_layer_thread=types.SimpleNamespace(is_alive=lambda: True),
        )
        cls._init_layerwise_reuse_state(worker)
        cls._mark_layer_reuse_pending(worker, 2)
        entered, released = threading.Event(), threading.Event()

        def wait():
            entered.set()
            cls.wait_for_layer_reuse(worker, 6)
            released.set()

        thread = threading.Thread(target=wait)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(released.wait(0.02))
        cls._complete_layer_reuse(worker, 2, None)
        thread.join(1)
        self.assertTrue(released.is_set())

    def test_overlay_groups_do_not_release_other_readers(self):
        cls = self.ns["MooncakeLayerwiseConnectorWorker"]
        meta = self.ns["LayerMetadata"]
        worker = types.SimpleNamespace(
            index_to_name={2: ["a"], 3: ["b"], 6: ["c"]},
            layer_metadata={
                "a": meta([0], [100], [8], [1]),
                "b": meta([1], [100], [8], [1]),
                "c": meta([0], [100], [8], [1]),
            },
            _pending_reuse_layers=set(),
            _storage_send_errors={},
            kv_send_layer_thread=types.SimpleNamespace(is_alive=lambda: True),
        )
        cls._init_layerwise_reuse_state(worker)
        self.assertEqual(worker.layer_storage_slots[2], worker.layer_storage_slots[6])
        self.assertNotEqual(worker.layer_storage_slots[2], worker.layer_storage_slots[3])
        cls._mark_layer_reuse_pending(worker, 2)
        cls._mark_layer_reuse_pending(worker, 3)
        cls._complete_layer_reuse(worker, 3, None)
        event = worker.storage_send_done_events[worker.layer_storage_slots[2][0]]
        self.assertFalse(event.is_set())
        cls._complete_layer_reuse(worker, 2, "WRITE failed")
        with self.assertRaisesRegex(RuntimeError, "WRITE failed"):
            cls.wait_for_layer_reuse(worker, 6)


if __name__ == "__main__":
    unittest.main()
