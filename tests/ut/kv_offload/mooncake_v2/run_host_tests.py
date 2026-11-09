# SPDX-License-Identifier: Apache-2.0
"""Run the real Push classes with framework initialization isolated.

Requires Python 3.11+, pytest, msgspec, pyzmq and httpx. No torch/NPU is
imported. This exercises control flow, sockets and CPU byte-copy mocks;
it does not validate framework imports or actual device transfer.
"""

import ast
import contextlib
import dataclasses
import importlib.util
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import types
import typing
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

import httpx
import msgspec
import pytest
import zmq

ROOT = Path(__file__).resolve().parents[4]
PREFIX = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake"


def package(name):
    for index in range(1, len(name.split(".")) + 1):
        part = ".".join(name.split(".")[:index])
        if part not in sys.modules:
            module = types.ModuleType(part)
            module.__path__ = []
            sys.modules[part] = module
        if "." in part:
            parent, child = part.rsplit(".", 1)
            setattr(sys.modules[parent], child, sys.modules[part])
    return sys.modules[name]


def socket(ctx, path, socket_type, bind=False):
    result = ctx.socket(socket_type)
    result.setsockopt(zmq.LINGER, 0)
    if bind:
        result.bind(path)
    else:
        result.connect(path)
    return result


class Results:
    def __init__(self):
        self.finished_sending = set()
        self.finished_recving = set()
        self.failed_recving = set()


def load(name, relative, extra=None):
    module = package(name)
    module.__file__ = str(ROOT / relative)
    namespace = dict(
        Any=typing.Any,
        Sequence=typing.Sequence,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        Future=Future,
        ThreadPoolExecutor=ThreadPoolExecutor,
        wait_futures=wait,
        OrderedDict=OrderedDict,
        defaultdict=defaultdict,
        deque=deque,
        Mapping=typing.Mapping,
        cast=typing.cast,
        cdiv=lambda numerator, denominator: (numerator + denominator - 1) // denominator,
        GET_META_MSG=b"get_meta_msg",
        ACK_MSG=b"ACK",
        DONE_RECVING_MSG=b"done_recving_msg",
        _METADATA_TIMEOUT_MS=1000,
        _KT=typing.TypeVar("_KT"),
        _VT=typing.TypeVar("_VT"),
        WorkerRank=tuple[int, int, int],
        contextlib=contextlib,
        httpx=httpx,
        msgspec=msgspec,
        zmq=zmq,
        queue=queue,
        threading=threading,
        time=time,
        math=math,
        os=os,
        logger=logging.getLogger(name),
        envs=types.SimpleNamespace(VLLM_MOONCAKE_CONNECTOR_TIMEOUT=1.0, VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=30.0),
        KVConnectorMetadata=type("KVConnectorMetadata", (), {}),
        KVConnectorHandshakeMetadata=type("KVConnectorHandshakeMetadata", (), {}),
        KVConnectorTransferResults=Results,
        make_zmq_socket=socket,
        make_zmq_path=lambda protocol, host, port: f"{protocol}://{host}:{port}",
        extract_layer_index=lambda name, modules=1: (
            int(re.search(r"layers\.(\d+)", name)[1]) * modules
            + (int(name.rsplit(".", 1)[1]) if modules > 1 and name.rsplit(".", 1)[1].isdigit() else 0)
        ),
        is_circular_kv_cache_spec=lambda spec: getattr(spec, "is_circular", False),
        torch=types.SimpleNamespace(),
        validate_dsv4_d2d_config=lambda config: None,
        json=__import__("json"),
        SocketOption=zmq.SocketOption,
        SocketType=zmq.SocketType,
        Context=zmq.Context,
        Again=zmq.Again,
        ZMQError=zmq.ZMQError,
    )
    namespace.update(extra or {})
    module.__dict__.update(namespace)
    tree = ast.parse((ROOT / relative).read_text())
    # Keep actual methods and decorators; replace only framework imports.
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    ast.fix_missing_locations(tree)
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    return module


def main():
    specs = {}
    for name, parent in (
        ("FullAttentionSpec", object),
        ("SlidingWindowSpec", object),
        ("MambaSpec", object),
        ("MLAAttentionSpec", object),
        ("SlidingWindowMLASpec", object),
        ("UniformTypeKVCacheSpecs", object),
        ("AscendSFAIndexerCacheSpec", object),
    ):
        specs[name] = type(name, (parent,), {})
    utils = load(PREFIX + ".utils", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py")
    metadata = load(PREFIX + ".metadata", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/metadata.py")
    shared = {**specs, **{key: value for key, value in vars(metadata).items() if key.startswith("Mooncake")}}
    shared.update({key: value for key, value in vars(utils).items() if callable(value) and not key.startswith("_")})
    base = load(PREFIX + ".base_worker", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py", shared)
    shared["MooncakeBaseConnectorWorker"] = base.MooncakeBaseConnectorWorker
    shared["PushSendReqMeta"] = metadata.PushSendReqMeta
    layout_name = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.dsv4_tuple_layout"
    package(layout_name.rsplit(".", 1)[0])
    spec = importlib.util.spec_from_file_location(
        layout_name, ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/dsv4_tuple_layout.py"
    )
    layout = importlib.util.module_from_spec(spec)
    sys.modules[layout_name] = layout
    spec.loader.exec_module(layout)
    snapshot_name = layout_name.rsplit(".", 1)[0] + ".dsv4_snapshot"
    snapshot_spec = importlib.util.spec_from_file_location(
        snapshot_name, ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/dsv4_snapshot.py"
    )
    snapshot = importlib.util.module_from_spec(snapshot_spec)
    sys.modules[snapshot_name] = snapshot
    snapshot_spec.loader.exec_module(snapshot)
    shared["validate_dsv4_d2d_config"] = layout.validate_dsv4_d2d_config
    load(PREFIX + ".push_worker", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/push_worker.py", shared)
    package("vllm.v1.kv_cache_interface").__dict__.update(specs)
    status = types.SimpleNamespace(FINISHED_LENGTH_CAPPED=1, FINISHED_STOPPED=2, FINISHED_ABORTED=3)
    package("vllm.v1.request").RequestStatus = status
    shared["RequestStatus"] = status
    base_scheduler = load(
        PREFIX + ".base_scheduler", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_scheduler.py", shared
    )
    shared.update(
        {name: getattr(base_scheduler, name) for name in ("MooncakeBaseConnectorScheduler", "MooncakeMetadataThread")}
    )
    load(PREFIX + ".pull_scheduler", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py", shared)
    load(PREFIX + ".push_scheduler", "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/push_scheduler.py", shared)
    tests = Path(__file__).parent

    class SocketAvailability:
        def pytest_collection_modifyitems(self, items):
            try:
                with zmq.Context() as ctx, ctx.socket(zmq.ROUTER) as probe:
                    probe.bind("tcp://127.0.0.1:0")
            except zmq.ZMQError:
                for item in items:
                    if (
                        item.path.name == "test_push_worker.py"
                        or any(name in item.name for name in ("real_sockets", "socket_timeout"))
                        or item.name.split("[")[0]
                        in (
                            "test_d_serves_all_ranks_and_snapshot_is_stable",
                            "test_invalid_queries_do_not_stop_listener",
                            "test_bind_failure_is_reported_and_thread_is_cleaned_up",
                            "test_shared_service_preserves_pull_query_and_completion_protocol",
                            "test_metaserver_dispatch_is_concurrent_and_snapshots_blocks",
                        )
                    ):
                        item.add_marker(pytest.mark.skip(reason="Loopback sockets unavailable in this sandbox"))

    return pytest.main(
        [
            "--confcutdir=" + str(tests),
            "-q",
            str(tests / "test_push_sending.py"),
            str(tests / "test_push_worker.py"),
            str(tests / "test_push_scheduler.py"),
            str(ROOT / "tests/ut/distributed/ascend_store/test_dsv4_snapshot.py"),
            *sys.argv[1:],
        ],
        plugins=[SocketAvailability()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
