# SPDX-License-Identifier: Apache-2.0
import json
import threading
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import timeline


class _Recorder:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, message, *args):
        self.lines.append(message % args if args else message)


class _Boom:
    def info(self, *args, **kwargs):
        raise RuntimeError("logging exploded")


def _payload(line: str) -> dict:
    assert timeline._PREFIX in line
    return json.loads(line.split("] ", 1)[1])


def test_emit_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(timeline, "logger", recorder)

    timeline.emit_timeline_event("layerwise_load_enqueued", layer_id=3)

    assert recorder.lines == []


def test_emit_writes_one_json_line_per_event(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE", "1")
    recorder = _Recorder()
    monkeypatch.setattr(timeline, "logger", recorder)

    timeline.emit_timeline_event("kvpp_broadcast_begin", layer_name="model.layers.3.self_attn.attn", attempt=2)

    assert len(recorder.lines) == 1
    payload = _payload(recorder.lines[0])
    assert payload["event"] == "kvpp_broadcast_begin"
    assert payload["layer_name"] == "model.layers.3.self_attn.attn"
    assert payload["attempt"] == 2
    assert isinstance(payload["monotonic_ns"], int)
    assert payload["thread"] == threading.current_thread().name
    assert "pid" in payload


def test_emit_serializes_non_json_values(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE", "1")
    recorder = _Recorder()
    monkeypatch.setattr(timeline, "logger", recorder)

    timeline.emit_timeline_event("kvpp_schedule_forward", handle=SimpleNamespace(name="s"))

    assert isinstance(_payload(recorder.lines[0])["handle"], str)


def test_emit_never_raises(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE", "1")
    monkeypatch.setattr(timeline, "logger", _Boom())

    # Instrumentation must never break the forward or transfer path.
    timeline.emit_timeline_event("kvpp_broadcast_end", layer_name="l0")
