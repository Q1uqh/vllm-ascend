# SPDX-License-Identifier: Apache-2.0
"""Compact timeline for the KVPP + layerwise prefetch pipeline.

Set ``VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE=1`` to emit one JSON line per event.
Every line carries a monotonic nanosecond timestamp, the pid, the rank and the
emitting thread name, so a single log stream is enough to reconstruct:

- when the pool queued a layer's H2D load and at which model layer it did so;
- whether that load was finished before the layer's KVPP broadcast started;
- how long the layer's attention waited for the broadcast;
- when a forward pass released a still-pending load wait.

Events are cheap and off by default. This helper must never alter the forward
or transfer path, hence the blanket exception guard.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs

_PREFIX = "[KVPP_LAYERWISE_TIMELINE]"
_invalid_switch_warned = False


def _timeline_enabled() -> bool:
    """Read the switch, reporting a bad value once instead of staying silent.

    The switch is strict (``0``/``1``) and this helper never raises, so without
    the one-shot warning a typo such as ``true`` would produce no output at all
    and look like a missing feature.
    """
    global _invalid_switch_warned
    try:
        return envs.VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE
    except ValueError as err:
        if not _invalid_switch_warned:
            _invalid_switch_warned = True
            try:
                logger.warning(
                    "VLLM_ASCEND_KVPP_LAYERWISE_TIMELINE must be '0' or '1' (%s); the KVPP timeline stays disabled.",
                    err,
                )
            except Exception:  # noqa: BLE001 - instrumentation must never break a transfer
                pass
        return False


def emit_timeline_event(event: str, **fields: Any) -> None:
    """Append one timeline event. Never raises, never blocks."""
    try:
        if not _timeline_enabled():
            return
        payload = {
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "rank": os.getenv("RANK"),
            "local_rank": os.getenv("LOCAL_RANK"),
            "thread": threading.current_thread().name,
            **fields,
        }
        logger.info("%s %s", _PREFIX, json.dumps(payload, separators=(",", ":"), default=str))
    except Exception:  # noqa: BLE001 - instrumentation must never break a transfer
        pass
