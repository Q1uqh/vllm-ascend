from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs

_PREFIX = "[KVPP_LAYERWISE_TIMELINE]"


def emit_timeline_event(event: str, **fields: Any) -> None:
    """Emit a process-wide monotonic timeline for KVPP/layerwise debugging.

    Reuse ``VLLM_ASCEND_KVPOOL_RANGE_DEBUG`` so production behavior and log
    volume remain unchanged unless the existing KV-pool debug switch is on.
    This helper must never alter the forward or transfer path.
    """
    try:
        if not envs.VLLM_ASCEND_KVPOOL_RANGE_DEBUG:
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
    except Exception:
        pass
