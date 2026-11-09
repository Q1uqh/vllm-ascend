# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional common lifecycle evidence for storage and PD slot readers."""

import json
import os
import socket
import time
import uuid

from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config

_TRACE_SESSION = uuid.uuid4().hex
_TRACE_HOST = socket.gethostname()


def emit_kv_lifecycle(event: str, **fields) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = dict(
            event=event,
            host=_TRACE_HOST,
            trace_session=_TRACE_SESSION,
            pid=os.getpid(),
            monotonic_ns=time.monotonic_ns(),
            **fields,
        )
        logger.info("[KVPOOL_RANGE_DEBUG] %s", json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Diagnostic failures must not change ownership or transfer results.
        pass


def kv_trace_enabled() -> bool:
    try:
        return bool(get_ascend_config().kv_pool_range_debug)
    except Exception:
        return False


def trace_requests(requests) -> list[dict]:
    """Copy identities, never retain a mutable request in a background trace."""
    if not kv_trace_enabled():
        return []
    entries = requests.items() if isinstance(requests, dict) else ((r.req_id, r) for r in requests)
    result = []
    for request_id, request in entries:
        load = getattr(request, "load_spec", None)
        result.append(
            dict(
                request=request_id,
                incarnation=getattr(request, "layerwise_request_id", None),
                endpoint=getattr(request, "target_token_len", getattr(request, "local_computed_tokens", None)),
                restore_endpoint=getattr(load, "kvpool_cached_tokens", None),
            )
        )
    return result
