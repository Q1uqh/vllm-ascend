# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional common lifecycle evidence for storage and PD slot readers."""

import json
import os
import time

from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


def emit_kv_lifecycle(event: str, **fields) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = dict(event=event, pid=os.getpid(), monotonic_ns=time.monotonic_ns(), **fields)
        logger.info("[KVPOOL_RANGE_DEBUG] %s", json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Diagnostic failures must not change ownership or transfer results.
        pass
