"""Time stock vLLM's LLM(...) constructor.

The request-loop clients in stock.py take a live engine and do not
own construction. This helper times the cold constructor wall and,
when vLLM's INFO logs expose them, splits weight loading from KV
cache profiling. Warm reuse is a zeroed dict for the kept instance.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any


# vLLM 0.26 startup lines that bracket weight load vs KV work.
# Matched case-insensitively against the assembled log record message.
_WEIGHT_DONE = re.compile(
    r"(finished loading|loading model weights|model loading took|"
    r"loaded weights)",
    re.I,
)
_KV_MARK = re.compile(
    r"(kv cache|gpu KV cache|profiling|available.+kv|"
    r"captured.+cudagraph|cuda graph)",
    re.I,
)


class _StartupStamp(logging.Handler):
    """Record wall times when known startup messages appear."""

    def __init__(self, t0: float):
        super().__init__(level=logging.DEBUG)
        self.t0 = t0
        self.weight_done_s: float | None = None
        self.kv_mark_s: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        elapsed = time.perf_counter() - self.t0
        if self.weight_done_s is None and _WEIGHT_DONE.search(msg):
            self.weight_done_s = elapsed
        if _KV_MARK.search(msg):
            # keep the first KV-related mark after weights (or any)
            if self.kv_mark_s is None:
                self.kv_mark_s = elapsed


def _attach_stamp(stamp: _StartupStamp) -> list[tuple[logging.Logger, int]]:
    """Raise vLLM loggers to INFO and attach the stamp handler.

    Returns prior levels so the caller can restore them.
    """
    names = ("vllm", "vllm.engine", "vllm.worker",
             "vllm.worker.worker", "vllm.v1", "vllm.v1.engine",
             "vllm.v1.core", "vllm.model_executor")
    restored = []
    for name in names:
        log = logging.getLogger(name)
        restored.append((log, log.level))
        if log.level == logging.NOTSET or log.level > logging.INFO:
            log.setLevel(logging.INFO)
        log.addHandler(stamp)
    return restored


def _detach_stamp(stamp: _StartupStamp,
                  restored: list[tuple[logging.Logger, int]]) -> None:
    for log, level in restored:
        log.removeHandler(stamp)
        log.setLevel(level)


def warm_boot_dict() -> dict:
    """Boot breakdown when LLM is already alive in this process."""
    return dict(kind="warm", llm_init_s=0.0, weight_load_s=None,
                kv_profile_s=None, boot_s=0.0)


def time_llm_boot(**llm_kwargs: Any) -> tuple[Any, dict]:
    """Cold-construct vLLM's LLM(...). Returns (llm, boot_dict).

    Always records the full constructor wall. weight_load_s and
    kv_profile_s are filled from log markers when present; otherwise
    they stay null (vLLM does not expose sub-timers on the public
    surface).
    """
    from vllm import LLM

    stamp = _StartupStamp(time.perf_counter())
    restored = _attach_stamp(stamp)
    t0 = time.perf_counter()
    try:
        llm = LLM(**llm_kwargs)
    finally:
        _detach_stamp(stamp, restored)
    total = time.perf_counter() - t0

    weight_load_s = None
    kv_profile_s = None
    if stamp.weight_done_s is not None:
        weight_load_s = round(stamp.weight_done_s, 2)
        # remainder of constructor after weights ≈ KV profile + pool
        kv_profile_s = round(max(0.0, total - stamp.weight_done_s), 2)
    elif stamp.kv_mark_s is not None:
        # only a KV mark: treat everything before it as weights
        weight_load_s = round(stamp.kv_mark_s, 2)
        kv_profile_s = round(max(0.0, total - stamp.kv_mark_s), 2)

    boot = dict(kind="cold",
                llm_init_s=round(total, 2),
                weight_load_s=weight_load_s,
                kv_profile_s=kv_profile_s,
                boot_s=round(total, 2))
    return llm, boot
