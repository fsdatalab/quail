"""Structured engine-step traces."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ScheduledChunkTrace:
    document_id: int
    filter_start: int
    k: int
    new_tokens: int
    cached_prefix_tokens: int
    kv_pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class EngineStepTrace:
    step: int
    started_ns: int
    ended_ns: int
    prefill_tokens: int
    decode_tokens: int
    sequence_count: int
    queued_documents: int
    queued_filter_calls: int
    hbm_used_bytes: int | None
    hbm_free_bytes: int | None
    token_capacity: int | None
    sequence_capacity: int | None
    planning_ns: int
    input_preparation_ns: int
    idle_before_ns: int
    cache_reset_confirmed: bool | None
    unused_capacity_reason: str | None
    request_ids: tuple[str, ...] = ()
    chunks: tuple[ScheduledChunkTrace, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.ended_ns - self.started_ns


class TraceRecorder:
    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path is not None else None
        self._records: list[EngineStepTrace] = []
        self._lock = threading.Lock()
        self._handle = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("x", encoding="utf-8")

    @property
    def records(self) -> tuple[EngineStepTrace, ...]:
        with self._lock:
            return tuple(self._records)

    def record(self, trace: EngineStepTrace) -> None:
        payload = asdict(trace)
        with self._lock:
            self._records.append(trace)
            if self._handle is not None:
                self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
                self._handle.flush()

    def extend(self, traces: Sequence[EngineStepTrace]) -> None:
        for trace in traces:
            self.record(trace)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
