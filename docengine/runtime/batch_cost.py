"""Measured primitive tables and analytical batch-time estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SharedPrefixGroup:
    prefix_tokens: int
    query_tokens: tuple[int, ...]

    @property
    def attention_pairs(self) -> int:
        return sum(
            query * self.prefix_tokens + query * (query + 1) // 2
            for query in self.query_tokens
        )


@dataclass(frozen=True)
class BatchFeatures:
    new_tokens: tuple[int, ...]
    cached_prefix_tokens: tuple[int, ...]
    output_tokens: int
    standard_query_tokens: tuple[int, ...] = ()
    shared_prefix_groups: tuple[SharedPrefixGroup, ...] = ()
    kv_transfer_bytes: int = 0

    def validate(self) -> None:
        if not self.new_tokens:
            raise ValueError("batch needs at least one sequence")
        if len(self.new_tokens) != len(self.cached_prefix_tokens):
            raise ValueError("new-token and cached-prefix rows must align")
        if any(value < 0 for value in self.new_tokens):
            raise ValueError("new tokens cannot be negative")
        if any(value < 0 for value in self.cached_prefix_tokens):
            raise ValueError("cached prefix tokens cannot be negative")
        if self.output_tokens < 0:
            raise ValueError("output tokens cannot be negative")
        if self.kv_transfer_bytes < 0:
            raise ValueError("KV transfer bytes cannot be negative")

    @property
    def total_new_tokens(self) -> int:
        return sum(self.new_tokens)

    @property
    def standard_attention_pairs(self) -> int:
        if self.standard_query_tokens:
            if len(self.standard_query_tokens) != len(
                self.cached_prefix_tokens
            ):
                raise ValueError(
                    "standard query-token rows must align with prefixes"
                )
            return sum(
                query * prefix + query * (query + 1) // 2
                for query, prefix in zip(
                    self.standard_query_tokens,
                    self.cached_prefix_tokens,
                )
            )
        return sum(
            query * prefix + query * (query + 1) // 2
            for query, prefix in zip(
                self.new_tokens,
                self.cached_prefix_tokens,
            )
        )

    @property
    def cascade_attention_pairs(self) -> int:
        return sum(group.attention_pairs for group in self.shared_prefix_groups)


@dataclass(frozen=True)
class PrimitivePoint:
    work: int
    time_ns: int


class PrimitiveTimeTable:
    def __init__(self, points: Sequence[PrimitivePoint]):
        if not points:
            raise ValueError("primitive table needs measured points")
        ordered = sorted(points, key=lambda point: point.work)
        if ordered[0].work < 0:
            raise ValueError("primitive work cannot be negative")
        if any(point.time_ns < 0 for point in ordered):
            raise ValueError("primitive time cannot be negative")
        if len({point.work for point in ordered}) != len(ordered):
            raise ValueError("primitive work points must be unique")
        if any(
            right.time_ns < left.time_ns
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("primitive time must be monotone")
        self.points = tuple(ordered)

    def estimate_ns(self, work: int) -> int:
        if work < 0:
            raise ValueError("work cannot be negative")
        if work == 0:
            return 0
        if work <= self.points[0].work:
            first = self.points[0]
            if first.work == 0:
                return first.time_ns
            return round(first.time_ns * work / first.work)
        for left, right in zip(self.points, self.points[1:]):
            if work <= right.work:
                span = right.work - left.work
                fraction = (work - left.work) / span
                return round(
                    left.time_ns
                    + fraction * (right.time_ns - left.time_ns)
                )
        last = self.points[-1]
        if last.work == 0:
            return last.time_ns
        return round(last.time_ns * work / last.work)

    def to_rows(self) -> list[dict[str, int]]:
        return [asdict(point) for point in self.points]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[dict[str, int]],
    ) -> "PrimitiveTimeTable":
        return cls([PrimitivePoint(**row) for row in rows])


@dataclass(frozen=True)
class AttentionShapePoint:
    groups: int
    k: int
    prefix_tokens: int
    tail_tokens: int
    time_ns: int


class AttentionShapeTable:
    def __init__(self, points: Sequence[AttentionShapePoint]):
        if not points:
            raise ValueError("attention shape table needs measured points")
        if any(
            point.groups <= 0
            or point.k <= 0
            or point.prefix_tokens < 0
            or point.tail_tokens <= 0
            or point.time_ns <= 0
            for point in points
        ):
            raise ValueError("invalid attention shape point")
        keys = [
            (
                point.groups,
                point.k,
                point.prefix_tokens,
                point.tail_tokens,
            )
            for point in points
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("attention shape points must be unique")
        self.points = tuple(points)

    def estimate_ns(
        self,
        *,
        groups: int,
        k: int,
        prefix_tokens: int,
        tail_tokens: int,
    ) -> int:
        measured_k = min(
            {point.k for point in self.points},
            key=lambda value: abs(value - k),
        )
        tail_rows = []
        measured_tails = sorted({
            point.tail_tokens
            for point in self.points
            if point.k == measured_k
        })
        for measured_tail in measured_tails:
            candidates = [
                point for point in self.points
                if point.k == measured_k
                and point.tail_tokens == measured_tail
            ]
            by_group = {}
            for point in candidates:
                by_group.setdefault(point.groups, []).append(point)
            group_rows = [
                (
                    measured_groups,
                    self._estimate_prefix(
                        points,
                        prefix_tokens,
                        measured_tail,
                    ),
                )
                for measured_groups, points in sorted(by_group.items())
            ]
            tail_rows.append((
                measured_tail,
                self._interpolate_axis(group_rows, groups),
            ))
        base = self._interpolate_axis(tail_rows, tail_tokens)
        return round(base * k / measured_k)

    def _estimate_prefix(
        self,
        points: Sequence[AttentionShapePoint],
        prefix_tokens: int,
        tail_tokens: int,
    ) -> float:
        ordered = sorted(points, key=lambda point: point.prefix_tokens)
        if len(ordered) == 1:
            point = ordered[0]
            work = max(1, prefix_tokens + tail_tokens)
            measured = max(
                1,
                point.prefix_tokens + point.tail_tokens,
            )
            return point.time_ns * work / measured
        if prefix_tokens <= ordered[0].prefix_tokens:
            left, right = ordered[0], ordered[1]
        elif prefix_tokens >= ordered[-1].prefix_tokens:
            left, right = ordered[-2], ordered[-1]
        else:
            left, right = next(
                (left, right)
                for left, right in zip(ordered, ordered[1:])
                if left.prefix_tokens <= prefix_tokens <= right.prefix_tokens
            )
        fraction = (
            (prefix_tokens - left.prefix_tokens)
            / (right.prefix_tokens - left.prefix_tokens)
        )
        return left.time_ns + fraction * (
            right.time_ns - left.time_ns
        )

    def _interpolate_axis(
        self,
        rows: Sequence[tuple[int, float]],
        value: int,
    ) -> float:
        if len(rows) == 1:
            measured_value, measured_time = rows[0]
            return measured_time * value / measured_value
        if value <= rows[0][0]:
            left, right = rows[0], rows[1]
        elif value >= rows[-1][0]:
            left, right = rows[-2], rows[-1]
        else:
            left, right = next(
                (left, right)
                for left, right in zip(rows, rows[1:])
                if left[0] <= value <= right[0]
            )
        fraction = (value - left[0]) / (right[0] - left[0])
        return left[1] + fraction * (right[1] - left[1])

    def to_rows(self) -> list[dict[str, int]]:
        return [asdict(point) for point in self.points]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[dict[str, int]],
    ) -> "AttentionShapeTable":
        return cls([AttentionShapePoint(**row) for row in rows])


@dataclass(frozen=True)
class BatchTimeEstimate:
    dense_ns: int
    standard_attention_ns: int
    cascade_attention_ns: int
    output_ns: int
    kv_transfer_ns: int
    fixed_ns: int

    @property
    def total_ns(self) -> int:
        return (
            self.dense_ns
            + self.standard_attention_ns
            + self.cascade_attention_ns
            + self.output_ns
            + self.kv_transfer_ns
            + self.fixed_ns
        )


class MeasuredBatchTimeEstimator:
    def __init__(
        self,
        *,
        dense: PrimitiveTimeTable,
        standard_attention: PrimitiveTimeTable,
        cascade_attention: PrimitiveTimeTable,
        output: PrimitiveTimeTable,
        kv_transfer: PrimitiveTimeTable,
        fixed_ns: int,
    ):
        if fixed_ns < 0:
            raise ValueError("fixed time cannot be negative")
        self.dense = dense
        self.standard_attention = standard_attention
        self.cascade_attention = cascade_attention
        self.output = output
        self.kv_transfer = kv_transfer
        self.fixed_ns = int(fixed_ns)

    def estimate(self, features: BatchFeatures) -> BatchTimeEstimate:
        features.validate()
        return BatchTimeEstimate(
            dense_ns=self.dense.estimate_ns(features.total_new_tokens),
            standard_attention_ns=self.standard_attention.estimate_ns(
                features.standard_attention_pairs
            ),
            cascade_attention_ns=self.cascade_attention.estimate_ns(
                features.cascade_attention_pairs
            ),
            output_ns=self.output.estimate_ns(features.output_tokens),
            kv_transfer_ns=self.kv_transfer.estimate_ns(
                features.kv_transfer_bytes
            ),
            fixed_ns=self.fixed_ns,
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "dense": self.dense.to_rows(),
            "standard_attention": self.standard_attention.to_rows(),
            "cascade_attention": self.cascade_attention.to_rows(),
            "output": self.output.to_rows(),
            "kv_transfer": self.kv_transfer.to_rows(),
            "fixed_ns": self.fixed_ns,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "MeasuredBatchTimeEstimator":
        payload = json.loads(Path(path).read_text())
        return cls(
            dense=PrimitiveTimeTable.from_rows(payload["dense"]),
            standard_attention=PrimitiveTimeTable.from_rows(
                payload["standard_attention"]
            ),
            cascade_attention=PrimitiveTimeTable.from_rows(
                payload["cascade_attention"]
            ),
            output=PrimitiveTimeTable.from_rows(payload["output"]),
            kv_transfer=PrimitiveTimeTable.from_rows(payload["kv_transfer"]),
            fixed_ns=payload["fixed_ns"],
        )


@dataclass(frozen=True)
class ValidationRow:
    predicted_ns: int
    measured_ns: int
    label: str

    @property
    def absolute_error_fraction(self) -> float:
        if self.measured_ns <= 0:
            raise ValueError("measured time must be positive")
        return abs(self.predicted_ns - self.measured_ns) / self.measured_ns

    @property
    def signed_error_fraction(self) -> float:
        if self.measured_ns <= 0:
            raise ValueError("measured time must be positive")
        return (self.predicted_ns - self.measured_ns) / self.measured_ns


@dataclass(frozen=True)
class ValidationSummary:
    count: int
    median_absolute_error: float
    p95_absolute_error: float
    mean_signed_error: float

    @property
    def passes(self) -> bool:
        return (
            self.median_absolute_error <= 0.05
            and self.p95_absolute_error <= 0.10
        )


def summarize_validation(
    rows: Iterable[ValidationRow],
) -> ValidationSummary:
    collected = tuple(rows)
    if not collected:
        raise ValueError("validation needs at least one row")
    absolute = sorted(row.absolute_error_fraction for row in collected)
    index = max(0, min(len(absolute) - 1, ceil_index(0.95, len(absolute))))
    signed = [row.signed_error_fraction for row in collected]
    return ValidationSummary(
        count=len(collected),
        median_absolute_error=median(absolute),
        p95_absolute_error=absolute[index],
        mean_signed_error=sum(signed) / len(signed),
    )


def ceil_index(fraction: float, count: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    if count <= 0:
        raise ValueError("count must be positive")
    position = fraction * count
    integer = int(position)
    if integer < position:
        integer += 1
    return max(0, integer - 1)
