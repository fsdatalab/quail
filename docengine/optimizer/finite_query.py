"""Exact small-query planning and measured batch selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from math import prod
from typing import Callable, Protocol, Sequence

from docengine.runtime.batching import (
    BatchPlan,
    VariableLengthBatchPacker,
    WorkItem,
)
from docengine.runtime.kv import KVPageAllocator


@dataclass(frozen=True)
class FiniteQueryProblem:
    document_lengths: tuple[int, ...]
    pass_probabilities: tuple[float, ...]
    max_batch_documents: int
    max_k: int

    @property
    def n_documents(self) -> int:
        return len(self.document_lengths)

    @property
    def n_filters(self) -> int:
        return len(self.pass_probabilities)

    def validate(self) -> None:
        if not self.document_lengths:
            raise ValueError("problem needs documents")
        if not self.pass_probabilities:
            raise ValueError("problem needs filters")
        if any(length <= 0 for length in self.document_lengths):
            raise ValueError("document lengths must be positive")
        if any(
            not 0.0 <= probability <= 1.0
            for probability in self.pass_probabilities
        ):
            raise ValueError("pass probabilities must be between zero and one")
        if self.max_batch_documents <= 0:
            raise ValueError("max_batch_documents must be positive")
        if self.max_k <= 0:
            raise ValueError("max_k must be positive")


@dataclass(frozen=True)
class FinitePlannerState:
    stages: tuple[int, ...]
    resident: tuple[bool, ...]

    def done(self, n_filters: int) -> bool:
        return all(stage < 0 or stage >= n_filters for stage in self.stages)


@dataclass(frozen=True)
class FiniteWork:
    document: int
    stage: int
    k: int


@dataclass(frozen=True)
class FiniteBatch:
    work: tuple[FiniteWork, ...]


FiniteBatchCost = Callable[
    [FinitePlannerState, FiniteBatch, FiniteQueryProblem],
    float,
]


@dataclass(frozen=True)
class ExactPlanResult:
    expected_seconds: float
    first_batch: FiniteBatch | None


class ExactFiniteQueryPlanner:
    def __init__(
        self,
        problem: FiniteQueryProblem,
        batch_cost: FiniteBatchCost,
    ):
        problem.validate()
        self.problem = problem
        self.batch_cost = batch_cost

    def initial_state(self) -> FinitePlannerState:
        return FinitePlannerState(
            stages=(0,) * self.problem.n_documents,
            resident=(False,) * self.problem.n_documents,
        )

    def solve(
        self,
        state: FinitePlannerState | None = None,
    ) -> ExactPlanResult:
        root = state or self.initial_state()

        @lru_cache(maxsize=None)
        def value(current: FinitePlannerState) -> tuple[float, FiniteBatch | None]:
            if current.done(self.problem.n_filters):
                return 0.0, None
            best = float("inf")
            best_batch = None
            for batch in self.legal_batches(current):
                immediate = self.batch_cost(current, batch, self.problem)
                expected = 0.0
                for probability, next_state in self.transitions(current, batch):
                    expected += probability * value(next_state)[0]
                total = immediate + expected
                if total < best:
                    best = total
                    best_batch = batch
            return best, best_batch

        seconds, first = value(root)
        return ExactPlanResult(seconds, first)

    def legal_batches(
        self,
        state: FinitePlannerState,
    ) -> tuple[FiniteBatch, ...]:
        active = [
            document
            for document, stage in enumerate(state.stages)
            if 0 <= stage < self.problem.n_filters
        ]
        batches = []
        for count in range(
            1,
            min(self.problem.max_batch_documents, len(active)) + 1,
        ):
            for documents in combinations(active, count):
                choices = []
                for document in documents:
                    stage = state.stages[document]
                    max_k = min(
                        self.problem.max_k,
                        self.problem.n_filters - stage,
                    )
                    choices.append(range(1, max_k + 1))
                for ks in product(*choices):
                    batches.append(FiniteBatch(tuple(
                        FiniteWork(
                            document=document,
                            stage=state.stages[document],
                            k=k,
                        )
                        for document, k in zip(documents, ks)
                    )))
        return tuple(batches)

    def transitions(
        self,
        state: FinitePlannerState,
        batch: FiniteBatch,
    ) -> tuple[tuple[float, FinitePlannerState], ...]:
        per_work = [
            self._work_outcomes(work)
            for work in batch.work
        ]
        combined = {}
        for outcomes in product(*per_work):
            probability = prod(item[0] for item in outcomes)
            stages = list(state.stages)
            resident = list(state.resident)
            for work, (_probability, passes) in zip(batch.work, outcomes):
                resident[work.document] = True
                if passes < work.k:
                    stages[work.document] = -1
                    resident[work.document] = False
                else:
                    stages[work.document] += work.k
                    if stages[work.document] >= self.problem.n_filters:
                        resident[work.document] = False
            next_state = FinitePlannerState(
                stages=tuple(stages),
                resident=tuple(resident),
            )
            combined[next_state] = combined.get(next_state, 0.0) + probability
        return tuple(
            (probability, next_state)
            for next_state, probability in combined.items()
            if probability > 0.0
        )

    def _work_outcomes(
        self,
        work: FiniteWork,
    ) -> tuple[tuple[float, int], ...]:
        rows = []
        survival = 1.0
        for offset in range(work.k):
            probability = self.problem.pass_probabilities[
                work.stage + offset
            ]
            rows.append((survival * (1.0 - probability), offset))
            survival *= probability
        rows.append((survival, work.k))
        return tuple(
            (probability, passes)
            for probability, passes in rows
            if probability > 0.0
        )


class BatchPlanEstimator(Protocol):
    def __call__(self, batch: BatchPlan) -> float:
        ...


class MeasuredBatchPlanner:
    def __init__(
        self,
        *,
        packer: VariableLengthBatchPacker,
        estimate_seconds: BatchPlanEstimator,
    ):
        self.packer = packer
        self.estimate_seconds = estimate_seconds

    def choose(
        self,
        ready: Sequence[WorkItem],
        kv: KVPageAllocator,
    ) -> BatchPlan:
        if not ready:
            return self.packer.pack(ready, kv)
        orderings = self._orderings(ready)
        candidates = []
        seen = set()
        for ordering in orderings:
            batch = self.packer.pack(ordering, kv)
            signature = tuple(
                (chunk.work_id, chunk.new_tokens)
                for chunk in batch.chunks
            )
            if not batch.chunks or signature in seen:
                continue
            seen.add(signature)
            useful = sum(
                chunk.new_tokens * chunk.useful_probability
                for chunk in batch.chunks
            )
            seconds = self.estimate_seconds(batch)
            if seconds <= 0.0:
                raise ValueError("batch estimate must be positive")
            candidates.append((
                -(useful / seconds),
                seconds,
                signature,
                batch,
            ))
        if not candidates:
            return self.packer.pack(ready, kv)
        candidates.sort(key=lambda row: row[:3])
        return candidates[0][3]

    def _orderings(
        self,
        ready: Sequence[WorkItem],
    ) -> tuple[tuple[WorkItem, ...], ...]:
        return (
            tuple(ready),
            tuple(sorted(
                ready,
                key=lambda item: (
                    -item.cached_prefix_tokens,
                    item.remaining_tokens,
                    item.document_id,
                ),
            )),
            tuple(sorted(
                ready,
                key=lambda item: (
                    item.remaining_tokens,
                    -item.useful_probability,
                    item.document_id,
                ),
            )),
            tuple(sorted(
                ready,
                key=lambda item: (
                    -item.useful_probability,
                    -item.remaining_tokens,
                    item.document_id,
                ),
            )),
        )
