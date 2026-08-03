import pytest

from docengine.optimizer.finite_query import (
    ExactFiniteQueryPlanner,
    FiniteQueryProblem,
    MeasuredBatchPlanner,
)
from docengine.runtime.batching import (
    BatchLimits,
    VariableLengthBatchPacker,
    WorkItem,
    WorkKind,
)
from docengine.runtime.kv import KVPageAllocator


def problem():
    return FiniteQueryProblem(
        document_lengths=(10,),
        pass_probabilities=(0.9, 0.9),
        max_batch_documents=1,
        max_k=2,
    )


def test_exact_planner_chooses_cheap_speculation():
    planner = ExactFiniteQueryPlanner(
        problem(),
        batch_cost=lambda _state, batch, _problem: (
            1.2 if batch.work[0].k == 2 else 1.0
        ),
    )
    result = planner.solve()
    assert result.expected_seconds == pytest.approx(1.2)
    assert result.first_batch.work[0].k == 2


def test_exact_planner_rejects_expensive_speculation():
    planner = ExactFiniteQueryPlanner(
        problem(),
        batch_cost=lambda _state, batch, _problem: (
            3.0 if batch.work[0].k == 2 else 1.0
        ),
    )
    result = planner.solve()
    assert result.expected_seconds == pytest.approx(1.9)
    assert result.first_batch.work[0].k == 1


def test_transition_probabilities_cover_all_outcomes():
    planner = ExactFiniteQueryPlanner(problem(), lambda *_args: 1.0)
    state = planner.initial_state()
    batch = next(
        batch for batch in planner.legal_batches(state)
        if batch.work[0].k == 2
    )
    transitions = planner.transitions(state, batch)
    assert sum(probability for probability, _ in transitions) == pytest.approx(1.0)
    assert {next_state.stages[0] for _, next_state in transitions} == {-1, 2}


def work(name, tokens, document, probability):
    return WorkItem(
        work_id=name,
        owner=name,
        document_id=document,
        filter_start=0,
        k=1,
        kind=WorkKind.PREFILL,
        token_offset=0,
        total_new_tokens=tokens,
        cached_prefix_tokens=0,
        useful_probability=probability,
    )


def test_measured_planner_selects_best_exact_packing():
    packer = VariableLengthBatchPacker(BatchLimits(
        max_new_tokens=8,
        max_sequences=1,
        max_temporary_bytes=0,
    ))
    planner = MeasuredBatchPlanner(
        packer=packer,
        estimate_seconds=lambda batch: (
            1.0 if batch.chunks[0].work_id == "short" else 10.0
        ),
    )
    batch = planner.choose(
        [
            work("long", 8, 0, 1.0),
            work("short", 4, 1, 1.0),
        ],
        KVPageAllocator(
            total_pages=8,
            page_size_tokens=4,
            bytes_per_token=8,
        ),
    )
    assert batch.chunks[0].work_id == "short"
