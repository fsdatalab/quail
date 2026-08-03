import copy
from time import perf_counter_ns

from docengine.runtime.batching import (
    BatchLimits,
    VariableLengthBatchPacker,
    WorkKind,
)
from docengine.runtime.custom import DocEngineRuntime, RunnerOutput
from docengine.runtime.kv import KVPageAllocator
from docengine.runtime.protocol import FilterQuery
from docengine.runtime.trace import TraceRecorder


class ScriptedRunner:
    def __init__(self, query, outcomes):
        self.query = query
        self.outcomes = outcomes
        self.calls = []

    def execute(self, batch, states):
        started = perf_counter_ns()
        answers = {}
        for chunk in batch.chunks:
            self.calls.append(chunk)
            if chunk.kind is WorkKind.PREFILL:
                continue
            state = states[chunk.document_id]
            boundary = state.body_tokens - chunk.cached_prefix_tokens
            total = sum(
                boundary + len(self.query.question_token_ids[stage])
                for stage in range(
                    state.stage,
                    state.stage + chunk.k,
                )
            )
            if chunk.token_end == total:
                answers[chunk.work_id] = tuple(
                    self.outcomes[chunk.document_id][stage]
                    for stage in range(
                        state.stage,
                        state.stage + chunk.k,
                    )
                )
        return RunnerOutput(
            answers=answers,
            started_ns=started,
            ended_ns=perf_counter_ns(),
        )


def query():
    return FilterQuery.from_sequences(
        body_token_ids=[
            [1] * 5,
            [2] * 20,
            [3] * 2,
        ],
        question_token_ids=[
            [10, 11],
            [12, 13],
        ],
        yes_token_ids=[99],
    )


def runtime(speculation_k=1, trace=None):
    q = query()
    outcomes = [
        [1, 1],
        [1, 0],
        [0, 1],
    ]
    runner = ScriptedRunner(q, outcomes)
    engine = DocEngineRuntime(
        query=q,
        runner=runner,
        packer=VariableLengthBatchPacker(BatchLimits(
            max_new_tokens=16,
            max_sequences=3,
            max_temporary_bytes=10_000,
        )),
        kv=KVPageAllocator(
            total_pages=16,
            page_size_tokens=4,
            bytes_per_token=8,
        ),
        trace=trace,
        speculation_k=speculation_k,
    )
    return engine, runner


def test_runtime_continuously_packs_variable_length_work():
    trace = TraceRecorder()
    engine, runner = runtime(trace=trace)
    result = engine.run()
    assert result.survivors == (0,)
    assert result.rejected == (1, 2)
    assert result.answers == {
        (0, 1): 1,
        (0, 2): 1,
        (1, 1): 1,
        (1, 2): 0,
        (2, 1): 0,
    }
    assert result.steps >= 3
    assert any(
        row.prefill_tokens and row.decode_tokens
        for row in trace.records
    )
    assert any(
        chunk.new_tokens < 20
        for chunk in runner.calls
        if chunk.kind is WorkKind.PREFILL
        and chunk.document_id == 1
    )


def test_fused_k_records_only_logically_required_answers():
    engine, _runner = runtime(speculation_k=2)
    result = engine.run()
    assert result.survivors == (0,)
    assert result.answers[(1, 2)] == 0
    assert (2, 2) not in result.answers


def test_hot_path_does_not_deepcopy(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("deepcopy called")

    monkeypatch.setattr(copy, "deepcopy", fail)
    engine, _runner = runtime()
    assert engine.run().survivors == (0,)
