from dataclasses import asdict
import gzip
import json

import pytest

from docengine.runtime.artifacts import RunMetadata, write_run_artifact
from docengine.runtime.protocol import (
    FilterQuery,
    GroundTruthLabels,
    evaluate_answers,
)
from docengine.runtime.trace import EngineStepTrace, TraceRecorder


def _query():
    return FilterQuery.from_sequences(
        body_token_ids=[[1, 2], [3, 4, 5]],
        question_token_ids=[[10], [11]],
        yes_token_ids=[99],
    )


def test_runtime_query_has_no_ground_truth():
    query = _query()
    assert set(asdict(query)) == {
        "body_token_ids",
        "question_token_ids",
        "yes_token_ids",
    }
    assert query.n_documents == 2
    assert query.n_filters == 2


def test_ground_truth_is_evaluator_only():
    query = _query()
    labels = GroundTruthLabels.from_sequences([[1, 0], [1, 1]])
    labels.validate_for(query)
    evaluation = evaluate_answers(
        {(0, 1): 1, (0, 2): 1, (1, 1): 1, (1, 2): 1},
        labels,
    )
    assert evaluation.attempted == 4
    assert evaluation.correct == 3
    assert evaluation.wrong == ((0, 2),)
    assert evaluation.expected_survivors == (1,)


def test_ground_truth_shape_must_match_query():
    with pytest.raises(ValueError):
        GroundTruthLabels.from_sequences([[1]]).validate_for(_query())


def test_run_artifacts_are_unique_and_immutable(tmp_path):
    metadata = RunMetadata.create(
        phase="unit",
        config={"n": 2},
        seeds={"workload": 7},
        model_revision="model-sha",
        dataset_revision="dataset-sha",
        command=("pytest",),
        repo=tmp_path,
    )
    directory = write_run_artifact(tmp_path, metadata, {"value": 1})
    assert (directory / "metadata.json").exists()
    with gzip.open(directory / "result.json.gz", "rt") as handle:
        assert json.load(handle) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_run_artifact(tmp_path, metadata, {"value": 2})


def test_trace_recorder_writes_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = EngineStepTrace(
        step=0,
        started_ns=10,
        ended_ns=20,
        prefill_tokens=8,
        decode_tokens=1,
        sequence_count=2,
        queued_documents=3,
        queued_filter_calls=4,
        hbm_used_bytes=100,
        hbm_free_bytes=200,
        token_capacity=16,
        sequence_capacity=8,
        planning_ns=2,
        input_preparation_ns=1,
        idle_before_ns=0,
        cache_reset_confirmed=True,
        unused_capacity_reason=None,
    )
    with TraceRecorder(path) as recorder:
        recorder.record(trace)
        assert recorder.records == (trace,)
    row = json.loads(path.read_text().strip())
    assert row["step"] == 0
    assert row["prefill_tokens"] == 8
