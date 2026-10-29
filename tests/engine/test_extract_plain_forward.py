"""CPU checks for the unpaged extraction-trace probe; no Modal invocation."""

import numpy as np
import pytest
import torch

from experiments.cells.extract_plain_forward import (
    answer_prefix,
    compare_scores,
    plain_chunk,
    trace_index,
)


def test_plain_probe_keeps_the_exact_sequence_and_has_no_kv_inputs():
    chunk = plain_chunk(torch, [101, 7, 9, 42], "cpu")
    assert chunk.input_ids.tolist() == [101, 7, 9, 42]
    assert chunk.positions.tolist() == [0, 1, 2, 3]
    assert chunk.final_indices.tolist() == [3]
    assert chunk.meta["cu_a"].tolist() == [0, 4]
    assert chunk.attention_mode == "unified"
    assert all(
        chunk.meta[key] is None
        for key in ("unified", "shared", "reads", "kv_src", "kv_dst")
    )
    with pytest.raises(ValueError, match="nonempty"):
        plain_chunk(torch, [], "cpu")


def test_trace_prefixes_reject_missing_steps_stops_and_duplicate_queries():
    records = [
        dict(document=3, question=2, index=i, chosen=token)
        for i, token in enumerate([101, 42, -1])
    ]
    indexed = trace_index(records)
    assert answer_prefix(indexed, 3, 2, 2) == [101, 42]
    with pytest.raises(ValueError, match="stopped"):
        answer_prefix(indexed, 3, 2, 3)
    with pytest.raises(ValueError, match="prefix step"):
        answer_prefix(trace_index(records[1:]), 3, 2, 2)
    with pytest.raises(ValueError, match="duplicate"):
        trace_index([*records, records[0]])


def test_plain_score_comparison_uses_the_trace_token_ids_and_grouped_stop():
    reference = dict(
        chosen=3,
        allowed=3,
        runner_up=2,
        top=3,
        allowed_lp=-0.7,
        top_lp=-0.7,
        margin=0.3,
        stop_lp=-3.1,
    )
    compared = compare_scores(np.array([-3, -2, -1, -0.5]), -2.8, reference)
    assert compared["tokens"]["allowed"]["delta"] == pytest.approx(0.2)
    assert compared["tokens"]["runner_up"]["delta"] == pytest.approx(0)
    assert compared["stop_delta"] == pytest.approx(0.3)
