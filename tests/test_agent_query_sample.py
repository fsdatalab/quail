"""Checks for the agent query sample experiment."""

from experiments.agent_query_sample import (
    _comparison,
    parse_function_calls,
    stable_sample_rows,
)


def test_stable_sample_rows_does_not_depend_on_input_order():
    rows = [{"id": f"d{index}"} for index in range(20)]

    forward = stable_sample_rows(rows, 5)
    reverse = stable_sample_rows(list(reversed(rows)), 5)

    assert forward == reverse
    assert len(forward) == 5


def test_comparison_treats_32b_as_reference():
    reference = {
        "selectivity": 0.5,
        "answers": [
            {"id": "a", "answer": True},
            {"id": "b", "answer": True},
            {"id": "c", "answer": False},
            {"id": "d", "answer": False},
        ],
    }
    candidate = {
        "selectivity": 0.5,
        "answers": [
            {"id": "a", "answer": True},
            {"id": "b", "answer": False},
            {"id": "c", "answer": True},
            {"id": "d", "answer": False},
        ],
    }

    result = _comparison(reference, candidate)

    assert result == {
        "agreement": 0.5,
        "selectivity_gap": 0.0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }


def test_parse_function_calls_requires_sample_and_both_models():
    value = (
        "sample=fc-sample,qwen3-32b-fp8=fc-32b,"
        "qwen3-4b-fp8=fc-4b"
    )

    assert parse_function_calls(value) == {
        "sample": "fc-sample",
        "qwen3-32b-fp8": "fc-32b",
        "qwen3-4b-fp8": "fc-4b",
    }
