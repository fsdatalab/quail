"""CPU checks for the Modal wrapper around the QUAIL-B labeling pass."""

import pytest

from quail.bench.judge_pass import parse_function_calls


def test_parse_function_calls_requires_every_workload():
    calls = parse_function_calls(
        "imdb=fc-imdb,biodex=fc-bio,fever=fc-fever,"
        "lepard=fc-lepard,agent=fc-agent")
    assert calls == {
        "imdb": "fc-imdb",
        "biodex": "fc-bio",
        "fever": "fc-fever",
        "lepard": "fc-lepard",
        "agent": "fc-agent",
    }


def test_parse_function_calls_rejects_a_missing_workload():
    with pytest.raises(ValueError, match="missing"):
        parse_function_calls("imdb=fc-imdb")
