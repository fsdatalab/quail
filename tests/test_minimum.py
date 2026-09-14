"""CPU checks for the prefix trie and the minimum input tokens of a run."""

import pyarrow as pa
import pytest

from quail_b.minimum import (
    DocumentTokens,
    minimum_input_tokens,
    prefix_trie_size,
    token_metrics,
    validate_prompt_pieces,
)
from quail_b.queries import QuerySpec
from quail_b.scoring import RunOutput
from tools.make_substrait_plans import Filter, Join, Scan, build_plan


def _spec(query_id, description, tree):
    return QuerySpec.from_plan(query_id, description, build_plan(tree))


def _encode(texts):
    return [[byte + 1 for byte in text.encode("utf-8")] for text in texts]


def _ids(text):
    return _encode([text])[0]


def _lcp(left, right):
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return n


PRE = _ids("DOCUMENT:\n")
QUESTION = _ids("\n\nuseful?\nANSWER:")
FRAME = _ids("\n\nDoes it mention the aspect?")
LABEL = _ids("\n\nASPECT:\n")
TAIL = _ids("\nANSWER:")


def test_prefix_trie_size_counts_shared_prefixes_once():
    assert prefix_trie_size([[1, 2, 3], [1, 2, 4], [9]]) == 5
    assert prefix_trie_size([[1, 2], [1, 2]]) == 2
    assert prefix_trie_size([(), (7,)]) == 1
    assert prefix_trie_size([]) == 0


def test_minimum_input_tokens_counts_each_document_prefix_once_and_pairs_apart():
    spec = _spec(
        "TEST-1",
        "one filter then one join",
        Join(
            Filter(Scan("docs", "d", "body"), "useful {0}"),
            Scan("aspects", "a", "name"),
            ("d", "a"),
            "{0} mentions {1}",
        ),
    )
    corpus = {
        "docs": pa.table({"id": ["a", "b", "c"],
                          "body": ["same start one", "same start two", "other"]}),
        "aspects": pa.table({"id": ["x", "y"], "name": ["aa", "ab"]}),
    }
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
        "joins": [{"id": "join-1", "anchor": "d", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    filter_answers = {"filter-1": pa.table({
        "d": ["a", "b", "c"], "answer": [True, True, False]})}
    join_answers = {"join-1": pa.table({
        "d": ["a", "a", "b", "b"], "a": ["x", "y", "x", "y"],
        "answer": [True, False, True, True]})}
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, join_answers,
        DocumentTokens(corpus, _encode))

    # the three documents share the preamble, and the first two share
    # "same start " (11 tokens) beyond it; every document gets the
    # question once, the two anchors get the frame once, sharing the
    # lead it has in common with the question; each pair gets its own
    # label, partner, and tail
    pre = len(PRE)
    documents = 3 * pre + 14 + 14 + 5 - (pre + pre + 11)
    anchored = len(QUESTION) + len(FRAME) - _lcp(QUESTION, FRAME)
    pairs = 2 * len(LABEL) + 4 + 2 * len(TAIL)
    assert minimum == documents + len(QUESTION) + 2 * anchored + 2 * pairs


def test_minimum_input_tokens_counts_a_document_once_across_uses():
    second = Filter(Scan("docs", "d2", "body"), "short {0}")
    second = Filter(second, "clear {0}")
    spec = _spec(
        "TEST-2",
        "a self join with two filter stages on one side",
        Join(Scan("docs", "d1", "body"), second, ("d1", "d2"), "{0} before {1}"),
    )
    corpus = {"docs": pa.table({"id": ["a", "b"], "body": ["alpha", "beta"]})}
    first, second = _ids("\n\nshort?\nANSWER:"), _ids("\n\nclear?\nANSWER:")
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": first},
                    {"id": "filter-2", "tail": second}],
        "joins": [{"id": "join-1", "anchor": "d1", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    # both rows pass the first stage, only "alpha" reaches the second;
    # the join anchors on d1, the same two documents
    filter_answers = {
        "filter-1": pa.table({"d2": ["a", "b"], "answer": [True, True]}),
        "filter-2": pa.table({"d2": ["a"], "answer": [True]}),
    }
    join_answers = {"join-1": pa.table({
        "d1": ["a", "a", "b", "b"], "d2": ["a", "b", "a", "b"],
        "answer": [True, True, False, True]})}
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, join_answers,
        DocumentTokens(corpus, _encode))

    documents = 2 * len(PRE) + 5 + 4 - len(PRE)
    alpha = len(first) + len(second) + len(FRAME) - sum((
        _lcp(first, second), max(_lcp(FRAME, first), _lcp(FRAME, second))))
    beta = len(first) + len(FRAME) - _lcp(first, FRAME)
    pairs = 2 * len(LABEL) + 9 + 2 * len(TAIL)
    assert minimum == documents + alpha + beta + 2 * pairs


def _filter_spec():
    return _spec(
        "TEST-0",
        "one filter",
        Filter(Scan("docs", "d", "body"), "useful {0}"),
    )


def _filter_output(**kwargs):
    return RunOutput(
        {"filter-1": pa.table({"d": ["a"], "answer": [True]})},
        {}, pa.table({"d": ["a"]}), runtime_s=1.0, **kwargs)


def test_token_metrics_record_fresh_tokens_without_prompt_pieces():
    spec = _filter_spec()
    output = _filter_output(measurements={"fresh_tokens": 40})
    assert token_metrics(spec, output, {}) == {
        "fresh_tokens": 40, "minimum_tokens": None, "regret_tokens": None,
    }
    assert token_metrics(spec, _filter_output(), {}) == {
        "fresh_tokens": None, "minimum_tokens": None, "regret_tokens": None,
    }


def test_token_metrics_require_fresh_tokens_with_prompt_pieces():
    spec = _filter_spec()
    pieces = {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    }
    output = _filter_output(prompt_pieces=pieces)
    with pytest.raises(ValueError, match="fresh_tokens"):
        token_metrics(spec, output, {})
    output.measurements = {"fresh_tokens": True}
    with pytest.raises(ValueError, match="nonnegative"):
        token_metrics(spec, output, {})
    output.prompt_pieces = None
    output.measurements = {"fresh_tokens": -1}
    with pytest.raises(ValueError, match="nonnegative"):
        token_metrics(spec, output, {})


def test_token_metrics_need_answers_with_prompt_pieces():
    spec = _filter_spec()
    pieces = {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"id": "filter-1", "tail": QUESTION}],
    }
    output = RunOutput(
        None, None, pa.table({"d": ["a"]}), runtime_s=1.0,
        measurements={"fresh_tokens": 10}, prompt_pieces=pieces)
    with pytest.raises(ValueError, match="filter and join answers"):
        token_metrics(spec, output, {})
