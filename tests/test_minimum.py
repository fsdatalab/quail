"""CPU checks for the prefix trie and the minimum input tokens of a run."""

import pyarrow as pa

from quail_b.minimum import (
    DocumentTokens,
    minimum_input_tokens,
    prefix_trie_size,
    validate_prompt_pieces,
)
from quail_b.queries import AliasSpec, JoinSpec, QuerySpec


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


def test_minimum_input_tokens_counts_each_distinct_prefix_once():
    spec = QuerySpec(
        "TEST-1", "one filter then one join",
        (AliasSpec("d", "docs", "body", ("useful {0}",)),
         AliasSpec("a", "aspects", "name")),
        (JoinSpec("{0} mentions {1}", ("d", "a")),),
        ("d.id", "a.id"))
    corpus = {
        "docs": pa.table({"id": ["a", "b", "c"],
                          "body": ["same start one", "same start two", "other"]}),
        "aspects": pa.table({"id": ["x", "y"], "name": ["aa", "ab"]}),
    }
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"alias": "d", "position": 0, "tail": QUESTION}],
        "joins": [{"position": 0, "anchor": "d", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    filter_answers = {("d", 0): pa.table({
        "d": ["a", "b", "c"], "answer": [True, True, False]})}
    join_answers = {0: pa.table({
        "d": ["a", "a", "b", "b"], "a": ["x", "y", "x", "y"],
        "answer": [True, False, True, True]})}
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, join_answers,
        DocumentTokens(corpus, _encode))

    # the three documents share the preamble, and the first two share
    # "same start " (11 tokens) beyond it; every document gets the
    # question once, the two anchors get the frame once, and each
    # anchor's pairs share the label and the partners' first token
    pre = len(PRE)
    documents = 3 * pre + 14 + 14 + 5 - (pre + pre + 11)
    anchored = len(QUESTION) + len(FRAME) - _lcp(QUESTION, FRAME)
    pairs = len(LABEL) + 3 + 2 * len(TAIL)
    assert minimum == documents + len(QUESTION) + 2 * anchored + 2 * pairs


def test_minimum_input_tokens_counts_a_document_once_across_uses():
    spec = QuerySpec(
        "TEST-2", "a self join with two filter stages on one side",
        (AliasSpec("d1", "docs", "body"),
         AliasSpec("d2", "docs", "body", ("short {0}", "clear {0}"))),
        (JoinSpec("{0} before {1}", ("d1", "d2")),),
        ("d1.id", "d2.id"))
    corpus = {"docs": pa.table({"id": ["a", "b"], "body": ["alpha", "beta"]})}
    first, second = _ids("\n\nshort?\nANSWER:"), _ids("\n\nclear?\nANSWER:")
    pieces = validate_prompt_pieces(spec, {
        "tokenizer": "test", "preamble": PRE,
        "filters": [{"alias": "d2", "position": 0, "tail": first},
                    {"alias": "d2", "position": 1, "tail": second}],
        "joins": [{"position": 0, "anchor": "d1", "frame": FRAME,
                   "label": LABEL, "tail": TAIL}],
    })
    # both rows pass the first stage, only "alpha" reaches the second;
    # the join anchors on d1, the same two documents
    filter_answers = {
        ("d2", 0): pa.table({"d2": ["a", "b"], "answer": [True, True]}),
        ("d2", 1): pa.table({"d2": ["a"], "answer": [True]}),
    }
    join_answers = {0: pa.table({
        "d1": ["a", "a", "b", "b"], "d2": ["a", "b", "a", "b"],
        "answer": [True, True, False, True]})}
    minimum = minimum_input_tokens(
        spec, pieces, filter_answers, join_answers,
        DocumentTokens(corpus, _encode))

    documents = 2 * len(PRE) + 5 + 4 - len(PRE)
    alpha = len(first) + len(second) + len(FRAME) - sum((
        _lcp(first, second), max(_lcp(FRAME, first), _lcp(FRAME, second))))
    beta = len(first) + len(FRAME) - _lcp(first, FRAME)
    pairs = len(LABEL) + 9 + 2 * len(TAIL)
    assert minimum == documents + alpha + beta + 2 * pairs
