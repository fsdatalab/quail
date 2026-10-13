"""Extraction costs: the executor's rounds priced like a classification."""

import pytest

from quail.cost.extract import (
    ExtractAssumptions,
    document_work,
    estimate,
)
from quail.cost.work import ask, scan
from quail.physical import ExtractSpec
from quail.specs import H100_SXM, QWEN3_4B_FP8

HEAD, TAIL = 2, 40
CHUNK = 10000


def _cost(lengths, **kwargs):
    options = dict(chunk=CHUNK, model=QWEN3_4B_FP8, device=H100_SXM,
                   max_answer_tokens=ExtractSpec.max_answer_tokens,
                   draft_tokens=ExtractSpec.draft_tokens)
    options.update(kwargs)
    live = options.pop("live", len(lengths))
    return estimate(live, lengths, HEAD, TAIL, **options)



def _document(length, **kwargs):
    return document_work(length, HEAD, TAIL,
                         max_answer_tokens=ExtractSpec.max_answer_tokens,
                         draft_tokens=ExtractSpec.draft_tokens, **kwargs)


def test_a_document_scans_once_then_writes_its_answer_in_rounds():
    work, fed, rows = _document(100)
    context = HEAD + 100 + TAIL
    answer = 1 + ExtractSpec.draft_tokens
    expected = scan(HEAD + 100, TAIL) + ask(context, answer)
    assert work == expected
    assert fed == TAIL + answer
    assert rows == 1 + answer
    # resident KV turns the scan into a question over it
    resident, _, _ = _document(100, resident=True)
    assert resident == ask(HEAD + 100, TAIL) + ask(context, answer)
    assert resident.tokens < work.tokens
    shared, _, _ = _document(100, shared=80)
    assert shared == ask(80, HEAD + 100 + TAIL - 80) + ask(context, answer)
    assert shared.tokens == work.tokens - 80
    assert _document(100, resident=True, shared=80)[0] == resident


def test_parallel_estimate_preserves_work_and_limits_used_workers():
    one = _cost([100, 100])
    two = _cost([100, 100], workers=2)
    assert two.work == one.work
    assert two.seconds == pytest.approx(one.seconds / 2)
    assert _cost([100], workers=8).seconds == _cost([100]).seconds


def test_cost_uses_plan_limits_and_separate_answer_assumptions():
    assumptions = ExtractAssumptions(answer_tokens=40)
    cost = _cost([100], max_answer_tokens=8, draft_tokens=3, assumptions=assumptions)
    # Both rounds read KV. The second also reads the first round's four tokens.
    assert cost.work.kv_read == 142 + 146
    assert cost.work.pairs == 142 * 143 / 2 + 4 * 142 + 10 + 4 * 146 + 10
    assert cost.work.tokens == 150
    assert cost.suffix_tokens == TAIL + 8
    assert cost.rows == 9
    shorter = _cost([100], max_answer_tokens=8, draft_tokens=3,
                    assumptions=ExtractAssumptions(answer_tokens=2))
    assert shorter.work == scan(HEAD + 100, TAIL) + ask(HEAD + 100 + TAIL, 4)
    capped = _cost([100], max_answer_tokens=1,
                   assumptions=assumptions)
    assert capped.work == scan(HEAD + 100, TAIL)
    windowed, _, _ = document_work(
        100, HEAD, TAIL, window=128, max_answer_tokens=8, draft_tokens=3,
        assumptions=assumptions)
    assert windowed.sliding_kv_read == 2 * 127
    assert windowed.sliding_pairs == 128 * 142 - 128 * 127 / 2 + 8 * 128
