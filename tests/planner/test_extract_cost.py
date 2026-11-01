"""Extraction costs: the executor's rounds priced like a classification."""

import pytest

from quail.cost.extract import (
    ExtractAssumptions,
    document_work,
    estimate,
)
from quail.cost.work import ask, decode, scan
from quail.physical import ExtractSpec
from quail.specs import H100_SXM, QWEN3_4B_FP8

HEAD, TAIL = 2, 40
CHUNK = 10000


def _cost(lengths, **kwargs):
    options = dict(chunk=CHUNK, model=QWEN3_4B_FP8, device=H100_SXM,
                   max_answer_tokens=ExtractSpec.max_answer_tokens,
                   draft_tokens=16, draft_policy="fixed")
    options.update(kwargs)
    expected_rows = options.pop("expected_rows", len(lengths))
    return estimate(expected_rows, lengths, HEAD, TAIL, **options)



def _document(length, **kwargs):
    options = dict(max_answer_tokens=ExtractSpec.max_answer_tokens,
                   draft_tokens=16, draft_policy="fixed")
    options.update(kwargs)
    return document_work(length, HEAD, TAIL, **options)


def _rounds_work(context, round_tokens, rounds):
    """The decode work of a fractional number of equal rounds."""
    whole = int(rounds)
    work = decode(context, [round_tokens] * whole)
    if rounds > whole:
        work += (decode(context + whole * round_tokens, [round_tokens])
                 * (rounds - whole))
    return work


def test_expected_rounds_follow_the_answer_length_curve():
    assumptions = ExtractAssumptions(answer_tokens=16)
    assert assumptions.survival(0) == 1.0
    assert assumptions.survival(16) == pytest.approx((15 / 16) ** 16)
    # with no draft, a round per token the answer may still have
    assert assumptions.rounds(64, 0) == pytest.approx(
        sum((15 / 16) ** r for r in range(63)))
    # a 17-token round covers most answers of mean 16 in one go
    rounds = assumptions.rounds(64, 16)
    assert 1 < rounds < 2
    assert assumptions.rounds(64, 63) == 1.0
    assert assumptions.rounds(1, 16) == 0.0
    assert ExtractAssumptions(answer_tokens=0).rounds(64, 16) == 0.0
    # longer answers take more rounds; longer drafts fewer
    assert ExtractAssumptions(answer_tokens=40).rounds(64, 16) > rounds
    assert assumptions.rounds(64, 32) < rounds


def test_a_document_scans_once_then_writes_its_answer_in_rounds():
    work, fed, rows = _document(100)
    context = HEAD + 100 + TAIL
    round_tokens = 1 + 16
    rounds = ExtractAssumptions().rounds(ExtractSpec.max_answer_tokens, 16)
    answer = _rounds_work(context, round_tokens, rounds)
    assert work == scan(HEAD + 100, TAIL) + answer
    assert fed == pytest.approx(TAIL + rounds * round_tokens)
    assert rows == pytest.approx(1 + rounds * round_tokens)
    # resident KV turns the scan into a question over it
    resident, _, _ = _document(100, resident=True)
    assert resident == ask(HEAD + 100, TAIL) + answer
    assert resident.tokens < work.tokens
    # a borrowed prefix includes the prompt head
    shared, _, _ = _document(100, shared=80)
    assert shared == ask(HEAD + 80, 100 + TAIL - 80) + answer
    assert shared.tokens == pytest.approx(work.tokens - 80 - HEAD)
    assert _document(100, resident=True, shared=80)[0] == resident


def test_parallel_estimate_preserves_work_and_limits_used_workers():
    one = _cost([100, 100])
    two = _cost([100, 100], gpus=2)
    assert two.work == one.work
    assert two.seconds == pytest.approx(one.seconds / 2)
    assert _cost([100], gpus=8).seconds == _cost([100]).seconds






