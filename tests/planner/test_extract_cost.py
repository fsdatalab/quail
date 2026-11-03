"""Extraction costs: the executor's rounds priced like a classification."""

from dataclasses import fields, replace

import numpy as np
import pytest

from quail.cost.draft_policy import DraftPolicy
from quail.cost.extract import (
    ExtractAssumptions,
    document_work,
    estimate,
    table_work,
)
from quail.cost.work import Work, ask, decode, scan
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


def _assert_same_work(work, expected):
    for field in fields(Work):
        assert getattr(work, field.name) == pytest.approx(
            getattr(expected, field.name), rel=1e-12), field.name


def _rounds_work(context, round_tokens, limit, assumptions):
    """The decode work of every round, each weighted by its chance to run."""
    work = Work()
    for index in range(-(-(limit - 1) // round_tokens)):
        step = decode(context + index * round_tokens, [round_tokens])
        work += step * assumptions.survival(index * round_tokens)
    return work


def _priced_round_by_round(length, shared, *, window, recurrent, cap, rho,
                           limit, assumptions):
    """One document's work, adding each round's ask weighted by its chance."""
    prefix = HEAD + length
    context = prefix + TAIL
    policy = DraftPolicy(cap, limit, assumptions.answer_tokens)
    round_tokens = 1 + policy.draft_length(rho * context, 1, num_streams=1)
    if shared:
        work = ask(HEAD + shared, context - HEAD - shared, window=window)
    else:
        work = scan(prefix, TAIL, window=window)
    if recurrent:
        work = replace(work, state_read=1 + bool(shared), state_written=2)
    for index in range(-(-(limit - 1) // round_tokens)):
        probability = assumptions.survival(index * round_tokens)
        if recurrent:
            replay = round_tokens if index else 0
            step = ask(context + max(0, index - 1) * round_tokens,
                       replay + round_tokens, window=window)
            step = replace(step, state_read=1 + bool(replay),
                           state_written=float(bool(replay)))
        else:
            step = ask(context + index * round_tokens, round_tokens, window=window)
            step = replace(step, state_written=step.state_read)
        work += step * probability
    return work


@pytest.mark.parametrize("window", [0, 512])
@pytest.mark.parametrize("recurrent", [False, True])
def test_a_table_costs_the_sum_of_its_documents_priced_round_by_round(
        window, recurrent):
    lengths = np.array([3, 40, 300, 480, 505, 900, 2500])
    shared = np.array([0, 20, 0, 400, 0, 600, 1000])
    assumptions = ExtractAssumptions(answer_tokens=64)
    options = dict(window=window, recurrent=recurrent, cap=16, rho=0.02,
                   limit=1024, assumptions=assumptions)
    expected = Work()
    for length, borrowed in zip(lengths, shared):
        expected += _priced_round_by_round(int(length), int(borrowed), **options)
    work, _, _ = table_work(
        lengths, HEAD, TAIL, window=window, resident=False, shared=shared,
        max_answer_tokens=1024, draft_tokens=16, draft_policy="adaptive",
        round_cost_per_token=0.02, assumptions=assumptions, recurrent=recurrent)
    _assert_same_work(work, expected)


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
    answer = _rounds_work(context, round_tokens, ExtractSpec.max_answer_tokens,
                          ExtractAssumptions())
    _assert_same_work(work, scan(HEAD + 100, TAIL) + answer)
    assert fed == pytest.approx(TAIL + rounds * round_tokens)
    assert rows == pytest.approx(1 + rounds * round_tokens)
    # resident KV turns the scan into a question over it
    resident, _, _ = _document(100, resident=True)
    _assert_same_work(resident, ask(HEAD + 100, TAIL) + answer)
    assert resident.tokens < work.tokens
    # a borrowed prefix includes the prompt head
    shared, _, _ = _document(100, shared=80)
    _assert_same_work(shared, ask(HEAD + 80, 100 + TAIL - 80) + answer)
    assert shared.tokens == pytest.approx(work.tokens - 80 - HEAD)
    assert _document(100, resident=True, shared=80)[0] == resident


def test_parallel_estimate_preserves_work_and_limits_used_workers():
    one = _cost([100, 100])
    two = _cost([100, 100], gpus=2)
    assert two.work == one.work
    assert two.seconds == pytest.approx(one.seconds / 2)
    assert _cost([100], gpus=8).seconds == _cost([100]).seconds






