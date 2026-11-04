"""Extraction costs: the executor's rounds priced like a classification."""

from dataclasses import fields, replace

import numpy as np
import pytest
from helpers.rust_cost import Cost, cost_model, native

from quail.common.draft_policy import DraftPolicy
from quail.common.work import Work, ask, decode, scan
from quail.physical import ExtractSpec
from quail.specs import H100_SXM, QWEN3_4B_FP8

HEAD, TAIL = 2, 40
CHUNK = 10000
# the mean answer length the cost model assumes unless told otherwise
ANSWER_TOKENS = 16


def _cost(lengths, *, gpus=1):
    return Cost(*cost_model().extract_estimate(
        len(lengths), lengths, [], HEAD, TAIL, CHUNK, QWEN3_4B_FP8.name,
        H100_SXM.name, False, ExtractSpec.max_answer_tokens, 16, "fixed", 0.0,
        gpus, ANSWER_TOKENS))


def _document(length, *, resident=False, shared=0):
    """One document's work fields, fed tokens, and readout rows."""
    [record] = native.CostModel.extract_document_work(
        [length], [shared], HEAD, TAIL, 0, resident,
        ExtractSpec.max_answer_tokens, 16, "fixed", 0.0, ANSWER_TOKENS, False)
    return record


def _survival(answer_tokens, tokens):
    return native.CostModel.extract_survival(answer_tokens, tokens)


def _rounds(answer_tokens, max_answer_tokens, draft_tokens):
    return native.CostModel.extract_rounds(
        answer_tokens, max_answer_tokens, draft_tokens)


def _assert_same_work(work, expected):
    for field in fields(Work):
        assert work[field.name] == pytest.approx(
            getattr(expected, field.name), rel=1e-12), field.name


def _rounds_work(context, round_tokens, limit, answer_tokens):
    """The decode work of every round, each weighted by its chance to run."""
    work = Work()
    for index in range(-(-(limit - 1) // round_tokens)):
        step = decode(context + index * round_tokens, [round_tokens])
        work += step * _survival(answer_tokens, index * round_tokens)
    return work


def _priced_round_by_round(length, shared, *, window, recurrent, cap, rho,
                           limit, answer_tokens):
    """One document's work, adding each round's ask weighted by its chance."""
    prefix = HEAD + length
    context = prefix + TAIL
    policy = DraftPolicy(cap, limit, answer_tokens)
    round_tokens = 1 + policy.draft_length(rho * context, 1, num_streams=1)
    if shared:
        work = ask(HEAD + shared, context - HEAD - shared, window=window)
    else:
        work = scan(prefix, TAIL, window=window)
    if recurrent:
        work = replace(work, state_read=1 + bool(shared), state_written=2)
    for index in range(-(-(limit - 1) // round_tokens)):
        probability = _survival(answer_tokens, index * round_tokens)
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
    options = dict(window=window, recurrent=recurrent, cap=16, rho=0.02,
                   limit=1024, answer_tokens=64)
    expected = Work()
    for length, borrowed in zip(lengths, shared):
        expected += _priced_round_by_round(int(length), int(borrowed), **options)
    work, _, _ = native.CostModel.extract_table_work(
        lengths.tolist(), shared.tolist(), HEAD, TAIL, window, False, 1024, 16,
        "adaptive", 0.02, 64, recurrent)
    _assert_same_work(work, expected)


def test_expected_rounds_follow_the_answer_length_curve():
    assert _survival(16, 0) == 1.0
    assert _survival(16, 16) == pytest.approx((15 / 16) ** 16)
    # with no draft, a round per token the answer may still have
    assert _rounds(16, 64, 0) == pytest.approx(
        sum((15 / 16) ** r for r in range(63)))
    # a 17-token round covers most answers of mean 16 in one go
    rounds = _rounds(16, 64, 16)
    assert 1 < rounds < 2
    assert _rounds(16, 64, 63) == 1.0
    assert _rounds(16, 1, 16) == 0.0
    assert _rounds(0, 64, 16) == 0.0
    # longer answers take more rounds; longer drafts fewer
    assert _rounds(40, 64, 16) > rounds
    assert _rounds(16, 64, 32) < rounds


def test_a_document_scans_once_then_writes_its_answer_in_rounds():
    work, fed, rows = _document(100)
    context = HEAD + 100 + TAIL
    round_tokens = 1 + 16
    rounds = _rounds(ANSWER_TOKENS, ExtractSpec.max_answer_tokens, 16)
    answer = _rounds_work(context, round_tokens, ExtractSpec.max_answer_tokens,
                          ANSWER_TOKENS)
    _assert_same_work(work, scan(HEAD + 100, TAIL) + answer)
    assert fed == pytest.approx(TAIL + rounds * round_tokens)
    assert rows == pytest.approx(1 + rounds * round_tokens)
    # resident KV turns the scan into a question over it
    resident, _, _ = _document(100, resident=True)
    _assert_same_work(resident, ask(HEAD + 100, TAIL) + answer)
    assert resident["tokens"] < work["tokens"]
    # a borrowed prefix includes the prompt head
    shared, _, _ = _document(100, shared=80)
    _assert_same_work(shared, ask(HEAD + 80, 100 + TAIL - 80) + answer)
    assert shared["tokens"] == pytest.approx(work["tokens"] - 80 - HEAD)
    assert _document(100, resident=True, shared=80)[0] == resident


def test_parallel_estimate_preserves_work_and_limits_used_workers():
    one = _cost([100, 100])
    two = _cost([100, 100], gpus=2)
    assert two.work == one.work
    assert two.seconds == pytest.approx(one.seconds / 2)
    assert _cost([100], gpus=8).seconds == _cost([100]).seconds
