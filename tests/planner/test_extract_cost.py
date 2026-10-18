"""Extraction costs: the executor's rounds priced like a classification."""

import math
from dataclasses import replace

import pytest

from quail.cost.extract import (
    ExtractAssumptions,
    document_work,
    estimate,
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
    live = options.pop("live", len(lengths))
    return estimate(live, lengths, HEAD, TAIL, **options)



def _document(length, **kwargs):
    options = dict(max_answer_tokens=ExtractSpec.max_answer_tokens,
                   draft_tokens=16, draft_policy="fixed")
    options.update(kwargs)
    return document_work(length, HEAD, TAIL, **options)


def _rounds_work(context, round_tokens, limit=64, mean=16):
    """Average work over every possible capped answer length."""
    total = Work()
    q = 1 - 1 / mean
    cap = math.ceil((limit - 1) / round_tokens)
    for length in range(1, limit + 1):
        probability = q ** (length - 1) * (1 if length == limit else 1 - q)
        count = min(cap, math.ceil(length / round_tokens))
        total += decode(context, [round_tokens] * count) * probability
    return total


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
    answer = _rounds_work(context, round_tokens)
    assert vars(work) == pytest.approx(vars(scan(HEAD + 100, TAIL) + answer))
    assert fed == pytest.approx(TAIL + rounds * round_tokens)
    assert rows == pytest.approx(1 + rounds * round_tokens)
    # resident KV turns the scan into a question over it
    resident, _, _ = _document(100, resident=True)
    assert vars(resident) == pytest.approx(vars(ask(HEAD + 100, TAIL) + answer))
    assert resident.tokens < work.tokens
    shared, _, _ = _document(100, shared=80)
    assert vars(shared) == pytest.approx(vars(ask(80, HEAD + 100 + TAIL - 80) + answer))
    assert shared.tokens == pytest.approx(work.tokens - 80)
    assert _document(100, resident=True, shared=80)[0] == resident


def test_parallel_estimate_preserves_work_and_limits_used_workers(monkeypatch):
    lengths, memo = [100, 100], {}
    one = _cost(lengths, memo=memo)
    with monkeypatch.context() as patch:
        patch.setattr("quail.cost.extract._document_steps",
                      lambda *args, **kwargs: pytest.fail("rescanned document lengths"))
        two = _cost(lengths, workers=2, memo=memo)
    assert two.work == one.work
    assert one.seconds / 2 < two.seconds <= one.seconds
    assert two.seconds == pytest.approx(_cost([100]).seconds)
    assert _cost([100], workers=8).seconds == _cost([100]).seconds


def test_cost_uses_plan_limits_and_separate_answer_assumptions():
    assumptions = ExtractAssumptions(answer_tokens=40)
    cost = _cost([100], max_answer_tokens=8, draft_tokens=3, assumptions=assumptions)
    context = HEAD + 100 + TAIL
    rounds = assumptions.rounds(8, 3)
    assert rounds == pytest.approx(1 + (39 / 40) ** 4)
    assert replace(cost.work, readout_flops=0, readout_bytes=0, batches=()) == (
        scan(HEAD + 100, TAIL) + _rounds_work(context, 4, limit=8, mean=40))
    assert cost.suffix_tokens == pytest.approx(TAIL + 4 * rounds)
    assert cost.rows == pytest.approx(1 + 4 * rounds)
    shorter = _cost([100], max_answer_tokens=8, draft_tokens=3,
                    assumptions=ExtractAssumptions(answer_tokens=2))
    assert shorter.work.tokens < cost.work.tokens
    capped = _cost([100], max_answer_tokens=1, assumptions=assumptions)
    assert replace(capped.work, readout_flops=0, readout_bytes=0, batches=()) == (
        scan(HEAD + 100, TAIL))
    assert capped.work.readout_flops == 2 * QWEN3_4B_FP8.hidden * QWEN3_4B_FP8.vocab
    windowed, _, _ = document_work(
        100, HEAD, TAIL, window=128, max_answer_tokens=8, draft_tokens=3,
        draft_policy="fixed", assumptions=assumptions)
    assert windowed.sliding_kv_read == pytest.approx(rounds * 127)


def test_adaptive_policy_drafts_longer_for_longer_documents():
    # the round's KV read grows with the document, so the policy drafts
    # more and takes fewer rounds than a fixed short draft; a scan reads
    # no KV, so the KV read counts the rounds alone
    per_token = 0.02
    long, long_fed, _ = _document(8000, draft_tokens=63, draft_policy="adaptive",
                                  round_cost_per_token=per_token)
    fixed, fixed_fed, _ = _document(8000, draft_tokens=4, draft_policy="fixed")
    assert long.kv_read < fixed.kv_read
    short, short_fed, _ = _document(400, draft_tokens=63, draft_policy="adaptive",
                                    round_cost_per_token=per_token)
    # a short document's cheaper KV read earns a shorter draft
    assert short_fed - TAIL < long_fed - TAIL
