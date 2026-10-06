"""Analytical classification costs against explicit label outcomes."""

import itertools
from dataclasses import asdict, replace

import pytest

from quail.cost.classify import estimate
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.work import Work
from quail.specs import H100_SXM, QWEN3_4B_FP8

LABELS = ((10,), (20, 21), (30, 31, 32, 33, 34))


def _cost(lengths, labels=LABELS, **kwargs):
    options = dict(lengths=lengths, shared=(), chunk=10000, capacity=100000,
                   model=QWEN3_4B_FP8, device=H100_SXM)
    options.update(kwargs)
    live = options.pop("live", len(lengths))
    return estimate("trie_decode", live, 2, 3, labels, **options)


def _enumerated_work(lengths, depths, window=0):
    """Count causal attention edges and retained KV reads for one outcome."""
    total = dict.fromkeys(asdict(Work()), 0.0)
    for length, depth in zip(lengths, depths):
        prompt = 2 + length + 3
        tokens = prompt + depth
        total["tokens"] += tokens
        total["kv_written"] += tokens
        total["pairs"] += sum(range(1, tokens + 1))
        total["kv_read"] += sum(range(prompt + 1, prompt + depth))
        if window:
            total["sliding_pairs"] += sum(min(i, window)
                                           for i in range(1, tokens + 1))
            total["sliding_kv_read"] += sum(min(i, window - 1)
                                             for i in range(prompt + 1, prompt + depth))
    return total


@pytest.mark.parametrize("lengths,labels,window", [
    ((10, 90), LABELS, 0),
    ((10,), ((10,), (20,), (30, 31, 32, 33, 34)), 16),
])
def test_expected_work_and_passes_match_all_label_combinations(lengths, labels, window):
    model = replace(QWEN3_4B_FP8, sliding_window=window, full_attention_period=2)
    cost = _cost(lengths, labels, model=model)
    outcomes = list(itertools.product(map(len, labels), repeat=len(lengths)))
    expected = dict.fromkeys(asdict(Work()), 0.0)
    for depths in outcomes:
        for key, value in _enumerated_work(lengths, depths, window).items():
            expected[key] += value / len(outcomes)
    assert asdict(cost.work) == pytest.approx(expected)
    assert cost.passes == pytest.approx(
        sum(max(depths) for depths in outcomes) / len(outcomes))
    assert cost.suffix_tokens == pytest.approx(
        sum(sum(depths) for depths in outcomes) / len(outcomes))
    assert cost.rounds == 5


def test_label_and_document_order_do_not_change_the_estimate():
    lengths = (17, 21, 99, 105)
    expected = _cost(lengths, chunk=140)
    for labels in itertools.permutations(LABELS):
        assert _cost(lengths[::-1], labels, chunk=140) == expected
    assert expected.suffix_tokens == pytest.approx(4 * 8 / 3)


@pytest.mark.parametrize("budget", ["chunk", "capacity"])
def test_smaller_batches_pay_more_weight_reads_without_changing_work(budget):
    device = replace(H100_SXM, peak_flops=float("inf"), bf16_flops=float("inf"))
    large = _cost((10, 20, 30), device=device)
    small = _cost((10, 20, 30), device=device, **{budget: 40})
    assert asdict(small.work) == pytest.approx(asdict(large.work))
    assert small.suffix_tokens == large.suffix_tokens
    assert small.passes > large.passes
    assert small.seconds > large.seconds
    weights = sum(component.bytes_moved for component in dense_decoder_components(
        Work(), QWEN3_4B_FP8, passes=1))
    head = 2 * QWEN3_4B_FP8.hidden * QWEN3_4B_FP8.vocab
    for cost in (large, small):
        kv = QWEN3_4B_FP8.kappa * (cost.work.kv_written + cost.work.kv_read)
        assert cost.seconds == pytest.approx(
            ((weights + head) * cost.passes + kv) / device.hbm_bw)


def test_kv_reserves_the_longest_continuation():
    # The first prompts occupy 16 + 26 tokens; five-token labels need 20 + 30.
    separate = _cost((10, 20), capacity=49)
    together = _cost((10, 20), capacity=50)
    assert separate.work == together.work
    assert separate.passes == pytest.approx(2 * 8 / 3)
    assert together.passes < separate.passes


@pytest.mark.parametrize("live", [0, 0.25, 1.8])
def test_expected_document_counts_are_not_rounded(live):
    cost = _cost((10,), live=live)
    single = _cost((10,))
    assert asdict(cost.work) == pytest.approx(
        {key: value * live for key, value in asdict(single.work).items()})
    assert cost.suffix_tokens == pytest.approx(live * 8 / 3)
    assert cost.seconds == pytest.approx(single.seconds * live)
    assert cost.passes == pytest.approx(single.passes * live)
    assert cost.rounds == (5 if live else 0)
