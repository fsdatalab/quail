"""Analytical classification costs against explicit label outcomes."""

import itertools
from dataclasses import asdict, replace

import pytest

from quail.cost.classify import SAMPLE_DOCUMENTS, estimate, estimate_chains
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.work import Work
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

LABELS = ((10,), (20, 21), (30, 31, 32, 33, 34))


def _cost(lengths, labels=LABELS, **kwargs):
    options = dict(lengths=lengths, shared=(), chunk=10000, capacity=100000,
                   model=QWEN3_4B_FP8, device=H100_SXM)
    options.update(kwargs)
    live = options.pop("live", len(lengths))
    return estimate("trie_decode", live, 2, 3, labels, **options)


def _enumerated_work(lengths, depths, window=0):
    """Count all causal attention edges and suffix KV reads for one outcome."""
    total = dict.fromkeys(asdict(Work()), 0.0)
    for length, depth in zip(lengths, depths):
        prompt = 2 + length + 3
        tokens = prompt + depth
        total["tokens"] += tokens
        total["kv_written"] += tokens
        total["pairs"] += sum(range(1, tokens + 1))
        total["kv_read"] += sum(range(prompt, prompt + depth))
        if window:
            total["sliding_pairs"] += sum(min(i, window)
                                           for i in range(1, tokens + 1))
            total["sliding_kv_read"] += sum(min(i, window - 1)
                                             for i in range(prompt, prompt + depth))
    return total


@pytest.mark.parametrize("lengths", [(10,), (10, 90)])
@pytest.mark.parametrize("labels", [LABELS, ((10,), (20,), (30, 31, 32, 33, 34))])
@pytest.mark.parametrize("window", [0, 16])
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


def test_length_variation_preserves_attention_work():
    varied = _cost((10, 90))
    uniform = _cost((50, 50))
    assert varied.work.tokens == uniform.work.tokens
    assert varied.work.kv_read == uniform.work.kv_read
    assert varied.work.pairs - uniform.work.pairs == pytest.approx(1600)


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


@pytest.mark.parametrize("live", [0, 0.25, 1.8, SAMPLE_DOCUMENTS * 2.5])
def test_expected_document_counts_are_not_rounded(live):
    cost = _cost((10,), live=live)
    single = _cost((10,))
    assert asdict(cost.work) == pytest.approx(
        {key: value * live for key, value in asdict(single.work).items()})
    assert cost.suffix_tokens == pytest.approx(live * 8 / 3)
    assert cost.seconds == pytest.approx(single.seconds * live)
    assert cost.passes == pytest.approx(single.passes * live)
    assert cost.rounds == (5 if live else 0)


def test_sampling_scales_expected_label_work():
    count = SAMPLE_DOCUMENTS + 17
    cost = _cost((10,) * count)
    assert cost.suffix_tokens == pytest.approx(count * 8 / 3)
    assert cost.work.tokens == pytest.approx(count * (15 + 8 / 3))
    empty = _cost((), live=12)
    assert empty.work == Work()
    assert empty.seconds == empty.passes == empty.suffix_tokens == empty.rounds == 0


@pytest.mark.parametrize("resident,shared", [(False, (0, 6)), (True, ())])
def test_resident_and_shared_prefixes_reduce_prefill_work(resident, shared):
    cost = estimate_chains(
        2, 3, [1], live=2, lengths=(10, 20), shared=shared,
        chunk=100, capacity=1000, model=QWEN3_4B_FP8, device=H100_SXM,
        resident=resident)
    prefixes = (12, 22)
    reused = prefixes if resident else (0, 8)
    assert cost.work.tokens == sum(prefixes) + 2 * 4 - sum(reused)
    assert cost.work.pairs == sum(
        sum(range(1, prefix + 5)) - sum(range(1, reuse + 1))
        for prefix, reuse in zip(prefixes, reused))
    assert cost.work.kv_read == sum(prefixes) + 2 * 3 + sum(reused)
    assert cost.suffix_tokens == 2
    assert cost.rounds == cost.passes == 1


def test_diffusion_draws_keep_canvas_and_sliding_attention_work():
    model = DIFFUSION_GEMMA_26B_FP8
    canvas = model.answer_canvas.rows
    options = dict(lengths=(100, 1200), shared=(), chunk=10000, capacity=100000,
                   model=model, device=H100_SXM)
    cost = estimate("letters", 2, 2, 3, ((10,), (20,)), draws=4, **options)
    assert cost.work.tokens == 1300 + 2 * (5 + 4 * (1 + canvas))
    assert cost.suffix_tokens == 2 * 4 * (1 + canvas)
    assert cost.work.sliding_pairs < cost.work.pairs
    assert cost.work.sliding_kv_read < cost.work.kv_read
    assert cost.passes == cost.rounds == 1
