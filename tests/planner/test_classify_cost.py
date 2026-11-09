"""Classification costs: expected work against label outcomes, priced like filters."""

import itertools
from dataclasses import asdict, replace

import pytest
from helpers.rust_cost import classify_estimate, cost_model, native
from helpers.work import scan, stream

from quail.common.work import Work
from quail.labels import label_trie
from quail.specs import H100_SXM, QWEN3_4B_FP8

# a one-token label; two that share a first token and split there; two
# that split after a forced token
LABELS = ((10,), (20, 21, 22), (20, 25, 26, 27, 28), (30, 31, 32), (30, 31, 33))
# the tokens a greedy decode feeds for each label, per round
HEAD, FRAME, CHUNK = 2, 3, 10000
# Qwen3 4B with a 16-token window on every other layer
WINDOWED = "qwen3-4b-fp8-window-16"


@pytest.fixture(scope="module")
def rust(tmp_path_factory):
    return cost_model(tmp_path_factory.mktemp("specs"), **{WINDOWED: dict(
        base=QWEN3_4B_FP8.name, sliding_window=16,
        layer_kinds=["sliding_attention", "full_attention"] * 18)})


def _cost(rust, lengths, labels=LABELS, scoring="trie_decode", **kwargs):
    options = dict(lengths=lengths, shared=(), chunk=CHUNK,
                   model=QWEN3_4B_FP8.name, device=H100_SXM.name)
    options.update(kwargs)
    expected_rows = options.pop("expected_rows", len(lengths))
    return classify_estimate(rust, scoring, expected_rows, HEAD, FRAME, labels,
                             **options)


def _head_bytes():
    return 2 * QWEN3_4B_FP8.hidden * QWEN3_4B_FP8.vocab


def _readout_seconds(rust, rows, passes):
    return max(_head_bytes() * rows / rust.gemm_flops(H100_SXM.name, "bf16"),
               _head_bytes() * passes / H100_SXM.hbm_bw)


def _decode_outcome_work(lengths, runs, window=0, shared=None, resident=False):
    """Count one label outcome's decode work from the first principles.

    The first request is the prefix, frame, cue, and the forced tokens
    to the first choice as one causal segment; each later round feeds
    one run that attends to everything before it and reads the
    document's KV. A shared or resident prefix is read instead of
    computed.
    """
    total = dict.fromkeys(asdict(Work()), 0.0)
    shared = shared or (0,) * len(lengths)
    for length, fed, borrowed in zip(lengths, runs, shared):
        prompt = HEAD + length + FRAME
        kept = HEAD + length if resident else HEAD + borrowed if borrowed else 0
        fresh = range(kept + 1, prompt + sum(fed) + 1)
        total["tokens"] += len(fresh)
        total["kv_written"] += len(fresh)
        total["pairs"] += sum(fresh)
        contexts = [prompt + sum(fed[:round_]) for round_ in range(1, len(fed))]
        total["kv_read"] += kept + sum(contexts)
        # The document remains saved while every round reads and writes
        # the answer's checkpoint. A fresh borrower also reads its parent.
        total["state_read"] += len(fed) + bool(borrowed and not resident)
        total["state_written"] += len(fed) + (not resident)
        if window:
            total["sliding_pairs"] += sum(min(i, window) for i in fresh)
            total["sliding_kv_read"] += sum(
                min(context, window - 1) for context in contexts)
            if kept:
                total["sliding_kv_read"] += min(kept, window - 1)
    return total


@pytest.mark.parametrize("lengths,labels,window,shared,resident", [
    ((10, 90), LABELS, 0, None, False),
    ((40, 7, 25), LABELS, 16, None, True),
])
def test_expected_work_matches_every_label_assignment(rust, lengths, labels, window,
                                                      shared, resident):
    model = WINDOWED if window else QWEN3_4B_FP8.name
    cost = _cost(rust, lengths, labels, model=model, shared=shared or (),
                 resident=resident)
    runs = native.CostModel.label_nodes([list(label) for label in labels], [])[2]
    outcomes = list(itertools.product(runs, repeat=len(lengths)))
    expected = dict.fromkeys(asdict(Work()), 0.0)
    for picked in outcomes:
        for key, value in _decode_outcome_work(lengths, picked, window,
                                               shared, resident).items():
            expected[key] += value / len(outcomes)
    assert cost.work == pytest.approx(expected)
    assert cost.passes == pytest.approx(cost.work["tokens"] / CHUNK)
    assert cost.suffix_tokens == pytest.approx(
        sum(sum(map(sum, picked)) for picked in outcomes) / len(outcomes))
    assert cost.rounds == max(map(len, runs))


def test_one_round_rules_are_priced_like_filters(rust):
    lengths = (10, 20, 30)
    for scoring, suffix in (("letters", 1), ("trie_tree", len(label_trie(LABELS)))):
        cost = _cost(rust, lengths, scoring=scoring)
        work = Work()
        for length in lengths:
            prefix = HEAD + length
            # the frame is written first; the request reads it and the document
            work += scan(prefix, FRAME) + stream(prefix + FRAME, [suffix])
        work = replace(work, state_read=2 * len(lengths),
                       state_written=2 * len(lengths))
        assert cost.work == pytest.approx(asdict(work))
        assert cost.passes == pytest.approx(work.tokens / CHUNK)
        assert cost.seconds == pytest.approx(
            rust.unrounded_seconds(asdict(work), QWEN3_4B_FP8.name, H100_SXM.name,
                                   CHUNK, 0.0, 0)
            + _readout_seconds(rust, len(lengths) * suffix, cost.passes))
        assert cost.suffix_tokens == len(lengths) * suffix
        assert cost.rounds == 1


@pytest.mark.parametrize("num_documents,gpus", [(8, 2)])
def test_parallel_classification_divides_time_by_the_copies_in_use(
        rust, num_documents, gpus):
    lengths = [100] * num_documents
    serial = _cost(rust, lengths)
    parallel = _cost(rust, lengths, gpus=gpus)
    assert parallel.work == serial.work
    assert parallel.seconds == pytest.approx(
        serial.seconds / min(num_documents, gpus))
