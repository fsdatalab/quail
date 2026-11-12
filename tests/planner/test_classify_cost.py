"""Classification costs: expected work against label outcomes, priced like filters."""

import itertools
from dataclasses import asdict, replace

import pytest

from quail.cost.classify import estimate
from quail.cost.sol import unrounded_seconds
from quail.cost.work import Work, scan, stream
from quail.labels import decode_round_tokens, read_nodes
from quail.specs import H100_SXM, QWEN3_4B_FP8

# a one-token label; two that share a first token and split there; two
# that split after a forced token
LABELS = ((10,), (20, 21, 22), (20, 25, 26, 27, 28), (30, 31, 32), (30, 31, 33))
# the tokens a greedy decode feeds for each label, per round
HEAD, FRAME, CHUNK = 2, 3, 10000


def _cost(lengths, labels=LABELS, scoring="trie_decode", **kwargs):
    options = dict(lengths=lengths, shared=(), chunk=CHUNK,
                   model=QWEN3_4B_FP8, device=H100_SXM)
    options.update(kwargs)
    expected_rows = options.pop("expected_rows", len(lengths))
    return estimate(scoring, expected_rows, HEAD, FRAME, labels, **options)


def _head_bytes():
    return 2 * QWEN3_4B_FP8.hidden * QWEN3_4B_FP8.vocab


def _readout_seconds(rows, passes):
    return max(_head_bytes() * rows / H100_SXM.gemm_flops("bf16"),
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
def test_expected_work_matches_every_label_assignment(lengths, labels, window,
                                                      shared, resident):
    kinds = ("sliding_attention", "full_attention") * 18 if window else ()
    model = replace(QWEN3_4B_FP8, sliding_window=window, layer_kinds=kinds)
    cost = _cost(lengths, labels, model=model, shared=shared or (),
                 resident=resident)
    runs = decode_round_tokens(labels)
    outcomes = list(itertools.product(runs, repeat=len(lengths)))
    expected = dict.fromkeys(asdict(Work()), 0.0)
    for picked in outcomes:
        for key, value in _decode_outcome_work(lengths, picked, window,
                                               shared, resident).items():
            expected[key] += value / len(outcomes)
    assert asdict(cost.work) == pytest.approx(expected)
    assert cost.passes == pytest.approx(cost.work.tokens / CHUNK)
    assert cost.suffix_tokens == pytest.approx(
        sum(sum(map(sum, picked)) for picked in outcomes) / len(outcomes))
    assert cost.rounds == max(map(len, runs))






def test_one_round_rules_are_priced_like_filters():
    lengths = (10, 20, 30)
    # the trie reads the root, (20,) and (30, 31), where labels split,
    # and also feeds (30,), the forced token before (30, 31)
    for scoring, suffix, rows in (("letters", 1, 1), ("trie_tree", 4, 3)):
        cost = _cost(lengths, scoring=scoring)
        work = Work()
        for length in lengths:
            prefix = HEAD + length
            # the frame is written first; the request reads it and the document
            work += scan(prefix, FRAME) + stream(prefix + FRAME, [suffix])
        work = replace(work, state_read=2 * len(lengths),
                       state_written=2 * len(lengths))
        assert asdict(cost.work) == pytest.approx(asdict(work))
        assert cost.passes == pytest.approx(work.tokens / CHUNK)
        assert cost.seconds == pytest.approx(
            unrounded_seconds(work, QWEN3_4B_FP8, H100_SXM, CHUNK)
            + _readout_seconds(len(lengths) * rows, cost.passes))
        assert cost.suffix_tokens == len(lengths) * suffix
        assert cost.rounds == 1






















@pytest.mark.parametrize("num_documents,gpus", [(8, 2)])
def test_parallel_classification_divides_time_by_the_copies_in_use(
        num_documents, gpus):
    lengths = [100] * num_documents
    serial = _cost(lengths)
    parallel = _cost(lengths, gpus=gpus)
    assert parallel.work == serial.work
    assert parallel.seconds == pytest.approx(
        serial.seconds / min(num_documents, gpus))


def test_trie_tree_reads_only_the_nodes_where_labels_split():
    # (30,) has one child, so the root already tells its label apart;
    # (20,) splits, and the label (40,) ends where (40, 41) continues
    labels = ((10,), (20, 21), (20, 22), (30, 31, 32), (40,), (40, 41))
    assert read_nodes(labels) == {(), (20,), (40,)}
