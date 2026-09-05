"""CPU tests for the scheduling loops the request backends share."""

from types import SimpleNamespace

import pytest

from quail.backends.request_scheduling import (
    join_regret_tokens,
    longest_common_prefix,
    run_filter_chain,
    run_filter_chain_waves,
    run_join_grouped,
    suffix_major_tiled_order,
)


def test_longest_common_prefix_counts_shared_leading_tokens():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix([], [1]) == 0


def test_join_regret_buckets_pair0_and_rest():
    # Two anchors, two suffixes each, 16-token blocks. Anchor 0 was
    # computed before (40 seen tokens): pair 0 misses everything
    # (regret 32, the block floor of 40) and pair 1 hits fully.
    # Anchor 1 is new: pair 0 owes nothing even though the engine
    # served 16 tokens, and pair 1 recomputes half its 40-token prefix
    # (regret 32 - 16 = 16).
    prefixes = [[7] * 40, [9] * 40]
    cached = [0, 32, 16, 16]

    assert join_regret_tokens(prefixes, 2, cached, [40, 0], 16) == 48


def test_join_regret_is_zero_when_cache_serves_every_would_hit():
    prefixes = [[7] * 32]
    assert join_regret_tokens(prefixes, 2, [32, 32], [32], 16) == 0


class _FakeFilterEngine:
    """Finishes one queued request per step and answers TRUE."""

    def __init__(self):
        self.pending = []
        self.events = []

    def add_request(self, request_id, prompt, _sampling_params):
        self.pending.append((request_id, prompt["prompt_token_ids"]))
        self.events.append(("add", request_id))

    def step(self):
        request_id, prompt_token_ids = self.pending.pop(0)
        self.events.append(("finish", request_id))
        return [SimpleNamespace(
            finished=True,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            num_cached_tokens=0,
            outputs=[SimpleNamespace(token_ids=[1], text="TRUE")],
        )]


def test_filter_chain_submits_next_stage_before_prior_stage_finishes():
    engine = _FakeFilterEngine()
    result = run_filter_chain(
        engine,
        sampling_params=object(),
        body_ids=[[1] * 5, [2] * 17],
        question_ids=[[3] * 3, [4] * 3],
        budget_tokens=100,
        tag="test",
        true_ids={1},
        block_size=16,
        max_num_seqs=2,
    )

    assert result["doc_cap"] == 2
    assert engine.events.index(("add", "test-0-1")) < \
        engine.events.index(("finish", "test-1-0"))
    assert result["survivors"] == [0, 1]


class _WaveRecordingClient:
    """Answers by a fixed (document, stage) truth table.

    Documents are identified by their body length, stages by the
    question length, so the recorded waves can be decoded.
    """

    def __init__(self, verdicts, body_lengths, question_lengths):
        self.verdicts = verdicts
        self.body_lengths = body_lengths
        self.question_lengths = question_lengths
        self.waves = []

    def generate(self, prompts, _sampling_params, use_tqdm=False):
        wave = []
        outs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            found = None
            for index, body in enumerate(self.body_lengths):
                for stage, tail in enumerate(self.question_lengths):
                    if len(ids) == body + tail:
                        found = (index, stage)
            wave.append(found)
            bit = 7 if self.verdicts[found] else 8
            outs.append(SimpleNamespace(
                prompt_token_ids=ids,
                num_cached_tokens=0,
                outputs=[SimpleNamespace(token_ids=[bit], text="")]))
        self.waves.append(wave)
        return outs


def test_wave_chain_advances_documents_under_the_cap():
    # Bodies 10/20/30 tokens, questions 3/4 tokens: every (doc, stage)
    # pair has a distinct prompt length.
    verdicts = {(0, 0): True, (0, 1): True,
                (1, 0): False,
                (2, 0): True, (2, 1): False}
    client = _WaveRecordingClient(verdicts, [10, 20, 30], [3, 4])

    # mean request = 20 + 4 + 1 = 25; budget 50 -> two admission
    # slots. The SGLang client sizes admission without page rounding.
    result = run_filter_chain_waves(
        client, {"temperature": 0.0},
        [[1] * 10, [2] * 20, [3] * 30], [[4] * 3, [5] * 4],
        50, true_ids={7}, block_size=1)

    assert result["doc_cap"] == 2
    # Wave 1: documents 0 and 1 on stage 1. Wave 2: document 0 has
    # advanced to stage 2 while document 2 takes the freed slot: one
    # request per live document, which is the pipelining property.
    assert client.waves[0] == [(0, 0), (1, 0)]
    assert client.waves[1] == [(0, 1), (2, 0)]
    assert client.waves[2] == [(2, 1)]
    assert result["survivors"] == [0]
    assert result["requests"] == 5


def _tiled_order_tiles(order, n_suffixes):
    """Split a suffix-major-tiled order back into its anchor tiles."""
    tiles = []
    position = 0
    while position < len(order):
        tile = []
        while (position + len(tile) < len(order)
               and order[position + len(tile)][1] == 0):
            tile.append(order[position + len(tile)][0])
        assert tile, "each tile must start with suffix 0"
        expected = [
            (anchor_index, suffix_index)
            for suffix_index in range(n_suffixes)
            for anchor_index in tile
        ]
        assert order[position:position + len(expected)] == expected
        position += len(expected)
        tiles.append(tile)
    return tiles


def test_suffix_major_tiles_cover_all_pairs_within_budget():
    prefixes = [[0] * length for length in (30, 30, 30, 50, 10, 90)]
    suffixes = [[0] * 5, [0] * 3]
    budget = 100

    order = suffix_major_tiled_order(prefixes, suffixes, budget)
    assert sorted(order) == sorted(
        (i, j) for i in range(len(prefixes))
        for j in range(len(suffixes)))

    tiles = _tiled_order_tiles(order, len(suffixes))
    assert [anchor for tile in tiles for anchor in tile] == list(
        range(len(prefixes)))
    for tile in tiles:
        cost = sum(len(prefixes[anchor]) + 5 for anchor in tile)
        assert cost <= budget or len(tile) == 1


class _ParityClient:
    def generate(self, prompts, _sp, use_tqdm=False):
        outs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            bit = 1 if (ids[0] + ids[-1]) % 2 == 0 else 0
            outs.append(SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[bit])],
                prompt_token_ids=ids,
                num_cached_tokens=0))
        return outs


def test_tiled_join_answers_match_anchor_major_order():
    prefixes = [[100 + i] * (4 + i) for i in range(5)]
    suffixes = [[200 + j] * 3 for j in range(4)]

    baseline = run_join_grouped(
        _ParityClient(), object(), prefixes, suffixes, {1})
    tiled = run_join_grouped(
        _ParityClient(), object(), prefixes, suffixes, {1},
        submission="suffix-major-tiled", tile_budget_tokens=20)

    assert baseline["answers"] == tiled["answers"]
    assert baseline["prompt_tokens"] == tiled["prompt_tokens"]
    assert tiled["submission"] == "suffix-major-tiled"

    with pytest.raises(ValueError):
        run_join_grouped(_ParityClient(), object(), prefixes, suffixes,
                         {1}, submission="suffix-major-tiled")
