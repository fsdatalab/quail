"""CPU tests for the scheduling loops the request backends share."""

import asyncio
from types import SimpleNamespace

import pytest

from quail.backends.request_scheduling import (
    join_regret_tokens,
    longest_common_prefix,
    run_filter_chain,
    run_filter_chain_async,
    run_join_grouped,
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


def test_async_filters_advance_and_refill_before_slow_request_finishes():
    async def run():
        release = asyncio.Event()
        events = []
        active = 0
        peak = 0

        async def generate(prompt, sampling_params):
            nonlocal active, peak
            document, question = prompt[0], prompt[-1]
            events.append(("start", document, question))
            active += 1
            peak = max(peak, active)
            if document == 2 and question == 8:
                await release.wait()
            else:
                await asyncio.sleep(0)
            if document == 3:
                release.set()
            answer = document == 2 or (document == 1 and question == 8)
            active -= 1
            events.append(("finish", document, question))
            return SimpleNamespace(
                prompt_token_ids=prompt, num_cached_tokens=0,
                outputs=[SimpleNamespace(token_ids=[1 if answer else 0])],
            )

        result = await asyncio.wait_for(run_filter_chain_async(
            generate, {}, [[1] * 10, [2] * 20, [3] * 30],
            [[8] * 3, [9] * 4], 64, true_ids={1}, block_size=16,
            max_num_seqs=10,
        ), timeout=2)
        assert result["doc_cap"] == 2
        assert peak == 2
        assert events.index(("start", 1, 9)) < events.index(("finish", 2, 8))
        assert events.index(("start", 3, 8)) < events.index(("finish", 2, 8))
        assert result["survivors"] == [1]
        assert result["requests"] == 5
        assert result["answers"] == {
            (0, 1): 1, (0, 2): 0, (1, 1): 1, (1, 2): 1, (2, 1): 0}

    asyncio.run(run())


def test_async_filter_failure_cancels_other_requests():
    async def run():
        pending = asyncio.Event()
        cancelled = asyncio.Event()

        async def generate(prompt, sampling_params):
            if prompt[0] == 1:
                await pending.wait()
                raise RuntimeError("request failed")
            pending.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with pytest.raises(ExceptionGroup, match="TaskGroup"):
            await run_filter_chain_async(
                generate, {}, [[1], [2]], [[9]], 100, true_ids={1}, max_num_seqs=2,
            )
        assert cancelled.is_set()

    asyncio.run(run())


def test_async_filter_empty_input_submits_nothing():
    async def generate(prompt, sampling_params):
        raise AssertionError("no documents")

    result = asyncio.run(
        run_filter_chain_async(generate, {}, [], [[9]], 100, true_ids={1}))
    assert result["requests"] == 0
    assert result["survivors"] == []


class _ParityClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, _sp, use_tqdm=False):
        self.calls.append(prompts)
        outs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            bit = 1 if (ids[0] + ids[-1]) % 2 == 0 else 0
            outs.append(SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[bit])],
                prompt_token_ids=ids,
                num_cached_tokens=ids[0] % 7))
        return outs


@pytest.mark.parametrize("submission", ["anchor-major", "suffix-major"])
def test_join_answers_preserve_anchor_major_order(submission):
    prefixes = [[100 + i] * (4 + i) for i in range(5)]
    suffixes = [[200 + j] * 3 for j in range(4)]
    client = _ParityClient()
    result = run_join_grouped(client, object(), prefixes, suffixes, {1},
                              submission=submission)
    expected = ([[100 + i, 200 + j] for i in range(5) for j in range(4)]
                if submission == "anchor-major" else
                [[100 + i, 200 + j] for j in range(4) for i in range(5)])
    assert len(client.calls) == 1
    assert [[p["prompt_token_ids"][0], p["prompt_token_ids"][-1]]
            for p in client.calls[0]] == expected
    assert result["answers"] == [
        int((i + j) % 2 == 0) for i in range(5) for j in range(4)]
    assert result["cached_per_request"] == [
        (100 + i) % 7 for i in range(5) for j in range(4)]
    assert result["cached_tokens"] == sum(result["cached_per_request"])
    assert result["fresh_tokens"] == sum(
        len(p) + len(s) for p in prefixes for s in suffixes
    ) - result["cached_tokens"]
    assert result["submission"] == submission
    with pytest.raises(ValueError, match="unknown join submission"):
        run_join_grouped(_ParityClient(), object(), prefixes, suffixes, {1},
                         submission="unknown")


@pytest.mark.parametrize("prefixes,suffixes",
                         [([], [[1]]), ([[1]], []), ([], [])])
def test_suffix_major_join_empty_input(prefixes, suffixes):
    result = run_join_grouped(_ParityClient(), object(), prefixes, suffixes, {1},
                              submission="suffix-major")
    assert result["answers"] == []
    assert result["cached_per_request"] == []
    assert result["fresh_tokens"] == 0
