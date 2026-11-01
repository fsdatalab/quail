"""CPU tests for the scheduling loops the request backends share."""

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest

from quail.baselines.request_scheduling import (
    run_filter_sequence,
    run_filter_sequence_async,
    true_bit,
)

READ_ANSWER = partial(true_bit, true_ids={1})


class _FakeFilterEngine:
    """Finishes one queued request per step and answers TRUE."""

    def __init__(self):
        self.pending = []
        self.events = []

    def add_request(self, request_id, prompt, _sampling_params):
        self.pending.append((request_id, prompt["prompt_token_ids"]))
        self.events.append(("add", request_id))

    def abort_request(self, request_ids):
        self.pending = [item for item in self.pending if item[0] not in request_ids]
        self.events.extend(("abort", request_id) for request_id in request_ids)

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


def test_filter_sequences_pipeline_refill_fail_and_take_empty_input():
    engine = _FakeFilterEngine()
    result = run_filter_sequence(
        engine,
        sampling_params=object(),
        body_ids=[[1] * 5, [2] * 17],
        question_ids=[[3] * 3, [4] * 3],
        budget_tokens=100,
        tag="test",
        read_answer=READ_ANSWER,
        block_size=16,
        max_num_seqs=2,
    )

    assert result["doc_cap"] == 2
    assert engine.events.index(("add", "test-0-1")) < \
        engine.events.index(("finish", "test-1-0"))
    assert result["survivors"] == [0, 1]

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
                prompt_token_ids=prompt, num_cached_tokens=document,
                outputs=[SimpleNamespace(token_ids=[1 if answer else 0])],
            )

        result = await asyncio.wait_for(run_filter_sequence_async(
            generate, {}, [[1] * 10, [2] * 20, [3] * 30],
            [[8] * 3, [9] * 4], 64, read_answer=READ_ANSWER, block_size=16,
        ), timeout=2)
        assert result["doc_cap"] == 2
        assert peak == 2
        assert events.index(("start", 1, 9)) < events.index(("finish", 2, 8))
        assert events.index(("start", 3, 8)) < events.index(("finish", 2, 8))
        assert result["survivors"] == [1]
        assert result["requests"] == 5
        assert result["prompt_tokens"] == 13 + 14 + 23 + 24 + 33
        assert result["cached_tokens"] == 1 + 1 + 2 + 2 + 3
        assert result["answers"] == {
            (0, 1): 1, (0, 2): 0, (1, 1): 1, (1, 2): 1, (2, 1): 0}

    asyncio.run(run())

    # a failed request cancels the rest; no documents means no requests
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
            await run_filter_sequence_async(
                generate, {}, [[1], [2]], [[9]], 100, read_answer=READ_ANSWER,
                max_num_seqs=2,
            )
        assert cancelled.is_set()

    asyncio.run(run())

    async def generate(prompt, sampling_params):
        raise AssertionError("no documents")

    result = asyncio.run(
        run_filter_sequence_async(
            generate, {}, [], [[9]], 100, read_answer=READ_ANSWER))
    assert result["requests"] == 0
    assert result["survivors"] == []
