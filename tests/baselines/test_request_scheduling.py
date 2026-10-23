"""CPU tests for the scheduling loops the request backends share."""

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest

from quail.baselines.request_scheduling import (
    extract_document_cap,
    run_extract_fields,
    run_filter_sequence,
    run_filter_sequence_async,
    run_join_grouped,
    true_bit,
)

READ_ANSWER = partial(true_bit, true_ids={1})

def test_join_submission_order_and_empty_inputs():
    for submission in ["anchor-major", "suffix-major"]:
        prefixes = [[100 + i] * (4 + i) for i in range(5)]
        suffixes = [[200 + j] * 3 for j in range(4)]
        client = _ParityClient()
        result = run_join_grouped(client, object(), prefixes, suffixes, READ_ANSWER,
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
        pairs = [(0, 1), (0, 3), (2, 0), (2, 2), (2, 3)]
        client = _ParityClient()
        result = run_join_grouped(client, object(), prefixes, suffixes, READ_ANSWER,
                                  submission=submission, pairs=pairs)
        heads = [(p["prompt_token_ids"][0] - 100, p["prompt_token_ids"][-1] - 200)
                 for p in client.calls[0]]
        assert heads == (pairs if submission == "anchor-major"
                         else sorted(pairs, key=lambda pair: pair[::-1]))
        # answers and cache counts come back in the listed order
        assert result["answers"] == [int((a + s) % 2 == 0) for a, s in pairs]
        assert result["cached_per_request"] == [(100 + a) % 7 for a, _ in pairs]
    with pytest.raises(ValueError, match="unknown join submission"):
        run_join_grouped(_ParityClient(), object(), prefixes, suffixes, READ_ANSWER,
                         submission="unknown")

    for prefixes, suffixes in [([], [[1]]), ([[1]], []), ([], [])]:
        result = run_join_grouped(
            _ParityClient(), object(), prefixes, suffixes, READ_ANSWER,
            submission="suffix-major")
        assert result["answers"] == []
        assert result["cached_per_request"] == []
        assert result["fresh_tokens"] == 0


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



class _FakeStepEngine:
    """Writes one token of every queued request per step; ends at the second."""

    def __init__(self):
        self.pending = {}
        self.events = []

    def add_request(self, request_id, prompt, _sampling_params):
        self.pending[request_id] = (prompt["prompt_token_ids"], 0)
        self.events.append(("add", request_id))

    def abort_request(self, request_ids):
        for request_id in request_ids:
            self.pending.pop(request_id, None)
            self.events.append(("abort", request_id))

    def step(self):
        outputs = []
        for request_id, (ids, written) in list(self.pending.items()):
            finished = written == 1
            if finished:
                del self.pending[request_id]
            else:
                self.pending[request_id] = (ids, 1)
            self.events.append(("finish" if finished else "token", request_id))
            outputs.append(SimpleNamespace(
                request_id=request_id, finished=finished,
                prompt_token_ids=ids, num_cached_tokens=0,
                outputs=[SimpleNamespace(token_ids=[1] * (written + 1))]))
        return outputs


def test_extract_fields_compute_each_body_before_its_other_fields():
    bodies = [[1] * 30, [2] * 30, [3] * 30]
    tails = [[7], [8], [9]]
    # each document holds 32 body tokens and three 32-token fields
    assert extract_document_cap(bodies, tails, 4, 256, block_size=16) == 2
    assert extract_document_cap(bodies, tails, 4, 10_000, block_size=16,
                                max_num_seqs=7) == 2
    assert extract_document_cap([], tails, 4, 160) == 0

    engine = _FakeStepEngine()
    result = run_extract_fields(engine, object(), bodies, tails, 256,
                                tag="x", answer_tokens=4, block_size=16)
    assert result["doc_cap"] == 2
    events = engine.events
    for document in range(3):
        first = events.index(("token", f"x-{document}-0"))
        for field in (1, 2):
            assert events.index(("add", f"x-{document}-{field}")) > first
    # the third document waits until a document has all its answers
    before = events[:events.index(("add", "x-2-0"))]
    assert sum(event == "add" for event, _ in before) == 6
    assert any(all(("finish", f"x-{d}-{f}") in before for f in range(3))
               for d in range(2))
    assert [[output.prompt_token_ids for output in field]
            for field in result["outputs"]] == [
        [body + tail for body in bodies] for tail in tails]
    assert not engine.pending


def test_a_failed_extract_run_aborts_its_queued_requests(monkeypatch):
    engine = _FakeStepEngine()
    step = engine.step

    def fail():
        if any(event == "finish" for event, _ in engine.events):
            raise RuntimeError("step failed")
        return step()

    monkeypatch.setattr(engine, "step", fail)
    with pytest.raises(RuntimeError, match="step failed"):
        run_extract_fields(engine, object(), [[1], [2]], [[7], [8]], 100,
                           tag="x")
    assert not engine.pending
    assert any(event == "abort" for event, _ in engine.events)

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


@pytest.mark.parametrize("failure", ["submit", "step", "answer"])
def test_a_failed_filter_sequence_aborts_its_queued_requests(monkeypatch, failure):
    engine = _FakeFilterEngine()
    read_answer = READ_ANSWER

    def fail(*args, **kwargs):
        raise RuntimeError("request failed")

    if failure == "submit":
        add_request = engine.add_request

        def submit(request_id, prompt, sampling_params):
            add_request(request_id, prompt, sampling_params)
            if request_id == "retry-1-0":
                fail()

        monkeypatch.setattr(engine, "add_request", submit)
    elif failure == "step":
        monkeypatch.setattr(engine, "step", fail)
    else:
        read_answer = fail
    with pytest.raises(RuntimeError, match="request failed"):
        run_filter_sequence(
            engine, sampling_params=object(), body_ids=[[1], [2]],
            question_ids=[[9]], budget_tokens=100, tag="retry",
            read_answer=read_answer, max_num_seqs=2)
    assert not engine.pending
    aborted = {request_id for event, request_id in engine.events
               if event == "abort"}
    assert aborted and aborted <= {"retry-0-0", "retry-1-0"}
