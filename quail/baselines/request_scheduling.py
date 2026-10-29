"""Request scheduling shared by the vLLM and SGLang backends.

Filter sequences advance on individual request completions under a shared
KV admission calculation. The fields of an extraction, and the maps of a
pipelined backend, share each document's KV the same way. Joins submit
one request per document pair.
"""

from __future__ import annotations

import asyncio
import re
import time

MAX_SEQUENCES = 4_096
MAX_BATCHED_TOKENS = 25_305


_ANSWER_WORD = re.compile(r"\b(TRUE|FALSE)\b", re.IGNORECASE)


def true_bit(output, true_ids) -> int:
    """Read one token from a request restricted to TRUE/FALSE tokens."""
    tokens = output.outputs[0].token_ids
    if not tokens:
        raise ValueError("request returned no answer token")
    return int(tokens[0] in true_ids)


def text_answer(output) -> int:
    """Read the first TRUE or FALSE word from generated text."""
    match = _ANSWER_WORD.search(output.outputs[0].text or "")
    if match is None:
        raise ValueError("request returned no TRUE/FALSE answer")
    return int(match.group(1).upper() == "TRUE")


def canvas_answer(output, *, true_ids, false_ids) -> int:
    """Compare TRUE/FALSE token scores at the first canvas position.

    Equal scores return FALSE.

    Raises:
        ValueError: A TRUE/FALSE token score is missing.
    """
    logprobs = output.outputs[0].logprobs
    entries = logprobs[0] if logprobs else {}
    if any(t not in entries for t in true_ids | false_ids):
        raise ValueError("vLLM omitted requested TRUE/FALSE scores")
    true = max(entries[t].logprob for t in true_ids)
    false = max(entries[t].logprob for t in false_ids)
    return int(true > false)


def filter_document_cap(
    body_ids,
    question_ids,
    budget_tokens: int,
    *,
    block_size: int = 1,
    max_num_seqs: int | None = None,
) -> int:
    """Return how many documents the KV budget keeps live at once.

    Each live document holds one request of its body, its longest
    question, and the one answer token, rounded up to whole KV blocks.
    """
    if not body_ids:
        return 0
    longest_tail = max(len(question) for question in question_ids)
    rounded_sizes = [
        (
            (len(body) + longest_tail + 1 + block_size - 1) // block_size
        ) * block_size
        for body in body_ids
    ]
    mean_request = sum(rounded_sizes) // max(1, len(rounded_sizes))
    document_cap = max(1, budget_tokens // mean_request)
    if max_num_seqs is not None:
        document_cap = min(document_cap, max_num_seqs)
    return document_cap


class _FilterSequence:
    """Answers, survivors, and counters for one filter sequence."""

    def __init__(self, body_ids, question_ids, read_answer):
        self.body_ids = body_ids
        self.question_ids = question_ids
        self.read_answer = read_answer
        self.answers = {}
        self.survivors = []
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0

    def record(self, document: int, stage: int, output) -> bool:
        """Record one answer; return True when the document advances."""
        self.requests += 1
        self.prompt_tokens += len(output.prompt_token_ids)
        self.cached_tokens += int(getattr(output, "num_cached_tokens", 0) or 0)
        answer = self.read_answer(output)
        self.answers[(document, stage + 1)] = answer
        if answer and stage + 1 < len(self.question_ids):
            return True
        if answer:
            self.survivors.append(document)
        return False

    def result(self, wall: float, **settings) -> dict:
        return {
            "wall": wall,
            "survivors": sorted(self.survivors),
            "answers": self.answers,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            **settings,
        }


def _cap_settings(document_cap, budget_tokens, block_size, max_num_seqs):
    return {
        "doc_cap": document_cap,
        "budget_tokens": budget_tokens,
        "block_size": block_size,
        "max_num_seqs": max_num_seqs,
    }


def run_filter_sequence(
    engine,
    sampling_params,
    body_ids,
    question_ids,
    budget_tokens,
    *,
    tag="q",
    read_answer,
    block_size=1,
    max_num_seqs=None,
    body_texts=None,
    question_texts=None,
    render_prompt=None,
):
    """Submit the next filter after each document passes.

    The engine exposes add_request and step, as vLLM's LLMEngine does,
    so a passing document's next stage enters the queue while other
    documents are still on earlier stages.
    """
    document_cap = filter_document_cap(
        body_ids, question_ids, budget_tokens,
        block_size=block_size, max_num_seqs=max_num_seqs,
    )
    sequence = _FilterSequence(body_ids, question_ids, read_answer)
    inflight = {}

    def submit(document, stage):
        request_id = f"{tag}-{document}-{stage}"
        inflight[request_id] = (document, stage)
        if body_texts is not None and question_texts is not None:
            prompt = body_texts[document] + question_texts[stage]
            if render_prompt is not None:
                prompt = render_prompt(prompt)
        else:
            prompt = {
                "prompt_token_ids": body_ids[document] + question_ids[stage]
            }
        engine.add_request(request_id, prompt, sampling_params)

    started = time.perf_counter()
    expected_rows = 0
    next_document = 0
    try:
        while next_document < len(body_ids) or inflight:
            while next_document < len(body_ids) and expected_rows < document_cap:
                submit(next_document, 0)
                expected_rows += 1
                next_document += 1
            for output in engine.step():
                if not output.finished or output.request_id not in inflight:
                    continue
                document, stage = inflight.pop(output.request_id)
                if sequence.record(document, stage, output):
                    submit(document, stage + 1)
                else:
                    expected_rows -= 1
    finally:
        if inflight:
            # the engine is reused by the next query; a request left
            # queued would run there and take KV from it
            engine.abort_request(list(inflight))
    return sequence.result(
        time.perf_counter() - started,
        **_cap_settings(document_cap, budget_tokens, block_size,
                        max_num_seqs),
    )



def extract_document_cap(
    body_ids,
    tail_ids,
    answer_tokens: int,
    budget_tokens: int,
    *,
    block_size: int = 1,
    max_num_seqs: int | None = None,
) -> int:
    """Return how many documents the KV budget keeps live at once.

    Each live document holds its body once and, for every field, the
    field's tail and its longest answer, each rounded up to whole KV
    blocks. Each field is one sequence.
    """
    if not body_ids:
        return 0

    def blocks(tokens):
        return (tokens + block_size - 1) // block_size * block_size

    fields = sum(blocks(len(tail) + answer_tokens + block_size)
                 for tail in tail_ids)
    sizes = [blocks(len(body)) + fields for body in body_ids]
    document_cap = max(1, budget_tokens // (sum(sizes) // len(sizes)))
    if max_num_seqs is not None:
        document_cap = min(document_cap, max(1, max_num_seqs // len(tail_ids)))
    return document_cap


def run_extract_fields(
    engine,
    sampling_params,
    body_ids,
    tail_ids,
    budget_tokens,
    *,
    tag="x",
    answer_tokens=0,
    block_size=1,
    max_num_seqs=None,
    body_texts=None,
    tail_texts=None,
    render_prompt=None,
):
    """Decode every field of each document, the first field before the rest.

    A document's first field computes the body's KV. The other fields
    enter the queue once that request has written a token, so they read
    the body from the prefix cache instead of computing it again. The
    engine exposes add_request and step, as vLLM's LLMEngine does.

    Args:
        sampling_params: One sampling setting for every field, or a
            list with one per field.
        answer_tokens: The answer tokens each field counts toward the
            document cap.

    Returns:
        A dict with the outputs of each field in document order, the
        wall seconds, and the admission settings.
    """
    fields = len(tail_ids)
    field_params = (list(sampling_params)
                    if isinstance(sampling_params, (list, tuple))
                    else [sampling_params] * fields)
    document_cap = extract_document_cap(
        body_ids, tail_ids, answer_tokens, budget_tokens,
        block_size=block_size, max_num_seqs=max_num_seqs,
    )
    outputs = [[None] * len(body_ids) for _ in range(fields)]
    inflight = {}
    released = set()
    left = {}

    def submit(document, field):
        request_id = f"{tag}-{document}-{field}"
        inflight[request_id] = (document, field)
        if body_texts is not None and tail_texts is not None:
            prompt = body_texts[document] + tail_texts[field]
            if render_prompt is not None:
                prompt = render_prompt(prompt)
        else:
            prompt = {
                "prompt_token_ids": body_ids[document] + tail_ids[field]
            }
        engine.add_request(request_id, prompt, field_params[field])

    started = time.perf_counter()
    expected_rows = next_document = 0
    try:
        while next_document < len(body_ids) or inflight:
            while next_document < len(body_ids) and expected_rows < document_cap:
                left[next_document] = fields
                submit(next_document, 0)
                expected_rows += 1
                next_document += 1
            for output in engine.step():
                if output.request_id not in inflight:
                    continue
                document, field = inflight[output.request_id]
                if field == 0 and document not in released:
                    # the body's KV is computed, so the other fields hit it
                    released.add(document)
                    for later in range(1, fields):
                        submit(document, later)
                if not output.finished:
                    continue
                del inflight[output.request_id]
                outputs[field][document] = output
                left[document] -= 1
                if not left[document]:
                    expected_rows -= 1
    finally:
        if inflight:
            # the engine is reused by the next query; a request left
            # queued would run there and take KV from it
            engine.abort_request(list(inflight))
    return {
        "outputs": outputs,
        "wall": time.perf_counter() - started,
        **_cap_settings(document_cap, budget_tokens, block_size, max_num_seqs),
    }

async def run_filter_sequence_async(
    generate,
    sampling_params,
    body_ids,
    question_ids,
    budget_tokens,
    *,
    read_answer,
    block_size=1,
    max_num_seqs=None,
):
    """Advance filters and refill admission on individual completions."""
    document_cap = filter_document_cap(
        body_ids, question_ids, budget_tokens,
        block_size=block_size, max_num_seqs=max_num_seqs,
    )
    sequence = _FilterSequence(body_ids, question_ids, read_answer)
    documents = iter(range(len(body_ids)))

    async def worker():
        for document in documents:
            for stage, question in enumerate(question_ids):
                output = await generate(body_ids[document] + question, sampling_params)
                if not sequence.record(document, stage, output):
                    break

    started = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for _ in range(min(document_cap, len(body_ids))):
            group.create_task(worker())
    return sequence.result(
        time.perf_counter() - started,
        **_cap_settings(document_cap, budget_tokens, block_size, max_num_seqs),
    )


def run_join_grouped(
    client,
    sampling_params,
    prefixes,
    suffixes,
    read_answer,
    *,
    submission="anchor-major",
    pairs=None,
    prefix_texts=None,
    suffix_texts=None,
):
    """Submit the join's requests; answers come back in anchor-major order.

    Args:
        client: The engine client with generate().
        sampling_params: Sampling settings for every request.
        prefixes: Per-anchor prefix token lists.
        suffixes: Per-partner suffix token lists.
        read_answer: Function that reads a Boolean answer from a response.
        submission: "anchor-major" or "suffix-major" request order.
        pairs: (anchor index, suffix index) list to evaluate, in
            anchor-major order; every anchor against every suffix when
            omitted.
        prefix_texts: Text form of each prefix when the engine tokenizes.
        suffix_texts: Text form of each suffix when the engine tokenizes.
    """
    if pairs is None:
        pairs = [(anchor, suffix) for anchor in range(len(prefixes))
                 for suffix in range(len(suffixes))]
    if submission == "anchor-major":
        order = list(range(len(pairs)))
    elif submission == "suffix-major":
        order = sorted(range(len(pairs)),
                       key=lambda index: (pairs[index][1], pairs[index][0]))
    else:
        raise ValueError(f"unknown join submission {submission!r}")
    if prefix_texts is not None and suffix_texts is not None:
        prompts = [
            prefix_texts[pairs[index][0]] + suffix_texts[pairs[index][1]]
            for index in order
        ]
    else:
        prompts = [
            {"prompt_token_ids": prefixes[pairs[index][0]]
             + suffixes[pairs[index][1]]}
            for index in order
        ]

    started = time.perf_counter()
    outputs = client.generate(prompts, sampling_params, use_tqdm=False)
    bits = [read_answer(output) for output in outputs]
    wall = time.perf_counter() - started
    cached_by_output = [
        int(getattr(output, "num_cached_tokens", 0) or 0)
        for output in outputs
    ]
    submitted_at = [0] * len(pairs)
    for position, index in enumerate(order):
        submitted_at[index] = position
    answers = [bits[position] for position in submitted_at]
    cached_per_request = [cached_by_output[position]
                          for position in submitted_at]
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    cached_tokens = sum(cached_per_request)
    return {
        "wall": wall,
        "answers": answers,
        "fresh_tokens": prompt_tokens - cached_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_per_request": cached_per_request,
        "submission": submission,
    }
