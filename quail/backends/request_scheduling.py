"""Request scheduling shared by the vLLM and SGLang backends.

Filter chains advance on individual request completions under a shared
KV admission calculation. Joins submit one request per document pair.
"""

from __future__ import annotations

import asyncio
import re
import time

MAX_SEQUENCES = 4_096
MAX_BATCHED_TOKENS = 25_305


_ANSWER_WORD = re.compile(r"\b(TRUE|FALSE)\b")
_WHOLE_ANSWER_WORD = re.compile(r"^\s*(TRUE|FALSE)\s*$", re.IGNORECASE)


def _ranked_answer(logprobs):
    """The likelier of TRUE and FALSE in the first position's logprobs.

    Each entry is judged by its decoded token. Returns 1 or 0, or None
    when neither answer word is among the entries.
    """
    if not logprobs:
        return None
    best = None
    for entry in logprobs[0].values():
        match = _WHOLE_ANSWER_WORD.match(
            getattr(entry, "decoded_token", None) or "")
        if match is None:
            continue
        if best is None or entry.logprob > best[0]:
            best = (entry.logprob, match.group(1).upper())
    return None if best is None else int(best[1] == "TRUE")


def true_bit(output, true_ids) -> int:
    """Whether the request answered TRUE.

    The first generated token decides when it is a TRUE id, which the
    allowed-token sampling of a decoder guarantees. A diffusion model's
    sampler cannot restrict its tokens: with logprobs at the answer
    row, TRUE and FALSE are ranked against each other; otherwise the
    answer is the first TRUE or FALSE word in the text. No answer word
    counts as FALSE.
    """
    completion = output.outputs[0]
    token_ids = completion.token_ids
    if token_ids and int(token_ids[0]) in true_ids:
        return 1
    ranked = _ranked_answer(getattr(completion, "logprobs", None))
    if ranked is not None:
        return ranked
    match = _ANSWER_WORD.search(getattr(completion, "text", "") or "")
    return int(match is not None and match.group(1) == "TRUE")


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


class _FilterChain:
    """Answers, survivors, and counters for one filter chain."""

    def __init__(self, body_ids, question_ids, true_ids):
        self.body_ids = body_ids
        self.question_ids = question_ids
        self.true_ids = true_ids
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
        answer = true_bit(output, self.true_ids)
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


def run_filter_chain(
    engine,
    sampling_params,
    body_ids,
    question_ids,
    budget_tokens,
    *,
    tag="q",
    true_ids,
    block_size=1,
    max_num_seqs=None,
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
    chain = _FilterChain(body_ids, question_ids, true_ids)
    inflight = {}

    def submit(document, stage):
        request_id = f"{tag}-{document}-{stage}"
        inflight[request_id] = (document, stage)
        engine.add_request(
            request_id,
            {
                "prompt_token_ids": (
                    body_ids[document] + question_ids[stage]
                )
            },
            sampling_params,
        )

    started = time.perf_counter()
    live = 0
    next_document = 0
    while next_document < len(body_ids) or inflight:
        while next_document < len(body_ids) and live < document_cap:
            submit(next_document, 0)
            live += 1
            next_document += 1
        for output in engine.step():
            if not output.finished or output.request_id not in inflight:
                continue
            document, stage = inflight.pop(output.request_id)
            if chain.record(document, stage, output):
                submit(document, stage + 1)
            else:
                live -= 1
    return chain.result(
        time.perf_counter() - started,
        **_cap_settings(document_cap, budget_tokens, block_size,
                        max_num_seqs),
    )


async def run_filter_chain_async(
    generate,
    sampling_params,
    body_ids,
    question_ids,
    budget_tokens,
    *,
    true_ids,
    block_size=1,
    max_num_seqs=None,
):
    """Advance filters and refill admission on individual completions."""
    document_cap = filter_document_cap(
        body_ids, question_ids, budget_tokens,
        block_size=block_size, max_num_seqs=max_num_seqs,
    )
    chain = _FilterChain(body_ids, question_ids, true_ids)
    documents = iter(range(len(body_ids)))

    async def worker():
        for document in documents:
            for stage, question in enumerate(question_ids):
                output = await generate(body_ids[document] + question, sampling_params)
                if not chain.record(document, stage, output):
                    break

    started = time.perf_counter()
    async with asyncio.TaskGroup() as group:
        for _ in range(min(document_cap, len(body_ids))):
            group.create_task(worker())
    return chain.result(
        time.perf_counter() - started,
        **_cap_settings(document_cap, budget_tokens, block_size, max_num_seqs),
    )


def run_join_grouped(
    client,
    sampling_params,
    prefixes,
    suffixes,
    true_ids,
    *,
    submission="anchor-major",
    pairs=None,
):
    """Submit the join's requests; answers come back in anchor-major order.

    Args:
        client: The engine client with generate().
        sampling_params: Sampling settings for every request.
        prefixes: Per-anchor prefix token lists.
        suffixes: Per-partner suffix token lists.
        true_ids: Token ids that mean TRUE.
        submission: "anchor-major" or "suffix-major" request order.
        pairs: (anchor index, suffix index) list to evaluate, in
            anchor-major order; every anchor against every suffix when
            omitted.
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
    prompts = [
        {"prompt_token_ids": prefixes[pairs[index][0]]
         + suffixes[pairs[index][1]]}
        for index in order
    ]

    started = time.perf_counter()
    outputs = client.generate(prompts, sampling_params, use_tqdm=False)
    wall = time.perf_counter() - started
    bits = [true_bit(output, true_ids) for output in outputs]
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


