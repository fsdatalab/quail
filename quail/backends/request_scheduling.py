"""Request scheduling shared by the vLLM and SGLang backends.

Filter chains advance on individual request completions under a shared
KV admission calculation. Joins submit one request per document pair.
"""

from __future__ import annotations

import asyncio
import time

MAX_SEQUENCES = 4_096
MAX_BATCHED_TOKENS = 25_305


def longest_common_prefix(left, right) -> int:
    """Return the length of the shared token prefix."""
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def true_bit(output, true_ids) -> int:
    """Return 1 when a request's first output token is a TRUE token."""
    token_ids = output.outputs[0].token_ids
    return int(bool(token_ids and int(token_ids[0]) in true_ids))


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


def split_cached_tokens(cached, could_hit, document_start, document_end):
    """Split one request's cached tokens by where they fell.

    Args:
        cached: Cached tokens the engine reported for the request.
        could_hit: Prefix length the document's own earlier request
            could explain, already rounded to KV blocks.
        document_start: Offset of the document's tokens in the prompt.
        document_end: Offset one past the document's last token.

    Returns:
        (own, shared_document, other): tokens the own prefix explains,
        tokens inside the document beyond it that another document's
        request computed, and tokens outside the document (preamble,
        labels, question, block rounding).
    """
    cached = int(cached)
    own = min(cached, could_hit)
    shared_start = max(could_hit, document_start)
    shared = max(0, min(cached, document_end) - shared_start)
    return own, shared, cached - own - shared


class _FilterChain:
    """Answers, survivors, and counters for one filter chain.

    Cached tokens split three ways. A request at stage 1 or later could
    hit the prefix its own document's previous request computed, up to
    the shared part of the two prompts rounded to KV blocks; a miss
    there is per document KV regret. Cached tokens beyond that but
    inside the document came from another document's request that
    shared a prefix. Cached tokens past the document are the question
    or block rounding and count for neither.
    """

    def __init__(self, body_ids, question_ids, true_ids, block_size=1):
        self.body_ids = body_ids
        self.question_ids = question_ids
        self.true_ids = true_ids
        self.block_size = block_size
        self.answers = {}
        self.survivors = []
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.regret_tokens = 0
        self.cross_row_cached_tokens = 0
        self.cached_own_tokens = 0
        self.cached_other_tokens = 0

    def _same_row_hit(self, document: int, stage: int) -> int:
        if stage == 0:
            return 0
        shared = len(self.body_ids[document]) + longest_common_prefix(
            self.question_ids[stage - 1], self.question_ids[stage])
        return (shared // self.block_size) * self.block_size

    def record(self, document: int, stage: int, output) -> bool:
        """Record one answer; return True when the document advances."""
        self.requests += 1
        self.prompt_tokens += len(output.prompt_token_ids)
        cached = int(getattr(output, "num_cached_tokens", 0) or 0)
        self.cached_tokens += cached
        could_hit = self._same_row_hit(document, stage)
        own, shared, other = split_cached_tokens(
            cached, could_hit, 0, len(self.body_ids[document]))
        self.regret_tokens += max(0, could_hit - cached)
        self.cached_own_tokens += own
        self.cross_row_cached_tokens += shared
        self.cached_other_tokens += other
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
            "regret_tokens": self.regret_tokens,
            "cross_row_cached_tokens": self.cross_row_cached_tokens,
            "cached_own_tokens": self.cached_own_tokens,
            "cached_other_tokens": self.cached_other_tokens,
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
    chain = _FilterChain(body_ids, question_ids, true_ids, block_size)
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
    chain = _FilterChain(body_ids, question_ids, true_ids, block_size)
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


def join_cache_accounting(
    prefixes,
    suffix_count: int,
    cached,
    seen_prefix_lengths,
    block_size: int,
    document_spans=None,
) -> dict:
    """Split a join's cached tokens into regret, cross row hits, and rest.

    The first suffix of an anchor could hit only the part of the
    prefix an earlier request already computed; later suffixes could
    hit the whole prefix. Cached tokens short of that are per document
    KV regret. On the first suffix, cached tokens beyond it that fall
    inside the anchor document were computed by another anchor's
    request that shared a prefix. Cached tokens anywhere else (the
    preamble, the label tokens after the prefix, the block that
    straddles the prefix end) count for neither.

    Args:
        prefixes: Per-anchor prefix token sequences.
        suffix_count: Requests (suffixes) per anchor: one count for
            every anchor, or a list with one count per anchor.
        cached: Cached token count per request, in anchor-major order.
        seen_prefix_lengths: Per anchor, the prefix length an earlier
            request had already computed.
        block_size: KV block size in tokens; a hit rounds down to it.
        document_spans: (start, end) of the anchor document inside each
            prefix. Defaults to the whole prefix.

    Returns:
        Dict with regret_tokens, cross_row_cached_tokens,
        cached_own_tokens, and cached_other_tokens.
    """
    regret = 0
    cross_row = 0
    own_total = 0
    other_total = 0
    counts = (
        [suffix_count] * len(prefixes) if isinstance(suffix_count, int)
        else list(suffix_count)
    )
    pair = 0
    for anchor_index, prefix in enumerate(prefixes):
        if document_spans is None:
            span = (0, len(prefix))
        else:
            span = document_spans[anchor_index]
        for suffix_index in range(counts[anchor_index]):
            could_hit = (
                seen_prefix_lengths[anchor_index]
                if suffix_index == 0 else len(prefix)
            )
            could_hit = (could_hit // block_size) * block_size
            regret += max(0, could_hit - int(cached[pair]))
            if suffix_index == 0:
                own, shared, other = split_cached_tokens(
                    cached[pair], could_hit, *span)
            else:
                own, shared, other = split_cached_tokens(
                    cached[pair], could_hit, 0, 0)
            own_total += own
            cross_row += shared
            other_total += other
            pair += 1
    return {
        "regret_tokens": regret,
        "cross_row_cached_tokens": cross_row,
        "cached_own_tokens": own_total,
        "cached_other_tokens": other_total,
    }


def join_regret_tokens(
    prefixes,
    suffix_count: int,
    cached,
    seen_prefix_lengths,
    block_size: int,
) -> int:
    """Count prefix tokens a request recomputed although it could have hit."""
    return join_cache_accounting(
        prefixes, suffix_count, cached, seen_prefix_lengths, block_size,
    )["regret_tokens"]
