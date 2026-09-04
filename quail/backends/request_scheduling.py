"""Request scheduling shared by the vLLM and SGLang backends.

A filter chain advances each document through its questions one
request at a time and keeps a bounded number of documents live. The
streaming form drives an engine that exposes add_request and step;
the wave form drives a client that only exposes a blocking generate.
Both use the same admission cap and the same bookkeeping.
"""

from __future__ import annotations

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
    """Answers, survivors, and counters for one filter chain.

    Cached tokens split two ways. A request at stage 1 or later could
    hit the prefix its own document's previous request computed, up to
    the shared part of the two prompts rounded to KV blocks; a miss
    there is per document KV regret. Cached tokens beyond that came
    from another document's request that shared a prefix.
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
        self.regret_tokens += max(0, could_hit - cached)
        self.cross_row_cached_tokens += max(0, cached - could_hit)
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


def run_filter_chain_waves(
    client,
    sampling_params,
    body_ids,
    question_ids,
    budget_tokens,
    *,
    true_ids,
    block_size=1,
    max_num_seqs=None,
):
    """Advance every live document by one filter stage per wave.

    The client exposes only a blocking generate, as SGLang's engine
    does, so wave boundaries stand in for the engine step loop. A
    document's next stage always finds its body cached because the
    previous stage finished first.
    """
    document_cap = filter_document_cap(
        body_ids, question_ids, budget_tokens,
        block_size=block_size, max_num_seqs=max_num_seqs,
    )
    chain = _FilterChain(body_ids, question_ids, true_ids, block_size)
    active = []
    next_document = 0
    started = time.perf_counter()
    while active or next_document < len(body_ids):
        while (
            next_document < len(body_ids)
            and len(active) < document_cap
        ):
            active.append((next_document, 0))
            next_document += 1
        prompts = [
            {
                "prompt_token_ids": (
                    body_ids[document] + question_ids[stage]
                )
            }
            for document, stage in active
        ]
        outputs = client.generate(prompts, sampling_params, use_tqdm=False)
        active = [
            (document, stage + 1)
            for (document, stage), output in zip(active, outputs)
            if chain.record(document, stage, output)
        ]
    return chain.result(
        time.perf_counter() - started,
        **_cap_settings(document_cap, budget_tokens, block_size,
                        max_num_seqs),
    )


def suffix_major_tiled_order(prefixes, suffixes, tile_budget_tokens):
    """Order pairs suffix-major within anchor tiles under a KV budget.

    Within one tile no two pairs share an anchor until the first
    suffix pass has completed and cached every anchor, so an engine
    that only caches finished requests (SGLang's radix cache) reuses
    anchors from the second pass on. The tile bound keeps a tile's
    anchors resident.

    Returns:
        List of (anchor_index, suffix_index) in submission order.
    """
    if tile_budget_tokens is None or tile_budget_tokens <= 0:
        raise ValueError("suffix major joins need a positive tile budget")
    max_suffix = max(len(suffix) for suffix in suffixes)
    tiles = []
    tile = []
    used = 0
    for anchor_index, prefix in enumerate(prefixes):
        cost = len(prefix) + max_suffix
        if tile and used + cost > tile_budget_tokens:
            tiles.append(tile)
            tile = []
            used = 0
        tile.append(anchor_index)
        used += cost
    if tile:
        tiles.append(tile)
    return [
        (anchor_index, suffix_index)
        for anchor_tile in tiles
        for suffix_index in range(len(suffixes))
        for anchor_index in anchor_tile
    ]


def run_join_grouped(
    client,
    sampling_params,
    prefixes,
    suffixes,
    true_ids,
    *,
    submission="anchor-major",
    tile_budget_tokens=None,
):
    """Submit one request for every tuple in a full cross product.

    Args:
        submission: "anchor-major" submits all suffixes of one anchor
            before the next anchor. "suffix-major-tiled" submits per
            suffix_major_tiled_order and needs tile_budget_tokens.

    Returns:
        Dict with wall time and counters. answers is in anchor-major
        pair order for either submission.
    """
    if submission == "anchor-major":
        order = None
        prompts = [
            {"prompt_token_ids": prefix + suffix}
            for prefix in prefixes
            for suffix in suffixes
        ]
    elif submission == "suffix-major-tiled":
        order = suffix_major_tiled_order(
            prefixes, suffixes, tile_budget_tokens
        )
        prompts = [
            {"prompt_token_ids": prefixes[anchor] + suffixes[suffix]}
            for anchor, suffix in order
        ]
    else:
        raise ValueError(f"unknown join submission {submission!r}")

    started = time.perf_counter()
    outputs = client.generate(prompts, sampling_params, use_tqdm=False)
    wall = time.perf_counter() - started
    bits = [true_bit(output, true_ids) for output in outputs]
    cached_by_output = [
        int(getattr(output, "num_cached_tokens", 0) or 0)
        for output in outputs
    ]
    if order is None:
        answers = bits
        cached_per_request = cached_by_output
    else:
        answers = [0] * len(bits)
        cached_per_request = [0] * len(bits)
        for output_index, (anchor, suffix) in enumerate(order):
            pair = anchor * len(suffixes) + suffix
            answers[pair] = bits[output_index]
            cached_per_request[pair] = cached_by_output[output_index]
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
) -> tuple[int, int]:
    """Split a join's cached tokens into regret and cross row hits.

    The first suffix of an anchor could hit only the part of the
    prefix an earlier request already computed; later suffixes could
    hit the whole prefix. Cached tokens short of that are per document
    KV regret. Cached tokens beyond it were computed by another
    document's request that shared a prefix.

    Returns:
        (regret_tokens, cross_row_cached_tokens)
    """
    regret = 0
    cross_row = 0
    for anchor_index, prefix in enumerate(prefixes):
        for suffix_index in range(suffix_count):
            could_hit = (
                seen_prefix_lengths[anchor_index]
                if suffix_index == 0 else len(prefix)
            )
            could_hit = (could_hit // block_size) * block_size
            pair = anchor_index * suffix_count + suffix_index
            regret += max(0, could_hit - int(cached[pair]))
            cross_row += max(0, int(cached[pair]) - could_hit)
    return regret, cross_row


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
    )[0]
