"""Request scheduling used by vLLM and SGLang backends."""

from __future__ import annotations

import time


def _true_bit(output, true_ids) -> int:
    token_ids = output.outputs[0].token_ids
    return int(bool(token_ids and int(token_ids[0]) in true_ids))


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
    """Submit the next filter after each document passes."""
    longest_tail = max(len(question) for question in question_ids)
    rounded_sizes = [
        (
            (len(body) + longest_tail + block_size - 1) // block_size
        ) * block_size
        for body in body_ids
    ]
    mean_request = sum(rounded_sizes) // max(1, len(rounded_sizes))
    document_cap = max(1, budget_tokens // mean_request)
    if max_num_seqs is not None:
        document_cap = min(document_cap, max_num_seqs)
    counters = {
        "requests": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "doc_cap": document_cap,
        "budget_tokens": budget_tokens,
        "block_size": block_size,
        "max_num_seqs": max_num_seqs,
    }
    answers = {}
    survivors = []
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
            counters["requests"] += 1
            counters["prompt_tokens"] += len(output.prompt_token_ids)
            counters["cached_tokens"] += int(
                getattr(output, "num_cached_tokens", 0) or 0
            )
            answer = _true_bit(output, true_ids)
            answers[(document, stage + 1)] = answer
            if answer and stage + 1 < len(question_ids):
                submit(document, stage + 1)
            else:
                if answer:
                    survivors.append(document)
                live -= 1
    return {
        "wall": time.perf_counter() - started,
        "survivors": sorted(survivors),
        "answers": answers,
        **counters,
    }


def suffix_major_tiled_order(prefixes, suffixes, tile_budget_tokens):
    """Order suffixes within anchor groups bounded by a KV budget."""
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
    """Submit one request for every tuple in a full cross product."""
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
    bits = [_true_bit(output, true_ids) for output in outputs]
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

__all__ = [
    "run_filter_chain",
    "run_join_grouped",
    "suffix_major_tiled_order",
]
