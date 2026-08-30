"""Stock vLLM filter and join clients.

Synchronous loops against the vLLM v1 LLMEngine surface.
"""

import itertools
import time

from quail.logical import join_anchor_prefix_ids, join_tuple_suffix_ids


def build_join_grouped_inputs(prompt, documents, anchor: int, tokenizer):
    """Build canonical join token parts for stock vLLM.

    Args:
        prompt: Bound join prompt.
        documents: One list of tokenized documents per placeholder.
        anchor: Index of the anchor placeholder.
        tokenizer: Tokenizer for encoding.

    Returns:
        Tuple of (prefixes, suffixes, members). Prefixes are in anchor
        document order, suffixes in Cartesian order of the remaining
        placeholders. members records each suffix's partner indices.
    """
    if len(documents) != len(prompt.args):
        raise ValueError(
            f"join has {len(prompt.args)} placeholders but received "
            f"{len(documents)} document tables")
    if anchor < 0 or anchor >= len(documents):
        raise ValueError(f"join anchor placeholder {anchor} is out of range")
    partner_slots = [i for i in range(len(documents)) if i != anchor]
    members = list(itertools.product(
        *(range(len(documents[i])) for i in partner_slots)))
    prefixes = [
        join_anchor_prefix_ids(prompt, anchor, ids, tokenizer)
        for ids in documents[anchor]
    ]
    suffixes = [
        join_tuple_suffix_ids(
            prompt,
            [(slot, documents[slot][member[j]])
             for j, slot in enumerate(partner_slots)],
            tokenizer)
        for member in members
    ]
    return prefixes, suffixes, members


def _true_bit(out, true_ids=None):
    """Extract the TRUE/FALSE answer bit from one vLLM output."""
    if true_ids is not None:
        toks = out.outputs[0].token_ids
        if toks:
            return 1 if int(toks[0]) in true_ids else 0
    t = out.outputs[0].text.upper()
    it = t.find("TRUE")
    if it < 0:
        return 0
    ifa = t.find("FALSE")
    return 1 if ifa < 0 or it < ifa else 0


def run_filter_chain(engine, sampling_params, body_ids, q_ids,
                     budget_tokens, tag="q", true_ids=None,
                     block_size=1, max_num_seqs=None):
    """Run a filter chain over stock vLLM with token-budget admission.

    Returns:
        Dict with wall time, per-(doc, stage) answers (stages 1-indexed),
        survivors, and request/token counters.
    """
    n = len(q_ids)
    longest_tail = max(len(q) for q in q_ids)
    request_sizes = [len(body) + longest_tail + 1 for body in body_ids]
    rounded_sizes = [
        ((tokens + block_size - 1) // block_size) * block_size
        for tokens in request_sizes
    ]
    mean_req = sum(rounded_sizes) // max(1, len(rounded_sizes))
    cap = max(1, budget_tokens // mean_req)
    if max_num_seqs is not None:
        cap = min(cap, max_num_seqs)
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0,
                    doc_cap=cap, budget_tokens=budget_tokens,
                    block_size=block_size,
                    max_num_seqs=max_num_seqs)
    answers = {}
    survivors = []
    inflight = {}                      # request id -> (doc, stage)

    def submit(i, j):
        rid = f"{tag}-{i}-{j}"
        inflight[rid] = (i, j)
        engine.add_request(
            rid, {"prompt_token_ids": body_ids[i] + q_ids[j]},
            sampling_params)

    t0 = time.time()
    live, next_doc = 0, 0
    while next_doc < len(body_ids) or inflight:
        # admit in workload order while live documents stay under
        # the cap
        while next_doc < len(body_ids) and live < cap:
            submit(next_doc, 0)
            live += 1
            next_doc += 1
        for out in engine.step():
            if not out.finished or out.request_id not in inflight:
                continue
            i, j = inflight.pop(out.request_id)
            counters["requests"] += 1
            counters["prompt_tokens"] += len(out.prompt_token_ids)
            counters["cached_tokens"] += (
                getattr(out, "num_cached_tokens", 0) or 0)
            got = _true_bit(out, true_ids)
            answers[(i, j + 1)] = got
            if got and j + 1 < n:
                submit(i, j + 1)      # the document keeps its slot
            else:
                if got:
                    survivors.append(i)
                live -= 1             # retire: the slot returns
    wall = time.time() - t0
    return dict(wall=wall, survivors=sorted(survivors),
                answers=answers, **counters)


def suffix_major_tiled_order(prefixes, suffixes, tile_budget_tokens):
    """Order pairs suffix-major within anchor tiles under a KV budget.

    Within one tile no two pairs share an anchor until the first
    suffix pass has completed and cached every anchor, so an engine
    that only caches finished requests (SGLang's radix cache) reuses
    anchors from the second pass on. The tile bound keeps a tile's
    anchors resident: a full pass over more anchor tokens than the KV
    pool holds would evict each anchor before its next use.

    Returns:
        List of (anchor_index, suffix_index) in submission order.
    """
    if tile_budget_tokens <= 0:
        raise ValueError("tile_budget_tokens must be positive")
    max_suffix = max(len(s) for s in suffixes)
    tiles, tile, used = [], [], 0
    for anchor_index, prefix in enumerate(prefixes):
        cost = len(prefix) + max_suffix
        if tile and used + cost > tile_budget_tokens:
            tiles.append(tile)
            tile, used = [], 0
        tile.append(anchor_index)
        used += cost
    if tile:
        tiles.append(tile)
    return [
        (anchor_index, suffix_index)
        for tile in tiles
        for suffix_index in range(len(suffixes))
        for anchor_index in tile
    ]


def run_join_grouped(llm, sampling_params, prefixes, suffixes,
                     true_ids, submission="anchor-major",
                     tile_budget_tokens=None):
    """Run a join as one request per pair over the full cross product.

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
        pair_prompts = [{"prompt_token_ids": p + s}
                        for p in prefixes for s in suffixes]
    elif submission == "suffix-major-tiled":
        if tile_budget_tokens is None:
            raise ValueError(
                "suffix-major-tiled needs tile_budget_tokens")
        order = suffix_major_tiled_order(
            prefixes, suffixes, tile_budget_tokens)
        pair_prompts = [
            {"prompt_token_ids": prefixes[i] + suffixes[j]}
            for i, j in order
        ]
    else:
        raise ValueError(f"unknown join submission {submission!r}")

    t0 = time.time()
    outputs = llm.generate(pair_prompts, sampling_params,
                           use_tqdm=False)
    wall = time.time() - t0
    bits = [1 if int(o.outputs[0].token_ids[0]) in true_ids else 0
            for o in outputs]
    cached_by_output = [
        int(getattr(o, "num_cached_tokens", 0) or 0) for o in outputs]
    if order is None:
        answers = bits
        cached_per_request = cached_by_output
    else:
        answers = [0] * len(bits)
        cached_per_request = [0] * len(bits)
        for position, (i, j) in enumerate(order):
            pair = i * len(suffixes) + j
            answers[pair] = bits[position]
            cached_per_request[pair] = cached_by_output[position]
    cached = sum(cached_per_request)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    return dict(wall=wall, answers=answers,
                fresh_tokens=prompt_tokens - cached,
                prompt_tokens=prompt_tokens, cached_tokens=cached,
                cached_per_request=cached_per_request,
                submission=submission)
