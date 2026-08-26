"""The strongest stock vLLM clients the exploration built, carried
over unchanged in substance:

- run_filter_chain: one request per (document, stage), gated
  client-side, admission by the SAME token budget the plan derives -
  the committed pipelined client (42.9 s vs chain's 39.8 s at 10k
  documents; the gap is block-boundary recompute and request-count
  overhead, not a handicapped client).

- run_join_grouped: one request per pair, ordered anchor-major so
  vLLM's prefix cache re-serves each anchor's KV across its whole
  partner stream (the arbitrary-order variant measured 1.87x worse
  and is never run).

Plain synchronous loops against the vLLM v1 LLMEngine surface; no
asyncio anywhere.
"""

import itertools
import time

from quail.logical import join_anchor_prefix_ids, join_tuple_suffix_ids


def build_join_grouped_inputs(prompt, documents, anchor: int, tokenizer):
    """Build canonical join token parts for stock vLLM.

    documents contains one list of tokenized documents per prompt
    placeholder. The returned prefixes are in anchor document order.
    The returned suffixes are in the Cartesian order of the remaining
    placeholders. `members` records each suffix's partner indices in
    placeholder order, excluding the anchor.
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
    """The answer bit. Under the one-token constrained sampler
    (allowed_token_ids = TRUE|FALSE ids, max_tokens=1) this is a
    token-id check, identical to the packed executor's answerer. The
    text scan below is the fallback for unconstrained models that
    restate the flag line or chatter - first decisive word wins."""
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
                     budget_tokens, tag="q", true_ids=None):
    """The stock filter baseline, the committed client's admission
    exactly: the token budget expressed as a DOCUMENT cap of
    budget // (mean document + longest question), and a document
    holds its slot from first stage to last verdict. Between a
    document's stages the engine serves other documents, so whether
    its KV is still resident is up to the prefix cache - that is the
    baseline the packed executor's kept KV replaces.

    Returns wall, per-(doc, stage) answers (stages 1-indexed),
    survivors, and the request/token counters."""
    n = len(q_ids)
    mean_req = (sum(len(b) for b in body_ids) // max(1, len(body_ids))
                + max(len(q) for q in q_ids))
    cap = max(1, budget_tokens // mean_req)
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0,
                    doc_cap=cap)
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


def run_join_grouped(llm, sampling_params, prefixes, suffixes,
                     true_ids):
    """The stock join baseline: one request per pair, anchor-major
    order (all of an anchor's pairs consecutive), prefix caching left
    to the engine. One generate() over the full list - the join has
    no gating, so nothing needs the pipelined client."""
    pair_prompts = [{"prompt_token_ids": p + s}
                    for p in prefixes for s in suffixes]
    t0 = time.time()
    outputs = llm.generate(pair_prompts, sampling_params,
                           use_tqdm=False)
    wall = time.time() - t0
    answers = [1 if int(o.outputs[0].token_ids[0]) in true_ids else 0
               for o in outputs]
    cached = sum(getattr(o, "num_cached_tokens", 0) or 0
                 for o in outputs)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    return dict(wall=wall, answers=answers,
                fresh_tokens=prompt_tokens - cached,
                prompt_tokens=prompt_tokens, cached_tokens=cached)
