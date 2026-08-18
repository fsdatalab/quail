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

import time


def _yes(out):
    """First decisive word wins: some models restate the flag line
    (the answer lands mid-text) or append chatter after it."""
    t = out.outputs[0].text.upper()
    iy = t.find("YES")
    if iy < 0:
        return 0
    ino = t.find("NO")
    return 1 if ino < 0 or iy < ino else 0


def run_filter_chain(engine, sampling_params, body_ids, q_ids,
                     budget_tokens, tag="q"):
    """The stock filter baseline. Between a document's stages the
    engine serves other documents, so whether the document's KV is
    still resident is up to the prefix cache - that is the baseline
    the packed executor's kept KV replaces.

    Returns wall, per-(doc, stage) answers (stages 1-indexed),
    survivors, and the request/token counters."""
    n = len(q_ids)
    q_cost = sum(len(q) for q in q_ids) + n
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    answers = {}
    survivors = []
    inflight = {}                      # request id -> (doc, stage, cost)

    def submit(i, j, cost):
        rid = f"{tag}-{i}-{j}"
        inflight[rid] = (i, j, cost)
        engine.add_request(
            rid, {"prompt_token_ids": body_ids[i] + q_ids[j]},
            sampling_params)

    t0 = time.time()
    used, next_doc = 0, 0
    while next_doc < len(body_ids) or inflight:
        # admit in workload order while the budget allows; an empty
        # engine always admits one, so a tiny budget serializes
        # instead of deadlocking
        while next_doc < len(body_ids):
            cost = len(body_ids[next_doc]) + q_cost
            if used + cost > budget_tokens and used > 0:
                break
            used += cost
            submit(next_doc, 0, cost)
            next_doc += 1
        for out in engine.step():
            if not out.finished or out.request_id not in inflight:
                continue
            i, j, cost = inflight.pop(out.request_id)
            counters["requests"] += 1
            counters["prompt_tokens"] += len(out.prompt_token_ids)
            counters["cached_tokens"] += (
                getattr(out, "num_cached_tokens", 0) or 0)
            got = _yes(out)
            answers[(i, j + 1)] = got
            if got and j + 1 < n:
                submit(i, j + 1, cost)
            else:
                if got:
                    survivors.append(i)
                used -= cost                    # retire: budget returns
    wall = time.time() - t0
    return dict(wall=wall, survivors=sorted(survivors),
                answers=answers, **counters)


def run_join_grouped(llm, sampling_params, prefixes, suffixes,
                     yes_ids):
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
    answers = [1 if int(o.outputs[0].token_ids[0]) in yes_ids else 0
               for o in outputs]
    cached = sum(getattr(o, "num_cached_tokens", 0) or 0
                 for o in outputs)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    return dict(wall=wall, answers=answers,
                fresh_tokens=prompt_tokens - cached,
                prompt_tokens=prompt_tokens, cached_tokens=cached)
