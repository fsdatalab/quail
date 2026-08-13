"""Clients that run a filter query against a vLLM engine.

Two executors, both under a plan-derived token budget:

  run_filter_chain         separate requests, one per (document,
                           stage), gated client-side. The document's
                           KV is whatever the engine's prefix cache
                           still holds when the next stage arrives.
                           This is the stock-engine baseline; with
                           `tags` it also carries pin and release
                           directives for the Quail scheduler.

  run_filter_chain_engine  chain mode: ONE living request per
                           document. The client registers the
                           questions once, then submits a single
                           request; the scheduler judges each answer,
                           rewinds to the document boundary, and
                           appends the next question. The document
                           prefills exactly once.

Admission is a token budget sized from the KV pool, not a request
count: a document enters only when its worst-case tokens fit, so the
pool never overflows and the prefix cache is never forced to evict.
Prompts are pre-tokenized once, so the engine never re-converts the
same document text.

The engine object must expose the vLLM AsyncLLM interface:
`generate(prompt, sampling_params, request_id)` returning an async
generator whose final item has `prompt_token_ids`, `num_cached_tokens`,
and `outputs[0].text`.
"""

import asyncio
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


class EngineTags:
    """Builds the de1| request ids for the in-engine scheduler: a pin
    directive on each document's first request, releases piggybacked on
    later submissions, and the end-of-run flush that releases every
    remaining pin."""

    def __init__(self, batch=40):
        self.pending = []
        self.batch = batch

    def rid(self, suffix, doc=None, pin_tokens=0, uses=1, extra=()):
        parts = ["de1", *extra]
        if pin_tokens and doc is not None:
            parts.append(f"p{pin_tokens}")
            parts.append(f"d{doc}")
            if uses > 1:
                parts.append(f"u{uses}")
        elif doc is not None:
            parts.append(f"d{doc}")
        if self.pending:
            take, self.pending = (self.pending[:self.batch],
                                  self.pending[self.batch:])
            parts.append("r" + ",".join(take))
        parts.append(suffix)
        return "|".join(parts)

    def release(self, doc):
        self.pending.append(str(doc))


async def run_filter_chain(engine, sampling_params, body_ids, q_ids,
                           budget_tokens, tag="q", tags=None,
                           use_priority=False):
    """Separate requests, one per (document, stage), gated client-side:
    a document's next stage is submitted only after the previous
    answer arrives, and a failed stage drops the document. Returns
    timings, counters, per-call answers keyed (doc, stage) with stages
    1-indexed, and the surviving document ids.

    Between a document's stages the engine serves other documents, so
    whether the document's KV is still resident is up to the prefix
    cache. That is the baseline this project's chain mode replaces.

    With `tags` set (an EngineTags), request ids carry pin and release
    directives for the in-engine scheduler and a flush request releases
    all pins at the end. With `use_priority`, resident-consumer requests
    (stage two onward) are submitted at a higher engine priority than
    first reads."""
    n = len(q_ids)
    q_cost = sum(len(q) for q in q_ids) + n
    used = 0
    cond = asyncio.Condition()
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    answers = {}
    survivors = []

    async def ask(ids, rid, pri=0, count=True):
        final = None
        kw = {"priority": pri} if use_priority else {}
        async for out in engine.generate({"prompt_token_ids": ids},
                                         sampling_params, rid, **kw):
            final = out
        if count:
            counters["requests"] += 1
            counters["prompt_tokens"] += len(final.prompt_token_ids)
            counters["cached_tokens"] += (
                getattr(final, "num_cached_tokens", 0) or 0)
        return _yes(final)

    # the pin must claim everything with a future consumer: the document
    # plus the shared prefix of the stage questions (their common
    # preamble straddles the boundary block and is reused by every
    # later stage)
    q_common = 0
    if len(q_ids) > 1:
        for col in zip(*q_ids):
            if len(set(col)) != 1:
                break
            q_common += 1

    def rid_for(i, j0):
        suffix = f"{tag}-{i}-{j0}"
        if tags is None:
            return suffix
        if j0 == 0:
            return tags.rid(suffix, doc=i,
                            pin_tokens=len(body_ids[i]) + q_common)
        return tags.rid(suffix)

    async def chain(i, cost):
        nonlocal used
        try:
            for j in range(n):
                got = await ask(body_ids[i] + q_ids[j], rid_for(i, j),
                                pri=0 if j > 0 else 1)
                answers[(i, j + 1)] = got
                if not got:
                    return
            survivors.append(i)
        finally:
            if tags is not None:
                tags.release(i)
            async with cond:
                used -= cost
                cond.notify_all()

    t0 = time.time()
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(chain(i, cost)))
    await asyncio.gather(*tasks)
    wall = time.time() - t0
    if tags is not None:
        # release every remaining pin so the engine can reset cleanly
        await ask(q_ids[0], f"de1|r*|{tag}-0-99", pri=0, count=False)
    return dict(wall=wall, survivors=sorted(survivors), answers=answers,
                **counters)


async def _register_query(engine, sampling_params, q_ids, yes_ids, no_ids,
                          qpart, suffix):
    """Send one query's chain-mode registration (question token lists
    plus the yes/no token ids in the request id) and wait it out."""
    reg = [len(q_ids)]
    for q in q_ids:
        reg += [len(q)] + list(q)
    npart = (f"N{','.join(map(str, sorted(no_ids)))}|" if no_ids else "")
    reg_rid = (f"de1|reg|{qpart}"
               f"Y{','.join(map(str, sorted(yes_ids)))}|{npart}"
               f"{suffix}")

    async def _reg():
        async for _ in engine.generate({"prompt_token_ids": reg},
                                       sampling_params, reg_rid):
            pass

    try:
        await asyncio.wait_for(_reg(), timeout=30)
    except (asyncio.TimeoutError, Exception):
        try:
            res = engine.abort(reg_rid)
            if hasattr(res, "__await__"):
                await res
        except Exception:
            pass


def _stage_tokens(snapshots, yes_ids, no_ids):
    """The per-stage answer tokens hidden in a chain's cumulative
    output record. The engine's rewind is invisible on the client
    side: the record only grows. Without no_ids every new token is
    one stage's answer; with them, only decisive tokens are answers
    and the rest is the model's trailing chatter, cut short
    engine-side."""
    decisive = set(yes_ids) | set(no_ids or ())
    stage_toks, seen = [], 0
    for snap in snapshots:
        for t in snap[seen:]:
            if not no_ids or t in decisive:
                stage_toks.append(t)
        seen = max(seen, len(snap))
    return stage_toks


async def run_filter_chain_engine(engine, sampling_params, body_ids, q_ids,
                                  budget_tokens, yes_ids, tag="c",
                                  no_ids=None):
    """Chain mode: the engine itself runs each document's whole filter
    chain. The client registers the question token lists and the gate
    token ids once, then submits ONE request per document; the
    scheduler judges each answer, rewinds to the document boundary,
    and appends the next question. Returns the same result shape as
    run_filter_chain.

    Because the document's KV belongs to a request that stays alive
    across every stage, it cannot be evicted between stages and is
    never re-prefilled - the read multiplier drops to the question
    tokens alone.

    With no_ids given, the gate runs in decisive-token mode for models
    that do not answer in one token: each stage may sample several
    tokens, the engine stops the stage at the first token in either
    set, and only those tokens count as stage answers. The simpler
    contract is a constrained sampler
    (SamplingParams(allowed_token_ids=yes|no ids, max_tokens=1)),
    which makes every stage answer in one token by construction."""
    n = len(q_ids)
    await _register_query(engine, sampling_params, q_ids, yes_ids, no_ids,
                          "", f"{tag}-reg")

    used = 0
    cond = asyncio.Condition()
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    answers = {}
    survivors = []
    raw0 = []
    q_cost = sum(len(q) for q in q_ids) + n

    async def chain(i, cost):
        nonlocal used
        toks = []
        try:
            final = None
            rid = f"de1|c|d{i}|{tag}-{i}-0"
            async for out in engine.generate(
                    {"prompt_token_ids": body_ids[i] + q_ids[0]},
                    sampling_params, rid):
                final = out
                ids = list(out.outputs[0].token_ids or ())
                if ids:
                    toks.append(ids)
            counters["requests"] += 1
            if final is not None:
                counters["prompt_tokens"] += len(final.prompt_token_ids)
                counters["cached_tokens"] += (
                    getattr(final, "num_cached_tokens", 0) or 0)
            stage_toks = _stage_tokens(toks, yes_ids, no_ids)
            if i == 0:
                raw0.extend(tuple(s) for s in toks[:8])
            for j, t in enumerate(stage_toks[:n]):
                answers[(i, j + 1)] = 1 if t in yes_ids else 0
            if len(stage_toks) >= n \
                    and all(t in yes_ids for t in stage_toks[:n]):
                survivors.append(i)
        finally:
            async with cond:
                used -= cost
                cond.notify_all()

    t0 = time.time()
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(chain(i, cost)))
    await asyncio.gather(*tasks)
    wall = time.time() - t0
    return dict(wall=wall, survivors=sorted(survivors), answers=answers,
                doc0_raw=raw0, **counters)
