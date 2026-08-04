"""Blocked streaming client scheduler (scheduler plan step three).

Runs a chain of yes or no filters over a document corpus against an
unmodified serving engine, combining the three wins measured in steps
one and two. Prompts are pre-tokenized once, so the engine never
re-converts the same document text. Admission is controlled by a token
budget sized to the engine's KV pool, so a document is admitted only
when its notes can stay resident, and because each document streams
through all of its filters the moment each answer arrives, it finishes
its whole chain while hot and returns its budget quickly. This yields
the blocked execution order without any stage barriers: the fastest
document never waits for the slowest.

Optional lookahead submits the next `lookahead` questions of a document
concurrently, paying for wasted questions when an earlier one fails.
Under streaming there are no stage barriers left for speculation to
hide, so its expected value is limited to filling the admission tail.

The engine object must expose the vLLM AsyncLLM interface:
`generate(prompt, sampling_params, request_id)` returning an async
generator whose final item has `prompt_token_ids`, `num_cached_tokens`,
and `outputs[0].text`.
"""

import asyncio
import time


def _yes(out):
    """First decisive word wins. The 4B model answers with the bare
    token; the 32B sometimes restates the flag line (the answer lands
    mid-text) or appends chatter after it."""
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

    def rid(self, suffix, doc=None, pin_tokens=0):
        parts = ["de1"]
        if pin_tokens and doc is not None:
            parts.append(f"p{pin_tokens}")
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
                           budget_tokens, lookahead=1, tag="q", tags=None,
                           use_priority=False, composition=None):
    """Run every document through the filter chain; return timings,
    counters, per-call answers keyed (doc, stage) with stages 1-indexed,
    and the surviving document ids.

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
            j = 0
            blocks = list(composition) if composition else None
            while j < n:
                kk = (blocks.pop(0) if blocks
                      else min(lookahead, n - j))
                kk = min(kk, n - j)
                got = await asyncio.gather(*[
                    ask(body_ids[i] + q_ids[j + off], rid_for(i, j + off),
                        pri=0 if j + off > 0 else 1)
                    for off in range(kk)])
                passes = 0
                for off in range(kk):
                    answers[(i, j + off + 1)] = got[off]
                    if got[off] and passes == off:
                        passes += 1
                if passes < kk:
                    return
                j += kk
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


async def run_query(engine, sampling_params, body_ids, q_ids,
                    budget_tokens, yes_ids=None, tag="q", no_ids=None):
    """The shipped plan. Multi-filter queries run in chain mode: one
    living engine request per document runs the whole filter chain,
    the scheduler judging answers and rewinding between filters
    (yes_ids, the token ids that mean yes, is required for the
    in-engine gate). A single-filter query has nothing to chain and
    runs as pinned, ranked requests."""
    if len(q_ids) >= 2:
        assert yes_ids, "chain mode needs the yes token ids for its gate"
        return await run_filter_chain_engine(engine, sampling_params,
                                             body_ids, q_ids,
                                             budget_tokens, yes_ids,
                                             tag=tag, no_ids=no_ids)
    return await run_filter_chain(engine, sampling_params, body_ids,
                                  q_ids, budget_tokens, lookahead=1,
                                  tag=tag, tags=EngineTags(),
                                  use_priority=True)


async def run_filter_chain_engine(engine, sampling_params, body_ids, q_ids,
                                  budget_tokens, yes_ids, tag="c",
                                  no_ids=None):
    """Chain mode: the engine itself runs each document's whole filter
    chain (register questions once, then one request per document; the
    scheduler judges answers, rewinds, and continues). Returns the same
    result shape as run_filter_chain.

    With no_ids given, the gate runs in decisive-token mode for models
    that do not answer in one token: each stage may sample several
    tokens, the engine stops the stage at the first token in either
    set, and only those tokens count as stage answers."""
    import asyncio
    import time as _time

    n = len(q_ids)
    reg = [n]
    for q in q_ids:
        reg += [len(q)] + list(q)
    npart = (f"N{','.join(map(str, sorted(no_ids)))}|" if no_ids else "")
    reg_rid = (f"de1|reg|Y{','.join(map(str, sorted(yes_ids)))}|"
               f"{npart}{tag}-reg")

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
            async for out in engine.generate(
                    {"prompt_token_ids": body_ids[i] + q_ids[0]},
                    sampling_params, f"de1|c|d{i}|{tag}-{i}-0"):
                final = out
                ids = list(out.outputs[0].token_ids or ())
                if ids:
                    toks.append(ids)
            counters["requests"] += 1
            if final is not None:
                counters["prompt_tokens"] += len(final.prompt_token_ids)
                counters["cached_tokens"] += (
                    getattr(final, "num_cached_tokens", 0) or 0)
            # the engine's rewind is invisible on this side: the
            # client's cumulative output record only grows. Without
            # no_ids every new token is one stage's answer; with them,
            # only decisive tokens are answers and the rest is the
            # model's trailing chatter, cut short engine-side.
            decisive = set(yes_ids) | set(no_ids or ())
            stage_toks, seen = [], 0
            for snap in toks:
                for t in snap[seen:]:
                    if not no_ids or t in decisive:
                        stage_toks.append(t)
                seen = max(seen, len(snap))
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

    t0 = _time.time()
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(chain(i, cost)))
    await asyncio.gather(*tasks)
    wall = _time.time() - t0
    return dict(wall=wall, survivors=sorted(survivors), answers=answers,
                doc0_raw=raw0, **counters)
