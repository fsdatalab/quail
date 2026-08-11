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

Speculation is not a client concern: it runs in-engine as the
speculative chain (run_filter_chain_engine with spec=True), one
request per document. The client-side concurrent-branch path was
measured against it and deleted (the ledger's speculative-chain
section; results/engine/spec_smoke.json).

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
    """Run every document through the filter chain as pinned, ranked
    requests, one question at a time, gated; return timings, counters,
    per-call answers keyed (doc, stage) with stages 1-indexed, and the
    surviving document ids.

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


async def run_map(engine, sampling_params, body_ids, prompt_ids,
                  budget_tokens, tag="m"):
    """Open-ended generation: every prompt against every document,
    one request per (document, prompt) pair, generated text returned
    keyed (doc, prompt) with prompts 1-indexed. Decode exists here.

    The document reads once: its first prompt's request commits the
    KV, and the remaining prompts launch on that request's first
    streamed token so they reuse it - simultaneous identical
    prefixes would each recompute the document (the measured prefill
    race, results/engine/reason_race.json). Multi-token outputs need
    their own streams, which is why this is not the one-request-per-
    document chain: routing sibling generations onto one stream is
    the open fork extension.

    Admission charges a document its full worst case up front - body,
    every prompt, the full generation cap per prompt - and every
    prompt launches on the first streamed token exactly as before:
    delaying a sibling for budget was measured to lose the shared
    document KV outright (the sibling arrives after the first prompt
    finished and the document's blocks were dropped; reads 2.35
    against the 1.37 floor, wall 3x). What admission releases early:
    each prompt's own charge (its tokens plus its cap) the moment
    that prompt finishes, instead of holding the whole document until
    its last prompt. The recycled budget admits the next documents
    sooner, which is where the decode width comes from."""
    n = len(prompt_ids)
    gen_budget = getattr(sampling_params, "max_tokens", 0) or 0
    q_cost = sum(len(p) for p in prompt_ids) + n * gen_budget
    used = 0
    cond = asyncio.Condition()
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    texts = {}

    async def gen(ids, rid, started=None):
        final = None
        try:
            async for out in engine.generate({"prompt_token_ids": ids},
                                             sampling_params, rid):
                final = out
                if started is not None and not started.is_set():
                    started.set()
        finally:
            if started is not None:
                started.set()
        counters["requests"] += 1
        counters["prompt_tokens"] += len(final.prompt_token_ids)
        counters["cached_tokens"] += (
            getattr(final, "num_cached_tokens", 0) or 0)
        return final.outputs[0].text

    async def one_doc(i, cost):
        nonlocal used
        held = cost          # this document's unreleased charge

        def release(amount):
            # under cond; never release more than the doc still holds
            nonlocal used
            nonlocal held
            amt = min(amount, held)
            held -= amt
            used -= amt

        async def one_prompt(j, started=None):
            try:
                return await gen(body_ids[i] + prompt_ids[j],
                                 f"de1|d{i}|{tag}-{i}-{j}",
                                 started=started)
            finally:
                async with cond:
                    release(len(prompt_ids[j]) + gen_budget)
                    cond.notify_all()

        try:
            ev = asyncio.Event()
            first = asyncio.create_task(one_prompt(0, started=ev))
            await ev.wait()
            rest = await asyncio.gather(*[one_prompt(j)
                                          for j in range(1, n)])
            texts[(i, 1)] = await first
            for j, t in enumerate(rest, start=2):
                texts[(i, j)] = t
        finally:
            async with cond:
                release(held)
                cond.notify_all()

    t0 = time.time()
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(one_doc(i, cost)))
    await asyncio.gather(*tasks)
    assert used == 0, f"admission accounting leaked: {used}"
    return dict(wall=time.time() - t0, texts=texts, **counters)


async def run_map_forked(engine, sampling_params, body_ids, prompt_ids,
                         budget_tokens, sep_id, tag="mf"):
    """Generative map on the one-request-per-document protocol: the
    document's first prompt generates on the parent request, the
    engine forks the remaining prompts as siblings, and their
    generations return spliced onto the parent's stream with the
    separator token between stages. Generations stop at the
    end-of-sequence token and never contain it, which makes the
    separator unambiguous. Returns token lists keyed (doc, prompt)
    with prompts 1-indexed; the caller detokenizes.

    Admission charges the full worst case up front - the engine may
    fork every sibling the moment the first token streams, so a
    partial reserve would under-count live generations. What it does
    release early: each separator on the stream marks a finished
    stage, and that stage's prompt and cap come off the charge
    immediately instead of waiting for the whole document."""
    return await _run_stream_chain(engine, sampling_params, body_ids,
                                   prompt_ids, budget_tokens, sep_id,
                                   tag, "c|s")


async def run_compose(engine, sampling_params, body_ids, stage_ids,
                      budget_tokens, sep_id, tag="cp"):
    """Composed map: stage k+1 reads stage k's output. One living
    request per document; each stage generates to its natural stop,
    the scheduler absorbs the output into the record (its KV stays,
    nothing rewinds) and appends the next stage's instruction in
    place. The client contract matches the forked map: one stream
    whose segments arrive separated by the registered separator
    token, which never enters the record. Returns token lists keyed
    (doc, stage) with stages 1-indexed; the caller detokenizes.
    Segments may carry a trailing stop token; the caller strips it.
    Engine-level parity against the resend execution (fresh prompt
    per stage) is required before any flight quotes this path."""
    return await _run_stream_chain(engine, sampling_params, body_ids,
                                   stage_ids, budget_tokens, sep_id,
                                   tag, "c|co")


async def _run_stream_chain(engine, sampling_params, body_ids,
                            prompt_ids, budget_tokens, sep_id, tag,
                            ridparts):
    n = len(prompt_ids)
    await _register_query(engine, sampling_params, prompt_ids, set(),
                          None, "", f"{tag}-reg", sep=sep_id)
    gen_budget = getattr(sampling_params, "max_tokens", 0) or 0
    q_cost = sum(len(p) for p in prompt_ids) + n * gen_budget
    used = 0
    cond = asyncio.Condition()
    outs = {}

    async def one_doc(i, cost):
        nonlocal used
        held = cost

        def release(amount):
            nonlocal used, held
            amt = min(amount, held)
            held -= amt
            used -= amt

        try:
            toks = []
            seps_released = 0
            async for out in engine.generate(
                    {"prompt_token_ids": body_ids[i] + prompt_ids[0]},
                    sampling_params, f"de1|{ridparts}|d{i}|{tag}-{i}-0"):
                ids = list(out.outputs[0].token_ids or ())
                if len(ids) > len(toks):
                    toks = ids
                seps = sum(1 for t in toks if t == sep_id)
                if seps > seps_released:
                    async with cond:
                        for k in range(seps_released,
                                       min(seps, n)):
                            release(len(prompt_ids[k]) + gen_budget)
                        seps_released = seps
                        cond.notify_all()
            parts, cur = [], []
            for t in toks:
                if t == sep_id:
                    parts.append(cur)
                    cur = []
                else:
                    cur.append(t)
            parts.append(cur)
            for k, p in enumerate(parts[:n], start=1):
                outs[(i, k)] = p
        finally:
            async with cond:
                release(held)
                cond.notify_all()

    t0 = time.time()
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(one_doc(i, cost)))
    await asyncio.gather(*tasks)
    return dict(wall=time.time() - t0, tokens=outs)


async def run_query(engine, sampling_params, body_ids, q_ids,
                    budget_tokens=None, yes_ids=None, tag="q", no_ids=None,
                    plan=None, class_map=None):
    """The one entry point for executing a query on one worker's engine.

    With `plan` given, the plan chooses the backend. Any object with
    the attributes mode ("chain" | "spec" | "requests"), budget_tokens,
    pin, and stage_token_window works; this module deliberately does
    not import the planner. Chain mode runs the whole filter chain
    inside the engine under the plan's budget; spec mode is the same
    one request per document with the gate ignored for advancement,
    so every stage answers (classifier sets); requests mode runs
    pinned, ranked requests (pins only when the plan says so), and
    decisive no-token ids are forwarded only when the plan's per-stage
    decode window is wider than one token. Sharding across workers happens
    above; each worker receives its shard's body_ids.

    Without `plan`, the shipped default applies. Multi-filter queries
    run in chain mode: one living engine request per document runs the
    whole filter chain, the scheduler judging answers and rewinding
    between filters (yes_ids, the token ids that mean yes, is required
    for the in-engine gate). A single-filter query has nothing to
    chain and runs as pinned, ranked requests under `budget_tokens`."""
    if class_map is not None:
        # classification takes the engine path at any prompt count:
        # the requests path judges answers as binary text, which a
        # multi-class query cannot use
        return await run_filter_chain_engine(
            engine, sampling_params, body_ids, q_ids,
            (plan.budget_tokens if plan is not None else budget_tokens),
            yes_ids, tag=tag, spec=True,
            forked=(getattr(plan, "operator", "") != "pipelined_map"
                    if plan is not None else True),
            class_map=class_map)
    if plan is not None:
        if plan.mode in ("chain", "spec"):
            spec = plan.mode == "spec"
            op = getattr(plan, "operator", "")
            return await run_filter_chain_engine(
                engine, sampling_params, body_ids, q_ids,
                plan.budget_tokens, yes_ids, tag=tag,
                no_ids=(no_ids if not spec
                        and plan.stage_token_window > 1 else None),
                spec=spec,
                # pipelined_map: every prompt, one per round, no forks
                forked=(op != "pipelined_map"),
                spec_after=(getattr(plan, "spec_after_stage", 0)
                            if not spec else 0))
        return await run_filter_chain(
            engine, sampling_params, body_ids, q_ids, plan.budget_tokens,
            tag=tag,
            tags=EngineTags() if plan.pin else None, use_priority=True)
    if len(q_ids) >= 2:
        assert yes_ids, "chain mode needs the yes token ids for its gate"
        return await run_filter_chain_engine(engine, sampling_params,
                                             body_ids, q_ids,
                                             budget_tokens, yes_ids,
                                             tag=tag, no_ids=no_ids)
    return await run_filter_chain(engine, sampling_params, body_ids,
                                  q_ids, budget_tokens,
                                  tag=tag, tags=EngineTags(),
                                  use_priority=True)


async def _register_query(engine, sampling_params, q_ids, yes_ids, no_ids,
                          qpart, suffix, sep=None):
    """Send one query's chain-mode registration (question token lists
    plus the yes/no token ids in the request id) and wait it out.
    `sep` registers a separator token for generative stages (the E
    part): the fork splices sibling generations onto the parent's
    stream with it between stages."""
    reg = [len(q_ids)]
    for q in q_ids:
        reg += [len(q)] + list(q)
    npart = (f"N{','.join(map(str, sorted(no_ids)))}|" if no_ids else "")
    epart = f"E{sep}|" if sep is not None else ""
    reg_rid = (f"de1|reg|{qpart}"
               f"Y{','.join(map(str, sorted(yes_ids)))}|{npart}{epart}"
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
                                  no_ids=None, spec=False, forked=True,
                                  spec_after=0, class_map=None):
    """Chain mode: the engine itself runs each document's whole filter
    chain (register questions once, then one request per document; the
    scheduler judges answers, rewinds, and continues). Returns the same
    result shape as run_filter_chain.

    With no_ids given, the gate runs in decisive-token mode for models
    that do not answer in one token: each stage may sample several
    tokens, the engine stops the stage at the first token in either
    set, and only those tokens count as stage answers.

    With spec=True the chain is speculative: the engine advances
    through every stage regardless of the gate, so the query gets all
    answers - still one request per document, no re-sent ids. Spec
    requires the one-token answer contract; a decisive-token window
    would misalign the stage record. For a model that chatters
    free-running (the 32B tier), constrain the sampler instead:
    SamplingParams(allowed_token_ids=yes|no ids, max_tokens=1) makes
    every stage answer in one token by construction. forked=False
    forces the sequential stage-by-stage path (the fork validation
    control)."""
    import time as _time

    assert not (spec and no_ids), \
        "spec mode requires one-token answers (no decisive-token window)"
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
            if spec:
                sq = "" if forked else "sq|"
                rid = f"de1|c|s|{sq}d{i}|{tag}-{i}-0"
            elif spec_after:
                # the hybrid: gate through stage spec_after, then fork
                # the remaining filters on the survivors
                rid = f"de1|c|sw{spec_after}|d{i}|{tag}-{i}-0"
            else:
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
                # class_map generalizes the binary judgment: the
                # sampled token maps to a class index (-1 = none)
                answers[(i, j + 1)] = (class_map.get(t, -1) if class_map
                                       else (1 if t in yes_ids else 0))
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


async def run_shared_scan(engine, sampling_params, body_ids, queries,
                          budget_tokens, tag="s"):
    """One corpus pass shared by several queries. Each query is a dict
    with q_ids (its question token lists), yes_ids, and optional
    no_ids. Every query's question set registers under its own query
    id; each admitted document then launches one chain request per
    query at the same time. The engine's prefix cache shares the
    document's KV across those chains, and every chain carries the
    same pin directive with the consumer count, so whichever chain
    finishes first pins the document's blocks for the siblings still
    queued; each finishing chain releases one mention, and the blocks
    free only after the last query is done with the document.

    Admission charges each document once: its body plus every query's
    question cost, held until all of its chains finish. Returns the
    total wall, one result dict per query (wall to that query's last
    answer, survivors, answers keyed (doc, stage)), combined counters,
    and the read multiplier ((prompt - cached) tokens over corpus
    tokens; near 1 plus question overhead means each document was
    prefilled once no matter how many queries read it)."""
    import time as _time

    nq = len(queries)
    for k, qy in enumerate(queries):
        await _register_query(engine, sampling_params, qy["q_ids"],
                              qy["yes_ids"], qy.get("no_ids"),
                              f"Q{k}|", f"{tag}-reg{k}")

    tags = EngineTags()
    used = 0
    cond = asyncio.Condition()
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    per = [dict(survivors=[], answers={}, wall=0.0) for _ in queries]
    t0 = _time.time()

    async def one_chain(i, k, started=None):
        qy = queries[k]
        n_k = len(qy["q_ids"])
        rid = tags.rid(f"{tag}-{i}-{k}", doc=i,
                       pin_tokens=len(body_ids[i]), uses=nq,
                       extra=("c", f"Q{k}"))
        try:
            toks = []
            final = None
            async for out in engine.generate(
                    {"prompt_token_ids": body_ids[i] + qy["q_ids"][0]},
                    sampling_params, rid):
                final = out
                if started is not None and not started.is_set():
                    started.set()
                ids = list(out.outputs[0].token_ids or ())
                if ids:
                    toks.append(ids)
            counters["requests"] += 1
            if final is not None:
                counters["prompt_tokens"] += len(final.prompt_token_ids)
                counters["cached_tokens"] += (
                    getattr(final, "num_cached_tokens", 0) or 0)
            stage_toks = _stage_tokens(toks, qy["yes_ids"],
                                       qy.get("no_ids"))
            rec = per[k]
            for j, t in enumerate(stage_toks[:n_k]):
                rec["answers"][(i, j + 1)] = (1 if t in qy["yes_ids"]
                                              else 0)
            if len(stage_toks) >= n_k and all(
                    t in qy["yes_ids"] for t in stage_toks[:n_k]):
                rec["survivors"].append(i)
            rec["wall"] = _time.time() - t0
        finally:
            # a chain that dies before yielding must still unblock the
            # sibling queries waiting on the document's first commit
            if started is not None:
                started.set()
            tags.release(i)

    async def doc_run(i, cost):
        nonlocal used
        try:
            if nq > 1:
                # commit the document's KV once before the sibling
                # queries launch: the prefix cache only dedups against
                # committed blocks, so simultaneous identical prefixes
                # would each prefill the document themselves
                ev = asyncio.Event()
                first = asyncio.create_task(one_chain(i, 0, started=ev))
                await ev.wait()
                await asyncio.gather(
                    first, *[one_chain(i, k) for k in range(1, nq)])
            else:
                await asyncio.gather(
                    *[one_chain(i, k) for k in range(nq)])
        finally:
            async with cond:
                used -= cost
                cond.notify_all()

    total_q_cost = sum(sum(len(q) for q in qy["q_ids"]) + len(qy["q_ids"])
                       for qy in queries)
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + total_q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(doc_run(i, cost)))
    await asyncio.gather(*tasks)
    wall = _time.time() - t0

    # drain the releases the last documents' chains had nothing to
    # ride on, then a release-all as the end-of-run safety net (a
    # no-op when the counts balanced)
    async def drain(rid):
        async for _ in engine.generate(
                {"prompt_token_ids": list(queries[0]["q_ids"][0])},
                sampling_params, rid):
            pass

    fi = 0
    while tags.pending:
        await drain(tags.rid(f"{tag}-fl{fi}"))
        fi += 1
    await drain(f"de1|r*|{tag}-flz")

    corpus = sum(len(b) for b in body_ids)
    mult = ((counters["prompt_tokens"] - counters["cached_tokens"])
            / max(1, corpus))
    return dict(wall=wall,
                queries=[dict(wall=p["wall"],
                              survivors=sorted(p["survivors"]),
                              answers=p["answers"]) for p in per],
                read_multiplier=mult, **counters)
