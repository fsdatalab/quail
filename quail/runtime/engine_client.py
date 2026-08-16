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

The client is a plain synchronous loop: admit documents while the
budget allows, run one engine step, route the step's outputs, repeat.
There is no asyncio anywhere in it. Concurrency across documents
lives where it already had to live - inside the engine's scheduler -
and a filter query is a batch operator, so the client has nobody else
to serve while it waits on a step.

The engine object must expose the vLLM v1 LLMEngine interface:
`add_request(request_id, prompt, sampling_params, priority=0)` and
`step()` returning outputs with `request_id`, `finished`,
`prompt_token_ids`, `num_cached_tokens`, and `outputs[0]`.
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


def _no_store_params(sampling_params):
    """A copy of the sampling params that tells the offload connector
    to store nothing for this request (max_offload_tokens=0 rides in
    extra_args["kv_transfer_params"]). Documents under a capped
    store's length threshold carry this so they never occupy capacity
    the plan reserved for longer ones."""
    import copy
    sp = (sampling_params.clone()
          if hasattr(sampling_params, "clone")
          else copy.copy(sampling_params))
    extra = dict(getattr(sp, "extra_args", None) or {})
    kv = dict(extra.get("kv_transfer_params") or {})
    kv["max_offload_tokens"] = 0
    extra["kv_transfer_params"] = kv
    sp.extra_args = extra
    return sp


def _run_to_completion(engine, request_id, max_seconds=30):
    """Step the engine until one specific request finishes, or abort
    it at the deadline. For requests submitted outside the measured
    loops (registration, the pin flush)."""
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        for out in engine.step():
            if out.request_id == request_id and out.finished:
                return True
    try:
        engine.abort_request([request_id])
    except Exception:
        pass
    return False


def run_filter_chain(engine, sampling_params, body_ids, q_ids,
                     budget_tokens, tag="q", tags=None,
                     use_priority=False, store_min_tokens=0):
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
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    answers = {}
    survivors = []

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

    inflight = {}                      # request id -> (doc, stage, cost)
    no_store = (_no_store_params(sampling_params)
                if store_min_tokens > 1 and sampling_params is not None
                else sampling_params)

    def submit(i, j, cost):
        rid = rid_for(i, j)
        inflight[rid] = (i, j, cost)
        kw = {"priority": 0 if j > 0 else 1} if use_priority else {}
        sp = (no_store if len(body_ids[i]) < store_min_tokens
              else sampling_params)
        engine.add_request(rid, {"prompt_token_ids": body_ids[i] + q_ids[j]},
                           sp, **kw)

    def retire(i, cost):
        if tags is not None:
            tags.release(i)
        return cost

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
                used -= retire(i, cost)
    wall = time.time() - t0
    if tags is not None:
        # release every remaining pin so the engine can reset cleanly
        flush_rid = f"de1|r*|{tag}-0-99"
        engine.add_request(flush_rid, {"prompt_token_ids": list(q_ids[0])},
                           sampling_params)
        _run_to_completion(engine, flush_rid)
    return dict(wall=wall, survivors=sorted(survivors), answers=answers,
                **counters)


def _register_query(engine, sampling_params, q_ids, yes_ids, no_ids,
                    qpart, suffix):
    """Send one query's chain-mode registration (question token lists
    plus the yes/no token ids in the request id) and run it out."""
    reg = [len(q_ids)]
    for q in q_ids:
        reg += [len(q)] + list(q)
    npart = (f"N{','.join(map(str, sorted(no_ids)))}|" if no_ids else "")
    reg_rid = (f"de1|reg|{qpart}"
               f"Y{','.join(map(str, sorted(yes_ids)))}|{npart}"
               f"{suffix}")
    engine.add_request(reg_rid, {"prompt_token_ids": reg}, sampling_params)
    _run_to_completion(engine, reg_rid)


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


def run_filter_chain_engine(engine, sampling_params, body_ids, q_ids,
                            yes_ids, tag="c", no_ids=None,
                            store_min_tokens=0):
    """Chain mode: the engine itself runs each document's whole filter
    chain. The client registers the question token lists and the gate
    token ids once, then submits ONE request per document - all of
    them, up front. Waiting requests hold no KV (measured: residency
    self-limits at sigma*B), so there is nothing for a client gate to
    protect; the plan's admission number is a boot-time feasibility
    floor, not a submission valve. The scheduler judges each answer,
    rewinds to the document boundary, and appends the next question.
    Returns the same result shape as run_filter_chain, which keeps
    its gate because the stock baseline's stages are conditional on
    verdicts and its in-flight requests do pin cache footprint.

    With no_ids given, the gate runs in decisive-token mode for models
    that do not answer in one token: each stage may sample several
    tokens, the engine stops the stage at the first token in either
    set, and only those tokens count as stage answers. The simpler
    contract is a constrained sampler
    (SamplingParams(allowed_token_ids=yes|no ids, max_tokens=1)),
    which makes every stage answer in one token by construction."""
    n = len(q_ids)
    _register_query(engine, sampling_params, q_ids, yes_ids, no_ids,
                    "", f"{tag}-reg")

    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    answers = {}
    survivors = []
    raw0 = []
    inflight = {}                    # request id -> (doc, snapshots)
    no_store = (_no_store_params(sampling_params)
                if store_min_tokens > 1 and sampling_params is not None
                else sampling_params)

    t0 = time.time()
    for i in range(len(body_ids)):
        rid = f"de1|c|d{i}|{tag}-{i}-0"
        inflight[rid] = (i, [])
        sp = (no_store if len(body_ids[i]) < store_min_tokens
              else sampling_params)
        engine.add_request(rid,
                           {"prompt_token_ids": body_ids[i] + q_ids[0]},
                           sp)
    while inflight:
        for out in engine.step():
            entry = inflight.get(out.request_id)
            if entry is None:
                continue
            i, toks = entry
            ids = list(out.outputs[0].token_ids or ())
            if ids:
                toks.append(ids)
            if not out.finished:
                continue
            del inflight[out.request_id]
            counters["requests"] += 1
            counters["prompt_tokens"] += len(out.prompt_token_ids)
            counters["cached_tokens"] += (
                getattr(out, "num_cached_tokens", 0) or 0)
            stage_toks = _stage_tokens(toks, yes_ids, no_ids)
            if i == 0:
                raw0.extend(tuple(s) for s in toks[:8])
            for j, t in enumerate(stage_toks[:n]):
                answers[(i, j + 1)] = 1 if t in yes_ids else 0
            if len(stage_toks) >= n \
                    and all(t in yes_ids for t in stage_toks[:n]):
                survivors.append(i)
    wall = time.time() - t0
    return dict(wall=wall, survivors=sorted(survivors), answers=answers,
                doc0_raw=raw0, **counters)
