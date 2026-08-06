"""The string-level entry point: query(engine, docs, filters).

Documents and filters are plain strings. Tokenization, yes-token
discovery, planning, and execution all happen here, so a caller
never touches token ids. The heavy imports (transformers, vllm) load
lazily inside the call, so the package stays importable without an
engine installed.
"""

import time

from .configs import DEVICES, MODELS
from .plan import plan_query
from .runtime.engine_client import run_map, run_map_forked, run_query

_AUTO = object()

_DEPLOYMENT = {"store": None}


def configure(store=None):
    """Deployment facts, set once by the operator, never per query: a
    StoreSpec when this installation has a persisted-KV store
    (measured read bandwidth; warm=True when the corpus KV is
    already saved). Queries pick it up automatically."""
    _DEPLOYMENT["store"] = store


def plan_engine_kwargs(plan, model_name, store=None):
    """The engine boot arguments a plan requires, as a plain dict for
    AsyncEngineArgs. This is the integration point for everything the
    plan can only enforce at boot: the derived sequence cap and its
    round-budget floor, the plan-owned scheduler, and the KV store
    connector when the plan's access is restore or spill (vLLM's
    tiering offload connector: RAM primary, disk secondary - the
    persist milestone's mechanism). Store access boots the stock
    scheduler exactly as the persist milestone banked it, so a
    store plan never carries forks - the planner zeroes the switch
    stage to match, and reconciling the store with plan-owned
    memory is the open follow-up.

    Note the free persist tier this function does not manage: on a
    live engine, the prefix cache already keeps document KV in GPU
    memory between queries, and on the 4B tier that is the only
    persist that beats recompute."""
    from vllm.config import KVTransferConfig

    # max_model_len is left to the model's own limit: the planner
    # already refuses any document the pool cannot hold, and a cap
    # derived from one corpus would refuse a later query's longer
    # documents on a reused engine. The per-length cost is block-table
    # metadata only, ~0.1 percent of memory at the worst sequence cap.
    # The step budget is plan-derived (Plan.engine_step_tokens): the
    # largest budget whose activation reservation stays a rounding
    # error against the KV pool. ROUND_TOKENS (the switch rule's
    # round size) is still the 2,048 measured under the old fixed
    # floor; it errs conservative until the step recorder re-anchors
    # it under derived budgets.
    kwargs = dict(
        model=model_name,
        max_num_seqs=plan.engine_max_seqs or None,
        max_num_batched_tokens=plan.engine_step_tokens or None)
    if getattr(plan, "access", "read") in ("restore", "spill"):
        # store access runs on the stock scheduler, exactly as the
        # persist milestone banked it: the tiering connector manages
        # the KV lifecycle, and reconciling it with plan-owned
        # memory (and therefore with forks) is open. The planner
        # already zeroed the plan's switch stage to match.
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config=dict(
                spec_name="TieringOffloadingSpec"))
    else:
        kwargs["scheduler_cls"] = ("docengine.engineext.scheduler."
                                   "DocEngineScheduler")
        if getattr(plan, "operator", "") in ("hybrid_filter",
                                             "hybrid_map"):
            kwargs["kv_transfer_config"] = KVTransferConfig(
                kv_connector="DocEngineForkConnector",
                kv_connector_module_path="docengine.engineext."
                                         "forkconnector",
                kv_role="kv_both")
    return {k: v for k, v in kwargs.items() if v is not None}

_YES_WORDS = ("YES", " YES", "Yes", " Yes", "Y", " Y")
_NO_WORDS = ("NO", " NO", "No", " No", "N", " N")


def _first_ids(tokenizer, words):
    ids = set()
    for w in words:
        got = tokenizer(w, add_special_tokens=False)["input_ids"]
        if got:
            ids.add(got[0])
    return ids


async def query(engine, docs, filters, est_selectivities=None,
                gated=True, policy=None, model="Qwen3-4B-FP8",
                device="H100-SXM-80GB", gpus=1, keep_order=False,
                tokenizer=None, sampling_params=_AUTO):
    """Run the filters, in order, over the documents.

    docs is a list of document strings. filters is a list of yes or
    no question strings, in gating order; each is appended verbatim
    after the document, so it must instruct the model to answer YES
    or NO. est_selectivities is the estimated fraction of documents
    that pass each filter, one number per filter; it only shapes the
    plan, never the answers (missing means 1.0 each; the switch rule
    uses the per-filter values, so a skewed chain switches where its
    selective filter actually sits). gated=True is a filter: a document that
    fails one prompt skips the rest. gated=False is a map: binary
    classification, every prompt on every document. The planner
    picks among four operators (pipelined_filter, hybrid_filter,
    pipelined_map, hybrid_map) with one underfill rule; policy
    forces "pipelined" or "hybrid" instead.

    tokenizer and sampling_params exist for tests; left alone, the
    model's tokenizer loads from transformers and the one-token
    greedy sampling parameters are built from vllm.

    Returns dict(answers={(doc_index, filter_index): 0 or 1} with
    filter_index starting at 1, survivors=[doc indexes that passed
    every filter], plan=the Plan, wall=seconds). Sharding across
    engines happens above this call: one engine, one call."""
    mc, dc = MODELS[model], DEVICES[device]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(mc.name)
    body_ids = tokenizer(list(docs), add_special_tokens=False)["input_ids"]
    q_ids = [tokenizer(f, add_special_tokens=False)["input_ids"]
             for f in filters]
    yes_ids = _first_ids(tokenizer, _YES_WORDS)
    no_ids = _first_ids(tokenizer, _NO_WORDS)

    # gated filters commute: survivors are identical under any order,
    # only the work changes, so execute cheapest-rejection-first
    # ((1 - selectivity) over prompt cost, descending). Answers are
    # reported under the caller's original filter indices. keep_order
    # opts out for prompts that must run as written. Maps skip this:
    # every prompt runs regardless, so order cannot change the work.
    order = list(range(len(filters)))
    if gated and est_selectivities and not keep_order:
        order = sorted(range(len(filters)),
                       key=lambda j: -(1 - est_selectivities[j])
                       / max(1, len(q_ids[j])))
        q_ids = [q_ids[j] for j in order]
        est_selectivities = [est_selectivities[j] for j in order]

    sel = (list(est_selectivities) if est_selectivities else 1.0)
    plan = plan_query(len(filters), [len(b) for b in body_ids], mc, dc,
                      gpus=gpus, selectivity=sel, gated=gated,
                      policy=policy, store=_DEPLOYMENT["store"])
    if not hasattr(plan, "mode"):          # a Refusal
        return dict(refusal=plan)

    if sampling_params is _AUTO:
        from vllm import SamplingParams
        allowed = (sorted(yes_ids | no_ids)
                   if plan.mode == "spec" else None)
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=plan.stage_token_window,
            allowed_token_ids=allowed)

    t0 = time.time()
    res = await run_query(engine, sampling_params, body_ids, q_ids,
                          budget_tokens=plan.budget_tokens,
                          yes_ids=yes_ids, no_ids=no_ids, plan=plan)
    # report answers under the caller's filter indices, whatever
    # order execution used
    answers = {(i, order[j - 1] + 1): v
               for (i, j), v in res["answers"].items()}
    return dict(answers=answers, survivors=res["survivors"],
                plan=plan, wall=time.time() - t0,
                filter_order=tuple(order))


async def classify(engine, docs, prompts, classes,
                   model="Qwen3-4B-FP8", device="H100-SXM-80GB",
                   gpus=1, tokenizer=None, sampling_params=_AUTO):
    """Multi-class, single-token classification: every prompt on
    every document, each answer exactly one of `classes`.

    classes is a list of label strings; each prompt must instruct
    the model to answer with one of them. The sampler is constrained
    to the classes' first tokens, so every answer is one token read
    from the prompt's own forward pass - no decode steps, the same
    execution tier as filters, with the same operators
    (pipelined_map or hybrid_map by corpus size). Two classes whose
    labels share a first token are ambiguous and refused.

    Returns dict(answers={(doc_index, prompt_index): class label},
    plan, wall). prompt_index starts at 1."""
    mc, dc = MODELS[model], DEVICES[device]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(mc.name)
    body_ids = tokenizer(list(docs), add_special_tokens=False)["input_ids"]
    q_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
             for p in prompts]
    class_ids = []
    for c in classes:
        got = tokenizer(c, add_special_tokens=False)["input_ids"]
        assert got, f"class {c!r} tokenizes to nothing"
        class_ids.append(got[0])
    assert len(set(class_ids)) == len(class_ids), \
        "two classes share a first token; pick distinguishable labels"
    class_map = {t: k for k, t in enumerate(class_ids)}

    plan = plan_query(len(prompts), [len(b) for b in body_ids], mc, dc,
                      gpus=gpus, gated=False,
                      store=_DEPLOYMENT["store"])
    if not hasattr(plan, "mode"):
        return dict(refusal=plan)
    if sampling_params is _AUTO:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=plan.stage_token_window,
            allowed_token_ids=sorted(class_ids))
    t0 = time.time()
    res = await run_query(engine, sampling_params, body_ids, q_ids,
                          yes_ids=set(class_ids), plan=plan,
                          class_map=class_map)
    answers = {k: (classes[v] if v >= 0 else None)
               for k, v in res["answers"].items()}
    return dict(answers=answers, plan=plan, wall=time.time() - t0)


async def map(engine, docs, prompts, max_output_tokens=256,
              model="Qwen3-4B-FP8", device="H100-SXM-80GB", gpus=1,
              tokenizer=None, sampling_params=_AUTO):
    """Open-ended generation: every prompt on every document,
    returning the generated text per pair. This is the generative
    tier - decode exists here, unlike filters and classify.

    Execution follows the plan's operator. hybrid_map (small corpus):
    one request per document - the first prompt generates on the
    parent, the engine forks the rest, and their generations return
    on the parent's stream with the end-of-sequence token as the
    stage separator. pipelined_map (large corpus): one request per
    (document, prompt), the document read once and reused through
    the cache, the first prompt committing the KV before the rest
    launch. The plan prices generation at the saturated decode rate
    (a banked constant), assuming every prompt hits the output cap.

    Returns dict(texts={(doc_index, prompt_index): generated string},
    plan, wall). prompt_index starts at 1."""
    mc, dc = MODELS[model], DEVICES[device]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(mc.name)
    body_ids = tokenizer(list(docs), add_special_tokens=False)["input_ids"]
    p_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
             for p in prompts]
    plan = plan_query(len(prompts), [len(b) for b in body_ids], mc, dc,
                      gpus=gpus, gated=False, gen_tokens=max_output_tokens,
                      store=_DEPLOYMENT["store"])
    if not hasattr(plan, "mode"):
        return dict(refusal=plan)
    if sampling_params is _AUTO:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_output_tokens)
    t0 = time.time()
    if plan.operator == "hybrid_map":
        sep = getattr(tokenizer, "eos_token_id", None)
        assert sep is not None, \
            "the tokenizer must define eos_token_id: it separates " \
            "sibling generations on the parent's stream"
        res = await run_map_forked(engine, sampling_params, body_ids,
                                   p_ids, plan.budget_tokens, sep)
        texts = {k: tokenizer.decode(v) for k, v in res["tokens"].items()}
    else:
        res = await run_map(engine, sampling_params, body_ids, p_ids,
                            plan.budget_tokens)
        texts = res["texts"]
    return dict(texts=texts, plan=plan, wall=time.time() - t0)
