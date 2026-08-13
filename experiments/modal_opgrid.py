"""Flight A of the operator grid (experiments/OPGRID.md): the filter
operators under mixed selectivities, on the CUDA 13 image, with the
step trace recording every engine step.

One container, one corpus (10,000 IMDB documents with a [FLAGS] line,
so every filter's pass rate is exact by construction), five filters,
three selectivity profiles:

  permissive       0.9 0.9 0.9 0.8 0.8   survivors stay high
  selective_early  0.2 0.5 0.7 0.9 0.9   survivors collapse at stage 1
  cliff            0.9 0.9 0.1 0.9 0.9   the cliff sits mid-chain

Executors per profile, interleaved rep-major so host drift hits
every cell equally: pipelined_filter (gated, one question per stage),
hybrid_filter (gated to the switch stage, then fork; the rule's stage,
forced to 1 where the rule says never), ask_everything (classifier
speculation on the same corpus - prices what gating saves), and
naive_vllm - the true baseline: a stock engine at the same caps, no
custom scheduler, no connector, no admission budget, no pins, plain
generate() per (document, stage), gated client-side.

Engine boots in one container: "new" uses the plan's derived
arguments (step budget from the activation rule, sequence cap from
the admitted worst case, bounded); "old" re-creates the old flight
recipe (500 x n_filters + 64 for both caps; the permissive-profile
rider, on request); "stock" is vLLM at the new boot's caps with none
of our machinery, running naive_vllm only. Parallel profile jobs:
pass `profile` to run one profile per container, so every
operator-and-boot comparison stays within its container.

Stated predictions, before the run:
  P1  pipelined lands within a few seconds of the read floor
      (fresh-read tokens / 97,889); the gap decomposes into c0 and
      ramp/drain, visible in the trace.
  P2  hybrid >= pipelined on selective_early and cliff (the rule
      fires mid-chain); ties or loses within noise on permissive
      (forced fork of fat survivor sets).
  P3  requests >= pipelined everywhere; the gap is the per-request
      tax.
  P4  ask_everything loses to pipelined in proportion to the tails
      gating skips.
  P5  boot new is within 2 percent of boot old, never worse beyond
      noise (steps are compute-bound past ~105 tokens; the budget
      only amortizes per-step costs).
  P6  every filter cell's trace shows zero decode sequences (a
      one-token answer samples from its own last prefill chunk), and
      zero heuristic evictions (the strict invariant).

Banks results/engine/opgrid_filters.json plus one gzipped step trace
per boot (opgrid_trace_new.jsonl.gz, opgrid_trace_old.jsonl.gz);
plot with plots/make_plot_steps.py, slicing on the banked per-cell
monotonic time bounds.

Run with:
  modal run experiments/modal_opgrid.py
"""

import gzip
import json
import os
import sys

import modal

# run from the repo root (so the docengine package resolves for
# mounting); modal_scale lives next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_scale import (FLAG_SEED, IMAGE_STAMP, MODEL,  # noqa: E402
                         MODEL32, _build_pool, _flags_line, _question,
                         hf_cache, image)

app = modal.App("docengine-opgrid")
opgrid_image = image.add_local_python_source("modal_scale")
# results also persist to a volume, so a dropped client connection
# (which killed this flight's second attempt mid-run) loses nothing:
#   modal volume get docengine-results opgrid_filters.json
results_vol = modal.Volume.from_name("docengine-results",
                                     create_if_missing=True)

PROFILES = dict(
    permissive=(0.9, 0.9, 0.9, 0.8, 0.8),
    selective_early=(0.2, 0.5, 0.7, 0.9, 0.9),
    cliff=(0.9, 0.9, 0.1, 0.9, 0.9),
)
N_FILTERS = 5


@app.function(image=opgrid_image, gpu="H100!", timeout=14400,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
async def opgrid_filters(n_docs: int = 10000, reps: int = 3,
                         model_key: str = "4b", profile: str = "",
                         old_rider: bool = False,
                         stock_only: bool = False,
                         classifier: bool = False) -> dict:
    import inspect
    import os
    import time as _time

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM as Engine

    from docengine.configs import DEVICES, MODELS
    from docengine.engineext.chainlogic import hybrid_switch_stage
    from docengine.plan import plan_query
    from docengine.plan.cost import ROUND_TOKENS
    from docengine.runtime.engine_client import run_filter_chain_engine

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    os.environ["DOCENGINE_STEPSTATS"] = "1"

    model_name = MODEL if model_key == "4b" else MODEL32
    cfg_name = "Qwen3-4B-FP8" if model_key == "4b" else "Qwen3-32B-FP8"
    tok = AutoTokenizer.from_pretrained(model_name)
    yes_ids, no_ids = set(), set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            yes_ids.add(ids[0])
    for w in ("NO", " NO", "No", " No", "N", " N"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            no_ids.add(ids[0])

    docs = _build_pool(n_docs)
    corpora = {}
    for k, (name, sels) in enumerate(PROFILES.items()):
        rng = np.random.default_rng(FLAG_SEED + 100 + k)
        flags = (rng.random((n_docs, N_FILTERS))
                 < np.array(sels)[None, :]).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        corpora[name] = (
            tok(bodies, add_special_tokens=False)["input_ids"],
            [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(N_FILTERS)],
            flags)

    # the plan's boot arguments, dogfooded from the planner itself;
    # the hybrid plan carries the largest sequence cap, so it sizes
    # the boot that must serve every cell
    doc_tokens = [len(b) for b in corpora["permissive"][0]]
    plan = plan_query(N_FILTERS, doc_tokens, MODELS[cfg_name],
                      DEVICES["H100-SXM-80GB"],
                      selectivity=list(PROFILES["selective_early"]),
                      policy="hybrid")
    budget = plan.budget_tokens
    # The plan's raw sequence cap (admitted worst case x filters)
    # OOM'd this flight's first boot: per-sequence engine overheads
    # (FlashInfer workspace, sampler buffers) scale with the cap and
    # live outside both the plan's and the engine's pool accounting -
    # the KV allocation came up 1.7 GiB short with ~6 GB unaccounted.
    # Cap at 4,096 (forks past the cap queue; correctness unaffected)
    # and pin max_model_len on both boots so the A/B isolates the
    # step budget. The planner's cap arithmetic needs this bound.
    boots = dict(
        new=dict(max_num_seqs=min(plan.engine_max_seqs, 4096),
                 max_num_batched_tokens=plan.engine_step_tokens,
                 max_model_len=4608),
        old=dict(max_num_seqs=500 * N_FILTERS + 64,
                 max_num_batched_tokens=max(2048, 500 * N_FILTERS + 64),
                 max_model_len=4608))

    # Constrain the sampler to the yes/no ids on BOTH tiers. The 32B
    # always needed it (free-running restates the flags line). The 4B
    # needs it for the baseline's sake: stock scores its one answer
    # token as text, so a "Y" token scores as NO and the cliff and
    # classifier stock cells read 25-55 percent wrong while our
    # id-judged cells read 0.3 percent on the same corpus. With the
    # constraint every executor emits only yes/no ids and the text
    # parse cannot diverge from the id judgment.
    kw = dict(allowed_token_ids=sorted(yes_ids | no_ids))
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        skip_clone=True, **kw)
    report = dict(n_docs=n_docs, n_filters=N_FILTERS, reps=reps,
                  profiles={k: list(v) for k, v in PROFILES.items()},
                  budget_tokens=budget, boots=boots,
                  plan_engine_max_seqs=plan.engine_max_seqs,
                  plan_engine_step_tokens=plan.engine_step_tokens,
                  model=model_name, image=IMAGE_STAMP, cells=[],
                  traces={})

    async def run_naive(engine, body_ids, q_ids, cap, tag,
                        gate=True):
        """The naive baseline: plain generate() per (doc, stage),
        gated client-side, no token budget, no pins, no priority.
        With gate=False this is the stock classifier baseline: every
        question on every document, nothing skipped. Outstanding
        documents are bounded by a semaphore at the engine's own
        sequence cap: the unbounded version (10,000 open requests)
        killed the stock engine core outright (EngineDeadError, the
        4b_stock job) - that crash is the banked unbounded result,
        and any real client bounds its concurrency."""
        import asyncio as _aio
        sem = _aio.Semaphore(cap)
        counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
        answers, survivors = {}, []

        async def ask(ids, rid):
            final = None
            async for out in engine.generate({"prompt_token_ids": ids},
                                             sp, rid):
                final = out
            counters["requests"] += 1
            counters["prompt_tokens"] += len(final.prompt_token_ids)
            counters["cached_tokens"] += (
                getattr(final, "num_cached_tokens", 0) or 0)
            t = final.outputs[0].text.upper()
            iy, ino = t.find("YES"), t.find("NO")
            return 1 if iy >= 0 and (ino < 0 or iy < ino) else 0

        async def chain(i):
            if not gate:
                # the COMPETENT classifier client: a document's
                # questions are independent, so co-submit them (the
                # maps access pattern, measured reads 1.36) instead
                # of one at a time (reads 4.9-5.7: the pool churns
                # between a document's questions - the sequential-
                # client baselines banked before the client audit).
                # The semaphore bounds requests, not documents, so
                # concurrency matches the engine cap like naive_map.
                async def one(j):
                    async with sem:
                        return await ask(body_ids[i] + q_ids[j],
                                         f"{tag}-{i}-{j}")
                got = await _aio.gather(*[one(j)
                                          for j in range(len(q_ids))])
                for j, g in enumerate(got):
                    answers[(i, j + 1)] = g
                if all(got):
                    survivors.append(i)
                return
            async with sem:
                for j in range(len(q_ids)):
                    got = await ask(body_ids[i] + q_ids[j],
                                    f"{tag}-{i}-{j}")
                    answers[(i, j + 1)] = got
                    if not got and gate:
                        return
                if all(answers[(i, j + 1)]
                       for j in range(len(q_ids))):
                    survivors.append(i)

        t0 = _time.time()
        await _aio.gather(*(chain(i) for i in range(len(body_ids))))
        return dict(wall=_time.time() - t0, answers=answers,
                    survivors=sorted(survivors), **counters)

    async def run_cell(engine, boot, prof, op, rep, seq_cap):
        body_ids, q_ids, flags = corpora[prof]
        sels = PROFILES[prof]
        tag = f"{boot}-{prof[:4]}-{op[:4]}-{rep}"
        switch = 0
        t0m = _time.monotonic()
        if op == "naive_classifier_cc":
            r = await run_naive(engine, body_ids, q_ids, seq_cap, tag,
                                gate=False)
        elif op == "naive_vllm":
            r = await run_naive(engine, body_ids, q_ids, seq_cap, tag)
        else:
            spec = op == "ask_everything"
            if op == "hybrid_filter":
                switch = hybrid_switch_stage(
                    n_docs, list(sels), 13, ROUND_TOKENS) or 1
            r = await run_filter_chain_engine(
                engine, sp, body_ids, q_ids, budget, yes_ids, tag=tag,
                spec=spec, spec_after=switch)
        t1m = _time.monotonic()
        wrong = sum(1 for (i, j), v in r["answers"].items()
                    if v != int(flags[i][j - 1]))
        reads = round((r.get("prompt_tokens", 0)
                       - r.get("cached_tokens", 0))
                      / max(1, sum(len(b) for b in body_ids)), 3)
        cell = dict(boot=boot, profile=prof, operator=op, rep=rep,
                    wall=round(r["wall"], 3), requests=r["requests"],
                    reads=reads, wrong=wrong,
                    answered=len(r["answers"]),
                    survivors=len(r["survivors"]),
                    switch_stage=switch,
                    t0m=round(t0m, 3), t1m=round(t1m, 3))
        report["cells"].append(cell)
        print(f"[opgrid] {cell}", flush=True)

    planned_ops = ("pipelined_filter", "hybrid_filter",
                   "ask_everything")
    sel_profiles = [profile] if profile else list(PROFILES)
    boot_plan = []       # (boot name, engine kwargs, planned?)
    if not stock_only:
        boot_plan.append(("new", boots["new"], True))
        if old_rider and "permissive" in sel_profiles:
            boot_plan.append(("old", boots["old"], True))
    boot_plan.append(("stock", boots["new"], False))

    for boot, kw, planned in boot_plan:
        # a failed boot (later ones re-boot in the same process)
        # banks what ran instead of losing the container's cells
        try:
            trace_path = f"/tmp/opgrid_trace_{boot}.jsonl"
            os.environ["DOCENGINE_STEPTRACE"] = trace_path
            extra = (dict(
                scheduling_policy="priority",
                kv_transfer_config=KVTransferConfig(
                    kv_connector="DocEngineForkConnector",
                    kv_connector_module_path=(
                        "docengine.engineext.forkconnector"),
                    kv_role="kv_both"),
                scheduler_cls=("docengine.engineext.scheduler."
                               "DocEngineScheduler"))
                if planned else {})
            # Stock boots at 0.88: its flash_attn workspace OOMed at
            # 0.92 (4.22 GiB tried, 2.79 free), and halving the step
            # budget instead is self-defeating - the boot profiler
            # hands the saved activation room straight to the KV
            # pool (measured: free VRAM fell to 138 MiB). The wrong
            # answers first blamed on 0.88 were the baseline's text
            # parser scoring a "Y" token as NO; the sampler
            # constraint above removes that failure mode for every
            # executor.
            engine = Engine.from_engine_args(AsyncEngineArgs(
                model=model_name, kv_cache_dtype="fp8",
                gpu_memory_utilization=0.92 if planned else 0.88,
                enable_prefix_caching=True,
                disable_log_stats=True, **extra, **kw))

            async def reset():
                res = engine.reset_prefix_cache()
                if inspect.isawaitable(res):
                    await res

            profs = (sel_profiles if boot != "old" else ["permissive"])
            # the old-boot rider answers one question - the step
            # budget's effect on the floor-bound executor - so it
            # runs pipelined only
            ops = (("pipelined_filter",) if boot == "old"
                   else planned_ops if planned
                   else ("naive_classifier_cc",) if classifier
                   else ("naive_vllm",))
            # For stock: set the semaphore from the KV pool capacity,
            # not max_num_seqs. This avoids overflowing the pool and
            # causing artificial prefix-cache eviction.
            if planned:
                sem_cap = kw["max_num_seqs"]
            else:
                # Set the semaphore from KV pool capacity so that
                # in-flight requests fit without eviction. Use the
                # permissive profile's documents (the largest corpus).
                _body_ids = corpora["permissive"][0]
                _q_ids = corpora["permissive"][1]
                mean_req = (int(sum(len(b) for b in _body_ids)
                                / len(_body_ids))
                            + max(len(q) for q in _q_ids))
                sem_cap = max(256, budget // mean_req)
                print(f"[opgrid] stock semaphore: budget {budget:,} / "
                      f"{mean_req} tok/req = {sem_cap}", flush=True)

            for rep in range(reps):
                for prof in profs:
                    for op in ops:
                        await reset()
                        try:
                            await run_cell(engine, boot, prof, op,
                                           rep, sem_cap)
                        except Exception as e:
                            # a dead cell banks its error WITH the
                            # traceback; the EngineDeadError string
                            # alone was undiagnosable last flight
                            import traceback as _tb
                            report["cells"].append(dict(
                                boot=boot, profile=prof, operator=op,
                                rep=rep,
                                error=f"{type(e).__name__}: {e}",
                                traceback=_tb.format_exc()[-4000:]))
                            print(f"[opgrid] cell {boot}/{prof}/{op}/"
                                  f"{rep} failed: {e}", flush=True)
                            _tb.print_exc()
            try:
                engine.shutdown()
            except Exception as e:
                print(f"[opgrid] shutdown: {e}", flush=True)
            if os.path.exists(trace_path):
                # the stock boot leaves no trace: only our scheduler
                # records steps
                with open(trace_path, "rb") as f:
                    report["traces"][boot] = gzip.compress(f.read())
        except Exception as e:
            import traceback as _tb
            report.setdefault("boot_errors", {})[boot] = (
                f"{type(e).__name__}: {e}")
            report.setdefault("boot_tracebacks", {})[boot] = (
                _tb.format_exc()[-4000:])
            print(f"[opgrid] boot {boot} failed: {e}", flush=True)
            _tb.print_exc()

    # persist to the volume before returning: the return value dies
    # with a dropped client, the volume does not
    sfx = (f"_{model_key}" + (f"_{profile}" if profile else "")
           + ("_classifier_cc" if classifier else "")
           + ("_stock" if stock_only else ""))
    traces = report.pop("traces")
    with open(f"/results/opgrid_filters{sfx}.json", "w") as f:
        json.dump(report, f)
    for boot, blob in traces.items():
        with open(f"/results/opgrid_trace_{boot}{sfx}.jsonl.gz",
                  "wb") as f:
            f.write(blob)
    results_vol.commit()
    print(f"[opgrid] banked opgrid_filters{sfx}.json to volume "
          "docengine-results", flush=True)
    report["traces"] = traces
    return report


MAP_PROMPTS = (
    "\n\nInstruction: give a one-sentence summary of the review "
    "above.\nSummary:",
    "\n\nInstruction: in one sentence, state the reviewer's overall "
    "sentiment and why.\nAnswer:",
    "\n\nInstruction: name the movie or show being reviewed, if "
    "stated; otherwise say unknown.\nAnswer:",
    "\n\nInstruction: guess the genre of the movie in at most five "
    "words.\nGenre:",
    "\n\nInstruction: quote the single phrase from the review that "
    "best captures its tone.\nQuote:",
)


@app.function(image=opgrid_image, gpu="H100!", timeout=14400,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
async def opgrid_maps(n_docs: int = 10000, model_key: str = "4b",
                      caps: str = "16,64,256",
                      boots: str = "new,stock",
                      profile: bool = False,
                      trace: bool = True) -> dict:
    """Flight B: open-ended maps. Every prompt on every document,
    free decode up to the cap, one repetition per cell. Two
    executors: pipelined_map (run_map: per-pair requests, the first
    prompt commits the document's KV before the rest launch,
    admission charges body + prompts + the full generation budget)
    and stock naive (plain generate per pair on a stock engine at
    the same caps, outstanding requests bounded at the sequence
    cap, no admission, no commit ordering - simultaneous identical
    prefixes race and recompute, the banked prefill race).

    Predictions, stated before the run: decode exists now - the
    trace's decode band is nonzero and grows with the cap; the
    pipelined reads multiplier stays near 1 while naive's grows with
    the prefill race; the planner's gen_tokens pricing is compared
    against every pipelined wall; zero heuristic evictions even at
    cap 256 (admission counts decode tokens)."""
    import inspect
    import os
    import time as _time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM as Engine

    from docengine.configs import DEVICES, MODELS
    from docengine.plan import plan_query
    from docengine.runtime.engine_client import run_map, run_map_rewind

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    if trace:
        os.environ["DOCENGINE_STEPSTATS"] = "1"
    if profile:
        # core-process CPU profile; the scheduler dumps a ranked
        # table at step 900 (walls under profiling are NOT quotable)
        os.environ["DOCENGINE_PROFILE"] = "1"

    caps = [int(c) for c in str(caps).split(",") if str(c).strip()]
    boot_names = [b.strip() for b in boots.split(",") if b.strip()]
    model_name = MODEL if model_key == "4b" else MODEL32
    cfg_name = "Qwen3-4B-FP8" if model_key == "4b" else "Qwen3-32B-FP8"
    tok = AutoTokenizer.from_pretrained(model_name)
    docs = _build_pool(n_docs)
    body_ids = tok(docs, add_special_tokens=False)["input_ids"]
    p_ids = [tok(p, add_special_tokens=False)["input_ids"]
             for p in MAP_PROMPTS]
    doc_tokens = [len(b) for b in body_ids]
    corpus = sum(doc_tokens)

    plans = {cap: plan_query(len(MAP_PROMPTS), doc_tokens,
                             MODELS[cfg_name],
                             DEVICES["H100-SXM-80GB"], gated=False,
                             gen_tokens=cap) for cap in caps}
    boot_plan = plans[max(caps)]
    # 4,096 is settled, not provisional: overlapped scheduling
    # halves effective decode width (running pinned at the cap,
    # half scheduled per step - the width-binder diagnostic), but
    # 8,192 OOMs at boot from per-sequence overheads and the
    # reachable width prices to a tie. The lever is closed.
    kw = dict(max_num_seqs=min(boot_plan.engine_max_seqs, 4096),
              max_num_batched_tokens=boot_plan.engine_step_tokens,
              max_model_len=4608)
    report = dict(n_docs=n_docs, n_prompts=len(MAP_PROMPTS),
                  caps=list(caps), model=model_name, boot=kw,
                  corpus_tokens=int(corpus),
                  predicted={cap: plans[cap].predicted_makespan_s
                             for cap in caps},
                  operators={cap: plans[cap].operator for cap in caps},
                  image=IMAGE_STAMP, cells=[], samples={}, traces={})

    async def naive_map(engine, sp, cap_seqs, tag):
        import asyncio as _aio
        sem = _aio.Semaphore(cap_seqs)
        counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
        texts = {}

        async def gen(i, j):
            async with sem:
                final = None
                async for out in engine.generate(
                        {"prompt_token_ids": body_ids[i] + p_ids[j]},
                        sp, f"{tag}-{i}-{j}"):
                    final = out
                counters["requests"] += 1
                counters["prompt_tokens"] += len(final.prompt_token_ids)
                counters["cached_tokens"] += (
                    getattr(final, "num_cached_tokens", 0) or 0)
                texts[(i, j + 1)] = final.outputs[0].text

        t0 = _time.time()
        await _aio.gather(*(gen(i, j) for i in range(len(body_ids))
                            for j in range(len(p_ids))))
        return dict(wall=_time.time() - t0, texts=texts, **counters)

    for boot, planned in (("new", True), ("stock", False)):
        if boot not in boot_names:
            continue
        try:
            trace_path = f"/tmp/opgrid_maps_trace_{boot}.jsonl"
            if trace:
                os.environ["DOCENGINE_STEPTRACE"] = trace_path
            else:
                os.environ.pop("DOCENGINE_STEPTRACE", None)
            extra = (dict(
                scheduling_policy="priority",
                kv_transfer_config=KVTransferConfig(
                    kv_connector="DocEngineForkConnector",
                    kv_connector_module_path=(
                        "docengine.engineext.forkconnector"),
                    kv_role="kv_both"),
                scheduler_cls=("docengine.engineext.scheduler."
                               "DocEngineScheduler"))
                if planned else {})
            # Maps stock keeps 0.92 at the full step budget - that
            # exact configuration completed all three caps in the
            # first flight (106/128/479 s), so it stays untouched
            # for comparability. Our maps boot runs 0.90: the
            # cap-256 death was mid-decode with the pool at 79%, so
            # the workspace outside the pool gets the extra room,
            # and the 0.90 cell measured clean (reads and step shape
            # match the 0.92 run).
            engine = Engine.from_engine_args(AsyncEngineArgs(
                model=model_name, kv_cache_dtype="fp8",
                gpu_memory_utilization=0.90 if planned else 0.92,
                enable_prefix_caching=True,
                disable_log_stats=True, **extra, **kw))

            async def reset():
                res = engine.reset_prefix_cache()
                if inspect.isawaitable(res):
                    await res

            eos_id = tok.eos_token_id
            ops = (["pipelined_map", "rewind_map"]
                   if planned else ["naive_vllm"])
            for cap in caps:
                for op in ops:
                    await reset()
                    sp = SamplingParams(temperature=0.0, max_tokens=cap)
                    tag = f"{boot}-{op[:3]}{cap}"
                    t0m = _time.monotonic()
                    try:
                        if op == "rewind_map":
                            r = await run_map_rewind(
                                engine, sp, body_ids, p_ids,
                                plans[cap].budget_tokens, eos_id,
                                tag=tag)
                            wall = r["wall"]
                            texts = {k: tok.decode(v) for k, v in
                                     r["tokens"].items()}
                            reads = -1.0
                            reqs = len(body_ids)
                        elif op == "pipelined_map":
                            r = await run_map(engine, sp, body_ids,
                                              p_ids,
                                              plans[cap].budget_tokens,
                                              tag=tag)
                            wall = r["wall"]
                            texts = r["texts"]
                            reads = round(
                                (r["prompt_tokens"]
                                 - r["cached_tokens"])
                                / max(1, corpus), 3)
                            reqs = r["requests"]
                        else:
                            r = await naive_map(engine, sp,
                                                kw["max_num_seqs"],
                                                tag)
                            wall = r["wall"]
                            texts = r["texts"]
                            reads = round(
                                (r["prompt_tokens"]
                                 - r["cached_tokens"])
                                / max(1, corpus), 3)
                            reqs = r["requests"]
                        t1m = _time.monotonic()
                        cell = dict(boot=boot, operator=op, cap=cap,
                                    wall=round(wall, 3),
                                    requests=reqs, reads=reads,
                                    t0m=round(t0m, 3),
                                    t1m=round(t1m, 3))
                        report["cells"].append(cell)
                        report["samples"][f"{op}-{cap}"] = [
                            texts.get((0, j + 1), "")[:160]
                            for j in range(len(p_ids))]
                        print(f"[opgrid-maps] {cell}", flush=True)
                    except Exception as e:
                        import traceback as _tb
                        report["cells"].append(dict(
                            boot=boot, operator=op, cap=cap,
                            error=f"{type(e).__name__}: {e}",
                            traceback=_tb.format_exc()[-4000:]))
                        print(f"[opgrid-maps] {boot}/{op}/m{cap} "
                              f"failed: {e}", flush=True)
                        _tb.print_exc()
            try:
                engine.shutdown()
            except Exception as e:
                print(f"[opgrid-maps] shutdown: {e}", flush=True)
            if os.path.exists(trace_path):
                with open(trace_path, "rb") as f:
                    report["traces"][boot] = gzip.compress(f.read())
        except Exception as e:
            import traceback as _tb
            report.setdefault("boot_errors", {})[boot] = (
                f"{type(e).__name__}: {e}")
            report.setdefault("boot_tracebacks", {})[boot] = (
                _tb.format_exc()[-4000:])
            print(f"[opgrid-maps] boot {boot} failed: {e}", flush=True)
            _tb.print_exc()

    traces = report.pop("traces")
    sfx = model_key if len(boot_names) == 2 else (
        f"{model_key}_" + "_".join(boot_names))
    with open(f"/results/opgrid_maps_{sfx}.json", "w") as f:
        json.dump({k: v for k, v in report.items()}, f)
    for boot, blob in traces.items():
        with open(f"/results/opgrid_maps_trace_{boot}_{sfx}"
                  ".jsonl.gz", "wb") as f:
            f.write(blob)
    results_vol.commit()
    print(f"[opgrid-maps] banked opgrid_maps_{sfx}.json",
          flush=True)
    report["traces"] = traces
    return report


# -----------------------------------------------------------------
# Sync filter comparison: both stock and ours through LLM.generate()
# so the submission path is identical and the comparison isolates
# KV rewind vs prefix cache.
# -----------------------------------------------------------------

@app.function(image=opgrid_image, gpu="H100!", timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def sync_filters(n_docs: int = 10000, reps: int = 3,
                 model_key: str = "4b") -> dict:
    """Sync LLM.generate() comparison: stage-major waves (stock)
    vs KV rewind (our scheduler), both through the sync API."""
    import gc
    import os
    import time as _time

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from docengine.configs import DEVICES, MODELS
    from docengine.plan import plan_query

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"

    model_name = MODEL if model_key == "4b" else MODEL32
    cfg_name = "Qwen3-4B-FP8" if model_key == "4b" else "Qwen3-32B-FP8"
    tok = AutoTokenizer.from_pretrained(model_name)
    yes_ids, no_ids = set(), set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            yes_ids.add(ids[0])
    for w in ("NO", " NO", "No", " No", "N", " N"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            no_ids.add(ids[0])

    rng = np.random.default_rng(FLAG_SEED + 100)
    flags = (rng.random((n_docs, N_FILTERS))
             < np.array(PROFILES["permissive"])[None, :]).astype(int)
    docs = _build_pool(n_docs)
    bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(N_FILTERS)]
    corpus = sum(len(b) for b in body_ids)
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=sorted(yes_ids | no_ids))

    plan = plan_query(N_FILTERS, [len(b) for b in body_ids],
                      MODELS[cfg_name], DEVICES["H100-SXM-80GB"],
                      selectivity=list(PROFILES["permissive"]),
                      policy="hybrid")
    B = plan.engine_step_tokens
    report = dict(n_docs=n_docs, model=model_name, B=B, cells=[])

    def _yes_sync(out):
        t = out.outputs[0].text.upper()
        iy, ino = t.find("YES"), t.find("NO")
        return 1 if iy >= 0 and (ino < 0 or iy < ino) else 0

    # --- ARM 1: stock stage-major waves ---
    for rep in range(reps):
        llm = LLM(model=model_name, kv_cache_dtype="fp8",
                   max_model_len=4608, max_num_seqs=4096,
                   max_num_batched_tokens=B,
                   gpu_memory_utilization=0.88,
                   enable_prefix_caching=True,
                   disable_log_stats=True)
        alive = list(range(n_docs))
        answers = {}
        total_prompt = 0
        total_cached = 0
        t0 = _time.time()
        for j in range(N_FILTERS):
            prompts = [{"prompt_token_ids": body_ids[i] + q_ids[j]}
                       for i in alive]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            for idx, (i, out) in enumerate(zip(alive, outs)):
                total_prompt += len(out.prompt_token_ids)
                total_cached += getattr(out, "num_cached_tokens", 0) or 0
                answers[(i, j + 1)] = _yes_sync(out)
            alive = [i for i in alive if answers[(i, j + 1)]]
        wall = _time.time() - t0
        uncached = total_prompt - total_cached
        reads = round(uncached / max(1, corpus), 3)
        wrong = sum(1 for (i, j), v in answers.items()
                    if v != int(flags[i][j - 1]))
        cell = dict(method="stock_waves", rep=rep,
                    wall=round(wall, 3), requests=sum(
                        len([i for i in range(n_docs)]) for _ in range(1)),
                    reads=reads, wrong=wrong, survivors=len(alive))
        # count actual requests
        cell["requests"] = len(answers)
        report["cells"].append(cell)
        print(f"[sync] {cell}", flush=True)
        del llm
        del llm; gc.collect(); torch.cuda.empty_cache()

    # --- ARM 2: KV rewind through our scheduler ---
    for rep in range(reps):
        llm = LLM(model=model_name, kv_cache_dtype="fp8",
                   max_model_len=4608, max_num_seqs=4096,
                   max_num_batched_tokens=B,
                   gpu_memory_utilization=0.92,
                   enable_prefix_caching=True,
                   disable_log_stats=True,
                   scheduler_cls=("docengine.engineext.scheduler."
                                  "DocEngineScheduler"))
        # register the query (questions + yes/no ids)
        reg = [len(q_ids)]
        for q in q_ids:
            reg += [len(q)] + list(q)
        reg_rid = (f"de1|reg|"
                   f"Y{','.join(map(str, sorted(yes_ids)))}|"
                   f"reg-{rep}")
        llm.generate([{"prompt_token_ids": reg}], sp,
                     request_id=[reg_rid], use_tqdm=False)

        # submit one chain request per document
        prompts = [{"prompt_token_ids": body_ids[i] + q_ids[0]}
                   for i in range(n_docs)]
        rids = [f"de1|c|d{i}|sync-{rep}-{i}-0" for i in range(n_docs)]
        t0 = _time.time()
        outs = llm.generate(prompts, sp, request_id=rids,
                            use_tqdm=False)
        wall = _time.time() - t0

        # parse chain outputs: each output has all stage answers
        answers = {}
        for i, out in enumerate(outs):
            toks = list(out.outputs[0].token_ids or ())
            for j, t in enumerate(toks[:N_FILTERS]):
                answers[(i, j + 1)] = 1 if t in yes_ids else 0
        survivors = [i for i in range(n_docs)
                     if all(answers.get((i, j+1), 0)
                            for j in range(N_FILTERS))]
        wrong = sum(1 for (i, j), v in answers.items()
                    if v != int(flags[i][j - 1]))
        cell = dict(method="kv_rewind", rep=rep,
                    wall=round(wall, 3), requests=n_docs,
                    reads=-1, wrong=wrong,
                    survivors=len(survivors))
        report["cells"].append(cell)
        print(f"[sync] {cell}", flush=True)
        del llm
        del llm; gc.collect(); torch.cuda.empty_cache()

    with open("/results/sync_filters.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return report


@app.local_entrypoint()
def main(n_docs: int = 10000, reps: int = 3, model: str = "4b",
         profile: str = "", old_rider: bool = False,
         stock_only: bool = False, classifier: bool = False,
         out: str = ""):
    import os
    data = opgrid_filters.remote(n_docs, reps, model, profile,
                                 old_rider, stock_only, classifier)
    sfx = (f"_{model}" + (f"_{profile}" if profile else "")
           + ("_classifier_cc" if classifier else "")
           + ("_stock" if stock_only else ""))
    traces = data.pop("traces", {})
    path = out or f"results/engine/opgrid_filters{sfx}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"saved {path}")
    for boot, blob in traces.items():
        tpath = f"results/engine/opgrid_trace_{boot}{sfx}.jsonl.gz"
        with open(tpath, "wb") as f:
            f.write(blob)
        print(f"saved {tpath}")
