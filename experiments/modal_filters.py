"""The filter comparison: KV rewind against stock vLLM pipelining.

Both runs execute the same five-filter query over the same documents,
through the same synchronous step interface, in the same container.
They differ in one thing: what happens to a document's KV between its
filter stages.

  stock       one request per (document, stage), gated client-side.
              Between stages the document's KV is whatever vLLM's
              prefix cache still holds.
  rewind      one living request per document. The scheduler judges
              each answer, erases the question KV back to the
              document boundary, and appends the next question. The
              document is prefilled exactly once.

Both are admitted by TOKEN budget, not request count. That matters:
the stock arm's concurrency is set to budget / mean-request-tokens
(about 2,048 documents at 10k docs on the 4B model), which keeps the
in-flight KV inside the pool. Running stock at the engine's 4,096
sequence cap instead overflows the pool by 1.6x, forces the prefix
cache to evict between stages, and costs it 2.40x corpus reads
against 1.23x - a configuration mistake that looks like a 2x win for
rewind and is not one. The measured gap on the fair setting is 1.08x
(42.9s against 39.8s), and it comes from block alignment: a prefix
cache matches in whole 16-token blocks, so the block straddling the
document boundary is recomputed on every stage, while a rewind cuts
by token position and keeps it.

The c0 anchor (modal run experiments/modal_filters.py::c0_anchor)
re-measures the per-query software residue with the host controlled:
query walls spread up to 45 percent across containers, so c0 must be
derived against the rate the same container actually serves, not the
fleet anchor. The probe runs on the stock boot - one full stage-1
pass, same submission machinery as the query - because single-tenant
mode on the planned boot refuses untagged requests, and container
speed is a property of the host, not the boot. Each arm's own
scheduler and gating costs then land in that arm's c0, which is what
a per-query residue means:

    c0 = wall - reads x corpus / rate_this_container

Run:
    modal run experiments/modal_filters.py
    modal run experiments/modal_filters.py --n-docs 2000 --reps 1
    modal run experiments/modal_filters.py::c0_anchor
"""

import json
import os

import modal

from workload import (CFG_NAME, DEVICE_NAME, MODEL, N_FILTERS,
                      SELECTIVITY, hf_cache, image, results_vol)

app = modal.App("quail-filters")
filters_image = image.add_local_python_source("workload")


@app.function(image=filters_image, gpu="H100!", timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def filter_cells(n_docs: int = 10000, reps: int = 3,
                 arms: str = "rewind,stock",
                 probe_rate: bool = False,
                 outname: str = "filter_cells.json",
                 kv: str = "fp8") -> dict:
    import time as _time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine as Engine

    from quail.configs import DEVICES, MODELS
    from quail.plan import plan_query
    from quail.runtime.engine_client import run_filter_chain_engine
    from workload import build_corpus, yes_no_ids

    os.environ["QUAIL_SINGLE_TENANT"] = "1"

    tok = AutoTokenizer.from_pretrained(MODEL)
    yes_ids, no_ids = yes_no_ids(tok)
    body_ids, q_ids, flags = build_corpus(tok, n_docs)
    corpus = sum(len(b) for b in body_ids)

    # KV format is a boot choice, independent of the fp8 compute path.
    # The plan's admission budget must use the matching bytes per
    # token, or a bf16 boot would admit twice the KV that fits.
    if kv not in ("fp8", "bf16"):
        raise ValueError(f"kv must be fp8 or bf16, got {kv!r}")
    model_cfg = (MODELS[CFG_NAME] if kv == "fp8"
                 else MODELS[CFG_NAME].with_kv_dtype(2))
    kv_cache_dtype = "fp8" if kv == "fp8" else "auto"

    plan = plan_query(N_FILTERS, [len(b) for b in body_ids],
                      model_cfg, DEVICES[DEVICE_NAME],
                      selectivity=list(SELECTIVITY))
    budget = plan.budget_tokens
    # One admission rule for both sides: the saturation-derived token
    # budget, expressed for stock as a document cap. The banked cells
    # ran the retired 0.8-pool rule (2,048 documents at 10k docs);
    # this setting is smaller and the next comparison run re-baselines
    # it (prediction: unchanged walls - the cap still exceeds the
    # ~425 concurrently live documents the step trace measured).
    mean_req = (corpus // n_docs) + max(len(q) for q in q_ids)
    stock_sem = max(1, budget // mean_req)

    # Constrain the sampler to the yes/no ids on both arms: every
    # stage then answers in exactly one token by construction, so no
    # decode step runs and neither arm can lose to the other's parser.
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=sorted(yes_ids | no_ids))

    report = dict(n_docs=n_docs, model=MODEL, kv=kv,
                  corpus_tokens=corpus,
                  corpus_sq_tokens=sum(len(b) ** 2 for b in body_ids),
                  n_filters=N_FILTERS, selectivity=list(SELECTIVITY),
                  budget_tokens=budget, stock_semaphore=stock_sem,
                  step_tokens=plan.engine_step_tokens,
                  max_num_seqs=plan.engine_max_seqs, cells=[])
    print(f"[filters] corpus {corpus:,} tokens over {n_docs} documents; "
          f"budget {budget:,}; stock cap {budget:,}/{mean_req} = "
          f"{stock_sem}", flush=True)

    def run_stock(engine, tag):
        """Pipelining on the stock engine: per-document sequential
        requests, gated client-side, bounded by the token-derived
        document cap. A plain admit-step-route loop; the concurrency
        lives in the engine's scheduler, not here."""
        counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
        answers, survivors = {}, []
        inflight = {}                     # request id -> (doc, stage)

        def ask(i, j):
            rid = f"{tag}-{i}-{j}"
            inflight[rid] = (i, j)
            engine.add_request(
                rid, {"prompt_token_ids": body_ids[i] + q_ids[j]}, sp)

        t0 = _time.time()
        live, next_doc = 0, 0
        while next_doc < n_docs or inflight:
            while next_doc < n_docs and live < stock_sem:
                live += 1
                ask(next_doc, 0)
                next_doc += 1
            for out in engine.step():
                if not out.finished or out.request_id not in inflight:
                    continue
                i, j = inflight.pop(out.request_id)
                counters["requests"] += 1
                counters["prompt_tokens"] += len(out.prompt_token_ids)
                counters["cached_tokens"] += (
                    getattr(out, "num_cached_tokens", 0) or 0)
                t = out.outputs[0].text.upper()
                iy, ino = t.find("YES"), t.find("NO")
                got = 1 if iy >= 0 and (ino < 0 or iy < ino) else 0
                answers[(i, j + 1)] = got
                if got and j + 1 < N_FILTERS:
                    ask(i, j + 1)
                else:
                    if got:
                        survivors.append(i)
                    live -= 1
        return dict(wall=_time.time() - t0, answers=answers,
                    survivors=sorted(survivors), **counters)

    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        planned = arm == "rewind"
        # the step trace is the only honest source of prefill work for
        # chain mode; it exists only on the planned boot
        trace_path = f"/tmp/quail_trace_{arm}.jsonl" if planned else None
        if trace_path:
            os.environ["QUAIL_STEPTRACE"] = trace_path
        else:
            os.environ.pop("QUAIL_STEPTRACE", None)
        extra = (dict(scheduler_cls="quail.engineext.scheduler."
                                    "QuailScheduler")
                 if planned else {})
        # Stock boots at 0.88: its flash-attention workspace wants
        # 4.22 GiB and only 2.79 GiB is free outside the KV pool at
        # 0.92, which OOMs mid-run. Our boot fits at 0.92 because the
        # plan sizes the step budget and sequence cap that decide the
        # workspace.
        engine = Engine.from_engine_args(EngineArgs(
            model=MODEL, kv_cache_dtype=kv_cache_dtype, max_model_len=4608,
            max_num_seqs=(plan.engine_max_seqs if planned else 4096),
            max_num_batched_tokens=plan.engine_step_tokens,
            gpu_memory_utilization=0.92 if planned else 0.88,
            enable_prefix_caching=True, disable_log_stats=True, **extra))
        try:
            if probe_rate and not planned:
                # The container's own serving rate: one full stage-1
                # pass through the same submission machinery as the
                # query. Fresh engine, so nothing is cached; the rep
                # loop resets the cache before rep 0, so the probe
                # warms nothing the query sees.
                pc = dict(prompt=0, cached=0)
                pending = set()
                t0p = _time.time()
                nd = 0
                while nd < n_docs or pending:
                    while nd < n_docs and len(pending) < stock_sem:
                        rid = f"probe-{nd}"
                        pending.add(rid)
                        engine.add_request(
                            rid,
                            {"prompt_token_ids": body_ids[nd] + q_ids[0]},
                            sp)
                        nd += 1
                    for out in engine.step():
                        if not out.finished or out.request_id not in pending:
                            continue
                        pending.discard(out.request_id)
                        pc["prompt"] += len(out.prompt_token_ids)
                        pc["cached"] += (
                            getattr(out, "num_cached_tokens", 0) or 0)
                probe_wall = _time.time() - t0p
                rate = (pc["prompt"] - pc["cached"]) / probe_wall
                report["probe"] = dict(
                    rate_tok_s=round(rate, 1),
                    probe_tokens=pc["prompt"] - pc["cached"],
                    probe_wall_s=round(probe_wall, 3),
                    probe_docs=n_docs, boot="stock")
                print(f"[filters] probe: {rate:,.0f} tok/s over "
                      f"{probe_wall:.1f}s", flush=True)
            for rep in range(reps):
                engine.reset_prefix_cache()
                tag = f"{arm}-{rep}"
                t0m = _time.monotonic()
                if planned:
                    r = run_filter_chain_engine(
                        engine, sp, body_ids, q_ids, budget, yes_ids,
                        tag=tag)
                else:
                    r = run_stock(engine, tag)
                t1m = _time.monotonic()
                # Two ways to count the prefill work, and only one of
                # them is right for each arm.
                #
                # Client side: (prompt tokens - cache hits) / corpus.
                # Correct for the stock arm, where every request is
                # submitted once and its final prompt IS what it asked
                # for. WRONG for chain mode: a rewind truncates and
                # re-extends prompt_token_ids, so the final snapshot
                # shows only [document + last question] and the
                # intermediate question prefills vanish. It reported
                # an identical 1.143 for three operators that do
                # visibly different amounts of work.
                #
                # Scheduler side: sum the step trace's prefill_tokens
                # over the cell's window. The scheduler sees every
                # prefill, so this is the honest number - but it only
                # exists for the planned boot, which is the one
                # running our scheduler.
                client_reads = round(
                    (r["prompt_tokens"] - r["cached_tokens"])
                    / max(1, corpus), 3)
                reads, source = client_reads, "client"
                if planned and trace_path and os.path.exists(trace_path):
                    pf = 0
                    with open(trace_path) as tf:
                        for line in tf:
                            if not line.strip():
                                continue
                            rec = json.loads(line)
                            if t0m <= rec["t"] <= t1m:
                                pf += rec["prefill_tokens"]
                    if pf:
                        reads = round(pf / max(1, corpus), 3)
                        source = "step-trace"
                wrong = sum(1 for (i, j), v in r["answers"].items()
                            if v != int(flags[i][j - 1]))
                cell = dict(arm=arm, rep=rep, wall=round(r["wall"], 3),
                            requests=r["requests"], reads=reads,
                            reads_source=source,
                            client_reads=client_reads,
                            wrong=wrong, answered=len(r["answers"]),
                            survivors=len(r["survivors"]))
                report["cells"].append(cell)
                print(f"[filters] {cell}", flush=True)
        finally:
            # the sync LLMEngine has no shutdown(); the core client
            # does, and it owns the GPU process - without this the
            # next boot in the same container finds the card occupied
            try:
                engine.engine_core.shutdown()
            except Exception as e:
                print(f"[filters] shutdown: {e}", flush=True)

    if "probe" in report:
        rate = report["probe"]["rate_tok_s"]
        for cell in report["cells"]:
            work_s = cell["reads"] * corpus / rate
            cell["token_work_s"] = round(work_s, 3)
            cell["c0_s"] = round(cell["wall"] - work_s, 3)

    with open(f"/results/{outname}", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return report


@app.function(image=filters_image, timeout=900,
              volumes={"/root/.cache/huggingface": hf_cache})
def corpus_stats(n_docs: int = 10000) -> dict:
    """Corpus shape for the estimator, no GPU: token totals and the
    squared-length sum the quadratic surcharge needs."""
    from transformers import AutoTokenizer

    from workload import build_corpus

    tok = AutoTokenizer.from_pretrained(MODEL)
    body_ids, _q, _f = build_corpus(tok, n_docs)
    lens = [len(b) for b in body_ids]
    return dict(n_docs=n_docs, corpus_tokens=sum(lens),
                corpus_sq_tokens=sum(x * x for x in lens),
                max_doc_tokens=max(lens),
                sq_per_token=round(sum(x * x for x in lens)
                                   / sum(lens), 2))


@app.local_entrypoint()
def stats(n_docs: int = 10000):
    print(json.dumps(corpus_stats.remote(n_docs), indent=2))


@app.local_entrypoint()
def c0_anchor(n_docs: int = 10000, reps: int = 3,
              out: str = "results/engine/c0_anchor.json"):
    data = filter_cells.remote(n_docs, reps, "rewind,stock",
                               probe_rate=True, outname="c0_anchor.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {out}")
    print(f"  probe {data['probe']['rate_tok_s']:,.0f} tok/s")
    for arm in ("stock", "rewind"):
        c0s = sorted(c["c0_s"] for c in data["cells"]
                     if c["arm"] == arm)
        if c0s:
            med = c0s[len(c0s) // 2]
            print(f"  {arm:<8} c0 median {med:.2f}s over {len(c0s)} reps "
                  f"(all: {c0s})")


@app.local_entrypoint()
def main(n_docs: int = 10000, reps: int = 3, arms: str = "rewind,stock",
         kv: str = "fp8", out: str = ""):
    tag = "filter_cells" if kv == "fp8" else f"filter_cells_{kv}"
    out = out or f"results/engine/{tag}.json"
    data = filter_cells.remote(n_docs, reps, arms, kv=kv,
                               outname=f"{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {out}")
    for arm in ("stock", "rewind"):
        walls = [c["wall"] for c in data["cells"] if c["arm"] == arm]
        if walls:
            reads = [c["reads"] for c in data["cells"] if c["arm"] == arm][0]
            print(f"  {arm:<8} {sum(walls)/len(walls):7.1f}s mean over "
                  f"{len(walls)} reps, reads {reads}")
