"""The filter comparison: KV rewind against stock vLLM pipelining.

Both arms run the same five-filter query over the same documents,
through the same async API, in the same container. They differ in one
thing: what happens to a document's KV between its filter stages.

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

Run:
    modal run experiments/modal_filters.py
    modal run experiments/modal_filters.py --n-docs 2000 --reps 1
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
async def filter_cells(n_docs: int = 10000, reps: int = 3,
                       arms: str = "rewind,stock") -> dict:
    import time as _time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM as Engine

    from quail.configs import DEVICES, MODELS
    from quail.plan import plan_query
    from quail.runtime.engine_client import run_filter_chain_engine
    from workload import build_corpus, yes_no_ids

    os.environ["QUAIL_SINGLE_TENANT"] = "1"

    tok = AutoTokenizer.from_pretrained(MODEL)
    yes_ids, no_ids = yes_no_ids(tok)
    body_ids, q_ids, flags = build_corpus(tok, n_docs)
    corpus = sum(len(b) for b in body_ids)

    plan = plan_query(N_FILTERS, [len(b) for b in body_ids],
                      MODELS[CFG_NAME], DEVICES[DEVICE_NAME],
                      selectivity=list(SELECTIVITY))
    budget = plan.budget_tokens
    # The concurrency a stock client should choose: the same token
    # budget the plan gives our admission, divided by what one request
    # costs. This is the fair setting; see the module docstring.
    mean_req = (corpus // n_docs) + max(len(q) for q in q_ids)
    stock_sem = max(256, budget // mean_req)

    # Constrain the sampler to the yes/no ids on both arms: every
    # stage then answers in exactly one token by construction, so no
    # decode step runs and neither arm can lose to the other's parser.
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=sorted(yes_ids | no_ids))

    report = dict(n_docs=n_docs, model=MODEL, corpus_tokens=corpus,
                  n_filters=N_FILTERS, selectivity=list(SELECTIVITY),
                  budget_tokens=budget, stock_semaphore=stock_sem,
                  step_tokens=plan.engine_step_tokens,
                  max_num_seqs=plan.engine_max_seqs, cells=[])
    print(f"[filters] corpus {corpus:,} tokens over {n_docs} documents; "
          f"budget {budget:,}; stock semaphore {budget:,}/{mean_req} = "
          f"{stock_sem}", flush=True)

    async def run_stock(engine, tag):
        """Pipelining on the stock engine: per-document sequential
        requests, gated client-side, bounded by the token-derived
        semaphore."""
        import asyncio as _aio
        sem = _aio.Semaphore(stock_sem)
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

        async def one_doc(i):
            async with sem:
                for j in range(N_FILTERS):
                    got = await ask(body_ids[i] + q_ids[j], f"{tag}-{i}-{j}")
                    answers[(i, j + 1)] = got
                    if not got:
                        return
                survivors.append(i)

        t0 = _time.time()
        await _aio.gather(*(one_doc(i) for i in range(n_docs)))
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
        engine = Engine.from_engine_args(AsyncEngineArgs(
            model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
            max_num_seqs=(plan.engine_max_seqs if planned else 4096),
            max_num_batched_tokens=plan.engine_step_tokens,
            gpu_memory_utilization=0.92 if planned else 0.88,
            enable_prefix_caching=True, disable_log_stats=True, **extra))
        try:
            for rep in range(reps):
                res = engine.reset_prefix_cache()
                if hasattr(res, "__await__"):
                    await res
                tag = f"{arm}-{rep}"
                t0m = _time.monotonic()
                if planned:
                    r = await run_filter_chain_engine(
                        engine, sp, body_ids, q_ids, budget, yes_ids,
                        tag=tag)
                else:
                    r = await run_stock(engine, tag)
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
            try:
                engine.shutdown()
            except Exception as e:
                print(f"[filters] shutdown: {e}", flush=True)

    with open("/results/filter_cells.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return report


@app.local_entrypoint()
def main(n_docs: int = 10000, reps: int = 3, arms: str = "rewind,stock",
         out: str = "results/engine/filter_cells.json"):
    data = filter_cells.remote(n_docs, reps, arms)
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
