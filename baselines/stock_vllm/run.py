"""Stock vLLM baseline: drives WorkerH100 over all 35 QUAIL-B queries.
Runs entirely on Modal.

    uv run modal run -m baselines.stock_vllm.run::main

Each query is a pipeline of filter and join steps executed naively on
stock vLLM. Filters run each stage as a separate generate_batch call;
survivors feed the next stage. Joins run the full cross product of
surviving documents via generate_join_batch. Multi-join queries
(star and chain shapes) run each join sequentially, thinning both
sides between steps so later joins see fewer documents.
"""

import json
import time
from pathlib import Path

import modal

from baselines.old_stock import operators
from baselines.old_stock.config import DATA_DIR, FILTER_MAX_TOKENS, MODEL_NAMES, SF
from baselines.old_stock.worker import WorkerH100, app, hf_cache_vol

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]", "pandas", "pyarrow",
                "numpy", "datasets", "transformers")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("quail")
    .add_local_python_source("baselines")
)

results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)

QUERY_ORDER = [
    "IMDB-1", "IMDB-2", "IMDB-3", "IMDB-4", "IMDB-5",
    "IMDB-6", "IMDB-7", "IMDB-8", "IMDB-9", "IMDB-10",
    "BIO-1", "BIO-2", "BIO-3", "BIO-4", "BIO-5",
    "BIO-6", "BIO-7", "BIO-8",
    "FEV-1", "FEV-2", "FEV-3", "FEV-4", "FEV-5", "FEV-6",
    "FEV-7", "FEV-8", "FEV-9",
    "LEP-1", "LEP-2", "LEP-3", "LEP-4", "LEP-5", "LEP-6",
    "LEP-7", "LEP-8",
]


def _tokenizer_for(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def define_all_queries():
    """All 35 QUAIL-B queries for the stock vLLM baseline.

    Each query is a dict with:
        aliases: {alias: (table_name, text_col)}
            Maps each alias to the parquet table and text column it reads.
        steps: list of step tuples, each either
            ("filter", alias, [template, ...])
                Run each template as a separate filter batch on the alias's
                live documents. Survivors of stage N feed stage N+1.
            ("join", template, left_alias, right_alias, anchor)
                Run the cross product of left and right live documents.
                anchor=0 means left is the prefix (cached); anchor=1 means
                right is the prefix. Both sides are thinned to indices that
                appear in at least one TRUE pair.
    """
    from quail.bench.quailb import (
        F1, F4, F5, F7, F8, F9, F11, F12, F13,
        LEP1, LEP2, LEP3, LEP4, LEP5, LEPS1,
        DISCUSS_ASPECT, ASPECT_SENTIMENT,
        REACTION, REACTION_SEVERE,
        SUPPORT, REFUTE, LEPJOIN,
    )

    rv = ("reviews", "body")
    asp = ("aspects", "aspect")
    rp = ("reports", "report")
    tm = ("terms", "term")
    cl = ("claims", "claim")
    ev = ("evidence", "text")
    dc = ("citations", "destination_context")
    pt = ("citations", "passage_text")

    Q = {}

    # ---- IMDB ----

    Q["IMDB-1"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1])])
    Q["IMDB-2"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-3"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1]),
                              ("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-4"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1, F4]),
                              ("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-5"] = dict(aliases={"r": rv, "a": asp},
                       steps=[("filter", "r", [F1, F4, F5]),
                              ("join", DISCUSS_ASPECT, "r", "a", 0)])
    Q["IMDB-6"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1, F4])])
    Q["IMDB-7"] = dict(aliases={"r": rv},
                       steps=[("filter", "r", [F1, F4, F5])])
    # 2-join star: both joins anchored on reviews
    Q["IMDB-8"] = dict(aliases={"r": rv, "a": asp, "a2": asp},
                       steps=[("join", DISCUSS_ASPECT, "r", "a", 0),
                              ("join", ASPECT_SENTIMENT, "r", "a2", 0)])
    # 3-join chain: r1-a1-r2-a2
    Q["IMDB-9"] = dict(
        aliases={"r1": rv, "a1": asp, "r2": rv, "a2": asp},
        steps=[("join", DISCUSS_ASPECT, "r1", "a1", 0),
               ("join", DISCUSS_ASPECT, "r2", "a1", 0),
               ("join", ASPECT_SENTIMENT, "r2", "a2", 0)])
    Q["IMDB-10"] = dict(
        aliases={"r1": rv, "a1": asp, "r2": rv, "a2": asp},
        steps=[("filter", "r1", [F1]),
               ("join", DISCUSS_ASPECT, "r1", "a1", 0),
               ("join", DISCUSS_ASPECT, "r2", "a1", 0),
               ("join", ASPECT_SENTIMENT, "r2", "a2", 0)])

    # ---- BioDEX ----

    Q["BIO-1"] = dict(aliases={"r": rp},
                      steps=[("filter", "r", [F7])])
    Q["BIO-2"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("join", REACTION, "r", "m", 0)])
    Q["BIO-3"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7]),
                             ("join", REACTION, "r", "m", 0)])
    Q["BIO-4"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7, F8]),
                             ("join", REACTION, "r", "m", 0)])
    Q["BIO-5"] = dict(aliases={"r": rp, "m": tm},
                      steps=[("filter", "r", [F7, F8, F9]),
                             ("join", REACTION, "r", "m", 0)])
    # 2-join star: both joins anchored on reports
    Q["BIO-6"] = dict(aliases={"r": rp, "m": tm, "m2": tm},
                      steps=[("join", REACTION, "r", "m", 0),
                             ("join", REACTION_SEVERE, "r", "m2", 0)])
    # 3-join chain: r1-m1-r2-m2
    Q["BIO-7"] = dict(
        aliases={"r1": rp, "m1": tm, "r2": rp, "m2": tm},
        steps=[("join", REACTION, "r1", "m1", 0),
               ("join", REACTION_SEVERE, "r2", "m1", 0),
               ("join", REACTION, "r2", "m2", 0)])
    Q["BIO-8"] = dict(
        aliases={"r1": rp, "m1": tm, "r2": rp, "m2": tm},
        steps=[("filter", "r1", [F7]),
               ("join", REACTION, "r1", "m1", 0),
               ("join", REACTION_SEVERE, "r2", "m1", 0),
               ("join", REACTION, "r2", "m2", 0)])

    # ---- FEVER ----
    # anchor=1 throughout: evidence pages are longer than claims

    Q["FEV-1"] = dict(aliases={"c": cl},
                      steps=[("filter", "c", [F11])])
    Q["FEV-2"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("join", SUPPORT, "c", "e", 1)])
    Q["FEV-3"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11]),
                             ("join", SUPPORT, "c", "e", 1)])
    Q["FEV-4"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11, F12]),
                             ("join", SUPPORT, "c", "e", 1)])
    # Two-sided pushdown: filter both sides before the join
    Q["FEV-5"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11]),
                             ("filter", "e", [F13]),
                             ("join", SUPPORT, "c", "e", 1)])
    Q["FEV-6"] = dict(aliases={"c": cl, "e": ev},
                      steps=[("filter", "c", [F11, F12]),
                             ("filter", "e", [F13]),
                             ("join", SUPPORT, "c", "e", 1)])
    # 2-join star: both joins anchored on claims
    Q["FEV-7"] = dict(aliases={"c": cl, "e": ev, "e2": ev},
                      steps=[("join", SUPPORT, "c", "e", 1),
                             ("join", REFUTE, "c", "e2", 1)])
    # 3-join chain: c1-e1-c2-e2
    Q["FEV-8"] = dict(
        aliases={"c1": cl, "e1": ev, "c2": cl, "e2": ev},
        steps=[("join", SUPPORT, "c1", "e1", 1),
               ("join", REFUTE, "c2", "e1", 1),
               ("join", SUPPORT, "c2", "e2", 1)])
    Q["FEV-9"] = dict(
        aliases={"c1": cl, "e1": ev, "c2": cl, "e2": ev},
        steps=[("filter", "c1", [F11]),
               ("join", SUPPORT, "c1", "e1", 1),
               ("join", REFUTE, "c2", "e1", 1),
               ("join", SUPPORT, "c2", "e2", 1)])

    # ---- LePaRD ----
    # Self-join: same table, two text columns. anchor=0 (destination_context
    # is the excerpt, typically longer than the quoted passage).

    Q["LEP-1"] = dict(aliases={"d": dc},
                      steps=[("filter", "d", [LEP1])])
    Q["LEP-2"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-3"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-4"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-5"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2, LEP3]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-6"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d",
                              [LEP1, LEP2, LEP3, LEP4, LEP5]),
                             ("join", LEPJOIN, "d", "s", 0)])
    # Two-sided pushdown: filter excerpts and passages separately
    Q["LEP-7"] = dict(aliases={"d": dc, "s": pt},
                      steps=[("filter", "d", [LEP1, LEP2]),
                             ("filter", "s", [LEPS1]),
                             ("join", LEPJOIN, "d", "s", 0)])
    Q["LEP-8"] = dict(aliases={"d": dc},
                      steps=[("filter", "d",
                              [LEP1, LEP2, LEP3, LEP4, LEP5])])

    return Q


def _load_alias_data(data_dir, sf, aliases):
    """Load table data for each alias in a query.

    Returns:
        {alias: (ids_list, texts_list)}. Aliases that share the same
        (table, text_col) get the same list objects.
    """
    cache = {}
    result = {}
    for alias, (table, text_col) in aliases.items():
        key = (table, text_col)
        if key not in cache:
            cache[key] = operators.read_table(
                data_dir, sf, table, "id", text_col)
        result[alias] = cache[key]
    return result


def run_query(worker, qid, query_def, data_dir, sf, tokenizer,
              true_ids, false_ids, do_profile=False):
    """Execute one multi-step QUAIL-B query on stock vLLM.

    Each filter template and each join is a separate worker call.
    Filter survivors thin the document set for subsequent steps.
    Join survivors thin both sides for subsequent steps.

    Returns:
        (summary_entry, per_request_rows)
    """
    alias_data = _load_alias_data(data_dir, sf, query_def["aliases"])
    live = {a: list(range(len(data[1])))
            for a, data in alias_data.items()}

    step_entries = []
    per_request_rows = []
    step_n = 0

    for step in query_def["steps"]:
        if step[0] == "filter":
            _, alias, templates = step
            ids_all, texts_all = alias_data[alias]

            for tmpl in templates:
                live_idx = live[alias]
                n_in = len(live_idx)
                if n_in == 0:
                    step_entries.append(dict(
                        kind="filter", alias=alias,
                        n_in=0, n_out=0,
                        generate_wall_time_s=0,
                        total_prompt_tokens=0,
                        total_output_tokens=0))
                    step_n += 1
                    continue

                live_texts = [texts_all[i] for i in live_idx]
                filt = operators.Filter(
                    f"{qid}-s{step_n}", tmpl)
                build_t0 = time.time()
                prompts = filt.build_prompts(live_texts, tokenizer)
                build_s = time.time() - build_t0

                t0 = time.time()
                result = worker.generate_batch.remote(
                    prompts, true_ids, false_ids, FILTER_MAX_TOKENS,
                    do_profile=(do_profile and step_n == 0))
                rpc_s = time.time() - t0

                new_live = [live_idx[j]
                            for j, req in
                            enumerate(result["per_request"])
                            if req["answer"] == 1]

                ptok = sum(r["prompt_tokens"]
                           for r in result["per_request"])
                otok = sum(r["output_tokens"]
                           for r in result["per_request"])
                fresh = ptok - result["oracle_regret"][
                    "oracle_hit_tokens"]

                entry = dict(
                    kind="filter", alias=alias,
                    n_in=n_in, n_out=len(new_live),
                    build_s=build_s,
                    rpc_wall_time_s=rpc_s,
                    generate_wall_time_s=result["wall_time_s"],
                    total_prompt_tokens=ptok,
                    fresh_prompt_tokens=fresh,
                    total_output_tokens=otok,
                    oracle_regret=result["oracle_regret"],
                    vllm_metrics=result["vllm_metrics"],
                    trace_path=result.get("trace_path"),
                )
                step_entries.append(entry)

                doc_ids = [ids_all[i] for i in live_idx]
                per_request_rows.append(dict(
                    entry=entry,
                    per_request=result["per_request"],
                    timeseries=result["timeseries"],
                    doc_ids=doc_ids))

                print(f"[stock_vllm] {qid} filter({alias}): "
                      f"{n_in}->{len(new_live)} "
                      f"wall={result['wall_time_s']:.2f}s "
                      f"ptok={ptok}", flush=True)
                live[alias] = new_live
                step_n += 1

        elif step[0] == "join":
            _, template, left_a, right_a, anchor = step
            left_ids_all, left_texts_all = alias_data[left_a]
            right_ids_all, right_texts_all = alias_data[right_a]
            left_live = live[left_a]
            right_live = live[right_a]
            left_texts = [left_texts_all[i] for i in left_live]
            right_texts = [right_texts_all[i] for i in right_live]
            nl, nr = len(left_texts), len(right_texts)

            if nl == 0 or nr == 0:
                step_entries.append(dict(
                    kind="join", left=left_a, right=right_a,
                    anchor=anchor, n_left=nl, n_right=nr,
                    n_pairs=0, n_true=0,
                    generate_wall_time_s=0,
                    total_prompt_tokens=0,
                    total_output_tokens=0))
                step_n += 1
                continue

            join_op = operators.Join(
                f"{qid}-s{step_n}", template, anchor=anchor)
            build_t0 = time.time()
            prefixes, suffixes, members = \
                join_op.build_grouped_inputs(
                    left_texts, right_texts, tokenizer)
            build_s = time.time() - build_t0
            n_pairs = len(prefixes) * len(suffixes)

            print(f"[stock_vllm] {qid} join({left_a}x{right_a}): "
                  f"{nl}x{nr}={n_pairs} pairs, submitting...",
                  flush=True)

            t0 = time.time()
            result = worker.generate_join_batch.remote(
                prefixes, suffixes, true_ids, false_ids,
                FILTER_MAX_TOKENS,
                do_profile=(do_profile and step_n == 0))
            rpc_s = time.time() - t0

            surviving_left, surviving_right = set(), set()
            n_true = 0
            idx = 0
            for anc_idx in range(len(prefixes)):
                for member in members:
                    if result["per_request"][idx]["answer"] == 1:
                        if anchor == 0:
                            surviving_left.add(anc_idx)
                            surviving_right.add(member[0])
                        else:
                            surviving_right.add(anc_idx)
                            surviving_left.add(member[0])
                        n_true += 1
                    idx += 1

            live[left_a] = sorted(
                left_live[i] for i in surviving_left)
            live[right_a] = sorted(
                right_live[i] for i in surviving_right)

            ptok = sum(r["prompt_tokens"]
                       for r in result["per_request"])
            otok = sum(r["output_tokens"]
                       for r in result["per_request"])
            fresh = ptok - result["oracle_regret"][
                "oracle_hit_tokens"]

            doc_ids = []
            for anc_idx in range(len(prefixes)):
                for member in members:
                    if anchor == 0:
                        li, ri = anc_idx, member[0]
                    else:
                        li, ri = member[0], anc_idx
                    doc_ids.append((
                        left_ids_all[left_live[li]],
                        right_ids_all[right_live[ri]]))

            entry = dict(
                kind="join", left=left_a, right=right_a,
                anchor=anchor, n_left=nl, n_right=nr,
                n_pairs=n_pairs, n_true=n_true,
                build_s=build_s,
                rpc_wall_time_s=rpc_s,
                generate_wall_time_s=result["wall_time_s"],
                total_prompt_tokens=ptok,
                fresh_prompt_tokens=fresh,
                total_output_tokens=otok,
                oracle_regret=result["oracle_regret"],
                vllm_metrics=result["vllm_metrics"],
                trace_path=result.get("trace_path"),
            )
            step_entries.append(entry)
            per_request_rows.append(dict(
                entry=entry,
                per_request=result["per_request"],
                timeseries=result["timeseries"],
                doc_ids=doc_ids))

            print(f"[stock_vllm] {qid} join({left_a}x{right_a}): "
                  f"{n_true}/{n_pairs} TRUE "
                  f"wall={result['wall_time_s']:.2f}s "
                  f"ptok={ptok}", flush=True)
            step_n += 1

    gen_wall = sum(s["generate_wall_time_s"] for s in step_entries)
    total_ptok = sum(s["total_prompt_tokens"] for s in step_entries)
    total_fresh = sum(s.get("fresh_prompt_tokens", 0)
                      for s in step_entries)
    total_otok = sum(s.get("total_output_tokens", 0)
                     for s in step_entries)

    summary = dict(
        query=qid, n_steps=len(step_entries),
        steps=step_entries,
        generate_wall_time_s=gen_wall,
        total_prompt_tokens=total_ptok,
        fresh_prompt_tokens=total_fresh,
        total_output_tokens=total_otok,
    )
    return summary, per_request_rows


@app.function(image=orchestrator_image, timeout=7200,
             volumes={"/root/.cache/huggingface": hf_cache_vol,
                      "/results": results_vol})
def run_baseline(model: str = "qwen3-4b", query_id: str | None = None,
                 gpu: str = "H100!", quantization: str = "fp8",
                 profile: bool = False) -> dict:
    """Build prompts and drive WorkerH100 for all QUAIL-B queries."""
    from quail.bench.quailb import build_sets

    if gpu != WorkerH100.GPU:
        raise ValueError(
            f"stock_vllm requires gpu={WorkerH100.GPU!r}; got {gpu!r}")

    build_sets(DATA_DIR, SF)
    tokenizer = _tokenizer_for(MODEL_NAMES[model])
    true_ids, false_ids = operators.true_false_ids(tokenizer)

    all_queries = define_all_queries()
    if query_id:
        ids = [q.strip() for q in query_id.split(",")]
        for qid in ids:
            if qid not in all_queries:
                raise ValueError(
                    f"unknown query {qid!r}; available: "
                    f"{sorted(all_queries)}")
    else:
        ids = [qid for qid in QUERY_ORDER if qid in all_queries]

    worker = WorkerH100(model=model, quantization=quantization)
    print(f"[stock_vllm] warming up {model} on {gpu} ({quantization})",
          flush=True)
    worker.warmup.remote(true_ids, false_ids)

    out_dir = Path("/results/stock_vllm") / time.strftime(
        "%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_queries = []
    for qid in ids:
        print(f"\n[stock_vllm] === {qid} ===", flush=True)
        qdef = all_queries[qid]
        try:
            entry, per_req = run_query(
                worker, qid, qdef, DATA_DIR, SF, tokenizer,
                true_ids, false_ids, do_profile=profile)
        except Exception as e:                              # noqa: BLE001
            entry = dict(query=qid,
                         error=f"{type(e).__name__}: {e}")
            per_req = []
            print(f"[stock_vllm] {qid} ERROR: {e}", flush=True)

        summary_queries.append(entry)

        if per_req:
            with open(out_dir / f"{qid}.jsonl", "w") as f:
                for row in per_req:
                    f.write(json.dumps(row) + "\n")

        with open(out_dir / "summary.json", "w") as f:
            json.dump(dict(
                model=model, gpu=gpu, quantization=quantization,
                sf=SF, queries=summary_queries), f, indent=2)
        results_vol.commit()

    print(f"\n[stock_vllm] saved {out_dir}/summary.json "
          f"({len(summary_queries)} queries)", flush=True)
    return dict(out_dir=str(out_dir), n_queries=len(summary_queries))


@app.local_entrypoint()
def main(model: str = "qwen3-4b", query: str = "", gpu: str = "H100!",
        quantization: str = "fp8", profile: bool = False):
    fc = run_baseline.spawn(
        model=model, query_id=(query or None), gpu=gpu,
        quantization=quantization, profile=profile)
    print(f"function call id: {fc.object_id}")
    print(fc.get())
