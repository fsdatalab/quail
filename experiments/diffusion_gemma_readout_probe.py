"""What the vLLM baseline's logprob readout sees for DiffusionGemma.

    uv run modal run experiments/diffusion_gemma_readout_probe.py \
      --prediction "..." 2>&1 | tee results/<stamp>-readout-probe.log

Boots the request backend's own vLLM engine for the one-row-canvas
model on IMDB reviews with the F1 filter prompt (--dataset imdb) or
on agent traces with the AGENT-1 prompt (--dataset agent), exactly as
the benchmark does, and records for each document whether logprobs
came back, how the TRUE and FALSE entries are decoded, the TRUE minus
FALSE logprob margin, what true_bit reads, and what Quail answered
for the same document in the run named by --quail-run.
--no-prefix-caching boots the engine without vLLM's prefix cache.
Writes /results/ablations/diffusion_gemma_readout_probe_<dataset>
_k<logprobs>[_nocache].json.
"""

import json

import modal

from quail.bench.images import gpu_image

image = gpu_image()
app = modal.App("quail-milestone1")
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}
MODEL = "diffusion-gemma-26b-a4b-fp8"
DATASETS = {
    "imdb": ("reviews.parquet", "body", "imdb/IMDB-1", "r"),
    "agent": ("agent_traces.parquet", "trace", "agent/AGENT-1", "t"),
}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def probe(prediction: str, n_docs: int = 256,
          quail_run: str = "20260919T220437Z-1550de75",
          logprobs: int = 0, dataset: str = "imdb",
          prefix_caching: bool = True) -> str:
    """Run the readout; a logprobs count above 0 overrides the backend's.

    The documents are every k-th of the dataset in token order, so
    the sample spans the lengths in the corpus.
    """
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from pathlib import Path

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from quail.backends.request_scheduling import _ranked_answer, true_bit
    from quail.backends.vllm import VLLMEngine
    from quail.logical import bind_prompt, render_filter_prompt_ids, true_false_ids
    from quail.specs import MODELS
    from quail_b.prompts import AGENT_RECOVERED, F1

    template = {"imdb": F1, "agent": AGENT_RECOVERED}[dataset]
    table_name, column, query_dir, id_column = DATASETS[dataset]
    spec = MODELS[MODEL]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    true_ids, false_ids = true_false_ids(tokenizer)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    docs = pq.read_table(f"/results/quailb_data/sf0.1/{table_name}").to_pandas()
    docs["tokens"] = [len(tok(text)) for text in docs[column]]
    docs = docs.sort_values("tokens")
    step = max(1, len(docs) // n_docs)
    docs = docs.iloc[::step].head(n_docs)
    ids = docs["id"].tolist()
    lengths = dict(zip(ids, docs["tokens"].tolist()))
    prompt = bind_prompt(template, (column,), tok, turn=spec.turn)
    prompts = [dict(prompt_token_ids=render_filter_prompt_ids(prompt, tok(b), tok))
               for b in docs[column]]
    if logprobs:
        import quail.backends.vllm as backend
        backend.DIFFUSION_LOGPROBS = logprobs

    class Engine(VLLMEngine):
        def llm_kwargs(self, spec):
            return {**super().llm_kwargs(spec),
                    "enable_prefix_caching": prefix_caching}

    state, boot = Engine().boot(spec, sorted(set(true_ids) | set(false_ids)))
    client, sampling = state["client"], state["sampling_params"]
    outputs = client.generate(prompts, sampling, use_tqdm=False)
    quail_answers = {}
    table = Path(f"/results/benchmarks/quailb/{quail_run}/quail/{query_dir}/"
                 "filters-0.parquet")
    if table.exists():
        for row in pq.read_table(table).to_pylist():
            quail_answers[row[id_column]] = bool(row["answer"])
    records = []
    counts = {"no_logprobs": 0, "neither_word": 0, "answer_ids_in_top": 0,
              "same_as_quail": 0, "compared": 0}
    decoded_forms = {}
    for doc_id, output in zip(ids, outputs):
        completion = output.outputs[0]
        rows = getattr(completion, "logprobs", None)
        record = {"id": doc_id, "tokens": lengths[doc_id],
                  "text": completion.text,
                  "sampled": list(completion.token_ids),
                  "read": true_bit(output, set(true_ids)),
                  "quail": quail_answers.get(doc_id)}
        if not rows:
            counts["no_logprobs"] += 1
            record["entries"] = None
        else:
            first = rows[0] or {}
            entries = [(int(t), lp.logprob, getattr(lp, "decoded_token", None),
                        getattr(lp, "rank", None))
                       for t, lp in first.items()]
            answer_entries = [e for e in entries if e[0] in true_ids | false_ids]
            record["answer_entries"] = answer_entries
            record["best_rank"] = min((e[3] for e in answer_entries
                                       if e[3] is not None), default=None)
            record["ranked"] = _ranked_answer(rows)
            best = {}
            for token, logprob, _, _ in answer_entries:
                side = "true" if token in true_ids else "false"
                best[side] = max(best.get(side, logprob), logprob)
            if len(best) == 2:
                record["margin"] = best["true"] - best["false"]
            if answer_entries:
                counts["answer_ids_in_top"] += 1
                for _, _, decoded, _ in answer_entries:
                    form = repr(decoded)
                    decoded_forms[form] = decoded_forms.get(form, 0) + 1
            if record["ranked"] is None:
                counts["neither_word"] += 1
        if record["quail"] is not None:
            counts["compared"] += 1
            counts["same_as_quail"] += int(record["quail"] == bool(record["read"]))
        records.append(record)
    ranks = sorted(r["best_rank"] for r in records if r.get("best_rank"))
    covered = {k: sum(rank <= k for rank in ranks)
               for k in (20, 50, 100, 200, 500, 1000, 2000, 5000)}
    report = {"prediction": prediction, "n_docs": n_docs, "boot": boot,
              "dataset": dataset, "prefix_caching": prefix_caching,
              "sampling": str(sampling), "capacity": state["capacity"],
              "counts": counts, "decoded_forms": decoded_forms,
              "best_rank_covered_by_k": covered, "worst_rank": max(ranks, default=None),
              "true_ids": sorted(true_ids), "false_ids": sorted(false_ids),
              "records": records}
    suffix = "" if prefix_caching else "_nocache"
    out = Path("/results/ablations/diffusion_gemma_readout_probe_"
               f"{dataset}_k{logprobs}{suffix}.json")
    out.write_text(json.dumps(report, indent=1, default=str))
    volumes["/results"].commit()
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     indent=1, default=str), flush=True)
    print(json.dumps(report["records"][:12], indent=1, default=str), flush=True)
    return str(out)


@app.local_entrypoint()
def main(prediction: str = "", docs: int = 256, logprobs: int = 0,
         dataset: str = "imdb", prefix_caching: bool = True):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    call = probe.spawn(prediction, docs, logprobs=logprobs, dataset=dataset,
                       prefix_caching=prefix_caching)
    print(f"function call id: {call.object_id} (readout probe)", flush=True)
    print(call.get(), flush=True)
