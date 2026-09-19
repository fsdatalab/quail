"""What the vLLM baseline's logprob readout sees for DiffusionGemma.

    uv run modal run experiments/diffusion_gemma_readout_probe.py \
      --prediction "..." 2>&1 | tee results/<stamp>-readout-probe.log

Boots the request backend's own vLLM engine for the one-row-canvas
model on IMDB reviews with the F1 filter prompt, exactly as the
benchmark does, and records for each review whether logprobs came
back, how the TRUE and FALSE entries are decoded, what true_bit
reads, and what Quail answered for the same review in the run named
by --quail-run. Writes /results/ablations/diffusion_gemma_readout_probe.json.
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


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def probe(prediction: str, n_docs: int = 256,
          quail_run: str = "20260919T193405Z-7f4d1afc") -> str:
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from pathlib import Path

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from quail.backends.request_scheduling import _ranked_answer, true_bit
    from quail.backends.vllm import VLLMEngine
    from quail.logical import bind_prompt, render_filter_prompt_ids, true_false_ids
    from quail.specs import MODELS
    from quail_b.prompts import F1

    spec = MODELS[MODEL]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    true_ids, false_ids = true_false_ids(tokenizer)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    reviews = pq.read_table("/results/quailb_data/sf0.1/reviews.parquet")
    ids = reviews["id"].to_pylist()[:n_docs]
    bodies = reviews["body"].to_pylist()[:n_docs]
    prompt = bind_prompt(F1, ("body",), tok, turn=spec.turn)
    prompts = [dict(prompt_token_ids=render_filter_prompt_ids(prompt, tok(b), tok))
               for b in bodies]
    state, boot = VLLMEngine().boot(spec, sorted(set(true_ids) | set(false_ids)))
    client, sampling = state["client"], state["sampling_params"]
    outputs = client.generate(prompts, sampling, use_tqdm=False)
    quail_answers = {}
    table = Path(f"/results/benchmarks/quailb/{quail_run}/quail/imdb/IMDB-1/"
                 "filters-0.parquet")
    if table.exists():
        for row in pq.read_table(table).to_pylist():
            quail_answers[row["r"]] = bool(row["answer"])
    records = []
    counts = {"no_logprobs": 0, "neither_word": 0, "answer_ids_in_top": 0,
              "same_as_quail": 0, "compared": 0}
    decoded_forms = {}
    for doc_id, output in zip(ids, outputs):
        completion = output.outputs[0]
        rows = getattr(completion, "logprobs", None)
        record = {"id": doc_id, "text": completion.text,
                  "sampled": list(completion.token_ids),
                  "read": true_bit(output, set(true_ids)),
                  "quail": quail_answers.get(doc_id)}
        if not rows:
            counts["no_logprobs"] += 1
            record["entries"] = None
        else:
            first = rows[0] or {}
            entries = [(int(t), lp.logprob, getattr(lp, "decoded_token", None))
                       for t, lp in first.items()]
            answer_entries = [e for e in entries if e[0] in true_ids | false_ids]
            record["answer_entries"] = answer_entries
            record["ranked"] = _ranked_answer(rows)
            if answer_entries:
                counts["answer_ids_in_top"] += 1
                for t, _, decoded in answer_entries:
                    decoded_forms[repr(decoded)] = decoded_forms.get(repr(decoded), 0) + 1
            if record["ranked"] is None:
                counts["neither_word"] += 1
        if record["quail"] is not None:
            counts["compared"] += 1
            counts["same_as_quail"] += int(record["quail"] == bool(record["read"]))
        records.append(record)
    report = {"prediction": prediction, "n_docs": n_docs, "boot": boot,
              "sampling": str(sampling), "capacity": state["capacity"],
              "counts": counts, "decoded_forms": decoded_forms,
              "true_ids": sorted(true_ids), "false_ids": sorted(false_ids),
              "records": records[:40]}
    out = Path("/results/ablations/diffusion_gemma_readout_probe.json")
    out.write_text(json.dumps(report, indent=1, default=str))
    volumes["/results"].commit()
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     indent=1, default=str), flush=True)
    print(json.dumps(report["records"][:12], indent=1, default=str), flush=True)
    return str(out)


@app.local_entrypoint()
def main(prediction: str = "", docs: int = 256):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    call = probe.spawn(prediction, docs)
    print(f"function call id: {call.object_id} (readout probe)", flush=True)
    print(call.get(), flush=True)
