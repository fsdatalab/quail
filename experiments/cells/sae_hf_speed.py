"""Time WRENCH-9's SAE features in Hugging Face transformers.

No serving engine (vLLM, SGLang) exposes the layer 11 residual stream
that the Qwen-Scope SAE reads, so the usual way to compute these
features is transformers with a forward hook. This cell runs the
WRENCH-9 prompts at scale factor 0.1 (3,454 steps) through
Qwen3.5-2B-Base in bf16, truncated to layers 0 to 11 as Quail's spec
is, in length-sorted batches padded to a token budget, hooks the
residual after layer 11, and encodes each prompt's last token with the
SAE, keeping the top 100 features. The timed region starts after the
model, the SAE, and the tokenized prompts are on the GPU.

Prediction: 5 to 20 times Quail's 9.184 s for the same query
(/results/ablations/sae-queries-sf01-20261025T005345Z.json), so 46 to
184 s.

    uv run modal run --detach -m experiments.cells.sae_hf_speed \
        2>&1 | tee /tmp/sae_hf_speed.log

Records: /results/ablations/sae-hf-speed-wrench9-<timestamp>.json.
"""

import json
from datetime import datetime, timezone

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image().add_local_python_source("experiments")
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}

MODEL = "qwen3.5-2b-sae-bf16"
QUERY = "WRENCH-9"
SCALE_FACTOR = 0.1
SAE_LAYER = 11
BATCH_TOKENS = (8192, 32768)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run(output_path: str) -> dict:
    """Seconds for WRENCH-9's features in transformers at each batch budget."""
    import time

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quail.specs import H100_USD_PER_HOUR, MODELS
    from quail_b.benchmark import load_benchmark
    from quail_b.rendering import render_sae_features_prompt

    suite = load_benchmark([QUERY], scale_factor=SCALE_FACTOR, accuracy=False)
    spec = suite.queries[0]
    operator = spec.info.sae_features[0]
    relation = next(r for r in spec.info.relations
                    if r.alias == operator.relation)
    documents = suite.tables[relation.table].column(
        relation.text_column).to_pylist()
    prompts = [render_sae_features_prompt(operator.prompt, document)
               for document in documents]

    model_spec = MODELS[MODEL]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.hf_name, revision=model_spec.revision or None)
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.hf_name, revision=model_spec.revision or None,
        torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    decoder = model.model if hasattr(model.model, "layers") else (
        model.model.language_model)
    decoder.layers = decoder.layers[:SAE_LAYER + 1]
    sae_path = hf_hub_download(model_spec.sae.hf_name, model_spec.sae.file,
                               revision=model_spec.sae.revision or None)
    sae = torch.load(sae_path, map_location="cuda", weights_only=True)
    encoder = sae["W_enc"].to(torch.bfloat16)
    bias = sae["b_enc"].to(torch.bfloat16)
    captured = {}

    def hook(_module, _inputs, output):
        captured["residual"] = output[0] if isinstance(output, tuple) \
            else output

    handle = decoder.layers[SAE_LAYER].register_forward_hook(hook)
    token_lists = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    input_tokens = sum(len(tokens) for tokens in token_lists)
    pad_id = tokenizer.pad_token_id or 0

    def batches(budget):
        batch = []
        for index in order:
            width = len(token_lists[index])
            if batch and width * (len(batch) + 1) > budget:
                yield batch
                batch = []
            batch.append(index)
        if batch:
            yield batch

    record = {"model": MODEL, "query": QUERY, "scale_factor": SCALE_FACTOR,
              "corpus_id": suite.corpus_id, "prompts": len(prompts),
              "input_tokens": input_tokens, "engine": "transformers",
              "runs": []}
    with torch.no_grad():
        for budget in BATCH_TOKENS:
            torch.cuda.synchronize()
            start = time.perf_counter()
            kept = 0
            for batch in batches(budget):
                width = max(len(token_lists[i]) for i in batch)
                ids = torch.full((len(batch), width), pad_id, device="cuda")
                mask = torch.zeros((len(batch), width), dtype=torch.long,
                                   device="cuda")
                for row, index in enumerate(batch):
                    tokens = token_lists[index]
                    ids[row, :len(tokens)] = torch.tensor(tokens)
                    mask[row, :len(tokens)] = 1
                decoder(input_ids=ids, attention_mask=mask)
                last = captured["residual"][
                    torch.arange(len(batch), device="cuda"),
                    mask.sum(dim=1) - 1]
                pre = (last @ encoder.T + bias).clamp(min=0.0)
                values, _features = torch.topk(pre, model_spec.sae.keep)
                kept += int((values > 0).sum())
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            run_record = {
                "batch_tokens": budget, "seconds": seconds,
                "input_tokens_per_second": input_tokens / seconds,
                "cost_usd": seconds / 3600 * H100_USD_PER_HOUR,
                "features_kept": kept}
            record["runs"].append(run_record)
            print(json.dumps(run_record), flush=True)
    handle.remove()
    results.reload()
    with open(output_path, "w") as out:
        json.dump(record, out, indent=2)
    results.commit()
    return record


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = f"/results/ablations/sae-hf-speed-wrench9-{stamp}.json"
    call = run.spawn(output_path)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"results: {output_path}", flush=True)
    print(json.dumps(call.get(), indent=2))
