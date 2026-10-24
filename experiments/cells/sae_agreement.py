"""Quail's layer 11 residual stream and SAE features against transformers.

Two steps, each in its own container. reference runs Qwen3.5-2B-Base
through transformers in bf16, hooks the residual stream after layer 11,
and records, for each document-plus-question prompt, the last token's
hidden state and its top 100 SAE features. quail loads the 12-layer
qwen3.5-2b-sae-bf16 spec through Quail's loader, runs the same prompts
as one-stage chunks, reads the residual rows the pipeline returns, and
records the same values.

Prediction: the last-token hidden states agree within bf16 tolerance
(median relative error under 1 percent, maximum under 3 percent), and
at least 90 percent of the prompts keep the same top 100 feature set.
The engines round their GEMMs differently, so boundary features may
swap; the comparison also reports the mean feature overlap.

    uv run modal run --detach -m experiments.cells.sae_agreement \
        2>&1 | tee /tmp/sae_agreement.log

Records: /results/ablations/qwen3.5-2b-sae-agreement-{reference,quail}.json
and the comparison in .../qwen3.5-2b-sae-agreement.json.
"""

import json

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
BUDGET = 8192
QUESTION = "\n\nWhat is this text about?"
HEAD = "DOCUMENT:\n"
TOPICS = ("a river", "an engineer", "a market", "a storm", "a violinist",
          "a bakery", "a mountain", "a teacher")
DOCUMENT_WORDS = (40, 200, 700)
SEQUENCES = 24


def _text(index: int) -> str:
    words = DOCUMENT_WORDS[index % len(DOCUMENT_WORDS)]
    topic = TOPICS[index % len(TOPICS)]
    sentence = (f"Entry {index} describes {topic} in detail, including the "
                f"place, the season, and what happened next. ")
    return (sentence * (words // 14 + 1))[: words * 6]


def _prompts(spec, tokenizer):
    """Return (document ids, question ids) per prompt, laid out as Quail's."""
    pairs = []
    for index in range(SEQUENCES):
        document = tokenizer(HEAD + _text(index),
                             add_special_tokens=False)["input_ids"]
        tail = tokenizer(QUESTION, add_special_tokens=False)["input_ids"]
        pairs.append((document, tail))
    del spec
    return pairs


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def reference() -> dict:
    """Transformers' layer 11 residual and features for each prompt."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quail.specs import MODELS

    spec = MODELS[MODEL]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name,
                                              revision=spec.revision or None)
    # device_map would need accelerate, which the image does not carry
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, revision=spec.revision or None,
        torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    sae_path = hf_hub_download(spec.sae_hf_name, spec.sae_file,
                               revision=spec.sae_revision or None)
    sae = torch.load(sae_path, map_location="cuda", weights_only=True)
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["residual"] = hidden.detach()

    layers = model.model.layers if hasattr(model, "model") else (
        model.language_model.model.layers)
    handle = layers[11].register_forward_hook(hook)
    record = {"model": MODEL, "hidden": [], "features": []}
    with torch.no_grad():
        for document, tail in _prompts(spec, tokenizer):
            ids = torch.tensor([document + tail], device="cuda")
            model(input_ids=ids)
            last = captured["residual"][0, -1].float()
            pre = last @ sae["W_enc"].T.float().cuda() + sae["b_enc"].float(
            ).cuda()
            values, index = torch.topk(pre.clamp(min=0.0), spec.sae_keep)
            record["hidden"].append(last.cpu().tolist())
            record["features"].append(sorted(
                int(feature) for feature, value in
                zip(index.tolist(), values.tolist()) if value > 0))
    handle.remove()
    results.reload()
    with open(f"/results/ablations/{MODEL}-agreement-reference.json",
              "w") as out:
        json.dump(record, out)
    results.commit()
    return {"prompts": len(record["features"])}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def quail() -> dict:
    """Quail's layer 11 residual and features, against the saved reference."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.sae import SaeEncoder, SaeFeaturesReadout
    from quail.backends.quail.executor.stages import Stage, run_stages
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost import budgets
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[MODEL]
    results.reload()
    with open(f"/results/ablations/{MODEL}-agreement-reference.json") as f:
        reference_record = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name,
                                              revision=spec.revision or None)
    prompts = _prompts(spec, tokenizer)
    loaded = load_model(spec.hf_name, spec.revision or None,
                        max_batched_tokens=BUDGET,
                        text_only=spec.vision_tower, num_layers=spec.layers)
    model_memory_bytes = torch.cuda.memory_allocated()
    full_pages, sliding = budgets.arena_pages(spec, H100_SXM, BUDGET)
    arena = KVArena(
        num_layers=spec.layers, num_pages=full_pages,
        page_tokens=budgets.PAGE_TOKENS, num_kv_heads=spec.num_kv_heads,
        d_head=spec.d_head, dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
        sliding_layers=spec.sliding_layer_set,
        sliding_window=spec.sliding_window, num_sliding_pages=sliding,
        **checkpoint_arena_args(spec, H100_SXM, BUDGET, full_pages))
    pipeline = build_pipeline(spec, loaded, arena)
    encoder = SaeEncoder.load(torch, functional, spec,
                              engine=pipeline.engine)
    hidden = spec.hidden

    class Recording(SaeFeaturesReadout):
        """Packs each answer row's features and its raw residual."""

        def submit(self, stream):
            packed = self.torch.cat(
                [self.encoder.packed_top_features(stream), stream.float()],
                dim=1)
            return self._copy(packed)

    readout = Recording(torch, encoder)
    tail = prompts[0][1]
    stage = Stage(suffixes=[[tail[-1]]], frame=list(tail[:-1]),
                  readout=readout, single=True, label="sae agreement")
    stage_answers, _, _ = run_stages(
        torch, arena, pipeline, [stage],
        [list(document) for document, _ in prompts],
        BUDGET, anchor_keys=[("sae", index) for index in range(len(prompts))],
        attention_mode="unified", default_attention="unified")
    keep = encoder.keep
    errors, overlaps, matches = [], [], 0
    for index in range(len(prompts)):
        row = np.asarray(stage_answers[0][index])[0]
        strengths = row[keep:2 * keep]
        kept = sorted(int(feature) for feature, value in
                      zip(row[:keep].astype(np.int64), strengths)
                      if value > 0)
        residual = row[2 * keep:2 * keep + hidden]
        expected = np.asarray(reference_record["hidden"][index],
                              dtype=np.float32)
        errors.append(float(np.linalg.norm(residual - expected)
                            / max(np.linalg.norm(expected), 1e-6)))
        reference_set = set(reference_record["features"][index])
        overlap = len(reference_set & set(kept)) / max(len(reference_set), 1)
        overlaps.append(overlap)
        matches += int(set(kept) == reference_set)
    record = {
        "model": MODEL, "prompts": len(prompts),
        "model_memory_bytes": model_memory_bytes,
        "hidden_relative_error_median": float(np.median(errors)),
        "hidden_relative_error_max": float(np.max(errors)),
        "top_feature_set_match_share": matches / len(prompts),
        "mean_feature_overlap": float(np.mean(overlaps)),
    }
    with open(f"/results/ablations/{MODEL}-agreement-quail.json", "w") as out:
        json.dump(record, out)
    results.commit()
    return record


@app.local_entrypoint()
def main(reuse_reference: bool = False):
    if not reuse_reference:
        print("reference:", reference.remote())
    record = quail.remote()
    print(json.dumps(record, indent=2))
