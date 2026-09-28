"""Compare Quail's FP8 forward pass with the bf16 Qwen3 4B, layer by layer.

Quail's AI.CLASSIFY label log probabilities differ from stock vLLM's on
the same Qwen3 4B FP8 checkpoint and token ids by a median of 0.86 nats
on a review's worst label (/results/ablations/classify-scores.json).
Both engines approximate the bf16 model the FP8 checkpoint was
quantized from. This cell runs the first IMDB-11 probe reviews through
Quail's forward pass and through Hugging Face transformers' bf16
Qwen3-4B, and records, for every layer, the relative distance between
the two residual streams; then the label log probabilities of Quail,
of the bf16 model, and of vLLM (from the earlier scores run).

    uv run modal run --detach experiments/cells/forward_check.py::layers \
      --vllm-call fc-01M3MB7YEMN8HXN8E08DH80MKX 2>&1 | tee forward-check.log

The summary goes to /results/ablations/forward-check.json.
"""

import json
from pathlib import Path

import modal

from quail.bench import labeling

app = labeling.app
RESULT_PATH = Path("/results/ablations/forward-check.json")
SAME_MODEL_PATH = Path("/results/ablations/classify-quail-same-model.json")
BF16_REPO = "Qwen/Qwen3-4B"
DOCUMENTS = 8
PREDICTION_TEXT = (
    "If Quail's forward pass is correct, its residual stream drifts from "
    "the bf16 model smoothly with depth, and its label log probabilities "
    "are about as far from the bf16 model's as vLLM's are. A jump at one "
    "layer, or labels much farther from bf16 than vLLM's, is a bug."
)


def _volumes() -> dict:
    return {"/root/.cache/huggingface": labeling.hf_cache,
            "/root/.cache/kernels": labeling.kernel_cache,
            "/results": labeling.results_vol}


def _reviews(tokenizer):
    """Return the probe reviews' ids, prompt token ids, and label token ids.

    The reviews are the first IMDB-11 reviews on which Quail and vLLM
    chose different labels, in the order the scores run used.
    """
    import quail_b
    from quail.logical import ColumnRef, bind_classify_prompt, label_text

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    query = quail_b.get_query("IMDB-11")
    (operator,) = query._info.classifies
    checks = json.loads(SAME_MODEL_PATH.read_text())["checks"]
    ids = [item["id"] for item in
           checks[f"IMDB-11:{operator.id}"]["disagreements"]][:DOCUMENTS]
    table = quail_b.load_table("reviews", scale_factor=0.1)
    texts = dict(zip(table.column("id").to_pylist(),
                     table.column("body").to_pylist()))
    prompt = bind_classify_prompt(
        operator.prompt, (ColumnRef("r", "reviews", "body"),),
        operator.labels, operator.descriptions, encode)
    prompts = [list(prompt.preamble_token_ids) + encode(texts[doc])
               + list(prompt.tail_token_ids) for doc in ids]
    labels = [tuple(encode(label_text(label))) for label in operator.labels]
    return ids, prompts, labels


@app.function(image=labeling.image, gpu="H100!", memory=98304, timeout=3600,
              volumes=_volumes())
def compare_layers() -> dict:
    """Run the probe reviews through Quail and bf16 Qwen3 4B; return distances."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import run_join
    from quail.backends.quail.executor.model import full_output_head, load_model
    from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline
    from quail.backends.quail.executor.readout import AsyncLabelLogprobs
    from quail.cost import budgets
    from quail.execution.labels import label_trie, trie_targets
    from quail.specs import DEVICES, MODELS

    labeling._mount()
    spec = MODELS["qwen3-4b-fp8"]
    device = DEVICES["h100-sxm"]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name,
                                              revision=spec.revision)
    model = load_model(spec.hf_name, revision=spec.revision)
    head = full_output_head(model)
    reference = AutoModelForCausalLM.from_pretrained(
        BF16_REPO, torch_dtype=torch.bfloat16).to("cuda").eval()
    chunk = budgets.chunk_budget(spec, device)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=budgets.arena_tokens(spec, device, chunk)
                    // budgets.PAGE_TOKENS // 4,
                    page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16)

    streams = []

    class Recording(Qwen3Pipeline):
        """Qwen3's forward pass, keeping the residual stream after each layer."""

        def forward_chunk(self, chunk):
            engine = self.engine
            chunk.meta["layer"] = 0
            hidden = self.embed(chunk.input_ids)
            residual = None
            layers = []
            for layer in self.layers:
                attn = layer.self_attn
                if residual is None:
                    residual = hidden
                    q_in, q_scale = engine.norm_quant(hidden,
                                                      layer.input_layernorm)
                else:
                    q_in, q_scale = engine.norm_quant(
                        hidden, layer.input_layernorm, residual)
                qkv = engine.gemm(q_in, q_scale, attn.qkv_proj)
                q, k = engine.qk_norm_rope(qkv, chunk.positions, attn)
                v = qkv[:, (engine.num_q_heads + engine.num_kv_heads)
                        * engine.head_dim:]
                o_in, o_scale = engine.attention(q, k, v, chunk)
                hidden = engine.gemm(o_in, o_scale, attn.o_proj)
                g_in, g_scale = engine.norm_quant(
                    hidden, layer.post_attention_layernorm, residual)
                gate_up = engine.gemm(g_in, g_scale, layer.mlp.gate_up_proj)
                d_in, d_scale = engine.activation_quant(gate_up)
                hidden = engine.gemm(d_in, d_scale, layer.mlp.down_proj)
                layers.append((residual + hidden).float())
            streams.append((chunk.positions.clone(), layers))
            final = chunk.final_indices
            normed, _ = engine.fused_add_rms_norm(
                hidden.index_select(0, final), residual.index_select(0, final),
                self.final_norm)
            return normed

    pipeline = Recording(model, arena, spec=spec)
    ids, prompts, labels = _reviews(tokenizer)
    targets = trie_targets(label_trie(labels))
    readout = AsyncLabelLogprobs(torch, torch.nn.functional, head, targets)
    label_columns = [targets.index(tokens[0]) for tokens in labels]

    out = {"ids": ids, "layers": [], "quail": [], "bf16": []}
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            streams.clear()
            answers, _, _ = run_join(
                torch, arena, pipeline, readout, [prompt[:-1]],
                [[[prompt[-1]]]], chunk, anchor_keys=[("check", index)],
                attention_mode="unified")
            quail = answers[0][0][0]
            out["quail"].append([float(quail[c]) for c in label_columns])
            result = reference(torch.tensor([prompt], device="cuda"),
                               output_hidden_states=True)
            logprobs = torch.log_softmax(result.logits[0, -1].float(), dim=0)
            out["bf16"].append([float(logprobs[ids[0]]) for ids in labels])
            # rows by position; a chunk holds this one prompt
            rows = {}
            for positions, layers in streams:
                for row, position in enumerate(positions.tolist()):
                    rows[position] = [layer[row] for layer in layers]
            ordered = [rows[position] for position in range(len(prompt))]
            distances = []
            for layer in range(len(ordered[0])):
                mine = torch.stack([row[layer] for row in ordered])
                theirs = result.hidden_states[layer + 1][0].float()
                if layer + 1 == len(result.hidden_states) - 1:
                    break    # transformers' last entry is after the final norm
                distances.append(float(
                    ((mine - theirs).norm(dim=1) / theirs.norm(dim=1)).median()))
            out["layers"].append(distances)
    return out


@app.function(image=labeling.publish_image, memory=8192, timeout=600,
              volumes=_volumes())
def summarize(quail_call: str, vllm_call: str) -> dict:
    """Compare the label log probabilities and save the summary."""
    import numpy as np

    layers = modal.FunctionCall.from_id(quail_call).get()
    vllm = modal.FunctionCall.from_id(vllm_call).get()["IMDB-11"]
    count = len(layers["ids"])
    assert vllm["ids"][:count] == layers["ids"]
    bf16 = np.asarray(layers["bf16"])
    quail = np.asarray(layers["quail"])
    reference = np.asarray(vllm["vllm"][:count])
    per_layer = np.asarray(layers["layers"])
    summary = {
        "prediction": PREDICTION_TEXT,
        "function_calls": {"layers": quail_call, "vllm": vllm_call},
        "documents": count,
        "relative_distance_by_layer": np.median(per_layer, axis=0).tolist(),
        "label_logprob_abs_difference": {
            "quail_vs_bf16": float(np.median(np.abs(quail - bf16))),
            "vllm_vs_bf16": float(np.median(np.abs(reference - bf16))),
            "quail_vs_vllm": float(np.median(np.abs(quail - reference))),
        },
        "label_logprobs": {"bf16": bf16.tolist(), "quail": quail.tolist(),
                           "vllm": reference.tolist()},
    }
    labeling._atomic_json(RESULT_PATH, summary)
    labeling.results_vol.commit()
    return summary


@app.local_entrypoint()
def layers(vllm_call: str):
    """Run the layer comparison and summarize it against vLLM's scores."""
    print(PREDICTION_TEXT, flush=True)
    call = compare_layers.spawn()
    print(f"[forward-check] layers function call id: {call.object_id}",
          flush=True)
    call.get()
    summary_call = summarize.spawn(call.object_id, vllm_call)
    print(f"[forward-check] summary function call id: "
          f"{summary_call.object_id}", flush=True)
    summary = summary_call.get()
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "label_logprobs"}, indent=2), flush=True)
    print(f"[forward-check] saved {RESULT_PATH}", flush=True)
