"""Compare Quail's fused Gemma add-and-norm with vLLM's norm modules.

Loads a hybrid checkpoint and runs the input norm of the first layer,
the post-attention norm of a middle layer, and the final norm, each as
vLLM's module (the path the pipeline used before) and as the engine's
one-kernel version, on bf16 rows with and without large outlier
channels. Reports the largest absolute difference, the share of
elements that match exactly, and the same for the residual.

Prediction: the new residual matches exactly (both add in float32 and
round once); the normalized rows match exactly on at least 99 percent
of elements and differ by at most one bf16 step elsewhere.

    uv run modal run experiments/cells/hybrid_norm_unit.py \
        --model qwen3.5-4b-bf16 2>&1 | tee /tmp/hybrid_norm_unit.log

Record: /results/ablations/<model>-norm-unit.json.
"""

import json
import os

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
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
ROWS = 4096


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def check(model: str) -> dict:
    """Return per-norm agreement between the module and the kernel."""
    from contextlib import nullcontext

    import torch
    from transformers import AutoTokenizer
    from vllm.config import set_current_vllm_config

    from quail.backends.quail.executor.attention import Engine
    from quail.backends.quail.executor.model import load_model
    from quail.logical import true_false_ids
    from quail.specs import MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    true_ids, false_ids = true_false_ids(tokenizer)
    loaded = load_model(spec.hf_name, spec.revision or None,
                        answer_token_ids=true_ids | false_ids,
                        max_batched_tokens=8192,
                        text_only=spec.vision_tower)
    text = getattr(loaded, "language_model", loaded)
    layers = text.model.layers
    config = getattr(loaded, "quail_vllm_config", None)
    # gemma_add_rms_norm reads only the engine's torch module
    engine = Engine.__new__(Engine)
    engine.torch = torch
    middle = len(layers) // 2
    norms = {"input_layernorm[0]": layers[0].input_layernorm,
             f"post_attention_layernorm[{middle}]":
                 layers[middle].post_attention_layernorm,
             "final_norm": text.model.norm}
    generator = torch.Generator(device="cuda").manual_seed(0)
    width = spec.hidden
    report = {}
    context = nullcontext() if config is None else set_current_vllm_config(
        config)
    with torch.inference_mode(), context:
        for name, norm in norms.items():
            for outliers in (False, True):
                def rows():
                    values = torch.randn((ROWS, width), device="cuda",
                                         generator=generator)
                    if outliers:
                        values[:, :8] *= 60.0
                    return values.to(torch.bfloat16)
                x, residual = rows(), rows()
                expected, expected_residual = norm(x.clone(),
                                                   residual.clone())
                got, got_residual = engine.gemma_add_rms_norm(
                    x, residual, norm.weight, norm.variance_epsilon)
                difference = (got.float() - expected.float()).abs()
                report[f"{name} outliers={outliers}"] = {
                    "max_abs_diff": difference.max().item(),
                    "exact_share": (got == expected).float().mean().item(),
                    "residual_exact_share": (
                        got_residual == expected_residual).float()
                        .mean().item(),
                    "nan": int(torch.isnan(got).sum().item()),
                }
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{model}-norm-unit.json", "w") as out:
        json.dump(report, out, indent=1)
    results.commit()
    return report


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16"):
    call = check.spawn(model)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1), flush=True)
