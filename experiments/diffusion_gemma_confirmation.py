r"""Confirm DiffusionGemma on the existing Modal app.

Run from the repository root and tee every line:

    uv run modal run experiments/diffusion_gemma_confirmation.py \
      --prediction "State the expected results before starting." \
      2>&1 | tee results/diffusion-gemma-confirmation.log

Three cells, each on its own H100:

- probe: load the fp8 checkpoint through Quail's loader, print the
  loaded footprint and the module facts the pipeline relies on.
- confirm: a filter and a join through a Quail session, with the
  answers of a small labeled corpus.
- reference: vLLM's own diffusion sampler on the same filter prompts,
  to compare its first answer token with Quail's. It renders the
  prompt three ways: bare text as Qwen3 gets it, inside a chat turn,
  and inside a chat turn with an empty thinking channel prefilled.

The cells write these files to the quail-results volume:

    /results/ablations/diffusion_gemma_confirmation_probe.json
    /results/ablations/diffusion_gemma_confirmation.json
    /results/ablations/diffusion_gemma_reference.json
"""

import json
import os
import time

import modal

from quail.bench.requirements import quail_b_requirement

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
MODEL = "diffusion-gemma-26b-a4b-fp8"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "vllm==0.26.0",
        "huggingface_hub[hf_transfer]",
        "transformers>=5.8.0",
        "pandas",
        "pyarrow",
        "numpy",
        "datasets>=5.0.1",
        "sqlglot>=27.0",
        "gigatoken>=0.10.0",
        quail_b_requirement(),
    )
    .env({
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    })
    .add_local_python_source("quail")
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True
)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

# A small labeled corpus: eight documents about food, eight about cars.
FOOD = [
    "The bakery sells sourdough loaves and cinnamon rolls every morning.",
    "Simmer the tomatoes with garlic and basil for a rich pasta sauce.",
    "This ramen shop is known for its slow-cooked pork broth.",
    "Fresh mangoes and papayas fill the market stalls in summer.",
    "The recipe calls for two eggs, flour, butter, and a pinch of salt.",
    "Grilled salmon with lemon and dill makes a quick weeknight dinner.",
    "The chef plated roasted vegetables beside a wedge of aged cheese.",
    "Street vendors serve tacos with pickled onions and lime.",
]
CARS = [
    "The sedan's turbocharged engine produces 250 horsepower.",
    "Rotate the tires every six thousand miles to even out wear.",
    "The dealership offers a five-year warranty on the powertrain.",
    "Electric vehicles charge overnight on a home wall connector.",
    "The mechanic replaced the brake pads and flushed the coolant.",
    "This hatchback gets forty miles per gallon on the highway.",
    "The pickup truck tows up to nine thousand pounds.",
    "Adaptive cruise control keeps a set distance from the car ahead.",
]
FILTER_TEMPLATE = "Is {0} about food or cooking?"
JOIN_TEMPLATE = "Are {0} and {1} about the same kind of thing?"
LEFT = FOOD[:2] + CARS[:2]
RIGHT = FOOD[2:4] + CARS[2:4]


def _save(name: str, value: dict) -> str:
    path = f"/results/ablations/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2)
    results_vol.commit()
    return path


def _session():
    import pyarrow as pa

    import quail
    from quail.planner.plan import EngineConfig

    session = quail.Session(EngineConfig(
        gpus=1, model=MODEL, backend="quail", device="h100-sxm",
    ))
    session.register("docs", quail.DocumentProvider.from_table(
        pa.table({
            "id": [f"f{i}" for i in range(8)] + [f"c{i}" for i in range(8)],
            "body": FOOD + CARS,
        }),
        id_col="id", identity="diffusion-gemma-confirmation-docs",
    ))
    session.register("left_docs", quail.DocumentProvider.from_table(
        pa.table({"id": [f"l{i}" for i in range(4)], "body": LEFT}),
        id_col="id", identity="diffusion-gemma-confirmation-left",
    ))
    session.register("right_docs", quail.DocumentProvider.from_table(
        pa.table({"id": [f"r{i}" for i in range(4)], "body": RIGHT}),
        id_col="id", identity="diffusion-gemma-confirmation-right",
    ))
    return session


def _run(query, gpu_count=1):
    from quail.execution.execute import execute_query
    from quail.specs import H100_USD_PER_HOUR

    result = execute_query(query)
    table = result.collect()
    report = result.report
    return {
        "rows": table.to_pylist(),
        "wall_s": report["wall_s"],
        "usd_per_query": report["wall_s"] / 3600 * gpu_count * H100_USD_PER_HOUR,
        "fresh_tokens": report.get("fresh_tokens"),
        "boot": report.get("boot"),
        "stages": report.get("stages"),
    }


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def probe(prediction: str) -> str:
    import torch

    from quail.backends.quail.executor.model import load_model
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    model = load_model(spec.hf_name)
    load_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated() - before
    layers = model.model.layers
    facts = {
        "class": type(model).__name__,
        "layers": len(layers),
        "sliding_layers": sum(1 for layer in layers if layer.self_attn.is_sliding),
        "full_layer_indices": [i for i, layer in enumerate(layers)
                               if not layer.self_attn.is_sliding],
        "head_geometry": sorted({
            (layer.self_attn.num_heads, layer.self_attn.num_kv_heads,
             layer.self_attn.head_dim) for layer in layers}),
        "moe_layers": sum(1 for layer in layers if layer.enable_moe_block),
        "sliding_window": model.model.config.sliding_window,
        "qkv_weight_dtype": str(layers[0].self_attn.qkv_proj.weight.dtype),
        "qkv_weight_shape": list(layers[0].self_attn.qkv_proj.weight.shape),
        "expert_module": type(layers[0].moe.experts).__name__,
        "answer_token_ids": list(model.quail_answer_token_ids),
        "self_conditioning_post_norm": type(
            model.self_conditioning.post_norm).__name__,
    }
    result = {
        "prediction": prediction,
        "gpu_device_name": torch.cuda.get_device_name(0),
        "load_model_s": round(load_s, 1),
        "loaded_bytes": loaded,
        "spec_w_mem_bytes": spec.w_mem_bytes,
        "facts": facts,
    }
    result["volume_path"] = _save("diffusion_gemma_confirmation_probe", result)
    return json.dumps(result, indent=2, default=str)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def confirm(prediction: str) -> str:
    import torch

    session = _session()
    started = time.time()
    filter_query = session.sql(
        "SELECT d.id FROM docs d "
        f"WHERE AI_FILTER(PROMPT('{FILTER_TEMPLATE}', d.body))"
    )
    filter_result = _run(filter_query)
    join_query = session.sql(
        "SELECT l.id, r.id FROM left_docs l JOIN right_docs r "
        f"ON AI_FILTER(PROMPT('{JOIN_TEMPLATE}', l.body, r.body), "
        "{'selectivity': 0.5})"
    )
    join_result = _run(join_query)
    kept = sorted(row["id"] for row in filter_result["rows"])
    result = {
        "prediction": prediction,
        "gpu_device_name": torch.cuda.get_device_name(0),
        "elapsed_s": time.time() - started,
        "filter": filter_result,
        "filter_expected": [f"f{i}" for i in range(8)],
        "filter_kept": kept,
        "filter_agreement": (
            sum(1 for i in range(8) if f"f{i}" in kept)
            + sum(1 for i in range(8) if f"c{i}" not in kept)) / 16,
        "join": join_result,
        "join_expected_pairs": sorted(
            [f"l{i}", f"r{j}"] for i in range(4) for j in range(4)
            if (i < 2) == (j < 2)),
    }
    result["volume_path"] = _save("diffusion_gemma_confirmation", result)
    return json.dumps(result, indent=2, default=str)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def reference(prediction: str) -> str:
    """Run vLLM's own diffusion sampler on Quail's filter prompts."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from quail.logical.prompts import (
        bind_prompt,
        render_filter_prompt_ids,
        true_false_ids,
    )
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    hf = AutoTokenizer.from_pretrained(spec.hf_name)

    def tok(text):
        return hf(text, add_special_tokens=False)["input_ids"]

    turns = {
        "bare": ("", ""),
        "turn": ("<bos><|turn>user\n", "<turn|>\n<|turn>model\n"),
        "turn+channel": spec.turn,
    }
    docs = FOOD + CARS
    prompts = {}
    for name, turn in turns.items():
        prompt = bind_prompt(FILTER_TEMPLATE, (SimpleNamespace(alias="d"),),
                             tok, turn=turn)
        prompts[name] = [render_filter_prompt_ids(prompt, tok(doc), tok)
                         for doc in docs]
    true_ids, false_ids = true_false_ids(hf)
    llm = LLM(
        model=spec.hf_name, max_num_seqs=4, gpu_memory_utilization=0.85,
        generation_config="vllm", max_model_len=4096,
        hf_overrides={"diffusion_sampler": "entropy_bound",
                      "diffusion_entropy_bound": 0.1},
        diffusion_config={"canvas_length": spec.canvas_tokens},
    )
    result = {
        "prediction": prediction,
        "true_ids": sorted(true_ids),
        "false_ids": sorted(false_ids),
        "layouts": {},
    }
    for name, ids in prompts.items():
        started = time.time()
        # the diffusion sampler owns temperature and rejects it as a
        # sampling parameter
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=row) for row in ids],
            SamplingParams(max_tokens=16))
        elapsed = time.time() - started
        answers = []
        for doc, row, output in zip(docs, ids, outputs):
            generated = list(output.outputs[0].token_ids)
            first = generated[0] if generated else None
            answers.append({
                "document": doc[:40],
                "prompt_tokens": len(row),
                "text": output.outputs[0].text,
                "tokens": generated[:6],
                "first_is_true": first in true_ids,
                "first_is_false": first in false_ids,
            })
        expected = [True] * 8 + [False] * 8
        agree = sum(
            1 for want, answer in zip(expected, answers)
            if answer["first_is_true" if want else "first_is_false"])
        result["layouts"][name] = {
            "prompt_text_example": hf.decode(ids[0]),
            "generate_s": elapsed,
            "first_token_agreement": agree / 16,
            "answers": answers,
        }
    result["volume_path"] = _save("diffusion_gemma_reference", result)
    return json.dumps(result, indent=2, default=str)


@app.local_entrypoint()
def main(prediction: str = "", runs: str = "probe,confirm,reference"):
    if not prediction:
        raise ValueError("pass --prediction before starting")
    functions = {
        "probe": lambda: probe.spawn(prediction),
        "confirm": lambda: confirm.spawn(prediction),
        "reference": lambda: reference.spawn(prediction),
    }
    selected = [name.strip() for name in runs.split(",") if name.strip()]
    unknown = set(selected) - set(functions)
    if unknown:
        raise ValueError(f"unknown runs: {sorted(unknown)}")
    calls = {name: functions[name]() for name in selected}
    for name, call in calls.items():
        print(f"function call id: {call.object_id} ({name})", flush=True)
    for name, call in calls.items():
        print(f"waiting for {name}", flush=True)
        try:
            print(call.get(), flush=True)
        except Exception as error:  # noqa: BLE001 - report every cell
            print(f"{name} failed: {error!r}", flush=True)
