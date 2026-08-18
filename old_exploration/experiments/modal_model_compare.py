"""Quick accuracy comparison: Qwen3-4B-FP8 vs Qwen3.5-4B on the
planted-flag filter task. Same 10k IMDB docs, 1 filter, constrained
YES/NO sampler. Reports wrong-answer count and throughput for each.

Run:
    modal run experiments/modal_model_compare.py
"""

import modal

from workload import IMAGE_BASE, hf_cache, results_vol

app = modal.App("quail-model-compare")

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("workload")
)

MODELS = [
    ("Qwen/Qwen3-4B-FP8", "fp8"),
    ("Qwen/Qwen3.5-4B", "bf16"),
]


@app.function(image=image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def compare():
    import gc
    import json
    import time

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from workload import (FLAG_SEED, N_FILTERS, SELECTIVITY, WORKLOAD_SEED,
                          build_pool, flags_line, question)

    n_docs = 10_000
    rng = np.random.default_rng(FLAG_SEED + 100)
    flags = (rng.random((n_docs, N_FILTERS))
             < np.array(SELECTIVITY)[None, :]).astype(int)
    docs = build_pool(n_docs)
    bodies = [d + flags_line(f) for d, f in zip(docs, flags)]
    expected = [int(flags[i][0]) for i in range(n_docs)]

    results = []

    for model_name, dtype_label in MODELS:
        print(f"\n[compare] loading {model_name} ({dtype_label})",
              flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
        q_ids = tok(question(1), add_special_tokens=False)["input_ids"]
        prompts = [{"prompt_token_ids": body_ids[i] + q_ids}
                   for i in range(n_docs)]
        total_tokens = sum(len(p["prompt_token_ids"]) for p in prompts)

        yes_ids, no_ids = set(), set()
        for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                yes_ids.add(ids[0])
        for w in ("NO", " NO", "No", " No", "N", " N"):
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                no_ids.add(ids[0])
        allowed = sorted(yes_ids | no_ids)
        print(f"[compare] yes_ids={sorted(yes_ids)}, no_ids={sorted(no_ids)}",
              flush=True)

        kv_dtype = "fp8" if dtype_label == "fp8" else "auto"
        max_seqs = 2048 if "3.5" in model_name else 4096
        llm = LLM(model=model_name,
                   kv_cache_dtype=kv_dtype,
                   max_model_len=4608,
                   max_num_seqs=max_seqs,
                   max_num_batched_tokens=25305,
                   gpu_memory_utilization=0.88,
                   enable_prefix_caching=False,
                   disable_log_stats=True)
        sp = SamplingParams(temperature=0.0, max_tokens=1,
                            allowed_token_ids=allowed)

        # warmup
        _ = llm.generate(prompts[:100], sp, use_tqdm=False)

        started = time.time()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        wall = time.time() - started

        def is_yes(out):
            t = out.outputs[0].text.upper()
            iy = t.find("YES")
            if iy < 0:
                return 0
            ino = t.find("NO")
            return 1 if ino < 0 or iy < ino else 0

        predictions = [is_yes(o) for o in outputs]
        wrong = sum(a != b for a, b in zip(predictions, expected))
        rate = total_tokens / wall

        row = {
            "model": model_name,
            "dtype": dtype_label,
            "n_docs": n_docs,
            "total_tokens": total_tokens,
            "wall": round(wall, 2),
            "rate": round(rate, 1),
            "wrong": wrong,
            "accuracy": round(1 - wrong / n_docs, 4),
        }
        results.append(row)
        print(f"[compare] {json.dumps(row)}", flush=True)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    outpath = "/results/model_compare.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"\nsaved {outpath}")
    for r in results:
        print(f"  {r['model']:<30} wrong={r['wrong']:>5}/{n_docs}  "
              f"({r['accuracy']:.1%})  {r['rate']:,.0f} tok/s")
    return json.dumps(results, indent=2)
