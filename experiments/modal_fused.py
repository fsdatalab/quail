"""Correctness gate for the fused shared-prefix attention patch.

One phase, "fusegate". It boots the 4B engine used across
experiments/modal_scale.py (Qwen3-4B-FP8, FP8 KV) with the FlashInfer
backend pinned and query quantization off, so the fused and unfused
paths read the same KV bytes with the same query dtype. For each of
three document lengths (about 300, 3,000, and 15,000 tokens, built by
concatenating pool reviews the way longdoc_run does) it runs 6
classifier questions over each of 50 documents twice: once with
ordinary attention, once with the fused cascade patch active. The
document KV is prefilled first, so the 6 question tails of a document
share its cached pages, which is the exact shape the fused kernel
groups on.

The gate passes only if every answer token id matches bit for bit
between the two runs, and only if the fused run actually planned
cascade steps (otherwise the comparison proved nothing). Walls are
printed per cell; the fused-versus-unfused question walls are the
speed signal.

Run with:
  modal run experiments/modal_fused.py --phase fusegate
  modal run experiments/modal_fused.py --phase fusegate --n-docs 10
"""

import json
import os
import time

import modal

app = modal.App("docengine-fused")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    # Pinned: the fused patch replaces a non-public builder method, so a
    # silent version jump on image rebuild could break it mid-study.
    # Every recorded result is stamped with this version.
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "yappi")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("docengine")
)
hf_cache = modal.Volume.from_name("docengine-hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen3-4B-FP8"
WORKLOAD_SEED = 20260731
FLAG_SEED = 424242
N_QUESTIONS = 6
LENGTHS = (300, 3_000, 15_000)


def _build_pool(n_docs):
    """Reproduce the repo's seeded 10k sample, then take its first n_docs."""
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "test"):
        path = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset")
        frames.append(pd.read_parquet(path)["text"])
    pool = list(frames[0]) + list(frames[1])
    rng = np.random.default_rng(WORKLOAD_SEED)
    idx = sorted(rng.choice(len(pool), size=10_000, replace=False))
    return [pool[i] for i in idx[:n_docs]]


def _flags_line(flags):
    return "\n\n[FLAGS] " + " ".join(
        f"FLAG_{j+1}={'YES' if f else 'NO'}" for j, f in enumerate(flags))


def _question(j):
    return (f"\n\nExample: if the line said [FLAGS] FLAG_9=NO, then FLAG_9 "
            f"has value NO.\nInstruction: output only the value of FLAG_{j} "
            f"from the [FLAGS] line above.\nFLAG_{j}=")


@app.function(image=image, gpu="H100!", timeout=5400,
              volumes={"/root/.cache/huggingface": hf_cache})
def fusegate_run(n_docs: int = 50) -> dict:
    # The patch lives in this process, so the engine must too.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from docengine.engineext.fused import (
        fused_stats,
        install_fused_patch,
        reset_fused_stats,
        set_fused_enabled,
    )

    install_fused_patch()
    set_fused_enabled(False)

    tok = AutoTokenizer.from_pretrained(MODEL)
    pool = _build_pool(10_000)
    pool_ids = tok(pool, add_special_tokens=False)["input_ids"]
    sep_ids = tok("\n\n", add_special_tokens=False)["input_ids"]

    cursor = 0

    def build_doc_ids(target):
        """Concatenate pool reviews like longdoc_run, then cut to the
        target so every cell's lengths are exact."""
        nonlocal cursor
        ids = []
        while len(ids) < target:
            if ids:
                ids.extend(sep_ids)
            ids.extend(pool_ids[cursor % len(pool_ids)])
            cursor += 1
        return ids[:target]

    llm = LLM(model=MODEL, kv_cache_dtype="fp8", max_model_len=16_384,
              gpu_memory_utilization=0.92, enable_prefix_caching=True,
              enforce_eager=True, disable_log_stats=True,
              max_num_batched_tokens=16_384, max_num_seqs=1_024,
              # Query quantization would give the unfused prefill an fp8
              # query while the cascade path always runs the model-dtype
              # query; identical answers require identical query dtype.
              attention_config={
                  "backend": "FLASHINFER",
                  "disable_flashinfer_q_quantization": True,
              })
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(N_QUESTIONS)]

    cells = []
    failures = []
    for target in LENGTHS:
        docs = [build_doc_ids(target) for _ in range(n_docs)]
        rng = np.random.default_rng(FLAG_SEED + target)
        flags = (rng.random((n_docs, N_QUESTIONS)) < 0.7).astype(int)
        bodies = [
            doc + tok(_flags_line(row),
                      add_special_tokens=False)["input_ids"]
            for doc, row in zip(docs, flags)
        ]
        prompts = [{"prompt_token_ids": body + q_ids[j]}
                   for body in bodies for j in range(N_QUESTIONS)]
        warm = [{"prompt_token_ids": body} for body in bodies]

        answers = {}
        walls = {}
        stats = {}
        for mode in ("unfused", "fused"):
            llm.reset_prefix_cache()
            set_fused_enabled(False)
            reset_fused_stats()
            t0 = time.time()
            llm.generate(warm, sp, use_tqdm=False)
            warm_wall = time.time() - t0
            set_fused_enabled(mode == "fused")
            t0 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            walls[mode] = dict(warm=warm_wall,
                               questions=time.time() - t0)
            set_fused_enabled(False)
            answers[mode] = [int(o.outputs[0].token_ids[0]) for o in outs]
            stats[mode] = fused_stats()

        flips = [i for i, (a, b) in enumerate(zip(answers["unfused"],
                                                  answers["fused"]))
                 if a != b]
        agree = {
            mode: sum(
                1 for i, token in enumerate(tokens)
                if tok.decode([token]).strip().upper().startswith("Y")
                == bool(flags[i // N_QUESTIONS][i % N_QUESTIONS])
            ) / len(tokens)
            for mode, tokens in answers.items()
        }
        cell = dict(target=target, n_docs=n_docs,
                    questions=N_QUESTIONS,
                    answers=n_docs * N_QUESTIONS,
                    walls=walls, stats=stats, flips=len(flips),
                    flip_indices=flips[:20], agreement=agree)
        cells.append(cell)
        print(f"[fusegate] {n_docs}x{target}: unfused warm "
              f"{walls['unfused']['warm']:.2f}s questions "
              f"{walls['unfused']['questions']:.2f}s | fused warm "
              f"{walls['fused']['warm']:.2f}s questions "
              f"{walls['fused']['questions']:.2f}s | cascade steps "
              f"{stats['fused']['cascade_steps']}, grouped requests "
              f"{stats['fused']['grouped_requests']}, fallback steps "
              f"{stats['fused']['fallback_steps']}, flips {len(flips)}",
              flush=True)
        if flips:
            failures.append(f"{target}: {len(flips)} answer tokens differ "
                            f"(first at {flips[0]})")
        if stats["unfused"]["cascade_steps"]:
            failures.append(f"{target}: unfused run planned cascade steps")
        if not stats["fused"]["cascade_steps"]:
            failures.append(f"{target}: fused run never planned a cascade "
                            f"step, so the comparison proved nothing")

    try:
        llm.shutdown()
    except Exception:
        pass
    import vllm
    result = dict(model=MODEL, phase="fusegate", n_docs=n_docs,
                  vllm_version=vllm.__version__,
                  kv_cache_dtype="fp8", cells=cells,
                  passed=not failures, failures=failures)
    print(f"[fusegate] {'PASS' if not failures else 'FAIL'}", flush=True)
    return result


@app.local_entrypoint()
def main(phase: str = "fusegate", n_docs: int = 0, out: str = ""):
    if phase != "fusegate":
        raise SystemExit(f"unknown phase {phase}")
    data = fusegate_run.remote(n_docs or 50)
    path = out or "results/engine/fusegate.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"saved {path}")
    # Saved first so a failing gate still leaves its evidence on disk.
    assert data["passed"], "; ".join(data["failures"])
