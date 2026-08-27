"""Forward-pass parity test: compare the packed executor's answers against stock vLLM on the same pairs."""

import json
import os

import modal

from corpus import MODEL

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "corpus", "baselines")
)

# House rule: attach to the existing app, never a fresh one.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})


@app.function(timeout=2400, **GPU_KW)
def accuracy(n_reports: int = 4, n_terms: int = 64) -> str:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from corpus import biodex_sample
    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import Answerer, pack_chunk, true_false_ids
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = biodex_sample(tokenizer, n_reports=n_reports)
    prefixes = data["prefixes"]
    suffixes = data["suffixes"][:n_terms]
    pairs = [(r, t) for r in range(n_reports) for t in range(n_terms)]

    # ---- stock vLLM: one request per pair, constrained TRUE/FALSE
    true, false = true_false_ids(tokenizer)
    llm = LLM(model=MODEL, max_num_batched_tokens=25_305,
              max_num_seqs=256, gpu_memory_utilization=0.45,
              kv_cache_dtype="bfloat16",
              enable_prefix_caching=False, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1,
                              allowed_token_ids=sorted(true | false),
                              logprobs=20)
    prompts = [prefixes[r] + suffixes[t] for r, t in pairs]
    outs = llm.generate(prompts, sampling)
    stock_answers = []
    stock_logps = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        true_lp = max((lp[t].logprob for t in true if t in lp),
                      default=-float("inf"))
        false_lp = max((lp[t].logprob for t in false if t in lp),
                       default=-float("inf"))
        stock_answers.append(int(true_lp > false_lp))
        stock_logps.append((true_lp, false_lp))
    del llm
    torch.cuda.empty_cache()

    # ---- the packed executor: same pairs, one chunk per report
    model = load_model(MODEL)
    spec = QWEN3_4B_FP8
    chunk = budgets.chunk_budget(spec, H100_SXM)
    arena_tok = budgets.arena_tokens(spec, H100_SXM, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model, arena,
                        attention_mode="merge_quant")
    answerer = Answerer(torch, F, model, tokenizer)

    packed_answers = []
    packed_margins = []
    with torch.inference_mode():
        for r in range(n_reports):
            key = f"r{r}"
            arena.alloc(key, len(prefixes[r]))
            chunk_d = pack_chunk(
                torch, arena,
                [dict(key=key, prefix=prefixes[r],
                      f=len(prefixes[r]), suffixes=suffixes)],
                attention_mode="merge_quant")
            normed = pipeline.forward_chunk(chunk_d)
            packed_answers.extend(answerer(normed))
            packed_margins.extend(answerer.margins(normed))
            arena.free_key(key)

    # ---- compare
    disagree = sum(a != b for a, b in zip(packed_answers, stock_answers))
    # hidden gap: packed final hidden vs stock's is not directly
    # comparable (stock does not expose hidden states), so the
    # numerical check is the answer margin: packed true-minus-false
    # against stock's logprob difference. Sign agreement is the gate.
    sign_agree = sum(
        (m > 0) == (s[0] > s[1])
        for m, s in zip(packed_margins, stock_logps))
    result = dict(
        cell="forward_accuracy", n_reports=n_reports, n_terms=n_terms,
        pairs=len(pairs),
        prediction=("0 answer disagreements; packed margin sign "
                    "matches stock logprob sign on every pair"),
        packed_true=int(sum(packed_answers)),
        stock_true=int(sum(stock_answers)),
        answer_disagreements=int(disagree),
        margin_sign_agreements=int(sign_agree),
        margin_sign_pairs=len(pairs),
    )
    result["pass"] = (disagree == 0 and sign_agree == len(pairs))
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/accuracy", exist_ok=True)
    with open("/results/accuracy/forward_accuracy.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


@app.local_entrypoint()
def run(n_reports: int = 4, n_terms: int = 64,
        out: str = "results/forward_accuracy.json"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(json.loads(accuracy.remote(n_reports, n_terms)), f,
                  indent=2)
    print(f"saved {out}")
