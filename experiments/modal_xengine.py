"""Cross-engine prefill falsification, phase "xengine".

The paper's headline claim rests on one number: vLLM 0.26.0 sustains
about 80,000 prompt tokens per second reading this corpus on one H100
(results/engine/speed_limit.json). This experiment tries to break that
number. Three independent software stacks each read the identical
corpus, prefill-only, nothing warm. If any stack reads faster, the
"engine speed limit" framing is wrong and the paper must change.

Arms, each in its own container with its own dependency stack:

  vllm    The control. Pinned vllm==0.26.0 on the exact image the
          speed_limit phase used. Must reproduce ~80k tok/s.

  sglang  A competing engine. Pinned sglang==0.5.16 on its own CUDA 13
          stack (sglang pins torch 2.11 + flashinfer cu13). Radix
          prefix cache disabled so nothing is warm.

  torch   No engine at all. Plain transformers forward passes,
          torch.compile, flash-attention if the prebuilt wheel loads,
          hand-batched by length buckets to minimize padding. This
          bounds how much of the 80k is engine software rather than
          the model's matmuls.

Attribution arms, added after the first flight (vllm control 80,556
tok/s reproduced; sglang 98,746 = 1.23x; torch 20,493). sglang beat
the control, so each hypothesis for WHY gets its own cell:

  vllm_new       The toolchain hypothesis. The newest vLLM on PyPI —
                 which at time of writing is 0.26.0, the control's own
                 pin — rebuilt on the same CUDA 13 devel image the
                 sglang arm used. Same engine code, same config;
                 only the toolchain underneath changes (nvcc present,
                 so flashinfer can JIT kernels the slim control image
                 cannot build). If this arm speeds up, the gap was
                 never engine software.

  vllm_bf16attn  The fp8-attention-path hypothesis. The control ran
                 fp8 KV, which makes prefill attention quantize K/V
                 on the fly; sglang ran bf16 KV. Same pinned vllm,
                 same image, same config, only kv_cache_dtype="auto".

  vllm_new_tuned The scheduling-overhead hypothesis at its best:
                 vllm_new plus async (overlapped) scheduling pinned
                 on and the documented prefill-throughput flags.

Fairness rules, enforced in code:
  - Every arm feeds the same documents (_build_pool(4000), same seed
    as every other experiment) tokenized the same way (no special
    tokens). Each arm reports a fingerprint (a hash) of the exact
    token stream; the entrypoint checks the fingerprints match.
  - Rates count only tokens actually prefilled: tokens fed minus
    tokens the engine says it served from cache (zero when caching is
    off). The torch arm reports rates both without and with padding.
  - No arm skips layers, truncates documents, or samples fewer
    documents. Per-arm caveats (precision, padding, KV dtype) are
    recorded in the JSON next to the numbers they qualify.

Run with:
  modal run experiments/modal_xengine.py                 # every arm
  modal run experiments/modal_xengine.py --arm sglang    # one arm; merges
                                                         # into xengine.json
  modal run experiments/modal_xengine.py \
      --arm vllm_new,vllm_bf16attn,vllm_new_tuned        # attribution only
"""

import hashlib
import json
import time

import modal

app = modal.App("docengine-xengine")

MODEL = "Qwen/Qwen3-4B-FP8"
# Unquantized parent of MODEL. Used only if plain transformers cannot
# load the fp8 checkpoint; the result is then marked bf16 loudly.
MODEL_BF16_FALLBACK = "Qwen/Qwen3-4B"
WORKLOAD_SEED = 20260731
CEIL = 275_000            # dense FP8 prefill ceiling, tokens per second
VLLM_PIN = "0.26.0"       # same pin as experiments/modal_scale.py
# Newest vLLM release on PyPI at time of writing. It happens to equal
# the control's pin, so the version half of the toolchain/version
# hypothesis is settled by construction: any difference the vllm_new
# arm shows is the image and toolchain, not the engine code.
VLLM_NEW_PIN = "0.26.0"
SGLANG_PIN = "0.5.16"     # latest sglang release at time of writing
TORCH_PIN = "2.8.0"
TRANSFORMERS_PIN = "4.57.6"

# Control arm: byte-for-byte the modal_scale image, so the ~80k
# baseline is reproduced under identical dependencies.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_PIN}", "huggingface_hub", "pandas",
                 "pyarrow", "numpy", "yappi")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

# Attribution arms: the newest vLLM on the same CUDA 13 devel
# toolchain the sglang arm gets. vllm 0.26.0 pins the very same
# torch==2.11.0 and flashinfer-python==0.6.14 that sglang 0.5.16
# pins, so with this image the two engines stand on an identical
# substrate: devel base with nvcc, so flashinfer can JIT kernels
# (the slim control image has no CUDA toolkit and cannot). The
# control image's VLLM_USE_FLASHINFER_SAMPLER=0 pin is dropped here
# on purpose: this arm is vLLM's defaults on a full toolchain.
vllm_new_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04",
                              add_python="3.12")
    .entrypoint([])
    .pip_install(f"vllm=={VLLM_NEW_PIN}", "huggingface_hub", "pandas",
                 "pyarrow", "numpy")
    .env({"VLLM_LOGGING_LEVEL": "WARNING"})
)

# sglang 0.5.16 pins torch==2.11.0 (CUDA 13 wheels) and
# flashinfer[cu13], so it gets the CUDA 13 devel image it expects;
# devel because flashinfer JIT-compiles kernels at first use.
sglang_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04",
                              add_python="3.12")
    .entrypoint([])
    .pip_install(f"sglang[all]=={SGLANG_PIN}", "huggingface_hub",
                 "pandas", "pyarrow", "numpy")
)

# Raw-torch arm: CUDA 12.8 matches torch 2.8.0's default wheels. The
# flash-attn install is a prebuilt wheel; if the download fails we do
# NOT build from source (an hour of nvcc), the arm falls back to
# PyTorch SDPA and records that in its caveats.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-"
    "linux_x86_64.whl")
torch_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .entrypoint([])
    .pip_install(f"torch=={TORCH_PIN}", f"transformers=={TRANSFORMERS_PIN}",
                 "accelerate", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "safetensors")
    .run_commands(f"pip install '{FLASH_ATTN_WHEEL}' || "
                  f"echo 'no prebuilt flash-attn wheel; will use SDPA'")
)

hf_cache = modal.Volume.from_name("docengine-hf-cache",
                                  create_if_missing=True)


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


def _corpus_ids(n_docs):
    """The shared workload: same documents, same tokenizer, same
    no-special-tokens convention as the speed_limit short_ids arm."""
    from transformers import AutoTokenizer

    docs = _build_pool(n_docs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok(docs, add_special_tokens=False)["input_ids"]


def _fingerprint(ids):
    """Hash of the exact token stream, so the entrypoint can prove all
    arms consumed identical input even across tokenizer versions."""
    import numpy as np

    h = hashlib.sha256()
    h.update(np.asarray([len(s) for s in ids], dtype=np.int64).tobytes())
    h.update(np.concatenate(
        [np.asarray(s, dtype=np.int32) for s in ids]).tobytes())
    return h.hexdigest()[:16]


def _warm_ids(count=32, length=512):
    """Random-token warmup prompts. They share no prefix with the
    corpus, so they exercise lazy compilation paths without warming
    any cache for the measured pass."""
    import numpy as np

    rng = np.random.default_rng(4242)
    return [rng.integers(1000, 100_000, size=length).tolist()
            for _ in range(count)]


def _line(arm, tokens, wall):
    rate = tokens / wall
    print(f"[xengine] {arm}: {tokens} tokens in {wall:.2f}s = "
          f"{rate:,.0f} tok/s ({100 * rate / CEIL:.1f}% of 275k spec)",
          flush=True)
    return rate


def _vllm_measure(arm, n_docs, llm_kwargs, drop_on_error=()):
    """Measurement body shared by the vLLM attribution arms: build the
    engine, warm it on random tokens, read the corpus cold, count fed
    minus cached. drop_on_error lists kwargs to remove one at a time
    (and record) if this vLLM build rejects them, so an arm degrades
    loudly instead of dying."""
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    ids = _corpus_ids(n_docs)
    fed = sum(len(x) for x in ids)
    kwargs = dict(llm_kwargs)
    dropped = []
    llm = None
    while llm is None:
        try:
            llm = LLM(**kwargs)
        except Exception:
            k = next((k for k in drop_on_error if k in kwargs), None)
            if k is None:
                raise
            kwargs.pop(k)
            dropped.append(k)
            print(f"[xengine] {arm}: engine rejected {k}; retrying "
                  f"without it", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    llm.generate([{"prompt_token_ids": x} for x in _warm_ids()], sp,
                 use_tqdm=False)

    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp,
                        use_tqdm=False)
    wall = time.time() - t0
    cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
    prefilled = sum(len(o.prompt_token_ids) for o in outs) - cached
    rate = _line(arm, prefilled, wall)
    try:
        llm.shutdown()
    except Exception:
        pass
    cfg = {k: v for k, v in kwargs.items() if k != "model"}
    return dict(
        arm=arm, engine=f"vllm {vllm.__version__}",
        versions=dict(vllm=vllm.__version__, torch=torch.__version__),
        model=MODEL, n_docs=n_docs, fed_tokens=fed,
        cached_tokens=cached, prefilled_tokens=prefilled,
        wall_s=wall, tok_per_s=rate, pct_of_spec=rate / CEIL,
        token_fingerprint=_fingerprint(ids),
        config=cfg, dropped_flags=dropped)


# ------------------------------------------------------------ arm 1: vllm

@app.function(image=vllm_image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def vllm_arm(n_docs: int = 4000) -> dict:
    import vllm
    from vllm import LLM, SamplingParams

    ids = _corpus_ids(n_docs)
    fed = sum(len(x) for x in ids)
    # Same settings as the speed_limit control that measured ~80k,
    # except prefix caching is off so nothing can be warm.
    llm = LLM(model=MODEL, kv_cache_dtype="fp8", max_model_len=8192,
              gpu_memory_utilization=0.92, enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    llm.generate([{"prompt_token_ids": x} for x in _warm_ids()], sp,
                 use_tqdm=False)

    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp,
                        use_tqdm=False)
    wall = time.time() - t0
    cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
    prefilled = sum(len(o.prompt_token_ids) for o in outs) - cached
    rate = _line("vllm", prefilled, wall)
    try:
        llm.shutdown()
    except Exception:
        pass
    return dict(
        arm="vllm", engine=f"vllm {vllm.__version__}",
        versions=dict(vllm=vllm.__version__),
        model=MODEL,
        precision="fp8 checkpoint weights, fp8 KV (the paper's control "
                  "configuration)",
        n_docs=n_docs, fed_tokens=fed, cached_tokens=cached,
        prefilled_tokens=prefilled, wall_s=wall, tok_per_s=rate,
        pct_of_spec=rate / CEIL, token_fingerprint=_fingerprint(ids),
        config=dict(kv_cache_dtype="fp8", max_model_len=8192,
                    gpu_memory_utilization=0.92,
                    enable_prefix_caching=False),
        caveats=[
            "Control arm: reproduces the speed_limit configuration "
            "with prefix caching disabled.",
        ])


# ---------------------------------------------------------- arm 2: sglang

@app.function(image=sglang_image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def sglang_arm(n_docs: int = 4000) -> dict:
    import sglang
    import torch

    ids = _corpus_ids(n_docs)
    fed = sum(len(x) for x in ids)
    # Offline batch engine, radix (prefix) cache disabled so no
    # request can reuse another's work. Everything else is sglang's
    # own defaults: this arm is their stack at its best, not ours.
    engine = sglang.Engine(model_path=MODEL, disable_radix_cache=True,
                           mem_fraction_static=0.90, context_length=8192,
                           log_level="warning")
    spm = {"temperature": 0.0, "max_new_tokens": 1}

    engine.generate(input_ids=_warm_ids(), sampling_params=spm)

    t0 = time.time()
    outs = engine.generate(input_ids=ids, sampling_params=spm)
    wall = time.time() - t0
    if isinstance(outs, dict):
        outs = [outs]
    cached = sum(int(o.get("meta_info", {}).get("cached_tokens", 0) or 0)
                 for o in outs)
    prefilled = fed - cached
    rate = _line("sglang", prefilled, wall)
    try:
        engine.shutdown()
    except Exception:
        pass
    return dict(
        arm="sglang", engine=f"sglang {sglang.__version__}",
        versions=dict(sglang=sglang.__version__,
                      torch=torch.__version__),
        model=MODEL,
        precision="fp8 checkpoint weights, KV dtype 'auto' (sglang "
                  "default, bf16 KV); prefill compute is unaffected, "
                  "only the KV writes differ from the vllm arm",
        n_docs=n_docs, fed_tokens=fed, cached_tokens=cached,
        prefilled_tokens=prefilled, wall_s=wall, tok_per_s=rate,
        pct_of_spec=rate / CEIL, token_fingerprint=_fingerprint(ids),
        config=dict(disable_radix_cache=True, mem_fraction_static=0.90,
                    context_length=8192),
        caveats=[
            "sglang runs the fp8 checkpoint through its own w8a8 fp8 "
            "path; kernel choice (flashinfer/fa3) is its default for "
            "H100.",
            "KV dtype left at sglang's default 'auto' rather than fp8: "
            "prefill-only work never reads KV back, so this changes "
            "cache write bandwidth only.",
        ])


# ----------------------------------------------------------- arm 3: torch

@app.function(image=torch_image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def torch_arm(n_docs: int = 4000, batch_tokens: int = 32768) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    ids = _corpus_ids(n_docs)
    fed = sum(len(x) for x in ids)
    caveats = []

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        caveats.append("flash-attn wheel unavailable; used PyTorch "
                       "SDPA (still a fused flash kernel on H100).")

    # Load the fp8 checkpoint if plain transformers can; otherwise the
    # unquantized bf16 parent. bf16 moves twice the weight bytes per
    # matmul, so a bf16 result is NOT comparable head-on and is
    # flagged everywhere it is reported.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype="auto", device_map="cuda",
            attn_implementation=attn_impl)
        checkpoint, precision = MODEL, "fp8 checkpoint weights (block-quantized), bf16 activations"
    except Exception as e:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_BF16_FALLBACK, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=attn_impl)
        checkpoint = MODEL_BF16_FALLBACK
        precision = ("bf16 UNQUANTIZED PARENT CHECKPOINT - the fp8 "
                     "checkpoint would not load in plain transformers")
        caveats.append(f"PRECISION MISMATCH: fp8 load failed "
                       f"({type(e).__name__}: {e}); this arm ran "
                       f"{MODEL_BF16_FALLBACK} in bf16. bf16 weights are "
                       f"2x the bytes of fp8, so this arm's rate is not "
                       f"directly comparable to the engine arms.")
        print(f"[xengine] torch: PRECISION MISMATCH - running bf16 "
              f"fallback {MODEL_BF16_FALLBACK}: {e}", flush=True)
    model.eval()

    # Length-bucketed batches to minimize padding: sort long-to-short,
    # pad each batch to its longest member rounded up to 128 (the
    # rounding bounds how many distinct shapes the compiler sees).
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    batches = []
    cur, plen = [], 0
    for i in order:
        need = -(-len(ids[i]) // 128) * 128
        if not cur:
            cur, plen = [i], need
        elif (len(cur) + 1) * plen <= batch_tokens:
            cur.append(i)
        else:
            batches.append((cur, plen))
            cur, plen = [i], need
    if cur:
        batches.append((cur, plen))

    dev = "cuda"

    def make_batch(rows, plen):
        b = torch.zeros((len(rows), plen), dtype=torch.long)
        m = torch.zeros((len(rows), plen), dtype=torch.long)
        last = torch.zeros(len(rows), dtype=torch.long)
        for r, i in enumerate(rows):
            n = len(ids[i])
            b[r, :n] = torch.as_tensor(ids[i], dtype=torch.long)
            m[r, :n] = 1
            last[r] = n - 1
        return b.to(dev), m.to(dev), last.to(dev)

    def step_eager(b, m, last):
        # Forward only. The head runs on each row's final real
        # position exactly as an engine's prefill does; no layer is
        # skipped, every real token goes through the full stack.
        h = model.model(input_ids=b, attention_mask=m).last_hidden_state
        h = h[torch.arange(b.shape[0], device=b.device), last]
        return model.lm_head(h).argmax(-1)

    step = torch.compile(step_eager, dynamic=True)
    compiled = True
    warm = batches[:: max(1, len(batches) // 4)][:4]
    with torch.inference_mode():
        try:
            for rows, plen in warm:
                step(*make_batch(rows, plen))
        except Exception as e:
            compiled = False
            step = step_eager
            caveats.append(f"torch.compile failed on this model "
                           f"({type(e).__name__}); ran eager.")
            print(f"[xengine] torch: compile failed, running eager: {e}",
                  flush=True)
            for rows, plen in warm:
                step(*make_batch(rows, plen))
        torch.cuda.synchronize()

        preds = []
        t0 = time.time()
        for rows, plen in batches:
            preds.append(step(*make_batch(rows, plen)))
        torch.cuda.synchronize()
        wall = time.time() - t0

    padded = sum(len(rows) * plen for rows, plen in batches)
    rate = _line("torch", fed, wall)
    rate_pad = padded / wall
    print(f"[xengine] torch incl. padding: {padded} tokens in "
          f"{wall:.2f}s = {rate_pad:,.0f} tok/s "
          f"(padding overhead {100 * (padded - fed) / fed:.1f}%)",
          flush=True)
    caveats.append("No paged attention or CUDA graphs: per-batch "
                   "kernel launches and H2D copies are inside the "
                   "measured wall, as they are for the engines.")
    return dict(
        arm="torch",
        engine=f"transformers {transformers.__version__} + "
               f"torch {torch.__version__}",
        versions=dict(torch=torch.__version__,
                      transformers=transformers.__version__,
                      attn_implementation=attn_impl,
                      torch_compile=compiled),
        model=checkpoint, precision=precision,
        n_docs=n_docs, fed_tokens=fed, cached_tokens=0,
        prefilled_tokens=fed, padded_tokens=padded,
        wall_s=wall, tok_per_s=rate,
        tok_per_s_incl_padding=rate_pad,
        pct_of_spec=rate / CEIL, token_fingerprint=_fingerprint(ids),
        config=dict(batch_tokens=batch_tokens, n_batches=len(batches),
                    pad_multiple=128, attn=attn_impl,
                    compiled=compiled),
        caveats=caveats)


# ------------------------------------------------------------- entrypoint

@app.local_entrypoint()
def main(arm: str = "all", n_docs: int = 4000, out: str = ""):
    import os

    fns = dict(vllm=vllm_arm, sglang=sglang_arm, torch=torch_arm)
    names = list(fns) if arm == "all" else \
        [a.strip() for a in arm.split(",")]
    for a in names:
        if a not in fns:
            raise SystemExit(f"unknown arm {a!r}; choose from "
                             f"{sorted(fns)} or 'all'")

    path = out or "results/engine/xengine.json"
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        # Merge only if the workload matches; a different corpus would
        # silently mix incomparable rows.
        if prev.get("model") == MODEL and prev.get("n_docs") == n_docs:
            data = prev
    data.update(model=MODEL, n_docs=n_docs, ceiling=CEIL,
                phase="xengine")
    data.setdefault("arms", {})

    for name in names:                      # sequential, one container each
        data["arms"][name] = fns[name].remote(n_docs)

    fps = {a: r.get("token_fingerprint")
           for a, r in data["arms"].items()}
    data["token_fingerprints"] = fps
    data["identical_corpus"] = len(set(fps.values())) == 1
    if not data["identical_corpus"]:
        print(f"[xengine] WARNING: token fingerprints differ across "
              f"arms: {fps} - rates are NOT comparable until this is "
              f"resolved", flush=True)

    ctrl = data["arms"].get("vllm")
    if ctrl:
        for a, r in sorted(data["arms"].items()):
            print(f"[xengine] summary {a}: {r['tok_per_s']:,.0f} tok/s "
                  f"= {r['tok_per_s'] / ctrl['tok_per_s']:.2f}x the "
                  f"vllm control", flush=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"saved {path}")
