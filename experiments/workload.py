"""The one workload every Quail experiment runs, and the Modal image.

The query: five gated yes/no filters over IMDB movie reviews. Each
document ends in a planted metadata line

    [FLAGS] FLAG_1=YES FLAG_2=YES FLAG_3=NO FLAG_4=YES FLAG_5=NO

and filter j asks for the value of FLAG_j. The answer is in the text,
so the query measures execution, not model reasoning: any wrong
answer is the checkpoint misreading a line it can see.

Selectivities (the probability each flag is YES, fixed by FLAG_SEED)
are 0.9, 0.9, 0.9, 0.8, 0.8, so about 47 percent of documents survive
all five stages and the survivor count falls stage by stage - which
is what makes gating worth executing rather than asking everything.

Corpus statistics at the default 10,000 documents (Qwen 4B
tokenizer): 3,203,917 tokens total, mean 320, median 246, min 44, max
2,947. The KV pool at 4B fp8 holds about 946,800 tokens, so the
corpus is 3.4 times the pool and cannot be held resident.
"""

import modal

# CUDA devel base: nvcc is present, so FlashInfer can JIT its kernels.
# The old slim image could not, and read 80,556 tok/s where this base
# reads 97,220. Every banked result carries IMAGE_STAMP.
IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
IMAGE_STAMP = dict(base=IMAGE_BASE, toolchain="cuda13-devel")

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    # Pinned: the scheduler subclass reaches into a non-public engine
    # interface, so a silent version jump on image rebuild could break
    # it mid-study. Every recorded result is stamped with this version.
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "yappi")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("quail")
)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-4B-FP8"
CFG_NAME = "Qwen3-4B-FP8"          # the key into quail.configs.MODELS
DEVICE_NAME = "H100-SXM-80GB"
WORKLOAD_SEED = 20260731
FLAG_SEED = 424242
CEIL = 275_000            # dense FP8 prefill ceiling, tokens per second
N_FILTERS = 5
SELECTIVITY = (0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8)


def build_pool(n_docs):
    """The seeded 10,000-document sample, truncated to n_docs. The
    sample is fixed by WORKLOAD_SEED, so every run in every experiment
    sees the same documents in the same order."""
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


def flags_line(flags):
    return "\n\n[FLAGS] " + " ".join(
        f"FLAG_{j+1}={'YES' if f else 'NO'}" for j, f in enumerate(flags))


def question(j):
    """Filter j's question. Every question shares a 33-token preamble,
    which the scheduler's rewind keeps resident (erasing it made every
    continuation recompute it)."""
    return (f"\n\nExample: if the line said [FLAGS] FLAG_9=NO, then FLAG_9 "
            f"has value NO.\nInstruction: output only the value of FLAG_{j} "
            f"from the [FLAGS] line above.\nFLAG_{j}=")


def build_corpus(tok, n_docs, seed_offset=100, n_filters=None):
    """Tokenized documents with their planted flag lines, the
    tokenized questions, and the flag truth table. n_filters defaults
    to N_FILTERS; the draw shape depends on it, so the default keeps
    every 5-filter corpus byte-identical to what the banked cells
    measured.

    Returns (body_ids, q_ids, flags)."""
    import numpy as np

    n = n_filters or N_FILTERS
    rng = np.random.default_rng(FLAG_SEED + seed_offset)
    flags = (rng.random((n_docs, n))
             < np.array(SELECTIVITY[:n])[None, :]).astype(int)
    docs = build_pool(n_docs)
    bodies = [d + flags_line(f) for d, f in zip(docs, flags)]
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(n)]
    return body_ids, q_ids, flags


def build_flat_pool(tok):
    """Every pool review joined by a blank line, tokenized once, and
    flattened into a single token list. The calibration sweeps slice
    documents of exact target lengths out of it; at 10,000 reviews it
    holds about 3.2 million tokens, and slicing wraps around when the
    sweep needs more."""
    docs = build_pool(10_000)
    doc_ids = tok(docs, add_special_tokens=False)["input_ids"]
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    flat = []
    for i, ids in enumerate(doc_ids):
        if i:
            flat.extend(sep)
        flat.extend(ids)
    return flat


def nonce_alphabet(tok):
    """Token ids the calibration nonce blocks are spelled in: single
    tokens for " a" through " p". Distinct single-token ids so a
    16-token block can encode a large counter."""
    ids = []
    for ch in "abcdefghijklmnop":
        t = tok(f" {ch}", add_special_tokens=False)["input_ids"]
        if len(t) == 1 and t[0] not in ids:
            ids.append(t[0])
    if len(ids) < 2:
        raise RuntimeError("nonce alphabet needs 2+ single-token ids")
    return ids


def yes_no_ids(tok):
    """The token ids that mean YES and NO. Constraining the sampler to
    their union makes every stage answer in exactly one token, which
    is what "zero decode" means here: the answer is sampled from the
    prefill pass and no decode step ever runs."""
    yes, no = set(), set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            yes.add(ids[0])
    for w in ("NO", " NO", "No", " No", "N", " N"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            no.add(ids[0])
    return yes, no


def kv_pool_tokens(llm):
    """The engine's actual KV pool size in tokens, if reachable."""
    for path in (("llm_engine", "cache_config"),
                 ("llm_engine", "vllm_config", "cache_config")):
        obj = llm
        try:
            for a in path:
                obj = getattr(obj, a)
            if obj.num_gpu_blocks:
                return int(obj.num_gpu_blocks) * int(obj.block_size)
        except Exception:
            continue
    return None


def sched_cfg(llm):
    try:
        sc = llm.llm_engine.vllm_config.scheduler_config
        return dict(max_num_batched_tokens=int(sc.max_num_batched_tokens),
                    max_num_seqs=int(sc.max_num_seqs))
    except Exception:
        return {}
