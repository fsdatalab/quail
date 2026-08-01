#!/usr/bin/env python3
"""Build the canonical 10,000-document workload (paper sec. 10.2, eq. 56).

Pool: the 50,000 labeled reviews of stanfordnlp/imdb (train + test splits),
fetched as parquet from the Hugging Face hub. doc_id = "<split>/<row>", which
makes IDs unique across splits (raw aclImdb filenames collide between splits).

Tokenizer: Qwen/Qwen3-4B-FP8 tokenizer.json. Qwen3-4B-FP8 and Qwen3-32B-FP8
ship byte-identical tokenizer.json files (sha256
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4), so one
realized d vector serves both models. d_i = len(encode(text,
add_special_tokens=False)) per eq. (1).

Notes recorded here so they are not lost:
  - 418 of the 50,000 pool texts are exact duplicates of another pool text
    (identical text_hash, distinct doc_id). They are kept as distinct rows:
    SQL row semantics. The v1 scheduler does NOT share KV across identical
    texts; that is a possible later refinement and would need its own rule.
  - No document exceeds L_ctx - max_j p_j (pool max d_i = 3112, sample max
    2924, L_ctx = 40960), so the context-window rule (eq. 23) rejects nothing
    at scale 1. Length-scaled sensitivity runs must re-check it.

Usage: python scripts/build_workload.py [--out workloads/documents.parquet]
"""

import argparse
import hashlib
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

SEED = 20260731
N_SAMPLE = 10_000
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
MODEL_REVISIONS = {
    "Qwen/Qwen3-4B-FP8": "96b30dc13593a244a5e59e84687309f53c375cfa",
    "Qwen/Qwen3-32B-FP8": "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
}
HF = "https://huggingface.co"
POOL_URLS = {
    "train": f"{HF}/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
    "test": f"{HF}/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet",
}
TOKENIZER_URL = f"{HF}/Qwen/Qwen3-4B-FP8/resolve/main/tokenizer.json"


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workloads/documents.parquet")
    ap.add_argument("--cache", default=os.environ.get("DOCENGINE_CACHE", "/tmp/docengine-cache"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    tok_path = fetch(TOKENIZER_URL, os.path.join(args.cache, "qwen3_tokenizer.json"))
    with open(tok_path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json sha256 mismatch: {got}")
    tok = Tokenizer.from_file(tok_path)

    frames = []
    for split, url in POOL_URLS.items():
        df = pd.read_parquet(fetch(url, os.path.join(args.cache, f"imdb_{split}.parquet")))
        df["doc_id"] = [f"{split}/{i}" for i in range(len(df))]
        frames.append(df[["doc_id", "text"]])
    pool = pd.concat(frames, ignore_index=True)
    assert len(pool) == 50_000, len(pool)

    encs = tok.encode_batch(pool["text"].tolist(), add_special_tokens=False)
    pool["d"] = [len(e.ids) for e in encs]
    pool["text_hash"] = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in pool["text"]]

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(pool), size=N_SAMPLE, replace=False))
    sample = pool.iloc[idx][["doc_id", "text_hash", "d"]].copy()
    sample["sample_seed"] = SEED
    sample["tokenizer_revision"] = TOKENIZER_SHA256

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample.to_parquet(args.out, index=False)

    q = np.percentile(sample["d"], [50, 90, 99])
    print(json.dumps({
        "n": len(sample),
        "sum_d": int(sample["d"].sum()),
        "mean": float(sample["d"].mean()),
        "p50": q[0], "p90": q[1], "p99": q[2],
        "max": int(sample["d"].max()),
        "seed": SEED,
        "model_revisions": MODEL_REVISIONS,
    }, indent=2))


if __name__ == "__main__":
    main()
