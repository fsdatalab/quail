"""Measure how rounding stored document KV to fp8 changes yes/no answers.

For each QUAIL-B filter document, the cell computes the document's KV
once in bf16 and answers every filter question on that document's set
against five versions of it:

- bf16: the KV as computed.
- plane_a: each value cut to e4m3 with a power-of-two scale per layer,
  K or V, and KV head. Values in e4m3's normal range keep their top
  three mantissa bits, which is the bf16 value with its low four
  mantissa bits zeroed. Smaller values go to the e4m3 subnormal grid
  and are counted as escapes.
- plane_a_restored: plane_a with the low bits put back; must equal bf16.
- fp8_rne: ordinary fp8 KV, round to nearest with an amax scale per
  layer, K or V, and KV head.
- chunked: bf16 KV computed in 512-token chunks instead of one pass,
  which measures how much answers move from kernel shapes alone.

Each answer's margin is the best TRUE logit minus the best FALSE logit,
computed in float32 from the final normed hidden state.

Run from the repository root and tee every line:

    uv run modal run experiments/kv_plane_margin.py \
      --prediction "State the expected results before starting." \
      2>&1 | tee results/kv-plane-margin.log

The cell writes /results/ablations/kv_plane_margin_4b.json to the
quail-results volume: one record per (document, question) and the
per-head exponent counts of the stored KV.
"""

import json
import math
import os
import time

import modal

from quail.bench.images import UV_VERSION, _cuda_base

# transformers runs the fp8 checkpoint only with these installed
image = (_cuda_base()
         .uv_sync(groups=["dev"], uv_version=UV_VERSION)
         .uv_pip_install("accelerate==1.10.1", "kernels==0.16.0")
         .add_local_python_source("quail"))

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/results": results_vol,
}

# QUAIL-B table -> the column its filter predicates read
TABLE_COLUMNS = {
    "reviews": "body",
    "citation_contexts": "destination_context",
    "agent_traces": "trace",
}
CHUNK_TOKENS = 512
E4M3_MAX = 448.0
E4M3_MIN_NORMAL_EXP = -6
E4M3_SUBNORMAL_STEP = 2.0 ** -9
THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 4.0)
VARIANTS = ("bf16", "plane_a", "plane_a_restored", "fp8_rne", "chunked")


def _kv_layers(cache):
    """Return the cache's per-layer objects holding keys and values."""
    return cache.layers


def _snapshot(cache):
    return [(layer.keys.clone(), layer.values.clone())
            for layer in _kv_layers(cache)]


def _install(cache, tensors):
    for layer, (k, v) in zip(_kv_layers(cache), tensors):
        layer.keys = k.clone()
        layer.values = v.clone()


def _pow2_scale(torch, x):
    """Per-head power-of-two scale so every value fits under E4M3_MAX.

    Args:
        torch: The torch module.
        x: KV of shape (1, heads, tokens, d_head).

    Returns:
        The scale exponent per head, shape (1, heads, 1, 1).
    """
    amax = x.float().abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-30)
    return torch.ceil(torch.log2(amax / E4M3_MAX))


def plane_a(torch, x):
    """Cut bf16 KV to e4m3 with a power-of-two scale per head.

    Args:
        torch: The torch module.
        x: bf16 KV of shape (1, heads, tokens, d_head).

    Returns:
        The plane A values as bf16, the low four mantissa bits, and a
        mask of values below e4m3's normal range (escapes).
    """
    shift = _pow2_scale(torch, x)
    y = x.float() * torch.exp2(-shift)
    normal = y.abs() >= 2.0 ** E4M3_MIN_NORMAL_EXP
    bits = x.view(torch.int16)
    low = bits & 0x000F
    cut = (bits & ~0x000F).view(torch.bfloat16)
    step = E4M3_SUBNORMAL_STEP
    sub = torch.sign(y) * torch.floor(y.abs() / step) * step
    sub = (sub * torch.exp2(shift)).to(torch.bfloat16)
    a = torch.where(normal, cut, sub)
    scaled = a.float() * torch.exp2(-shift)
    if not torch.equal(scaled.to(torch.float8_e4m3fn).float(), scaled):
        raise AssertionError("plane A holds a value e4m3 cannot represent")
    escapes = (~normal) & (x != 0)
    return a, low, escapes


def restore(torch, a, low, escapes, exact):
    """Put the low bits back into plane A; escapes take their exact value."""
    joined = (a.view(torch.int16) | low).view(torch.bfloat16)
    return torch.where(escapes, exact, joined)


def fp8_rne(torch, x):
    """Round bf16 KV to e4m3 with an amax scale per head, then back."""
    amax = x.float().abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-30)
    scale = amax / E4M3_MAX
    q = (x.float() / scale).to(torch.float8_e4m3fn).float() * scale
    return q.to(torch.bfloat16)


def _entropy(counts):
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c) \
        if total else 0.0


class Model:
    """Qwen3 4B fp8 with a float32 TRUE/FALSE margin readout."""

    def __init__(self, torch, spec):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from quail.logical.prompts import true_false_token_ids

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(spec.hf_name,
                                                 revision=spec.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_name, revision=spec.revision, dtype=torch.bfloat16,
            device_map="cuda").eval()
        true_ids, false_ids = true_false_token_ids(self.encode)
        weight = self.model.lm_head.weight
        self.true_rows = weight[true_ids].float()
        self.false_rows = weight[false_ids].float()

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def forward(self, ids, cache):
        torch = self.torch
        tensor = torch.tensor([ids], device="cuda")
        out = self.model.model(input_ids=tensor, past_key_values=cache,
                               use_cache=True)
        return out.last_hidden_state[0, -1].float()

    def prefill(self, ids, chunk=None):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config)
        step = chunk or len(ids)
        for start in range(0, len(ids), step):
            self.forward(ids[start:start + step], cache)
        return cache

    def margin(self, cache, prefix_tokens, tail_ids):
        """Answer one tail against the cache, then rewind the cache."""
        h = self.forward(tail_ids, cache)
        cache.crop(prefix_tokens)
        return float((self.true_rows @ h).max() - (self.false_rows @ h).max())


def _documents(table, limit):
    from quail_b.data import load_table

    column = TABLE_COLUMNS[table]
    rows = load_table(table, scale_factor=0.1, limit=limit)
    return [str(text) for text in rows.column(column).to_pylist()]


def _questions(table, model):
    from quail.logical.prompts import bind_prompt
    from quail_b.predicates import PREDICATES

    out = []
    for spec in PREDICATES:
        if spec.kind == "filter" and spec.left_table == table:
            prompt = bind_prompt(spec.template, ("l",), tokenizer=model.encode)
            out.append((spec.key, list(prompt.preamble_token_ids),
                        list(prompt.tail_token_ids)))
    return out


def _run_document(torch, model, doc_ids, questions, head_counts, stats):
    """Answer every question on one document under every KV variant."""
    preamble = questions[0][1]
    if any(q[1] != preamble for q in questions):
        raise ValueError("questions on one table must share a preamble")
    prefix = preamble + doc_ids
    cache = model.prefill(prefix)
    exact = _snapshot(cache)

    planes = [[plane_a(torch, t) for t in pair] for pair in exact]
    variants = {
        "bf16": exact,
        "plane_a": [tuple(p[0] for p in pair) for pair in planes],
        "plane_a_restored": [
            tuple(restore(torch, *p, t) for p, t in zip(pair, exact_pair))
            for pair, exact_pair in zip(planes, exact)],
        "fp8_rne": [tuple(fp8_rne(torch, t) for t in pair) for pair in exact],
    }
    for pair, exact_pair in zip(variants["plane_a_restored"], exact):
        for got, want in zip(pair, exact_pair):
            if not torch.equal(got, want):
                raise AssertionError("plane A plus plane B is not the bf16 KV")
    for layer, pair in enumerate(planes):
        for kv, (_a, _low, escapes) in enumerate(pair):
            stats["escapes"] += int(escapes.sum())
            stats["values"] += escapes.numel()
            exps = (exact[layer][kv].view(torch.int16) >> 7) & 0xFF
            for head in range(exps.shape[1]):
                head_counts[layer][kv][head] += torch.bincount(
                    exps[0, head].reshape(-1).long(), minlength=256)
    del planes

    margins = {}
    for name, tensors in variants.items():
        _install(cache, tensors)
        margins[name] = [model.margin(cache, len(prefix), q[2])
                         for q in questions]
    del variants, cache

    chunked = model.prefill(prefix, chunk=CHUNK_TOKENS)
    stats["chunked_kv_differs"] += sum(
        int((layer.keys != k).sum() + (layer.values != v).sum())
        for layer, (k, v) in zip(_kv_layers(chunked), exact))
    margins["chunked"] = [model.margin(chunked, len(prefix), q[2])
                          for q in questions]
    return len(prefix), margins


def _summary(records, stats, head_counts):
    lines = []
    lines.append(f"items: {len(records)}")
    lines.append(f"escapes: {stats['escapes']} of {stats['values']} values")
    lines.append(f"chunked KV values that differ from one-pass KV: "
                 f"{stats['chunked_kv_differs']} of {stats['values']}")
    entropies = [
        (sum(counts), _entropy(counts))
        for layer in head_counts for kv in layer for counts in kv]
    total = sum(n for n, _ in entropies)
    mean_h = sum(n * h for n, h in entropies) / total
    lines.append(f"exponent entropy per head, weighted mean: {mean_h:.3f} bits;"
                 f" plane A estimate: {1 + mean_h + 3:.3f} bits of 16")
    base = [r["margins"]["bf16"] for r in records]
    for name in VARIANTS[1:]:
        got = [r["margins"][name] for r in records]
        flips = sum((a > 0) != (b > 0) for a, b in zip(base, got))
        shifts = sorted(abs(a - b) for a, b in zip(base, got))
        p99 = shifts[int(0.99 * (len(shifts) - 1))]
        lines.append(f"{name}: {flips} flips; |margin shift| max "
                     f"{shifts[-1]:.4f}, p99 {p99:.4f}")
    got = [r["margins"]["plane_a"] for r in records]
    for tau in THRESHOLDS:
        below = sum(abs(m) < tau for m in got)
        wrong = sum(abs(m) >= tau and (m > 0) != (b > 0)
                    for m, b in zip(got, base))
        lines.append(f"plane_a tau {tau}: {below} of {len(got)} fall back; "
                     f"{wrong} accepted answers differ from bf16")
    return "\n".join(lines)


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=3600,
    volumes=volumes,
)
def measure(prediction: str, docs: str, output_name: str) -> str:
    """Run every document and question, save the records, return a summary."""
    import torch

    from quail.specs import MODELS

    print(f"prediction: {prediction}", flush=True)
    spec = MODELS["qwen3-4b-fp8"]
    t0 = time.perf_counter()
    model = Model(torch, spec)
    print(f"model loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    head_counts = [[[torch.zeros(256, dtype=torch.int64, device="cuda")
                     for _ in range(spec.n_kv)] for _ in range(2)]
                   for _ in range(spec.layers)]
    stats = {"escapes": 0, "values": 0, "chunked_kv_differs": 0}
    records = []
    limits = dict(part.split("=") for part in docs.split(","))
    with torch.inference_mode():
        for table, limit in limits.items():
            questions = _questions(table, model)
            texts = _documents(table, int(limit))
            t1 = time.perf_counter()
            for index, text in enumerate(texts):
                prefix_tokens, margins = _run_document(
                    torch, model, model.encode(text), questions,
                    head_counts, stats)
                for q, (key, _pre, tail) in enumerate(questions):
                    records.append({
                        "table": table, "document": index, "predicate": key,
                        "prefix_tokens": prefix_tokens,
                        "tail_tokens": len(tail),
                        "margins": {name: margins[name][q] for name in VARIANTS},
                    })
            print(f"{table}: {len(texts)} documents, {len(questions)} "
                  f"questions, {time.perf_counter() - t1:.1f} s", flush=True)
    counts = [[[c.tolist() for c in kv] for kv in layer] for layer in head_counts]
    path = f"/results/ablations/{output_name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump({"prediction": prediction, "model": spec.name,
                   "revision": spec.revision, "docs": limits,
                   "chunk_tokens": CHUNK_TOKENS, "stats": stats,
                   "exponent_counts": counts, "records": records}, file)
    results_vol.commit()
    summary = _summary(records, stats, counts)
    return f"saved {path}\n{summary}"


@app.local_entrypoint()
def main(
    prediction: str = "",
    docs: str = "reviews=200,citation_contexts=100,agent_traces=40",
    output_suffix: str = "",
):
    """Start the cell, print its function call id, then its summary."""
    if not prediction:
        raise ValueError("pass --prediction before starting")
    call = measure.spawn(prediction, docs, f"kv_plane_margin_4b{output_suffix}")
    print(f"function call id: {call.object_id}", flush=True)
    print(call.get(), flush=True)
