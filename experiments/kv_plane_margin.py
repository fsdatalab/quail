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
- plane_a_rne: e4m3 rounded to nearest with the power-of-two scale of
  plane_a, so the bf16 value is the rounded value plus a remainder of at
  most 8 bf16 steps.
- plane_a_rne_keep: plane_a_rne with the shared preamble's positions
  kept exact.
- chunked: bf16 KV computed in 512-token chunks instead of one pass,
  which measures how much answers move from kernel shapes alone.

Each answer's margin is the best TRUE logit minus the best FALSE logit,
computed in float32 from the final normed hidden state: the tail's last
row for Qwen3, the canvas row for DiffusionGemma. DiffusionGemma runs
the bf16 checkpoint that Quail's fp8 checkpoint was quantized from.
A sliding-window
layer's cache holds only the window, and every version of the KV covers
exactly what the cache holds.

Run from the repository root and tee every line:

    uv run modal run experiments/kv_plane_margin.py \
      --model qwen3-4b-fp8 \
      --prediction "State the expected results before starting." \
      2>&1 | tee results/kv-plane-margin.log

--offset skips that many rows of each table, so a run can use
documents an earlier run did not. The cell writes
/results/ablations/kv_plane_margin_<slug><suffix>.json to the
quail-results volume, with <slug> 4b or dgemma26b: one record per
(document, question) and the per-head exponent counts of the stored KV.
"""

import json
import math
import os
import time

import modal

from quail.bench.images import UV_VERSION, _cuda_base

# transformers runs the fp8 checkpoints only with these installed
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
SLUGS = {"qwen3-4b-fp8": "4b", "diffusion-gemma-26b-a4b-fp8": "dgemma26b"}
DGEMMA_BF16 = "google/diffusiongemma-26B-A4B-it"
DGEMMA_BF16_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
CHUNK_TOKENS = 512
E4M3_MAX = 448.0
E4M3_MIN_NORMAL_EXP = -6
E4M3_SUBNORMAL_STEP = 2.0 ** -9
THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
VARIANTS = ("bf16", "plane_a", "plane_a_restored", "fp8_rne", "plane_a_rne",
            "plane_a_rne_keep", "chunked")
GATED = ("plane_a", "fp8_rne", "plane_a_rne", "plane_a_rne_keep")


def _layers(cache):
    """The cache layers that hold KV."""
    return [layer for layer in cache.layers
            if getattr(layer, "keys", None) is not None]


def _snapshot(cache):
    return [(layer.keys.clone(), layer.values.clone()) for layer in _layers(cache)]


def _lengths(cache):
    return [getattr(layer, "cumulative_length", None) for layer in _layers(cache)]


def _install(cache, tensors, lengths):
    """Put one version of the document KV back into the cache."""
    for layer, (k, v), length in zip(_layers(cache), tensors, lengths):
        layer.keys = k.clone()
        layer.values = v.clone()
        if length is not None:
            layer.cumulative_length = length


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


def plane_a_rne(torch, x):
    """Round bf16 KV to e4m3 with a power-of-two scale per head.

    Args:
        torch: The torch module.
        x: bf16 KV of shape (1, heads, tokens, d_head).

    Returns:
        The rounded values as bf16, and how many normal-range values sit
        more than 8 bf16 steps of the rounded value away from it.
    """
    shift = _pow2_scale(torch, x)
    y = (x.float() * torch.exp2(-shift)).to(torch.float8_e4m3fn).float()
    a = y * torch.exp2(shift)
    normal = y.abs() >= 2.0 ** E4M3_MIN_NORMAL_EXP
    step = torch.exp2(torch.floor(torch.log2(a.abs().clamp_min(1e-30))) - 7)
    wide = normal & ((x.float() - a).abs() > 8 * step)
    return a.to(torch.bfloat16), int(wide.sum())


def _keep_first(tensors, lengths, keep):
    """Put the exact KV back for the first keep positions still cached."""
    out = []
    for (k, v), (k0, v0), length in zip(tensors, keep[1], lengths):
        first = 0 if length is None else length - k0.shape[2]
        rows = max(0, keep[0] - first)
        k, v = k.clone(), v.clone()
        k[:, :, :rows] = k0[:, :, :rows]
        v[:, :, :rows] = v0[:, :, :rows]
        out.append((k, v))
    return out


def _entropy(counts):
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c) \
        if total else 0.0


class Model:
    """A causal model with a float32 TRUE/FALSE margin readout."""

    def __init__(self, torch, spec):
        from transformers import AutoTokenizer

        from quail.logical.prompts import true_false_token_ids

        self.torch = torch
        self.spec = spec
        self.tok = AutoTokenizer.from_pretrained(spec.hf_name,
                                                 revision=spec.revision or None)
        self.model = self.load().eval()
        true_ids, false_ids = true_false_token_ids(self.encode)
        weight = self.model.lm_head.weight
        self.true_rows = weight[true_ids].float()
        self.false_rows = weight[false_ids].float()

    def load(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.spec.hf_name, revision=self.spec.revision,
            dtype=self.torch.bfloat16, device_map="cuda")

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def extend(self, ids, cache):
        """Append ids to the cache; return the last row's normed hidden state."""
        tensor = self.torch.tensor([ids], device="cuda")
        out = self.model.model(input_ids=tensor, past_key_values=cache,
                               use_cache=True)
        return out.last_hidden_state[0, -1].float()

    def answer_row(self, tail_ids, cache):
        return self.extend(tail_ids, cache)

    def prefill(self, ids, chunk=None):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config)
        step = chunk or len(ids)
        for start in range(0, len(ids), step):
            self.extend(ids[start:start + step], cache)
        return cache

    def margin(self, cache, tensors, lengths, tail_ids):
        """Answer one tail against one version of the document KV."""
        _install(cache, tensors, lengths)
        h = self.answer_row(tail_ids, cache)
        return float((self.true_rows @ h).max() - (self.false_rows @ h).max())


class DiffusionGemma(Model):
    """DiffusionGemma: the encoder writes KV, one canvas row reads the answer."""

    def load(self):
        from transformers import DiffusionGemmaForBlockDiffusion

        from quail.backends.quail.executor.models.diffusion_gemma import (
            canvas_token_ids,
        )

        self.canvas = list(canvas_token_ids(self.spec.vocab,
                                            self.spec.canvas_tokens))
        # transformers loads the fp8 checkpoint's attention weights as
        # zeros, so this runs the bf16 checkpoint it was quantized from
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            DGEMMA_BF16, revision=DGEMMA_BF16_REVISION,
            dtype=self.torch.bfloat16, device_map="cuda")
        attn = model.model.encoder.language_model.layers[0].self_attn
        if not attn.k_proj.weight.abs().max() > 0:
            raise AssertionError("DiffusionGemma loaded with zero attention weights")
        return model

    def extend(self, ids, cache):
        tensor = self.torch.tensor([ids], device="cuda")
        self.model.model.encoder(input_ids=tensor, past_key_values=cache,
                                 use_cache=True)

    def answer_row(self, tail_ids, cache):
        self.extend(tail_ids, cache)
        canvas = self.torch.tensor([self.canvas], device="cuda")
        out = self.model.model.decoder(decoder_input_ids=canvas,
                                       past_key_values=cache)
        return out.last_hidden_state[0, self.spec.canvas_answer_row].float()


def _documents(table, limit, offset):
    from quail_b.data import load_table

    column = TABLE_COLUMNS[table]
    rows = load_table(table, scale_factor=0.1, limit=offset + limit)
    return [str(text) for text in rows.column(column).to_pylist()[offset:]]


def _questions(table, model):
    from quail.logical.prompts import bind_prompt
    from quail_b.predicates import PREDICATES

    out = []
    for spec in PREDICATES:
        if spec.kind == "filter" and spec.left_table == table:
            prompt = bind_prompt(spec.template, ("l",), tokenizer=model.encode,
                                 turn=model.spec.turn)
            out.append((spec.key, list(prompt.preamble_token_ids),
                        list(prompt.tail_token_ids)))
    return out


def _count_exponents(torch, exact, head_counts):
    """Add each head's bf16 exponent counts for one document's KV."""
    for layer, pair in enumerate(exact):
        for kv, x in enumerate(pair):
            exps = (x.view(torch.int16) >> 7) & 0xFF
            key = f"{layer}/{kv}"
            if key not in head_counts:
                head_counts[key] = torch.zeros(
                    (exps.shape[1], 256), dtype=torch.int64, device="cuda")
            for head in range(exps.shape[1]):
                head_counts[key][head] += torch.bincount(
                    exps[0, head].reshape(-1).long(), minlength=256)


def _run_document(torch, model, doc_ids, questions, head_counts, stats):
    """Answer every question on one document under every KV version."""
    preamble = questions[0][1]
    if any(q[1] != preamble for q in questions):
        raise ValueError("questions on one table must share a preamble")
    prefix = preamble + doc_ids
    cache = model.prefill(prefix)
    exact = _snapshot(cache)
    lengths = _lengths(cache)
    if not any(k.abs().max() > 0 for k, _v in exact):
        raise AssertionError("the document KV is all zeros")

    planes = [[plane_a(torch, t) for t in pair] for pair in exact]
    variants = {
        "bf16": exact,
        "plane_a": [tuple(p[0] for p in pair) for pair in planes],
        "plane_a_restored": [
            tuple(restore(torch, *p, t) for p, t in zip(pair, exact_pair))
            for pair, exact_pair in zip(planes, exact)],
        "fp8_rne": [tuple(fp8_rne(torch, t) for t in pair) for pair in exact],
    }
    rounded = [[plane_a_rne(torch, t) for t in pair] for pair in exact]
    stats["rne_wide"] += sum(w for pair in rounded for _a, w in pair)
    variants["plane_a_rne"] = [tuple(a for a, _w in pair) for pair in rounded]
    variants["plane_a_rne_keep"] = _keep_first(
        variants["plane_a_rne"], lengths, (len(preamble), exact))
    del rounded
    for pair, exact_pair in zip(variants["plane_a_restored"], exact):
        for got, want in zip(pair, exact_pair):
            if not torch.equal(got, want):
                raise AssertionError("plane A plus plane B is not the bf16 KV")
    for pair in planes:
        for _a, _low, escapes in pair:
            stats["escapes"] += int(escapes.sum())
            stats["values"] += escapes.numel()
    del planes
    _count_exponents(torch, exact, head_counts)

    margins = {name: [model.margin(cache, tensors, lengths, q[2])
                      for q in questions]
               for name, tensors in variants.items()}
    del variants, cache

    chunked = model.prefill(prefix, chunk=CHUNK_TOKENS)
    chunked_kv, chunked_lengths = _snapshot(chunked), _lengths(chunked)
    stats["chunked_kv_differs"] += sum(
        int((k2 != k).sum() + (v2 != v).sum())
        for (k2, v2), (k, v) in zip(chunked_kv, exact))
    margins["chunked"] = [
        model.margin(chunked, chunked_kv, chunked_lengths, q[2])
        for q in questions]
    return len(prefix), margins


def _summary(records, stats, head_counts):
    lines = []
    lines.append(f"items: {len(records)}")
    lines.append(f"escapes: {stats['escapes']} of {stats['values']} values")
    lines.append(f"chunked KV values that differ from one-pass KV: "
                 f"{stats['chunked_kv_differs']} of {stats['values']}")
    entropies = [(sum(counts), _entropy(counts))
                 for heads in head_counts.values() for counts in heads]
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
    lines.append(f"rounded values more than 8 bf16 steps off: {stats['rne_wide']}")
    for name in GATED:
        got = [r["margins"][name] for r in records]
        for tau in THRESHOLDS:
            below = sum(abs(m) < tau for m in got)
            wrong = sum(abs(m) >= tau and (m > 0) != (b > 0)
                        for m, b in zip(got, base))
            lines.append(f"{name} tau {tau}: {below} of {len(got)} fall back;"
                         f" {wrong} accepted answers differ from bf16")
    return "\n".join(lines)


@app.function(
    image=image,
    gpu="H100!",
    memory=131072,
    timeout=10800,
    volumes=volumes,
)
def measure(prediction: str, model_name: str, docs: str,
            output_name: str, offset: int = 0) -> str:
    """Run every document and question, save the records, return a summary."""
    import torch

    from quail.specs import MODELS

    print(f"prediction: {prediction}", flush=True)
    spec = MODELS[model_name]
    t0 = time.perf_counter()
    cls = DiffusionGemma if spec.arch == "diffusion_gemma" else Model
    model = cls(torch, spec)
    print(f"model loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    head_counts = {}
    stats = {"escapes": 0, "values": 0, "chunked_kv_differs": 0,
             "rne_wide": 0}
    records = []
    limits = dict(part.split("=") for part in docs.split(","))
    with torch.inference_mode():
        for table, limit in limits.items():
            questions = _questions(table, model)
            texts = _documents(table, int(limit), offset)
            t1 = time.perf_counter()
            for index, text in enumerate(texts):
                prefix_tokens, margins = _run_document(
                    torch, model, model.encode(text), questions,
                    head_counts, stats)
                for q, (key, _pre, tail) in enumerate(questions):
                    records.append({
                        "table": table, "document": offset + index,
                        "predicate": key,
                        "prefix_tokens": prefix_tokens,
                        "tail_tokens": len(tail),
                        "margins": {name: margins[name][q] for name in VARIANTS},
                    })
            print(f"{table}: {len(texts)} documents, {len(questions)} "
                  f"questions, {time.perf_counter() - t1:.1f} s", flush=True)
    counts = {key: heads.tolist() for key, heads in head_counts.items()}
    path = f"/results/ablations/{output_name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump({"prediction": prediction, "model": spec.name,
                   "revision": spec.revision, "docs": limits, "offset": offset,
                   "chunk_tokens": CHUNK_TOKENS, "stats": stats,
                   "exponent_counts": counts, "records": records}, file)
    results_vol.commit()
    summary = _summary(records, stats, counts)
    return f"saved {path}\n{summary}"


@app.local_entrypoint()
def main(
    prediction: str = "",
    model: str = "qwen3-4b-fp8",
    docs: str = "reviews=200,citation_contexts=100,agent_traces=40",
    offset: int = 0,
    output_suffix: str = "",
):
    """Start the cell, print its function call id, then its summary."""
    if not prediction:
        raise ValueError("pass --prediction before starting")
    name = f"kv_plane_margin_{SLUGS[model]}{output_suffix}"
    call = measure.spawn(prediction, model, docs, name, offset)
    print(f"function call id: {call.object_id}", flush=True)
    print(call.get(), flush=True)


# ---- layer-0 bound test ---------------------------------------------
# Run with:
#
#     uv run modal run experiments/kv_plane_margin.py::bound \
#       --prediction "State the expected results before starting." \
#       2>&1 | tee results/kv-bound-layer0.log
#
# For DiffusionGemma's first layer, where the question rows' queries are
# exact, it compares the actual change in each row's attention output
# under plane_a_rne_keep KV with a limit computed from the rounded KV
# and stored per-token error sizes. "oracle" is given each score's exact
# change, which a proof cannot know; it shows how loose the rest is. It
# writes
# /results/ablations/kv_bound_layer0_dgemma26b.json.

BOUND_RANKS = (8, 16, 32)
BOUND_NAMES = ("naive",) + tuple(f"r{r}" for r in BOUND_RANKS) + ("oracle",)
CAPTURE = {"on": False, "calls": []}


def _install_capture():
    """Record layer 0's attention inputs while CAPTURE["on"] is set."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
    sdpa = mapping["sdpa"]
    if getattr(sdpa, "quail_capture", False):
        return

    def capture(module, query, key, value, attention_mask, **kwargs):
        if CAPTURE["on"] and module.layer_idx == 0:
            scaling = kwargs.get("scaling")
            CAPTURE["calls"].append((
                query.detach().double(), key.detach().double(),
                value.detach().double(),
                float(module.scaling if scaling is None else scaling),
                bool(module.is_causal)))
        return sdpa(module, query, key, value, attention_mask, **kwargs)

    capture.quail_capture = True
    mapping["sdpa"] = capture


def _captured(model, cache, tensors, lengths, tail_ids):
    """Layer 0's attention inputs for one question: tail rows, then canvas."""
    CAPTURE["calls"] = []
    CAPTURE["on"] = True
    try:
        model.margin(cache, tensors, lengths, tail_ids)
    finally:
        CAPTURE["on"] = False
    return CAPTURE["calls"]


def _mask(torch, rows, keys, causal, device):
    """Which keys each row may read: the last rows are the query rows."""
    allowed = torch.ones((rows, keys), dtype=torch.bool, device=device)
    if causal:
        first = keys - rows
        pos = torch.arange(keys, device=device)
        row_pos = first + torch.arange(rows, device=device)
        allowed = pos[None, :] <= row_pos[:, None]
    return allowed


def attention_bounds(torch, exact, approx, prefix, bases):
    """Actual attention output change and its limits for one layer-0 call.

    Args:
        torch: The torch module.
        exact: (query, key, value, scaling, causal) with the bf16 KV.
        approx: The same with the rounded document KV.
        prefix: How many leading key rows are document KV.
        bases: Per query head, a list of (d_head, r) orthonormal bases.

    Returns:
        A dict of (rows, heads) tensors: the actual change, the limit
        from sizes alone ("naive"), and the limits using each basis.
    """
    q, k_e, v_e, scaling, causal = exact
    q2, k_a, v_a, _s, _c = approx
    if not torch.equal(q, q2):
        raise AssertionError("layer 0 queries must not depend on the document KV")
    heads, kv_heads = q.shape[1], k_e.shape[1]
    group = heads // kv_heads
    rows, keys = q.shape[2], k_e.shape[2]
    allowed = _mask(torch, rows, keys, causal, q.device)
    out = {"actual": [], "naive": [], "oracle": []}
    out.update({f"r{r}": [] for r in BOUND_RANKS})
    for h in range(heads):
        qh, kv = q[0, h], h // group
        ke, ka, ve, va = k_e[0, kv], k_a[0, kv], v_e[0, kv], v_a[0, kv]
        dk, dv = ke - ka, ve - va
        if dk[prefix:].abs().max() > 0 or dv[prefix:].abs().max() > 0:
            raise AssertionError("only document KV may differ")
        s_a = (qh @ ka.T * scaling).masked_fill(~allowed, float("-inf"))
        p_a = torch.softmax(s_a, dim=-1)
        o_a = p_a @ va
        s_e = (qh @ ke.T * scaling).masked_fill(~allowed, float("-inf"))
        o_e = torch.softmax(s_e, dim=-1) @ ve
        out["actual"].append((o_e - o_a).norm(dim=-1))
        dv_norm = dv.norm(dim=-1)
        spread = (va[None, :, :] - o_a[:, None, :]).norm(dim=-1)
        eps = {"naive": scaling * qh.norm(dim=-1)[:, None] * dk.norm(dim=-1)[None]}
        for r, basis in zip(BOUND_RANKS, bases[h]):
            known = (qh @ basis) @ (dk @ basis).T
            q_rest = (qh - qh @ basis @ basis.T).norm(dim=-1)
            k_rest = (dk - dk @ basis @ basis.T).norm(dim=-1)
            eps[f"r{r}"] = scaling * (known.abs() + q_rest[:, None] * k_rest[None])
        # not computable without the exact KV: isolates the summing step
        eps["oracle"] = scaling * (qh @ dk.T).abs()
        for name, e in eps.items():
            e = e.masked_fill(~allowed, 0.0)
            low = (p_a * torch.exp(-e)).sum(-1, keepdim=True)
            high = (p_a * torch.exp(e)).sum(-1, keepdim=True)
            rho = torch.maximum(torch.exp(e) / low - 1, 1 - torch.exp(-e) / high)
            key_term = (p_a * rho * spread).sum(-1)
            value_term = (p_a * torch.exp(e) / low * dv_norm[None]).sum(-1)
            out[name].append(key_term + value_term)
            if name == "r16":
                out.setdefault("r16_key", []).append(key_term)
    return {name: torch.stack(v, dim=1) for name, v in out.items()}


def _query_bases(torch, calls_by_item, heads):
    """Per head, orthonormal bases spanning the most common query directions."""
    bases = []
    for h in range(heads):
        rows = torch.cat([q[0, h] for calls in calls_by_item for q, *_ in calls])
        _u, _s, vt = torch.linalg.svd(rows, full_matrices=False)
        bases.append([vt[:r].T.contiguous() for r in BOUND_RANKS])
    return bases


def _bound_items(torch, model, table, limit, offset):
    """Yield (document, question, prefix, exact calls, approx calls)."""
    questions = _questions(table, model)
    for index, text in enumerate(_documents(table, limit, offset)):
        preamble = questions[0][1]
        prefix = preamble + model.encode(text)
        if len(prefix) + max(len(q[2]) for q in questions) + 1 >= 1000:
            continue
        cache = model.prefill(prefix)
        exact, lengths = _snapshot(cache), _lengths(cache)
        rounded = [tuple(plane_a_rne(torch, t)[0] for t in pair) for pair in exact]
        approx = _keep_first(rounded, lengths, (len(preamble), exact))
        for key, _pre, tail in questions:
            yield (offset + index, key, len(prefix),
                   _captured(model, cache, exact, lengths, tail),
                   _captured(model, cache, approx, lengths, tail))


def _quantile(values, p):
    ordered = sorted(values)
    return ordered[int(p * (len(ordered) - 1))]


@app.function(
    image=image,
    gpu="H100!",
    memory=131072,
    timeout=7200,
    volumes=volumes,
)
def measure_bound(prediction: str, docs: str, offset: int,
                  calibration_offset: int, calibration_docs: int) -> str:
    """Compare layer-0 attention limits with the actual change."""
    import torch

    from quail.specs import MODELS

    print(f"prediction: {prediction}", flush=True)
    model = DiffusionGemma(torch, MODELS["diffusion-gemma-26b-a4b-fp8"])
    _install_capture()
    limits = {t: int(n) for t, n in (p.split("=") for p in docs.split(","))}
    records, entries = [], {name: [] for name in ("actual", "naive", "oracle",
                                                  "r16_key")}
    entries.update({f"r{r}": [] for r in BOUND_RANKS})
    violations = 0
    with torch.inference_mode():
        calibration = [item[3] for table in limits for item in _bound_items(
            torch, model, table, calibration_docs, calibration_offset)]
        heads = calibration[0][0][0].shape[1]
        bases = _query_bases(torch, calibration, heads)
        print(f"bases from {len(calibration)} calibration questions", flush=True)
        for table, limit in limits.items():
            for doc, key, prefix, exact, approx in _bound_items(
                    torch, model, table, limit, offset):
                for part, (e, a) in enumerate(zip(exact, approx)):
                    got = attention_bounds(torch, e, a, prefix, bases)
                    for name in BOUND_NAMES:
                        violations += int((got[name] < got["actual"]
                                           * (1 - 1e-9)).sum())
                    flat = {n: t.flatten().tolist() for n, t in got.items()}
                    for n, values in flat.items():
                        entries[n].extend(values)
                    records.append({"table": table, "document": doc,
                                    "predicate": key, "prefix_tokens": prefix,
                                    "part": "canvas" if part else "tail",
                                    **flat})
            print(f"{table}: {len(records)} calls so far", flush=True)
    path = "/results/ablations/kv_bound_layer0_dgemma26b.json"
    with open(path, "w") as file:
        json.dump({"prediction": prediction, "docs": limits, "offset": offset,
                   "calibration_offset": calibration_offset,
                   "calibration_docs": calibration_docs,
                   "ranks": BOUND_RANKS, "violations": violations,
                   "records": records}, file)
    results_vol.commit()
    actual = entries["actual"]
    lines = [f"saved {path}", f"row-head entries: {len(actual)}",
             f"limit below actual (must be 0): {violations}"]
    for name in BOUND_NAMES:
        ratios = [b / a for b, a in zip(entries[name], actual) if a > 0]
        lines.append(
            f"{name}: limit / actual median {_quantile(ratios, 0.5):.2f}, "
            f"p90 {_quantile(ratios, 0.9):.2f}; sum of limits / sum of "
            f"actual {sum(entries[name]) / sum(actual):.2f}")
    key_share = sum(entries["r16_key"]) / sum(entries["r16"])
    lines.append(f"r16: key term share of the limit {key_share:.2f}")
    return "\n".join(lines)


@app.local_entrypoint()
def bound(
    prediction: str = "",
    docs: str = "reviews=60,citation_contexts=40",
    offset: int = 400,
    calibration_offset: int = 600,
    calibration_docs: int = 10,
):
    """Start the layer-0 bound test, print its call id, then its summary."""
    if not prediction:
        raise ValueError("pass --prediction before starting")
    call = measure_bound.spawn(prediction, docs, offset, calibration_offset,
                               calibration_docs)
    print(f"function call id: {call.object_id}", flush=True)
    print(call.get(), flush=True)
