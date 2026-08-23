"""The retired "split" attention path, kept for comparisons only.

split is the pre-fusion implementation of the two-call pattern: the
same two FlashAttention-3 calls as the engine's merge_quant path,
but with the online-softmax LSE merge (Milakov & Gimelshein 2018,
"Online normalizer calculation for softmax") as plain PyTorch ops
and FP8 quantization as a separate step afterward. The engine ships
only the two production paths - unified (filters) and merge_quant
(joins) - and this module holds split because its merge is readable
line by line: it is the reference the fused merge_attn_quant Triton
kernel is checked against. History and evidence:
reports/2026-08-21-attention-paths.md.

Build chunks for it with pack_chunk(..., attention_mode=
"merge_quant") - split reads the same cross layout and ignores the
extra "source" vector. Call attention_split directly on a chunk's
meta, or run it through a full forward pass with set_path:

    set_path(pipeline, "split")        # installs the override
    run_filter(...)                    # forward_chunk calls split
    set_path(pipeline, "merge_quant")  # back to a production path

The gather fallback the engine's split once carried (contiguous KV
copies instead of paged reads) was removed with it; this reference
is paged-only, like the production paths.
"""

import functools


def _tokens_first(lse):
    # FA3 varlen returns (heads, tokens); transpose to (tokens, heads)
    return lse.transpose(0, 1).contiguous()


def attention_split(pipeline, q, k, v, meta):
    """Two FA3 calls merged by softmax state in plain PyTorch.

    Returns bf16 rows for o_proj, like the engine's unified path;
    quantization is the caller's (or forward_chunk's) separate step.
    """
    torch = pipeline.torch
    n = q.shape[0]
    H, KH, D = (pipeline.num_q_heads, pipeline.num_kv_heads,
                pipeline.head_dim)
    q3 = q.view(n, H, D)
    k3 = k.view(n, KH, D)
    v3 = v.contiguous().view(n, KH, D)
    layer = meta["layer"]

    if meta["kv_src"] is not None:
        pipeline.kv_row_scatter(k3, v3, meta["kv_src"],
                                meta["kv_dst"], layer)

    out_a, lse_a = pipeline._fa(
        q3, k3, v3, meta["cu_a"], meta["cu_a"],
        meta["max_a"], meta["max_a"], causal=True)

    cross = meta["cross"]
    if cross is None:
        # no kept-context reads in this chunk: whole-pair segments
        # (probe reference) or a cache-only pass
        meta["layer"] += 1
        return out_a.view(n, H * D)

    rows = cross["rows"]
    q_suf = q3.index_select(0, rows)
    kp, vp = pipeline.arena.paged_kv(layer)
    out_b, lse_b = pipeline._fa(
        q_suf, kp, vp, cross["cu_q"], None,
        cross["max_q"], cross["max_used"], causal=False,
        block_table=cross["table"], seqused_k=cross["used"])

    la = _tokens_first(lse_a).index_select(0, rows)
    lb = _tokens_first(lse_b)
    # online-softmax merge (Milakov & Gimelshein 2018):
    # (wa*A + wb*B)/(wa+wb) == A + (B-A)*sigmoid(lse_b - lse_a),
    # same merge, no fp32 copies of the row tensors
    w = torch.sigmoid(lb - la).to(torch.bfloat16)[..., None]
    merged = torch.lerp(out_a.index_select(0, rows), out_b, w)
    out = out_a.index_copy_(0, rows, merged)
    meta["layer"] += 1
    return out.view(n, H * D)


def set_path(pipeline, mode):
    """Point a Pipeline at a path by name, "split" included.

    Production modes assign attention_mode and clear the override;
    "split" keeps the merge_quant chunk layout and installs this
    module's implementation through the pipeline's override hook.
    """
    if mode == "split":
        pipeline.attention_mode = "merge_quant"
        pipeline.attention_override = functools.partial(
            attention_split, pipeline)
    else:
        pipeline.attention_override = None
        pipeline.attention_mode = mode
