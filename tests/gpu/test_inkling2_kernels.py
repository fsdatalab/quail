"""Inkling 2 Mini's kernels against plain PyTorch math, on a GPU.

Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword inkling2_kernels
--device b200`.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

HEADS, KV_HEADS, DIM, RANK, EXTENT, PAGE = 24, 8, 128, 16, 1024, 128


def _dense_attention(q, k, v, rel, sink, proj, window):
    """One sequence's attention: queries at the last positions of the keys."""
    rows, used = q.shape[0], k.shape[0]
    group = HEADS // KV_HEADS
    kk = k.float().repeat_interleave(group, dim=1)
    vv = v.float().repeat_interleave(group, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), kk) * DIM ** -0.5
    qpos = torch.arange(used - rows, used, device=q.device)
    dist = qpos[:, None] - torch.arange(used, device=q.device)[None, :]
    profile = torch.einsum("qhr,re->hqe", rel.float(), proj.float())
    bias = profile.gather(2, dist.clamp(0, EXTENT - 1).expand(HEADS, -1, -1))
    bias = bias.masked_fill((dist < 0) | (dist >= EXTENT), 0.0)
    logits = logits + bias
    keep = dist >= 0
    if window:
        keep &= dist < window
    logits = logits.masked_fill(~keep, -math.inf)
    full = torch.cat([logits, sink.float().t()[:, :, None]], dim=2)
    weights = full.softmax(-1)[..., :-1]
    return torch.einsum("hqk,khd->qhd", weights, vv)


@pytest.mark.parametrize("window", [0, 1024])
def test_relative_attention_matches_dense_attention(window):
    from quail.backends.quail.executor.models import inkling2_kernels as kernels

    torch.manual_seed(0)
    # (queries, keys): a long prefill past the window, a short question
    # after a resident document, and two one-token steps
    shapes = [(1300, 1500), (40, 700), (1, 2100), (1, 9)]
    pages_per = [-(-used // PAGE) for _, used in shapes]
    num_pages = sum(pages_per) + 3
    kp = torch.randn(num_pages, PAGE, KV_HEADS, DIM, device="cuda",
                     dtype=torch.bfloat16)
    vp = torch.randn_like(kp)
    order = torch.randperm(num_pages, device="cuda")
    table = torch.zeros(len(shapes), max(pages_per), dtype=torch.int32,
                        device="cuda")
    start = 0
    for index, count in enumerate(pages_per):
        table[index, :count] = order[start:start + count].int()
        start += count
    total = sum(rows for rows, _ in shapes)
    q = torch.randn(total, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    q = q / q.float().norm(dim=-1, keepdim=True).to(q.dtype) * 4
    rel = torch.randn(total, HEADS, RANK, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(total, HEADS, device="cuda", dtype=torch.bfloat16)
    proj = torch.randn(RANK, EXTENT, device="cuda", dtype=torch.bfloat16) * 0.3
    cu = [0]
    for rows, _ in shapes:
        cu.append(cu[-1] + rows)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    used = torch.tensor([u for _, u in shapes], dtype=torch.int32, device="cuda")
    for block_rows in (16, 64, 128):
        seqs, firsts = kernels.attention_tiles(cu, block_rows // 3)
        tiles = (torch.from_numpy(seqs).cuda(), torch.from_numpy(firsts).cuda())
        out = torch.empty_like(q)
        kernels.rel_attention(torch, q, kp, vp, rel, sink, proj, out, tiles,
                              cu_q, used, table, window=window,
                              max_keys=max(u for _, u in shapes),
                              block_rows=block_rows)
        for index, (rows, keys) in enumerate(shapes):
            pages = table[index, :pages_per[index]].long()
            k = kp[pages].reshape(-1, KV_HEADS, DIM)[:keys]
            v = vp[pages].reshape(-1, KV_HEADS, DIM)[:keys]
            span = slice(cu[index], cu[index + 1])
            expected = _dense_attention(q[span], k, v, rel[span], sink[span],
                                        proj, window)
            error = (out[span].float() - expected).abs().max().item()
            assert error < 2e-2, (block_rows, index, error)


def test_the_block_mix_matches_its_definition():
    from quail.backends.quail.executor.models import inkling2_kernels as kernels

    torch.manual_seed(1)
    rows, hidden = 37, 2304
    blocks = torch.randn(rows, 4, hidden, device="cuda", dtype=torch.bfloat16)
    partial = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    addend = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(3, hidden, device="cuda") * hidden ** -0.5
    weight = torch.rand(3, hidden, device="cuda", dtype=torch.bfloat16) + 0.5
    out = kernels.mix(torch, blocks, 2, partial, query, addend=addend,
                      weight=weight)
    states = torch.stack([blocks[:, 0].float(), blocks[:, 1].float(),
                          partial.float() + addend.float()], dim=1)
    normed = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + 1e-6)
    weights = torch.einsum("rsh,qh->rqs", normed, query).softmax(-1)
    mixed = torch.einsum("rqs,rsh->qrh", weights, normed)
    expected = (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + 1e-6)
                * weight.float()[:, None, :])
    assert (out.float() - expected).abs().max().item() < 3e-2


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _unswizzle(scales, rows, blocks):
    """Row-major (rows, blocks) values of CUTLASS's swizzled scales."""
    padded_rows = -(-rows // 128) * 128
    padded_blocks = -(-blocks // 4) * 4
    tiles = scales[:padded_rows].reshape(
        padded_rows // 128, padded_blocks // 4, 32, 4, 4)
    return tiles.permute(0, 3, 2, 1, 4).reshape(
        padded_rows, padded_blocks)[:rows, :blocks]


def _dequantize(packed, scales, row_scale):
    table = torch.tensor(E2M1, device=packed.device)
    codes = packed.long()
    values = torch.stack([table[codes & 15], table[codes >> 4]], -1)
    values = values.reshape(packed.shape[0], -1, 16) * scales.float()[..., None]
    return values.reshape(packed.shape[0], -1) * row_scale[:, None]


def test_nvfp4_rows_round_trip_and_the_expert_gemm_matches_dequantized_math():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    from quail.backends.quail.executor.models import inkling2_kernels as kernels

    torch.manual_seed(2)
    tokens, experts, topk, hidden, inter = 300, 16, 4, 2304, 1152
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, device="cuda")[:topk]
                       for _ in range(tokens)]).int()
    rows = tokens * topk
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
    sf_offsets = torch.empty_like(offsets)
    sizes1 = torch.empty((experts, 3), dtype=torch.int32, device="cuda")
    sizes2 = torch.empty_like(sizes1)
    a_map = torch.empty(rows, dtype=torch.int32, device="cuda")
    c_map = torch.empty_like(a_map)
    ops.get_cutlass_moe_mm_data(ids, offsets, sizes1, sizes2, a_map, c_map,
                                experts, inter, hidden, sf_offsets)
    q, sf, row_scale = kernels.quantize_tokens(
        torch, x, c_map, ids, offsets, sf_offsets, nvfp4=True,
        sf_rows=rows + 128 * experts)
    # each expert's scales start at its own 128-row tile
    flat = torch.empty((rows, hidden // 16), dtype=torch.float8_e4m3fn,
                       device="cuda")
    starts, sf_starts = offsets.tolist(), sf_offsets.tolist()
    blocks = _unswizzle(sf, sf.shape[0], hidden // 16)
    for e in range(experts):
        count = starts[e + 1] - starts[e]
        flat[starts[e]:starts[e + 1]] = blocks[sf_starts[e]:sf_starts[e] + count]
    restored = _dequantize(q, flat, row_scale)
    reference = x.float()[a_map.long()]
    relative = ((restored - reference).norm() / reference.norm()).item()
    assert relative < 0.12, relative

    # the grouped GEMM over random NVFP4 weights against dequantized math
    w = torch.randint(0, 256, (experts, 2 * inter, hidden // 2),
                      dtype=torch.uint8, device="cuda")
    w_scales = (torch.rand(experts, 2 * inter, hidden // 16, device="cuda")
                + 0.5).to(torch.float8_e4m3fn)
    alpha = torch.rand(experts, device="cuda") + 0.5
    out = torch.empty((rows, 2 * inter), dtype=torch.bfloat16, device="cuda")
    ops.cutlass_fp4_moe_mm(out, q, w, sf, swizzle_blockscale(w_scales), alpha,
                           sizes1, offsets[:-1], sf_offsets[:-1])
    weights = _dequantize(w.reshape(-1, hidden // 2),
                          w_scales.reshape(-1, hidden // 16),
                          torch.ones(experts * 2 * inter, device="cuda"))
    weights = weights.reshape(experts, 2 * inter, hidden)
    restored_unit = _dequantize(q, flat, torch.ones(rows, device="cuda"))
    for e in range(experts):
        part = slice(starts[e], starts[e + 1])
        expected = restored_unit[part] @ weights[e].t() * alpha[e]
        error = ((out[part].float() - expected).norm()
                 / expected.norm().clamp(min=1e-6)).item()
        assert error < 1e-2, (e, error)


def test_the_message_gates_match_their_definition():
    from quail.backends.quail.executor.models import inkling2_kernels as kernels

    torch.manual_seed(3)
    gates, rows, hidden, rank = 3, 29, 2304, 64
    raw = torch.randn(gates, rows, hidden, device="cuda").bfloat16()
    prev_block = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    prev_out = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(gates, hidden, device="cuda") * hidden ** -0.5
    down_t = (torch.randn(gates, 2 * hidden, rank, device="cuda")
              * 0.02).bfloat16()
    gains_t = (torch.randn(gates, rank, 2 * hidden, device="cuda")
               * 0.3).bfloat16()
    weight = torch.rand(gates, hidden, device="cuda", dtype=torch.bfloat16) + 0.5

    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    a, b = prev_block.float(), prev_out.float()
    states = torch.stack([norm(a), norm(b)], 1)
    weights = torch.einsum("rsh,gh->grs", states, query).softmax(-1)
    message = torch.einsum("grs,rsh->grh", weights, states)

    def expected(hidden_rows):
        inputs = torch.cat([norm(hidden_rows.float()).expand(gates, -1, -1),
                            norm(message)], -1)
        down = (inputs @ down_t.float()).bfloat16().float()
        gains = down @ gains_t.float()
        mixed = ((1 - gains[..., :hidden].tanh()) * hidden_rows.float()
                 + gains[..., hidden:].tanh() * message)
        return norm(mixed) * weight.float()[:, None, :]

    # the messages' products, as a loop pass forms them once
    down_m = down_t[:, hidden:].float()
    pa = torch.cat(list(a @ down_m), -1)
    pb = torch.cat(list(b @ down_m), -1)
    scores = torch.stack([a @ query.t(), b @ query.t()], -1).flatten(1)
    stats = torch.stack([(a * a).sum(-1), (b * b).sum(-1), (a * b).sum(-1)],
                        -1)
    ph = torch.bmm(raw.float(), down_t[:, :hidden].float())
    squares = raw.float().pow(2).sum(-1)
    out = kernels.gate_out(torch, raw, squares, prev_block, prev_out, ph, pa,
                           pb, scores, stats, gains_t, weight)
    assert (out.float() - expected(raw)).abs().max().item() < 3e-2
    # one hidden row shared by every gate
    ph = torch.cat(list(raw[0].float() @ down_t[:, :hidden].float()), -1)
    shared = kernels.gate_out(torch, raw[0], squares[:1], prev_block, prev_out,
                              ph, pa, pb, scores, stats, gains_t, weight)
    assert (shared.float() - expected(raw[0])).abs().max().item() < 3e-2


def test_the_fused_experts_agree_with_quails_expert_path(monkeypatch):
    from types import SimpleNamespace

    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    from quail.backends.quail.executor.models import inkling2
    from quail.backends.quail.executor.models.inkling2 import Inkling2Pipeline

    torch.manual_seed(5)
    experts, topk, hidden, inter, tokens = 128, 8, 2304, 1152, 4200

    def weight(rows, cols):
        values = torch.randint(0, 256, (experts, rows, cols // 2),
                               dtype=torch.uint8, device="cuda")
        scales = (torch.rand(experts, rows, cols // 16, device="cuda") * 0.5
                  + 0.25).to(torch.float8_e4m3fn)
        return values, scales

    # FC1 in the checkpoint's order: (gate, up) row pairs
    w13, w13_s = weight(2 * inter, hidden)
    w2, w2_s = weight(hidden, inter)
    w13 = inkling2._fused_order(torch, w13)
    w13_s = swizzle_blockscale(inkling2._fused_order(torch, w13_s))
    w2_s = swizzle_blockscale(w2_s)
    layer = SimpleNamespace(
        nvfp4=True, w13=w13, w2=w2, w13_scale=w13_s, w2_scale=w2_s,
        w13_mma=convert_sf_to_mma_layout(w13_s.view(experts * 2 * inter, -1),
                                         m=2 * inter, k=hidden,
                                         num_groups=experts),
        w2_mma=convert_sf_to_mma_layout(w2_s.view(experts * hidden, -1),
                                        m=hidden, k=inter, num_groups=experts),
        w13_alpha=torch.rand(experts, device="cuda") * 0.02 + 0.01,
        w2_alpha=torch.rand(experts, device="cuda") * 0.02 + 0.01,
        router=torch.randn(experts, hidden, device="cuda",
                           dtype=torch.bfloat16) * 0.05,
        router_bias=torch.zeros(experts, device="cuda"))
    pipeline = Inkling2Pipeline.__new__(Inkling2Pipeline)
    pipeline.torch, pipeline.hidden = torch, hidden
    pipeline._fp32_out = pipeline._fc2_scale = None
    pipeline.config = dict(n_routed_experts=experts, num_experts_per_tok=topk,
                           intermediate_size=inter, route_scale=8.0,
                           mlp_gate_softcap=4.0, mlp_up_softcap=25.0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    fused = pipeline._experts(layer, x).float()
    monkeypatch.setattr(inkling2, "FUSED_MIN_TOKENS", tokens)
    quail = pipeline._experts(layer, x).float()
    cosine = torch.nn.functional.cosine_similarity(
        fused.flatten(), quail.flatten(), dim=0).item()
    assert cosine > 0.99, cosine


def test_routing_picks_the_top_experts_by_biased_score():
    from quail.backends.quail.executor.models import inkling2_kernels as kernels

    torch.manual_seed(6)
    rows, experts, top_k = 1000, 128, 8
    logits = torch.randn(rows, experts, device="cuda") * 2
    bias = torch.randn(experts, device="cuda") * 0.1
    ids, weights = kernels.route(torch, logits, bias, top_k, 8.0)
    scores = logits.sigmoid()
    expected = torch.topk(scores + bias, top_k, dim=-1).indices
    assert torch.equal(ids.long().sort(-1).values, expected.sort(-1).values)
    picked = scores.gather(1, ids.long())
    assert torch.allclose(weights, picked / picked.sum(-1, keepdim=True) * 8,
                          rtol=1e-5, atol=1e-6)
