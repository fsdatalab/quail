"""Verifiable GPU evaluator for fast Triton launches, PDL, and split-KV decode."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")


def _dense_causal_paged(q3, kp, vp, lengths, page_lists, used, scale,
                        window=None):
    """Exact fp32 causal paged attention reference."""
    heads, kv_heads, dim = q3.shape[1], kp.shape[2], q3.shape[2]
    group = heads // kv_heads
    rows, start = [], 0
    for q_len, pages, seen in zip(lengths, page_lists, used):
        flat_k = kp[pages].reshape(-1, kv_heads, dim)[:seen].repeat_interleave(
            group, dim=1)
        flat_v = vp[pages].reshape(-1, kv_heads, dim)[:seen].repeat_interleave(
            group, dim=1)
        for offset in range(q_len):
            visible = seen - q_len + offset + 1
            low = 0 if window is None else max(0, visible - 1 - window[0])
            scores = torch.einsum(
                "hd,khd->hk", q3[start + offset].float(),
                flat_k[low:visible].float()) * scale
            weights = scores.softmax(-1)
            rows.append(torch.einsum(
                "hk,khd->hd", weights, flat_v[low:visible].float()))
        start += q_len
    return torch.stack(rows)


def test_fast_launch_is_bitwise_identical_to_standard_triton():
    """Verify direct compiled.run matches standard kernel[grid] bit-for-bit."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

    from quail.backends.quail.executor import (
        decode_attention,
        gdn_kernels,
        triton_launch,
    )
    from quail.backends.quail.executor.models import qwen3_5_kernels as k

    torch.manual_seed(101)
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    with set_current_vllm_config(VllmConfig()):
        rotary = MRotaryEmbedding(
            head_size=256, rotary_dim=64, max_position_embeddings=2048,
            base=10_000_000, is_neox_style=True, dtype=torch.bfloat16,
            mrope_section=[11, 11, 10]).to("cuda")
    attn = SimpleNamespace(
        num_heads=24, num_kv_heads=4, head_dim=256, rotary_emb=rotary,
        q_norm=SimpleNamespace(weight=torch.randn(256, device="cuda",
                                                  dtype=torch.bfloat16),
                               variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.randn(256, device="cuda",
                                                  dtype=torch.bfloat16),
                               variance_epsilon=1e-6),
    )
    kp = torch.randn(32, 128, 4, 256, device="cuda", dtype=torch.bfloat16) * 0.2
    vp = torch.randn_like(kp)
    saved_fast = triton_launch.USE_FAST_LAUNCH
    try:
        for rows in (1, 7, 16, 45, 131):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            res = torch.randn_like(x)
            w = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
            core = torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
            z = torch.randn(rows, 16384, device="cuda",
                            dtype=torch.bfloat16)[:, 10240:]
            gw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            qkv = torch.randn(rows, 24 * 512 + 8 * 256, device="cuda",
                              dtype=torch.bfloat16)
            pos = torch.arange(rows, device="cuda", dtype=torch.int64)
            out_att = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(out_att)
            pool = torch.randn(rows + 2, 48, 128, 128, device="cuda") * 0.05
            conv_out = torch.randn(rows, 10240, device="cuda",
                                   dtype=torch.bfloat16)
            a = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
            a_log = torch.randn(48, device="cuda")
            dt_bias = torch.randn(48, device="cuda")
            state_rows = torch.arange(1, rows + 1, device="cuda")
            q3 = torch.randn(rows, 24, 256, device="cuda", dtype=torch.bfloat16)
            cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
            used = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
            table = (torch.arange(rows, device="cuda", dtype=torch.int32)
                     % 32).view(rows, 1)

            results = []
            for fast in (False, True):
                triton_launch.USE_FAST_LAUNCH = fast
                r_copy = res.clone()
                fp8, bf16, _ = k.gemma_add_norm(
                    x, r_copy, w, 1e-6, inv_scale=scale, write_bf16=True)
                gn = k.gdn_gated_norm(core, z, gw, 1e-6, inv_scale=scale)
                q, kk, vv, gg = k.qkv_gemma_norm_rope_gate(qkv, pos, attn)
                sg = k.sigmoid_gate(out_att, gate, inv_scale=scale)
                p_copy = pool.clone()
                gdn_out = torch.empty(rows, 48, 128, device="cuda",
                                      dtype=torch.bfloat16)
                gdn_kernels.gdn_decode_step(
                    conv_out, a, b, a_log, dt_bias, p_copy, state_rows,
                    gdn_out, num_key_heads=16, scale=128 ** -0.5)
                split_out = decode_attention.triton_splitkv_decode(
                    q3, kp, vp, cu, 1, used, 128, table)
                results.append((fp8, bf16, r_copy, gn, q, kk, vv, gg, sg,
                                gdn_out, p_copy, split_out))
            for ref, got in zip(*results):
                assert torch.equal(got, ref)
    finally:
        triton_launch.USE_FAST_LAUNCH = saved_fast


def test_pdl_dependent_chains_are_race_free_and_bitwise_identical():
    """Run 200 randomized dependent kernel chains with PDL on vs off."""
    from quail.backends.quail.executor import (
        decode_attention,
        gdn_kernels,
        triton_launch,
    )
    from quail.backends.quail.executor.models import qwen3_5_kernels as k

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL requires Hopper or Blackwell")
    torch.manual_seed(202)
    rows = 45
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    w_norm = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
    w_gdn = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(48, device="cuda")
    dt_bias = torch.randn(48, device="cuda")
    kp = torch.randn(64, 128, 4, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    vp = torch.randn_like(kp)
    table = torch.arange(64, device="cuda", dtype=torch.int32).view(16, 4)
    used = torch.full((16,), 500, device="cuda", dtype=torch.int32)
    cu = torch.arange(17, device="cuda", dtype=torch.int32)

    def run_chain(use_pdl, x, res, conv_out, a, b, z, pool, q3, gate):
        saved = triton_launch.USE_PDL
        triton_launch.USE_PDL = use_pdl
        try:
            r = res.clone()
            p = pool.clone()
            fp8, bf16, _ = k.gemma_add_norm(
                x, r, w_norm, 1e-6, inv_scale=scale, write_bf16=True)
            gdn_out = torch.empty(rows, 48, 128, device="cuda",
                                  dtype=torch.bfloat16)
            state_rows = torch.arange(1, rows + 1, device="cuda")
            gdn_kernels.gdn_decode_step(
                conv_out, a, b, a_log, dt_bias, p, state_rows, gdn_out,
                num_key_heads=16, scale=128 ** -0.5)
            gn = k.gdn_gated_norm(gdn_out, z, w_gdn, 1e-6, inv_scale=scale)
            attn_out = decode_attention.triton_splitkv_decode(
                q3, kp, vp, cu, 1, used, 500, table)
            gated = k.sigmoid_gate(attn_out.view(16, -1), gate, inv_scale=scale)
            return fp8, bf16, r, gn, p, gated
        finally:
            triton_launch.USE_PDL = saved

    for iteration in range(200):
        torch.manual_seed(1000 + iteration)
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        res = torch.randn_like(x)
        conv_out = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
        z = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
        pool = torch.randn(rows + 2, 48, 128, 128, device="cuda") * 0.05
        q3 = torch.randn(16, 24, 256, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(16, 6144, device="cuda", dtype=torch.bfloat16)
        ref = run_chain(False, x, res, conv_out, a, b, z, pool, q3, gate)
        got = run_chain(True, x, res, conv_out, a, b, z, pool, q3, gate)
        for lhs, rhs in zip(ref, got):
            assert torch.equal(lhs, rhs), f"PDL mismatch at iteration {iteration}"


def test_triton_splitkv_decode_accuracy_zero_recompiles_and_speed():
    """Verify accuracy against fp32 dense attention and zero mid-query compiles."""
    from quail.backends.quail.executor import decode_attention as d

    torch.manual_seed(303)
    heads, kv_heads, dim, page = 24, 4, 256, 128
    scale = dim ** -0.5
    total_pages = 3000
    kp = torch.randn(total_pages, page, kv_heads, dim, device="cuda",
                     dtype=torch.bfloat16) * 0.25
    vp = torch.randn_like(kp)
    workspace = {}

    # Reject small page sizes that violate page_tokens % TILE_N == 0.
    assert not d.supports_splitkv_decode(
        4, 256, 16, 24, 4, causal=True, max_q=1, dtype=torch.bfloat16)
    small_kp = torch.randn(2, 16, kv_heads, dim, device="cuda",
                           dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="page_tokens divisible"):
        d.triton_splitkv_decode(
            torch.randn(1, heads, dim, device="cuda", dtype=torch.bfloat16),
            small_kp, small_kp,
            torch.tensor([0, 1], device="cuda", dtype=torch.int32), 1,
            torch.tensor([16], device="cuda", dtype=torch.int32), 16,
            torch.tensor([[0]], device="cuda", dtype=torch.int32))

    def run_case(lengths, page_lists, used, *, slice_offset=0, window=None):
        width = max(len(pages) for pages in page_lists)
        raw_table = torch.zeros(len(lengths) + slice_offset, width,
                                device="cuda", dtype=torch.int32)
        for index, pages in enumerate(page_lists):
            raw_table[slice_offset + index, :len(pages)] = torch.tensor(
                pages, dtype=torch.int32)
        raw_used = torch.zeros(len(lengths) + slice_offset,
                               device="cuda", dtype=torch.int32)
        raw_used[slice_offset:] = torch.tensor(used, dtype=torch.int32)
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        raw_cu = torch.zeros(len(cu) + slice_offset,
                             device="cuda", dtype=torch.int32)
        raw_cu[slice_offset:] = torch.tensor(cu, dtype=torch.int32)
        q3 = torch.randn(sum(lengths), heads, dim, device="cuda",
                         dtype=torch.bfloat16)
        out = d.triton_splitkv_decode(
            q3, kp, vp, raw_cu[slice_offset:], max(lengths),
            raw_used[slice_offset:], max(used), raw_table[slice_offset:],
            softmax_scale=scale, window=window, workspace=workspace)
        ref = _dense_causal_paged(
            q3, kp, vp, lengths, page_lists, used, scale, window=window)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    # Warm the two split regimes (single-split and multi-split).
    run_case([1], [[0]], [64])
    run_case([1], [[0, 1]], [256])
    device = torch.cuda.current_device()
    part_cache = d._kernels()["splitkv_decode_part"].device_caches[device][0]
    reduce_cache = d._kernels()["splitkv_decode_reduce"].device_caches[device][0]
    part_count_before = len(part_cache)
    reduce_count_before = len(reduce_cache)

    # Diverse batch sizes, ragged query lengths, unaligned slices, and contexts.
    cases = [
        ([1], [[5]], [1], 1),
        ([1], [[5]], [127], 3),
        ([1], [[5, 6, 7, 8]], [512], 2),
        ([1, 1, 1], [[3, 4], [7], [9, 10, 11]], [200, 90, 300], 1),
        ([3, 1, 4, 2], [[1, 2], [3], [4, 5, 6], [7, 8]], [180, 65, 340, 210], 3),
        ([1] * 16, [[i * 4 + j for j in range(4)] for i in range(16)],
         [450 + i for i in range(16)], 1),
        ([1] * 45, [[i * 5 + j for j in range(5)] for i in range(45)],
         [520 + i * 2 for i in range(45)], 2),
        ([1] * 160, [[i * 2 + j for j in range(2)] for i in range(160)],
         [200 + (i % 50) for i in range(160)], 3),
        ([16, 8, 12], [[10, 11], [12, 13, 14], [15, 16]], [220, 310, 190], 1),
    ]
    for lengths, page_lists, used, offset in cases:
        run_case(lengths, page_lists, used, slice_offset=offset)
    # Zero recompilations across every single shape, stride, and alignment!
    assert len(part_cache) == part_count_before
    assert len(reduce_cache) == reduce_count_before

    # Sliding window accuracy check.
    run_case([1, 3], [[1, 2, 3], [4, 5, 6]], [320, 350], window=(128, 0))

    # Measure B200 latency on CUAD-20 (45x8000), CUAD-1 (16x8192), SALES-8 (160x1280).
    for label, seqs, ctx in (
        ("CUAD-20", 45, 7936),
        ("CUAD-1", 16, 8192),
        ("SALES-8", 160, 1280),
    ):
        pages_per = (ctx + page - 1) // page
        table = torch.arange(seqs * pages_per, device="cuda",
                             dtype=torch.int32).view(seqs, pages_per) % total_pages
        used = torch.full((seqs,), ctx, device="cuda", dtype=torch.int32)
        cu = torch.arange(seqs + 1, device="cuda", dtype=torch.int32)
        q3 = torch.randn(seqs, heads, dim, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(q3)
        for _ in range(5):
            d.triton_splitkv_decode(q3, kp, vp, cu, 1, used, ctx, table,
                                    softmax_scale=scale, output=out,
                                    workspace=workspace)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            d.triton_splitkv_decode(q3, kp, vp, cu, 1, used, ctx, table,
                                    softmax_scale=scale, output=out,
                                    workspace=workspace)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) / 30 * 1000.0
        print(f"splitkv_decode {label} ({seqs}x{ctx}): {us:.1f} us/layer",
              flush=True)
