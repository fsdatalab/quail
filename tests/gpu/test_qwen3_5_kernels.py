"""Fused Qwen3.5 and Qwen3.8 GPU kernels against their reference math."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")


def _static_fp8_reference(x, scale):
    scaled = x.float() / scale.float()
    return scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def test_gemma_add_norm_matches_reference_in_bf16_and_static_fp8():
    from quail.backends.quail.executor.models.qwen3_5_kernels import gemma_add_norm

    torch.manual_seed(0)
    rows, width = 73, 5120
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(width, device="cuda", dtype=torch.float32) * 0.1).to(
        torch.bfloat16)
    scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)
    inv_scale = (1.0 / scale).reshape(1)
    eps = 1e-6

    total = x.float() + residual.float()
    expected_res = total.to(torch.bfloat16)
    expected_bf16 = (
        total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps)
        * (1.0 + weight.float())
    ).to(torch.bfloat16)
    expected_fp8 = _static_fp8_reference(expected_bf16, scale)

    res_in = residual.clone()
    fp8_out, bf16_out, new_res = gemma_add_norm(
        x, res_in, weight, eps, inv_scale=inv_scale, write_bf16=True)
    assert new_res.data_ptr() == res_in.data_ptr()
    torch.testing.assert_close(new_res, expected_res, rtol=0, atol=0)
    torch.testing.assert_close(bf16_out, expected_bf16, rtol=1e-2, atol=1e-2)
    assert (fp8_out.float() - expected_fp8.float()).abs().max().item() <= 0.25

    x_inplace = x.clone()
    res_inplace = residual.clone()
    _, bf16_in, _ = gemma_add_norm(
        x_inplace, res_inplace, weight, eps, inplace_x=True)
    assert bf16_in.data_ptr() == x_inplace.data_ptr()
    torch.testing.assert_close(bf16_in, bf16_out, rtol=0, atol=0)


def test_gdn_gated_norm_reads_strided_gate_and_matches_rms_norm_gated():
    from vllm.model_executor.layers.layernorm import RMSNormGated

    from quail.backends.quail.executor.models.qwen3_5_kernels import gdn_gated_norm

    torch.manual_seed(1)
    rows, heads, dim, conv_dim = 55, 48, 128, 10240
    core = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    mixed = torch.randn(rows, conv_dim + heads * dim, device="cuda",
                        dtype=torch.bfloat16)
    z = mixed[:, conv_dim:]
    assert not z.is_contiguous()
    weight = torch.ones(dim, device="cuda", dtype=torch.bfloat16) * 1.15
    scale = torch.tensor(0.08, device="cuda", dtype=torch.float32)
    inv_scale = (1.0 / scale).reshape(1)
    eps = 1e-6

    for activation in ("silu", "sigmoid"):
        expected = RMSNormGated.forward_static(
            core.reshape(-1, dim), z.reshape(-1, dim), weight, eps,
            torch.bfloat16, norm_before_gate=True, activation=activation,
        ).reshape(rows, heads * dim)
        actual_bf16 = gdn_gated_norm(
            core, z, weight, eps, activation=activation)
        torch.testing.assert_close(actual_bf16, expected, rtol=1e-2, atol=1e-2)
        actual_fp8 = gdn_gated_norm(
            core, z, weight, eps, inv_scale=inv_scale, activation=activation)
        expected_fp8 = _static_fp8_reference(actual_bf16, scale)
        torch.testing.assert_close(
            actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)


def test_qkv_gemma_norm_rope_gate_matches_unfused_1d_and_mrope():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

    from quail.backends.quail.executor.models.qwen3_5_kernels import (
        qkv_gemma_norm_rope_gate,
    )

    torch.manual_seed(2)
    rows, q_heads, kv_heads, head_dim, rotary_dim = 41, 24, 4, 256, 64
    mrope_section = [11, 11, 10]
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        rotary = MRotaryEmbedding(
            head_size=head_dim, rotary_dim=rotary_dim,
            max_position_embeddings=512, base=10000.0, is_neox_style=True,
            dtype=torch.bfloat16, mrope_section=mrope_section,
            mrope_interleaved=True,
        ).cuda()
        qw = (torch.randn(head_dim, device="cuda", dtype=torch.float32)
              * 0.1).to(torch.bfloat16)
        kw = (torch.randn(head_dim, device="cuda", dtype=torch.float32)
              * 0.1).to(torch.bfloat16)
        eps = 1e-6

        def gemma_norm(t, w):
            tf = t.float()
            return (tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)
                    * (1.0 + w.float())).to(torch.bfloat16)

        attn = SimpleNamespace(
            num_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim,
            rotary_emb=rotary,
            q_norm=SimpleNamespace(weight=qw, variance_epsilon=eps),
            k_norm=SimpleNamespace(weight=kw, variance_epsilon=eps),
        )
        q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
        qkv = torch.randn(rows, 2 * q_size + 2 * kv_size, device="cuda",
                          dtype=torch.bfloat16)

        for positions in (
            torch.arange(rows, device="cuda", dtype=torch.int64),
            torch.stack([
                torch.arange(rows, device="cuda", dtype=torch.int64),
                torch.arange(rows, device="cuda", dtype=torch.int64) + 3,
                torch.arange(rows, device="cuda", dtype=torch.int64) + 7,
            ]),
        ):
            q_gate, k_raw, v_raw = qkv.split(
                [2 * q_size, kv_size, kv_size], dim=-1)
            q_raw, gate_ref = q_gate.view(
                rows, q_heads, 2 * head_dim).chunk(2, dim=-1)
            q_ref = gemma_norm(q_raw.contiguous(), qw).reshape(rows, q_size)
            k_ref = gemma_norm(k_raw.view(rows, kv_heads, head_dim).contiguous(),
                               kw).reshape(rows, kv_size)
            q_ref, k_ref = rotary(positions, q_ref.clone(), k_ref.clone())
            q, k, v, gate = qkv_gemma_norm_rope_gate(qkv, positions, attn)
            torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(k, k_ref, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(v, v_raw.contiguous(), rtol=0, atol=0)
            torch.testing.assert_close(
                gate, gate_ref.reshape(rows, q_size), rtol=0, atol=0)


def test_sigmoid_gate_matches_bf16_and_static_fp8_reference():
    from quail.backends.quail.executor.models.qwen3_5_kernels import sigmoid_gate

    torch.manual_seed(3)
    out = torch.randn(63, 6144, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(63, 6144, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.04, device="cuda", dtype=torch.float32)
    inv_scale = (1.0 / scale).reshape(1)
    expected_bf16 = out * torch.sigmoid(gate)
    expected_fp8 = _static_fp8_reference(expected_bf16, scale)

    actual_fp8 = sigmoid_gate(out, gate, inv_scale=inv_scale)
    torch.testing.assert_close(
        actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)
    inplace = out.clone()
    actual_bf16 = sigmoid_gate(inplace, gate)
    assert actual_bf16.data_ptr() == inplace.data_ptr()
    torch.testing.assert_close(actual_bf16, expected_bf16, rtol=1e-2, atol=1e-2)

    import vllm._custom_ops  # noqa: F401

    from quail.backends.quail.executor.models.qwen3_5_kernels import silu_mul

    for dtype in (torch.bfloat16, torch.float16):
        gate_up = torch.randn(63, 2 * 512, device="cuda", dtype=dtype) * 3
        expected_silu = torch.empty(63, 512, device="cuda", dtype=dtype)
        torch.ops._C.silu_and_mul(expected_silu, gate_up)
        torch.testing.assert_close(silu_mul(gate_up), expected_silu,
                                   rtol=0, atol=0)


def test_decode_conv_and_pool_gdn_match_reference():
    from experiments.hybrid import causal_conv_reference, gated_delta_reference
    from quail.backends.quail.executor.gdn import (
        GdnPhase,
        _decode_conv_kernel,
        _move_states,
        _pool_delta_rule,
    )

    torch.manual_seed(4)
    channels, width = 256, 4
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    arena = SimpleNamespace(
        conv_state=torch.randn(8, 2, width - 1, channels, device="cuda",
                               dtype=torch.bfloat16),
        gdn_state=torch.randn(8, 2, 1, 4, 4, device="cuda"),
    )
    # Include slot 0 (must not be masked as null) and a copied slot (2 -> 5).
    reads = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int64)
    writes = torch.tensor([0, 5, 4], device="cuda", dtype=torch.int64)
    history_ref = arena.conv_state[reads, 1].clone()
    x = torch.randn(3, channels, device="cuda", dtype=torch.bfloat16)
    expected_conv, expected_tails = causal_conv_reference(
        x, weight, (1, 1, 1), history=history_ref, bias=bias, activate=True)

    phase = GdnPhase(
        rows=torch.arange(3, device="cuda", dtype=torch.int64),
        cu_seqlens=torch.arange(4, device="cuda", dtype=torch.int32),
        lengths=(1, 1, 1),
        read_slots=reads, write_slots=writes,
        state_rows=writes + 1,
        copy_reads=torch.tensor([2], device="cuda", dtype=torch.int64),
        copy_writes=torch.tensor([5], device="cuda", dtype=torch.int64),
        write_slots_i32=writes.to(torch.int32),
        moves_in_place=True,
    )
    gdn_before = arena.gdn_state[2, 1].clone()
    _move_states(arena, phase, 1)
    torch.testing.assert_close(arena.gdn_state[5, 1], gdn_before, rtol=0, atol=0)
    actual_conv = _decode_conv_kernel(x, weight, bias, arena, phase, 1)
    torch.testing.assert_close(actual_conv, expected_conv, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        arena.conv_state[writes, 1], expected_tails, rtol=1e-2, atol=1e-2)

    if torch.cuda.get_device_capability()[0] < 9:
        return
    spec = SimpleNamespace(gdn_value_heads=48, gdn_key_heads=16,
                           gdn_key_dim=128, gdn_value_dim=128)
    lengths = (70, 128, 45)
    total = sum(lengths)
    q = torch.nn.functional.normalize(
        torch.randn(total, 16, 128, device="cuda").float(), dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(total, 16, 128, device="cuda").float(), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(total, 48, 128, device="cuda", dtype=torch.bfloat16)
    log_g = -torch.rand(total, 48, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(total, 48, device="cuda", dtype=torch.float32))
    cu = torch.tensor([0, 70, 198, 243], device="cuda", dtype=torch.int32)
    scale = 128 ** -0.5

    ref_state = torch.zeros(3, 48, 128, 128, device="cuda", dtype=torch.float32)
    ref_out = gated_delta_reference(q, k, v, log_g, beta, ref_state, lengths, scale)

    gdn_arena = SimpleNamespace(
        gdn_state=torch.zeros(8, 2, 48, 128, 128, device="cuda", dtype=torch.float32)
    )
    w_slots = torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)
    prefill_phase = GdnPhase(
        rows=torch.arange(total, device="cuda"), cu_seqlens=cu, lengths=lengths,
        write_slots=w_slots.long(), write_slots_i32=w_slots, unique_writes=True)
    actual_out = _pool_delta_rule(
        spec, q, k, v, log_g.exp().contiguous(), beta, gdn_arena,
        prefill_phase, 1)
    torch.testing.assert_close(actual_out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        gdn_arena.gdn_state[w_slots.long(), 1], ref_state, rtol=2e-2, atol=2e-2)

    # Read-only suffix phase with repeated read slots ([3, 0, 3]).
    r_slots = torch.tensor([3, 0, 3], device="cuda", dtype=torch.int32)
    suffix_ref_state = gdn_arena.gdn_state[r_slots.long(), 1].clone()
    suffix_ref = gated_delta_reference(
        q, k, v, log_g, beta, suffix_ref_state, lengths, scale)
    suffix_phase = GdnPhase(
        rows=torch.arange(total, device="cuda"), cu_seqlens=cu, lengths=lengths,
        read_slots=r_slots.long(), read_slots_i32=r_slots)
    suffix_out = _pool_delta_rule(
        spec, q, k, v, log_g.exp().contiguous(), beta, gdn_arena,
        suffix_phase, 1)
    torch.testing.assert_close(suffix_out, suffix_ref, rtol=2e-2, atol=2e-2)

    # Read+write extension phase updating slots in place.
    ext_ref_state = gdn_arena.gdn_state[w_slots.long(), 1].clone()
    ext_ref = gated_delta_reference(
        q, k, v, log_g, beta, ext_ref_state, lengths, scale)
    ext_phase = GdnPhase(
        rows=torch.arange(total, device="cuda"), cu_seqlens=cu, lengths=lengths,
        read_slots=w_slots.long(), read_slots_i32=w_slots,
        write_slots=w_slots.long(), write_slots_i32=w_slots, unique_writes=True)
    ext_out = _pool_delta_rule(
        spec, q, k, v, log_g.exp().contiguous(), beta, gdn_arena, ext_phase, 1)
    torch.testing.assert_close(ext_out, ext_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        gdn_arena.gdn_state[w_slots.long(), 1], ext_ref_state,
        rtol=2e-2, atol=2e-2)


def test_sm100_nvfp4_silu_and_linear_match_both_row_regimes():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100 GPU")
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    from quail.backends.quail.executor.models.qwen3_5_kernels import (
        nvfp4_linear,
        nvfp4_quant,
        silu_mul_nvfp4,
    )

    torch.manual_seed(5)
    in_dim, out_dim = 512, 256
    weight = torch.randint(0, 255, (out_dim, in_dim // 2), device="cuda",
                           dtype=torch.uint8)
    weight_scale = swizzle_blockscale(
        torch.ones((out_dim, in_dim // 16), device="cuda",
                   dtype=torch.float8_e4m3fn))
    module = SimpleNamespace(
        weight=weight, weight_scale=weight_scale,
        alpha=torch.tensor([0.01], device="cuda", dtype=torch.float32),
        input_global_scale_inv=torch.tensor([8.0], device="cuda",
                                            dtype=torch.float32),
        output_size_per_partition=out_dim, weights_padding_cols=0,
    )
    for rows in (64, 2560):
        gate_up = torch.randn(rows, 2 * in_dim, device="cuda",
                              dtype=torch.bfloat16)
        q_silu, s_silu = silu_mul_nvfp4(gate_up, module)
        out_silu = nvfp4_linear(q_silu, s_silu, module)
        assert out_silu.shape == (rows, out_dim)
        assert out_silu.dtype == torch.bfloat16
        assert torch.isfinite(out_silu).all()

        x = (torch.nn.functional.silu(gate_up[:, :in_dim].float())
             * gate_up[:, in_dim:].float()).to(torch.bfloat16)
        q_ref, s_ref = nvfp4_quant(x, module)
        out_ref = nvfp4_linear(q_ref, s_ref, module)
        torch.testing.assert_close(out_silu, out_ref, rtol=5e-2, atol=5e-2)


def test_gdn_gather_scatter_and_transient_plan_match_reference():
    import numpy as np

    from quail.backends.quail.executor.gdn import (
        gather_gdn_inputs,
        plan_phases,
        scatter_gdn_rows,
    )

    torch.manual_seed(6)
    mixed = torch.randn(37, 10240 + 6144, device="cuda", dtype=torch.bfloat16)
    qkv = mixed[:, :10240]
    ba = torch.randn(37, 96, device="cuda", dtype=torch.bfloat16)
    b, a = ba.chunk(2, dim=-1)
    rows = torch.tensor([5, 0, 36, 12, 5, 19], device="cuda", dtype=torch.int64)

    gq, ga, gb = gather_gdn_inputs(qkv, a, b, rows)
    torch.testing.assert_close(gq, qkv.index_select(0, rows), rtol=0, atol=0)
    torch.testing.assert_close(ga, a.index_select(0, rows), rtol=0, atol=0)
    torch.testing.assert_close(gb, b.index_select(0, rows), rtol=0, atol=0)

    core = torch.randn(5, 48, 128, device="cuda", dtype=torch.bfloat16)
    dst_rows = torch.tensor([31, 4, 18, 0, 25], device="cuda", dtype=torch.int64)
    out_actual = torch.zeros(37, 48, 128, device="cuda", dtype=torch.bfloat16)
    out_ref = out_actual.clone()
    scatter_gdn_rows(core, dst_rows, out_actual)
    out_ref.index_copy_(0, dst_rows, core)
    torch.testing.assert_close(out_actual, out_ref, rtol=0, atol=0)

    arena = SimpleNamespace(
        read_state=lambda keys: None,
        done_reading=lambda keys: None,
        accounting=SimpleNamespace(),
    )
    groups = [
        dict(key=0, fresh=True, f=10, start=0, row0=0, prefix_end=10,
             suffix_spans=(np.array([10]), np.array([14])),
             transient_state=True),
        dict(key=1, fresh=True, f=8, start=0, row0=14, prefix_end=22,
             suffix_spans=(np.array([22]), np.array([27])),
             transient_state=True),
    ]
    phases = plan_phases(
        arena, groups,
        lambda _name, vals, dtype: torch.as_tensor(vals, dtype=dtype,
                                                   device="cuda"))
    assert len(phases) == 1
    assert phases[0].contiguous_from_zero
    assert phases[0].read_slots is None
    assert phases[0].write_slots is None
    assert phases[0].lengths == (14, 13)




def test_gdn_decode_step_and_slot_copy_match_prep_and_recurrence():
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    from quail.backends.quail.executor.gdn import copy_slots, gdn_decode_step

    torch.manual_seed(7)
    heads, value_heads, dim, seqs = 16, 48, 128, 5
    channels = 2 * heads * dim + value_heads * dim
    # a strided pool: several layers per state row, as the arena keeps them
    rows_all = torch.randn(9, 3, value_heads, dim, dim, device="cuda") * 0.05
    pool = rows_all[:, 1]
    conv_out = torch.randn(seqs, channels + 64, device="cuda",
                           dtype=torch.bfloat16)[:, :channels]
    a = torch.randn(seqs, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(seqs, value_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(value_heads, device="cuda")
    dt_bias = torch.randn(value_heads, device="cuda")
    state_rows = torch.tensor([3, 1, 8, 5, 2], device="cuda")
    out_rows = torch.tensor([9, 0, 4, 11, 6], device="cuda")
    scale = dim ** -0.5

    saved = pool.clone()
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out.contiguous(), a=a, b=b, A_log=a_log,
        dt_bias=dt_bias, num_k_heads=heads, head_k_dim=dim, head_v_dim=dim,
        apply_l2norm=True, output_g_exp=False)
    expected, _ = fused_recurrent_gated_delta_rule(
        q[None], k[None], v[None], g[None], beta[None], scale=scale,
        initial_state=pool, inplace_final_state=True,
        cu_seqlens=torch.arange(seqs + 1, device="cuda", dtype=torch.int32),
        ssm_state_indices=state_rows)
    expected_state = pool.clone()
    pool.copy_(saved)

    out = torch.zeros(12, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    gdn_decode_step(conv_out, a, b, a_log, dt_bias, pool, state_rows, out,
                    num_key_heads=heads, scale=scale, out_rows=out_rows)
    torch.testing.assert_close(out[out_rows], expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool, expected_state, rtol=1e-4, atol=1e-5)
    untouched = torch.ones(12, dtype=torch.bool, device="cuda")
    untouched[out_rows] = False
    assert not out[untouched].any()

    src = torch.tensor([0, 4], device="cuda")
    dst = torch.tensor([6, 7], device="cuda")
    copy_slots(pool, src, dst)
    torch.testing.assert_close(pool[dst], pool[src], rtol=0, atol=0)
    conv_pool = torch.randn(6, 2, 3, channels, device="cuda",
                            dtype=torch.bfloat16)[:, 0]
    copy_slots(conv_pool, src[:1], torch.tensor([5], device="cuda"))
    torch.testing.assert_close(conv_pool[5], conv_pool[0], rtol=0, atol=0)


def _dense_attention(q, keys, values, scale):
    """Attention of each query row over its own visible keys, in fp32."""
    scores = torch.einsum("hd,khd->hk", q.float(), keys.float()) * scale
    weights = scores.softmax(-1)
    return torch.einsum("hk,khd->hd", weights, values.float())


def test_short_query_paged_attention_and_shared_reads_match_dense():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100 GPU")
    from quail.backends.quail.executor.attention import Engine

    torch.manual_seed(8)
    heads, kv_heads, dim, page = 24, 4, 256, 128
    group = heads // kv_heads
    scale = dim ** -0.5
    kp = torch.randn(40, page, kv_heads, dim, device="cuda",
                     dtype=torch.bfloat16)
    vp = torch.randn_like(kp)
    engine = Engine.__new__(Engine)
    engine.torch = torch
    engine.fa_version = 4
    engine.is_fp8 = False
    engine.fp8 = torch.float8_e4m3fn
    engine.use_ue8m0 = False
    engine.arena = SimpleNamespace(paged_kv=lambda layer: (kp, vp),
                                   sliding_layers=())

    def keys_of(pages, used):
        flat_k = kp[pages].reshape(-1, kv_heads, dim)[:used]
        flat_v = vp[pages].reshape(-1, kv_heads, dim)[:used]
        return (flat_k.repeat_interleave(group, dim=1),
                flat_v.repeat_interleave(group, dim=1))

    def table_of(page_lists):
        width = max(len(pages) for pages in page_lists)
        return torch.tensor([pages + [0] * (width - len(pages))
                             for pages in page_lists],
                            device="cuda", dtype=torch.int32)

    def reference(q3, lengths, page_lists, used):
        rows, start = [], 0
        for length, pages, seen in zip(lengths, page_lists, used):
            keys, values = keys_of(pages, seen)
            for j in range(length):
                visible = seen - length + j + 1
                rows.append(_dense_attention(q3[start + j], keys[:visible],
                                             values[:visible], scale))
            start += length
        return torch.stack(rows)

    def unified(lengths, page_lists, used, prefill_seqs=0):
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        meta = dict(
            cu_q=torch.tensor(cu, device="cuda", dtype=torch.int32),
            max_q=max(lengths), table=table_of(page_lists),
            used=torch.tensor(used, device="cuda", dtype=torch.int32),
            max_used=max(used), prefill_seqs=prefill_seqs, causal=True)
        if prefill_seqs:
            decode = lengths[prefill_seqs:]
            row0 = cu[prefill_seqs]
            meta.update(
                prefill_max_q=max(lengths[:prefill_seqs]), decode_row0=row0,
                decode_max_q=max(decode),
                decode_cu_q=meta["cu_q"][prefill_seqs:] - row0)
        return meta

    # one-token rows of few sequences, then of more than 32 sequences, then a
    # chunk of one long prefill and short decode rows
    cases = [
        ([1, 1, 1], [[3, 4], [7], [9, 10, 11]], [200, 90, 300], 0),
        ([1] * 36, [[i] for i in range(36)], [60 + i for i in range(36)], 0),
        ([3, 2, 4], [[3, 4], [7], [9, 10, 11]], [200, 90, 300], 0),
        ([40, 1, 2], [[20, 21], [7], [30, 31]], [170, 50, 140], 1),
    ]
    for lengths, page_lists, used, prefill_seqs in cases:
        q3 = torch.randn(sum(lengths), heads, dim, device="cuda",
                         dtype=torch.bfloat16)
        meta = dict(layer=0, kv_written=True,
                    unified=unified(lengths, page_lists, used, prefill_seqs))
        out = engine.attention_unified(q3, None, None, meta).view_as(q3)
        torch.testing.assert_close(
            out.float(), reference(q3, lengths, page_lists, used),
            rtol=2e-2, atol=2e-2)

    # trie decode branches: each row reads its own pages causally and its
    # document's shared pages, merged by log-sum-exp weights
    own_pages, own_used = [[12], [13], [14]], [3, 2, 4]
    shared_pages, shared_used = [5, 6], 200
    q3 = torch.randn(3, heads, dim, device="cuda", dtype=torch.bfloat16)
    meta = dict(
        layer=0, kv_written=True, unified=unified([1, 1, 1], own_pages, own_used),
        shared=dict(
            rows=torch.arange(3, device="cuda"),
            cu_q=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
            max_q=3, table=table_of([shared_pages]),
            used=torch.tensor([shared_used], device="cuda", dtype=torch.int32),
            max_used=shared_used,
            source=torch.arange(3, device="cuda", dtype=torch.int32)))
    out, _ = engine.attention_shared(q3, None, None, meta)
    shared_k, shared_v = keys_of(shared_pages, shared_used)
    expected = []
    for row, (pages, seen) in enumerate(zip(own_pages, own_used)):
        keys, values = keys_of(pages, seen)
        expected.append(_dense_attention(
            q3[row], torch.cat([shared_k, keys]), torch.cat([shared_v, values]),
            scale))
    torch.testing.assert_close(out.view(3, heads, dim).float(),
                               torch.stack(expected), rtol=2e-2, atol=2e-2)
