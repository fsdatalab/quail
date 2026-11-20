"""Fused Triton and SM100 kernels for Qwen3.5 and Qwen3.8 layers."""

from quail.backends.quail.executor.attention import launch_triton, pdl_enabled

_KERNELS = {}
_SM100_CAPABLE = {}
# Measured B200 (SM100) crossovers for Qwen3.8 27B widths: CUTLASS leads up to
# 2,048 rows; above 16,384 rows, 8,192-row slices give 10-14% higher throughput.
_FP4_CUTLASS_MAX_ROWS = 2_048
_ROW_TILE_ABOVE = 16_384
ROW_TILE = 8_192


def _kernels() -> dict:
    """Compile the Qwen3.5 Triton kernels on first use inside a GPU process."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    # PDL invariant: loads before gdc_wait read only weights, scales, or positions.
    @triton.jit
    def gemma_add_norm_fp8(
        x_ptr, res_ptr, w_ptr, fp8_ptr, bf16_ptr, stride_x, stride_res,
        stride_fp8, stride_bf16, inv_scale_ptr, eps, H: tl.constexpr,
        BLOCK: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
        WRITE_FP8: tl.constexpr, WRITE_BF16: tl.constexpr, PDL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < H
        scale = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if WRITE_FP8:
            inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
        total = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                        other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            total += tl.load(res_ptr + row * stride_res + offs,
                             mask=mask, other=0.0).to(tl.float32)
        normed = (total * tl.math.rsqrt(tl.sum(total * total, axis=0) / H + eps)
                  * scale).to(tl.bfloat16)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        if HAS_RESIDUAL:
            tl.store(res_ptr + row * stride_res + offs,
                     total.to(res_ptr.dtype.element_ty), mask=mask)
        if WRITE_BF16:
            tl.store(bf16_ptr + row * stride_bf16 + offs, normed, mask=mask)
        if WRITE_FP8:
            clamped = tl.minimum(tl.maximum(
                normed.to(tl.float32) * inv_scale, -448.0), 448.0)
            tl.store(fp8_ptr + row * stride_fp8 + offs,
                     clamped.to(fp8_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def gdn_gated_norm(
        core_ptr, z_ptr, w_ptr, out_ptr, inv_scale_ptr, stride_core,
        stride_z, stride_out, eps, HEADS: tl.constexpr, D: tl.constexpr,
        HEADS_PER_BLOCK: tl.constexpr, BLOCK: tl.constexpr,
        SILU_GATE: tl.constexpr, WRITE_FP8: tl.constexpr, PDL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        h_offs = (tl.program_id(1).to(tl.int64) * HEADS_PER_BLOCK
                  + tl.arange(0, HEADS_PER_BLOCK))
        d_offs = tl.arange(0, BLOCK)
        d_mask = d_offs < D
        mask = (h_offs < HEADS)[:, None] & d_mask[None, :]
        offsets = h_offs[:, None] * D + d_offs[None, :]
        w = tl.load(w_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        if WRITE_FP8:
            inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
        x = tl.load(core_ptr + row * stride_core + offsets,
                    mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row * stride_z + offsets,
                    mask=mask, other=0.0).to(tl.float32)
        normed = (x * tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)[:, None]
                  * w[None, :])
        sig = tl.sigmoid(z)
        gated = (normed * (z * sig if SILU_GATE else sig)).to(tl.bfloat16)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        if WRITE_FP8:
            clamped = tl.minimum(tl.maximum(
                gated.to(tl.float32) * inv_scale, -448.0), 448.0)
            tl.store(out_ptr + row * stride_out + offsets,
                     clamped.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            tl.store(out_ptr + row * stride_out + offsets, gated, mask=mask)

    @triton.jit
    def qkv_gemma_norm_rope_gate(
        qkv_ptr, q_out_ptr, k_out_ptr, v_out_ptr, gate_out_ptr,
        qw_ptr, kw_ptr, cs_ptr, positions_ptr, stride_qkv, stride_q,
        stride_k, stride_v, stride_gate, stride_cs, stride_pm, stride_pt,
        eps, QH: tl.constexpr, KH: tl.constexpr, HD: tl.constexpr,
        ROT: tl.constexpr, HALF_ROT: tl.constexpr, HEAD_BLOCK: tl.constexpr,
        ROT_BLOCK: tl.constexpr, V_BLOCK: tl.constexpr, HAS_PASS: tl.constexpr,
        HAS_MROPE: tl.constexpr, MROPE_H: tl.constexpr, MROPE_W: tl.constexpr,
        PDL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        q_gate_cols, k_cols = QH * 2 * HD, KH * HD
        if head >= QH + KH:
            v_offs = (head - (QH + KH)) * V_BLOCK + tl.arange(0, V_BLOCK)
            v_mask = v_offs < k_cols
            if PDL:
                tl.extra.cuda.gdc_wait()
            vals = tl.load(qkv_ptr + row * stride_qkv + q_gate_cols + k_cols
                           + v_offs, mask=v_mask, other=0.0)
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(v_out_ptr + row * stride_v + v_offs, vals, mask=v_mask)
            return
        head_offs, rot_offs = tl.arange(0, HEAD_BLOCK), tl.arange(0, ROT_BLOCK)
        head_mask, rot_mask = head_offs < HD, rot_offs < HALF_ROT
        is_k = head >= QH
        local_head = tl.where(is_k, head - QH, head)
        if is_k:
            in_base = qkv_ptr + row * stride_qkv + q_gate_cols + local_head * HD
            w_ptr, out_base = kw_ptr, k_out_ptr + row * stride_k + local_head * HD
        else:
            in_base = qkv_ptr + row * stride_qkv + local_head * 2 * HD
            w_ptr, out_base = qw_ptr, q_out_ptr + row * stride_q + local_head * HD
        w = 1.0 + tl.load(w_ptr + head_offs, mask=head_mask,
                          other=0.0).to(tl.float32)
        w1 = 1.0 + tl.load(w_ptr + rot_offs, mask=rot_mask,
                           other=0.0).to(tl.float32)
        w2 = 1.0 + tl.load(w_ptr + HALF_ROT + rot_offs, mask=rot_mask,
                           other=0.0).to(tl.float32)
        position = tl.load(positions_ptr + row * stride_pt).to(tl.int64)
        if HAS_MROPE:
            position_h = tl.load(positions_ptr + stride_pm + row * stride_pt).to(
                tl.int64)
            position_w = tl.load(
                positions_ptr + 2 * stride_pm + row * stride_pt).to(tl.int64)
            position = tl.where(
                (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_H), position_h,
                tl.where((rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_W),
                         position_w, position))
        cs_base = cs_ptr + position * stride_cs
        cos = tl.load(cs_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        sin = tl.load(cs_base + HALF_ROT + rot_offs, mask=rot_mask,
                      other=0.0).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
        x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
        inv_rms = tl.math.rsqrt(tl.sum(x * x, axis=0) / HD + eps)
        if HAS_PASS:
            tl.store(out_base + head_offs, (x * inv_rms * w).to(tl.bfloat16),
                     mask=head_mask & (head_offs >= ROT))
        x1 = (tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
              * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
        x2 = (tl.load(in_base + HALF_ROT + rot_offs, mask=rot_mask,
                      other=0.0).to(tl.float32)
              * inv_rms * w2).to(tl.bfloat16).to(tl.float32)
        tl.store(out_base + rot_offs, (x1 * cos - x2 * sin).to(tl.bfloat16),
                 mask=rot_mask)
        tl.store(out_base + HALF_ROT + rot_offs,
                 (x2 * cos + x1 * sin).to(tl.bfloat16), mask=rot_mask)
        if not is_k:
            gate = tl.load(in_base + HD + head_offs, mask=head_mask, other=0.0)
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(gate_out_ptr + row * stride_gate + local_head * HD + head_offs,
                     gate, mask=head_mask)
        elif PDL:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def sigmoid_gate(
        out_ptr, gate_ptr, dst_ptr, inv_scale_ptr, stride_out, stride_gate,
        stride_dst, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
        WRITE_FP8: tl.constexpr, PDL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        if WRITE_FP8:
            inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
        vals = tl.load(out_ptr + row * stride_out + offs, mask=mask,
                       other=0.0).to(tl.bfloat16)
        sig = tl.sigmoid(tl.load(gate_ptr + row * stride_gate + offs, mask=mask,
                                 other=0.0).to(tl.float32)).to(tl.bfloat16)
        gated = (vals.to(tl.float32) * sig.to(tl.float32)).to(tl.bfloat16)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        if WRITE_FP8:
            clamped = tl.minimum(tl.maximum(
                gated.to(tl.float32) * inv_scale, -448.0), 448.0)
            tl.store(dst_ptr + row * stride_dst + offs,
                     clamped.to(dst_ptr.dtype.element_ty), mask=mask)
        else:
            tl.store(dst_ptr + row * stride_dst + offs, gated, mask=mask)

    _KERNELS.update(
        gemma_add_norm_fp8=gemma_add_norm_fp8,
        gdn_gated_norm=gdn_gated_norm,
        qkv_gemma_norm_rope_gate=qkv_gemma_norm_rope_gate,
        sigmoid_gate=sigmoid_gate,
    )
    return _KERNELS


def is_static_fp8(module) -> bool:
    """Return whether a linear module runs static per-tensor ModelOpt FP8."""
    return (type(getattr(module, "quant_method", None)).__name__
            == "ModelOptFp8LinearMethod"
            and getattr(module, "input_scale", None) is not None
            and getattr(module, "weight_scale", None) is not None
            and module.input_scale.numel() == 1
            and module.weight_scale.numel() == 1)


def is_nvfp4(module) -> bool:
    """Return whether a linear module runs SM100 ModelOpt NVFP4."""
    weight = getattr(module, "weight", None)
    if (type(getattr(module, "quant_method", None)).__name__
            != "ModelOptNvFp4LinearMethod"
            or weight is None or not weight.is_cuda):
        return False
    import torch

    index = (weight.device.index if weight.device.index is not None
             else torch.cuda.current_device())
    sm100 = _SM100_CAPABLE.get(index)
    if sm100 is None:
        sm100 = _SM100_CAPABLE[index] = (
            torch.cuda.get_device_capability(index)[0] == 10)
    return (sm100
            and getattr(module, "input_global_scale_inv", None) is not None
            and getattr(module, "weight_scale", None) is not None
            and getattr(module, "alpha", None) is not None
            and getattr(module, "weights_padding_cols", 0) == 0)


def inv_input_scale(module):
    """Return and cache the 1-element float32 inverse static FP8 input scale."""
    cached = getattr(module, "_quail_inv_input_scale", None)
    if cached is None:
        cached = module._quail_inv_input_scale = (
            1.0 / module.input_scale.float()).reshape(1)
    return cached


def fp8_linear(x_fp8, module, bias=None):
    """Run a static per-tensor FP8 GEMM via cuBLASLt, tiled above 16,384 rows."""
    import torch

    rows = x_fp8.shape[0]
    scale_a = module.input_scale.reshape(1)
    scale_b = module.weight_scale.reshape(1)
    if rows <= _ROW_TILE_ABOVE:
        if rows == 0:
            return torch.empty((0, module.weight.shape[1]), dtype=torch.bfloat16,
                               device=x_fp8.device)
        return torch._scaled_mm(
            x_fp8, module.weight, scale_a=scale_a, scale_b=scale_b,
            out_dtype=torch.bfloat16, bias=bias)
    out = torch.empty((rows, module.weight.shape[1]), dtype=torch.bfloat16,
                      device=x_fp8.device)
    for low in range(0, rows, ROW_TILE):
        high = min(rows, low + ROW_TILE)
        torch._scaled_mm(x_fp8[low:high], module.weight, scale_a=scale_a,
                         scale_b=scale_b, out_dtype=torch.bfloat16, bias=bias,
                         out=out[low:high])
    return out


def nvfp4_quant(x, module):
    """Quantize bf16 rows to swizzled NVFP4 for module."""
    from vllm._custom_ops import scaled_fp4_quant

    return scaled_fp4_quant(
        x, module.input_global_scale_inv, is_sf_swizzled_layout=True,
        backend=("cutlass" if x.shape[0] <= _FP4_CUTLASS_MAX_ROWS
                 else "flashinfer-cutedsl"))


def nvfp4_linear(x_fp4, x_scale, module, bias=None, *, out=None):
    """Run an SM100 NVFP4 GEMM: CUTLASS at small M, CuTe-DSL at large M."""
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        slice_nvfp4_output,
    )

    rows, output_size = x_fp4.shape[0], module.output_size_per_partition
    if rows == 0:
        return out if out is not None else torch.empty(
            (0, output_size), dtype=torch.bfloat16, device=x_fp4.device)
    x_fp4 = x_fp4.view(torch.uint8)
    x_scale = x_scale.view(torch.float8_e4m3fn)
    if (rows > _FP4_CUTLASS_MAX_ROWS and module.weight.shape[0] == output_size
            and bias is None):
        import flashinfer

        return flashinfer.mm_fp4(
            x_fp4, module.weight.t(), x_scale, module.weight_scale.t(),
            module.alpha, torch.bfloat16, out=out, backend="cute-dsl")
    if rows <= _FP4_CUTLASS_MAX_ROWS:
        from vllm._custom_ops import cutlass_scaled_fp4_mm

        result = cutlass_scaled_fp4_mm(
            x_fp4, module.weight, x_scale, module.weight_scale, module.alpha,
            torch.bfloat16)
    else:
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

        result = flashinfer_scaled_fp4_mm(
            x_fp4, module.weight, x_scale, module.weight_scale, module.alpha,
            torch.bfloat16, backend="cute-dsl")
    result = slice_nvfp4_output(result, output_size)
    if bias is not None:
        result = result + bias
    if out is not None:
        out.copy_(result)
        return out
    return result


def silu_mul_nvfp4(gate_up, module):
    """Fuse SwiGLU and swizzled NVFP4 quantization in one SM100 CuTe-DSL launch."""
    import flashinfer

    return flashinfer.silu_and_mul_nvfp4_quantize(
        gate_up, module.input_global_scale_inv, is_sf_swizzled_layout=True)


def nvfp4_mlp(x, gate_up, down):
    """Run a SwiGLU MLP of two NVFP4 linears, in row tiles above 16,384 rows."""
    import torch

    def tile(rows, out=None):
        x_fp4, x_scale = nvfp4_quant(rows, gate_up)
        hidden = nvfp4_linear(x_fp4, x_scale, gate_up,
                              getattr(gate_up, "bias", None))
        d_fp4, d_scale = silu_mul_nvfp4(hidden, down)
        return nvfp4_linear(d_fp4, d_scale, down, getattr(down, "bias", None),
                            out=out)

    rows = x.shape[0]
    if rows <= _ROW_TILE_ABOVE:
        return tile(x)
    out = torch.empty((rows, down.output_size_per_partition),
                      dtype=torch.bfloat16, device=x.device)
    for low in range(0, rows, ROW_TILE):
        tile(x[low:min(rows, low + ROW_TILE)], out[low:min(rows, low + ROW_TILE)])
    return out


def gemma_add_norm(x, residual, weight, eps, *, inv_scale=None,
                   write_bf16=True, inplace_x=False):
    """Add x into residual in place, normalize with (1 + weight), and quantize."""
    import torch

    rows, width = x.shape
    has_residual, write_fp8 = residual is not None, inv_scale is not None
    new_residual = residual if has_residual else x
    bf16_out = (None if not write_bf16
                else (x if inplace_x and has_residual else torch.empty_like(x)))
    fp8_out = (torch.empty((rows, width), dtype=torch.float8_e4m3fn,
                           device=x.device) if write_fp8 else None)
    if rows == 0:
        return fp8_out, bf16_out, new_residual
    block, pdl = 1 << (width - 1).bit_length(), pdl_enabled(x.device, rows)
    launch_triton(
        _kernels()["gemma_add_norm_fp8"], (rows,),
        (x, new_residual, weight, fp8_out if write_fp8 else x,
         bf16_out if write_bf16 else x, x.stride(0), new_residual.stride(0),
         fp8_out.stride(0) if write_fp8 else 0,
         bf16_out.stride(0) if write_bf16 else 0,
         inv_scale if write_fp8 else weight, eps),
        key=(width, block, has_residual, write_fp8, write_bf16, pdl),
        kwargs=dict(H=width, BLOCK=block, HAS_RESIDUAL=has_residual,
                    WRITE_FP8=write_fp8, WRITE_BF16=write_bf16, PDL=pdl,
                    num_warps=8, launch_pdl=pdl))
    return fp8_out, bf16_out, new_residual


def gdn_gated_norm(core, z, weight, eps, *, inv_scale=None, activation="silu"):
    """RMS-normalize per-head core outputs, gate by z, and optionally quantize."""
    import torch

    rows, heads, width = core.shape
    total_width, write_fp8 = heads * width, inv_scale is not None
    out = torch.empty((rows, total_width),
                      dtype=torch.float8_e4m3fn if write_fp8 else core.dtype,
                      device=core.device)
    if rows == 0:
        return out
    z_2d, core_2d = z.view(rows, total_width), core.reshape(rows, total_width)
    heads_per_block = 8 if heads >= 8 else (1 << (heads - 1).bit_length())
    block = 1 << (width - 1).bit_length()
    silu_gate, pdl = activation != "sigmoid", pdl_enabled(core.device, rows)
    launch_triton(
        _kernels()["gdn_gated_norm"],
        (rows, (heads + heads_per_block - 1) // heads_per_block),
        (core_2d, z_2d, weight, out, inv_scale if write_fp8 else weight,
         core_2d.stride(0), z_2d.stride(0), out.stride(0), eps),
        key=(heads, width, heads_per_block, block, silu_gate, write_fp8, pdl),
        kwargs=dict(HEADS=heads, D=width, HEADS_PER_BLOCK=heads_per_block,
                    BLOCK=block, SILU_GATE=silu_gate, WRITE_FP8=write_fp8,
                    PDL=pdl, num_warps=4, launch_pdl=pdl))
    return out


def qkv_gemma_norm_rope_gate(qkv, positions, attn, *, q_out=None, k_out=None,
                             v_out=None, gate_out=None):
    """Split qkv, apply Gemma Q/K RMSNorm and partial RoPE/MRoPE, and copy V/gate."""
    import torch

    rows = qkv.shape[0]
    num_q_heads, num_kv_heads, head_dim = (
        attn.num_heads, attn.num_kv_heads, attn.head_dim)
    rotary = attn.rotary_emb
    rotary_dim = rotary.rotary_dim
    q_width, kv_width = num_q_heads * head_dim, num_kv_heads * head_dim
    dtype, device = qkv.dtype, qkv.device
    q = q_out if q_out is not None else torch.empty((rows, q_width), dtype=dtype,
                                                    device=device)
    k = k_out if k_out is not None else torch.empty((rows, kv_width), dtype=dtype,
                                                    device=device)
    v = v_out if v_out is not None else torch.empty((rows, kv_width), dtype=dtype,
                                                    device=device)
    gate = gate_out if gate_out is not None else torch.empty(
        (rows, q_width), dtype=dtype, device=device)
    if rows == 0:
        return q, k, v, gate
    mrope_section = getattr(rotary, "mrope_section", None)
    if positions.ndim == 2 and not mrope_section:
        positions = positions[0]
    has_mrope = positions.ndim == 2
    stride_pm, stride_pt = positions.stride() if has_mrope else (0, positions.stride(0))
    mrope_h, mrope_w = (mrope_section[1], mrope_section[2]) if has_mrope else (0, 0)
    half_rot = rotary_dim // 2
    head_block = 1 << (head_dim - 1).bit_length()
    rot_block = max(16, 1 << (half_rot - 1).bit_length())
    v_block = 512
    has_pass, warps = rotary_dim < head_dim, max(1, head_block // 128)
    pdl = pdl_enabled(device, rows)
    launch_triton(
        _kernels()["qkv_gemma_norm_rope_gate"],
        (rows, num_q_heads + num_kv_heads + (kv_width + v_block - 1) // v_block),
        (qkv, q, k, v, gate, attn.q_norm.weight, attn.k_norm.weight,
         rotary.cos_sin_cache, positions, qkv.stride(0), q.stride(0),
         k.stride(0), v.stride(0), gate.stride(0),
         rotary.cos_sin_cache.stride(0), stride_pm, stride_pt,
         attn.q_norm.variance_epsilon),
        key=(num_q_heads, num_kv_heads, head_dim, rotary_dim, half_rot,
             head_block, rot_block, v_block, has_pass, has_mrope, mrope_h,
             mrope_w, warps, pdl),
        kwargs=dict(QH=num_q_heads, KH=num_kv_heads, HD=head_dim,
                    ROT=rotary_dim, HALF_ROT=half_rot, HEAD_BLOCK=head_block,
                    ROT_BLOCK=rot_block, V_BLOCK=v_block, HAS_PASS=has_pass,
                    HAS_MROPE=has_mrope, MROPE_H=mrope_h, MROPE_W=mrope_w,
                    PDL=pdl, num_warps=warps, launch_pdl=pdl))
    return q, k, v, gate


def sigmoid_gate(out, gate, *, inv_scale=None):
    """Scale attention output by sigmoid(gate) and optionally quantize to FP8."""
    import torch

    rows, width = out.shape
    write_fp8 = inv_scale is not None
    dst = (torch.empty((rows, width), dtype=torch.float8_e4m3fn,
                       device=out.device) if write_fp8 else out)
    if rows == 0:
        return dst
    block, pdl = 512, pdl_enabled(out.device, rows)
    launch_triton(
        _kernels()["sigmoid_gate"], (rows, (width + block - 1) // block),
        (out, gate, dst, inv_scale if write_fp8 else out, out.stride(0),
         gate.stride(0), dst.stride(0)),
        key=(width, block, write_fp8, pdl),
        kwargs=dict(WIDTH=width, BLOCK=block, WRITE_FP8=write_fp8, PDL=pdl,
                    num_warps=4, launch_pdl=pdl))
    return dst
