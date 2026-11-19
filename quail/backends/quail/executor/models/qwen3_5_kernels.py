"""Fused Triton and Blackwell SM100 kernels for Qwen3.5 and Qwen3.8 layers.

Each kernel fuses the normalization, gating, rotary embedding, or
activation step around a projection with its static FP8 or NVFP4
input quantization, and routes SM100 FP8 and NVFP4 matrix multiplies
to the fastest kernel at each row count.
"""

_KERNELS = {}
_FP4_CUTLASS_MAX_ROWS = 2_048
# Above _ROW_TILE_ABOVE rows a GEMM's activations and outputs outgrow the
# B200's L2 cache, and tiles of ROW_TILE rows run 10-14% faster than one call.
_ROW_TILE_ABOVE = 16_384
ROW_TILE = 8_192


def _kernels() -> dict:
    """Compile the Triton kernels on first use inside a GPU process."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def gemma_add_norm_fp8(
        x_ptr, res_ptr, w_ptr, fp8_ptr, bf16_ptr, stride_x, stride_res,
        stride_fp8, stride_bf16, inv_scale_ptr, eps, H: tl.constexpr,
        BLOCK: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
        WRITE_FP8: tl.constexpr, WRITE_BF16: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < H
        total = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                        other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            total += tl.load(res_ptr + row * stride_res + offs,
                             mask=mask, other=0.0).to(tl.float32)
            tl.store(res_ptr + row * stride_res + offs,
                     total.to(res_ptr.dtype.element_ty), mask=mask)
        variance = tl.sum(total * total, axis=0) / H
        scale = 1.0 + tl.load(w_ptr + offs, mask=mask,
                              other=0.0).to(tl.float32)
        normed = (total * tl.math.rsqrt(variance + eps) * scale).to(tl.bfloat16)
        if WRITE_BF16:
            tl.store(bf16_ptr + row * stride_bf16 + offs, normed, mask=mask)
        if WRITE_FP8:
            inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
            scaled = normed.to(tl.float32) * inv_scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
            tl.store(fp8_ptr + row * stride_fp8 + offs,
                     clamped.to(fp8_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def gdn_gated_norm(
        core_ptr, z_ptr, w_ptr, out_ptr, inv_scale_ptr, stride_core,
        stride_z, stride_out, eps, HEADS: tl.constexpr, D: tl.constexpr,
        HEADS_PER_BLOCK: tl.constexpr, BLOCK: tl.constexpr,
        SILU_GATE: tl.constexpr, WRITE_FP8: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        head_group = tl.program_id(1).to(tl.int64)
        h_offs = head_group * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK)
        d_offs = tl.arange(0, BLOCK)
        h_mask = h_offs < HEADS
        d_mask = d_offs < D
        mask = h_mask[:, None] & d_mask[None, :]
        offsets = h_offs[:, None] * D + d_offs[None, :]
        x = tl.load(core_ptr + row * stride_core + offsets,
                    mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row * stride_z + offsets,
                    mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / D
        normed = x * tl.math.rsqrt(variance + eps)[:, None] * w[None, :]
        sig = tl.sigmoid(z)
        gate = z * sig if SILU_GATE else sig
        gated = (normed * gate).to(tl.bfloat16)
        if WRITE_FP8:
            inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
            scaled = gated.to(tl.float32) * inv_scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
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
    ):
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        q_gate_cols = QH * 2 * HD
        k_cols = KH * HD
        if head >= QH + KH:
            v_chunk = head - (QH + KH)
            v_offs = v_chunk * V_BLOCK + tl.arange(0, V_BLOCK)
            v_mask = v_offs < KH * HD
            v_in = qkv_ptr + row * stride_qkv + q_gate_cols + k_cols + v_offs
            v_out = v_out_ptr + row * stride_v + v_offs
            vals = tl.load(v_in, mask=v_mask, other=0.0)
            tl.store(v_out, vals, mask=v_mask)
            return
        head_offs = tl.arange(0, HEAD_BLOCK)
        head_mask = head_offs < HD
        is_k = head >= QH
        local_head = tl.where(is_k, head - QH, head)
        if is_k:
            in_base = qkv_ptr + row * stride_qkv + q_gate_cols + local_head * HD
            w_ptr = kw_ptr
            out_base = k_out_ptr + row * stride_k + local_head * HD
        else:
            in_base = qkv_ptr + row * stride_qkv + local_head * 2 * HD
            w_ptr = qw_ptr
            out_base = q_out_ptr + row * stride_q + local_head * HD

        x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
        inv_rms = tl.math.rsqrt(tl.sum(x * x, axis=0) / HD + eps)
        w = 1.0 + tl.load(w_ptr + head_offs, mask=head_mask,
                          other=0.0).to(tl.float32)
        x_norm = (x * inv_rms * w).to(tl.bfloat16)
        if HAS_PASS:
            pass_mask = head_mask & (head_offs >= ROT)
            tl.store(out_base + head_offs, x_norm, mask=pass_mask)

        rot_offs = tl.arange(0, ROT_BLOCK)
        rot_mask = rot_offs < HALF_ROT
        x1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        x2 = tl.load(in_base + HALF_ROT + rot_offs, mask=rot_mask,
                     other=0.0).to(tl.float32)
        w1 = 1.0 + tl.load(w_ptr + rot_offs, mask=rot_mask,
                           other=0.0).to(tl.float32)
        w2 = 1.0 + tl.load(w_ptr + HALF_ROT + rot_offs, mask=rot_mask,
                           other=0.0).to(tl.float32)
        x1 = (x1 * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 * inv_rms * w2).to(tl.bfloat16).to(tl.float32)

        position_t = tl.load(positions_ptr + row * stride_pt).to(tl.int64)
        if HAS_MROPE:
            position_h = tl.load(
                positions_ptr + stride_pm + row * stride_pt).to(tl.int64)
            position_w = tl.load(
                positions_ptr + 2 * stride_pm + row * stride_pt).to(tl.int64)
            is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_H)
            is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_W)
            position = tl.where(is_h, position_h,
                                tl.where(is_w, position_w, position_t))
        else:
            position = position_t
        cs_base = cs_ptr + position * stride_cs
        cos = tl.load(cs_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        sin = tl.load(cs_base + HALF_ROT + rot_offs, mask=rot_mask,
                      other=0.0).to(tl.float32)
        tl.store(out_base + rot_offs,
                 (x1 * cos - x2 * sin).to(tl.bfloat16), mask=rot_mask)
        tl.store(out_base + HALF_ROT + rot_offs,
                 (x2 * cos + x1 * sin).to(tl.bfloat16), mask=rot_mask)

        if not is_k:
            gate = tl.load(in_base + HD + head_offs, mask=head_mask, other=0.0)
            tl.store(gate_out_ptr + row * stride_gate + local_head * HD + head_offs,
                     gate, mask=head_mask)

    @triton.jit
    def sigmoid_gate(
        out_ptr, gate_ptr, dst_ptr, inv_scale_ptr, stride_out, stride_gate,
        stride_dst, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
        WRITE_FP8: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1)
        offs = col_block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        vals = tl.load(out_ptr + row * stride_out + offs, mask=mask,
                       other=0.0).to(tl.bfloat16)
        gate = tl.load(gate_ptr + row * stride_gate + offs, mask=mask,
                       other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate).to(tl.bfloat16)
        gated = (vals.to(tl.float32) * sig.to(tl.float32)).to(tl.bfloat16)
        if WRITE_FP8:
            inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
            scaled = gated.to(tl.float32) * inv_scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
            tl.store(dst_ptr + row * stride_dst + offs,
                     clamped.to(dst_ptr.dtype.element_ty), mask=mask)
        else:
            tl.store(dst_ptr + row * stride_dst + offs, gated, mask=mask)

    @triton.jit
    def gather_gdn_inputs(
        qkv_ptr, a_ptr, b_ptr, rows_ptr, qkv_out_ptr, a_out_ptr, b_out_ptr,
        stride_qkv, stride_a, stride_b, stride_qo, stride_ao, stride_bo,
        CONV_DIM: tl.constexpr, HV: tl.constexpr, QKV_BLOCKS: tl.constexpr,
        BLOCK: tl.constexpr, HV_BLOCK: tl.constexpr,
    ):
        dst_row = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1)
        src_row = tl.load(rows_ptr + dst_row).to(tl.int64)
        if col_block < QKV_BLOCKS:
            offs = col_block * BLOCK + tl.arange(0, BLOCK)
            mask = offs < CONV_DIM
            vals = tl.load(qkv_ptr + src_row * stride_qkv + offs,
                           mask=mask, other=0.0)
            tl.store(qkv_out_ptr + dst_row * stride_qo + offs, vals, mask=mask)
        else:
            hv_offs = tl.arange(0, HV_BLOCK)
            hv_mask = hv_offs < HV
            a_vals = tl.load(a_ptr + src_row * stride_a + hv_offs,
                             mask=hv_mask, other=0.0)
            b_vals = tl.load(b_ptr + src_row * stride_b + hv_offs,
                             mask=hv_mask, other=0.0)
            tl.store(a_out_ptr + dst_row * stride_ao + hv_offs,
                     a_vals, mask=hv_mask)
            tl.store(b_out_ptr + dst_row * stride_bo + hv_offs,
                     b_vals, mask=hv_mask)

    @triton.jit
    def scatter_gdn_rows(
        src_ptr, rows_ptr, dst_ptr, stride_src, stride_dst,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        src_row = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1)
        dst_row = tl.load(rows_ptr + src_row).to(tl.int64)
        offs = col_block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        vals = tl.load(src_ptr + src_row * stride_src + offs,
                       mask=mask, other=0.0)
        tl.store(dst_ptr + dst_row * stride_dst + offs, vals, mask=mask)

    @triton.jit
    def gdn_decode_step(
        x_ptr, a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, state_ptr, state_rows_ptr,
        out_ptr, out_rows_ptr, stride_x, stride_a, stride_b, stride_state,
        stride_out, scale, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
        V: tl.constexpr, BV: tl.constexpr, HAS_OUT_ROWS: tl.constexpr,
    ):
        v_block = tl.program_id(0)
        seq_head = tl.program_id(1)
        seq = (seq_head // HV).to(tl.int64)
        hv = seq_head % HV
        h = hv // (HV // H)
        state_row = tl.load(state_rows_ptr + seq).to(tl.int64)
        offs_k = tl.arange(0, K)
        offs_v = v_block * BV + tl.arange(0, BV)
        row = x_ptr + seq * stride_x
        q = tl.load(row + h * K + offs_k).to(tl.float32)
        k = tl.load(row + H * K + h * K + offs_k).to(tl.float32)
        v = tl.load(row + 2 * H * K + hv * V + offs_v).to(tl.float32)
        # The bf16 rounding matches the separate prep kernel's stored q and k.
        q = (q / tl.sqrt(tl.sum(q * q) + 1e-6)).to(tl.bfloat16).to(tl.float32)
        k = (k / tl.sqrt(tl.sum(k * k) + 1e-6)).to(tl.bfloat16).to(tl.float32)
        q = q * scale
        x = (tl.load(a_ptr + seq * stride_a + hv).to(tl.float32)
             + tl.load(dt_bias_ptr + hv).to(tl.float32))
        softplus = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)),
                            tl.log(1.0 + tl.exp(x)))
        softplus = tl.where(x <= 20.0, softplus, x)
        g = -tl.exp(tl.load(a_log_ptr + hv).to(tl.float32)) * softplus
        beta = tl.sigmoid(tl.load(b_ptr + seq * stride_b + hv).to(tl.float32))
        p_state = (state_ptr + state_row * stride_state + hv * V * K
                   + offs_v[:, None] * K + offs_k[None, :])
        state = tl.load(p_state) * tl.exp(g)
        v = (v - tl.sum(state * k[None, :], 1)) * beta
        state += v[:, None] * k[None, :]
        o = tl.sum(state * q[None, :], 1)
        tl.store(p_state, state)
        out_row = tl.load(out_rows_ptr + seq).to(tl.int64) if HAS_OUT_ROWS else seq
        tl.store(out_ptr + out_row * stride_out + hv * V + offs_v,
                 o.to(out_ptr.dtype.element_ty))

    @triton.jit
    def copy_slots(pool_ptr, src_ptr, dst_ptr, stride_slot,
                   WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        pair = tl.program_id(0)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        src = tl.load(src_ptr + pair).to(tl.int64)
        dst = tl.load(dst_ptr + pair).to(tl.int64)
        vals = tl.load(pool_ptr + src * stride_slot + offs, mask=mask)
        tl.store(pool_ptr + dst * stride_slot + offs, vals, mask=mask)

    _KERNELS.update(
        copy_slots=copy_slots,
        gemma_add_norm_fp8=gemma_add_norm_fp8,
        gdn_gated_norm=gdn_gated_norm,
        qkv_gemma_norm_rope_gate=qkv_gemma_norm_rope_gate,
        sigmoid_gate=sigmoid_gate,
        gather_gdn_inputs=gather_gdn_inputs,
        scatter_gdn_rows=scatter_gdn_rows,
        gdn_decode_step=gdn_decode_step,
    )
    return _KERNELS


def is_static_fp8(module) -> bool:
    """Return whether a linear module runs static per-tensor ModelOpt FP8."""
    method = getattr(module, "quant_method", None)
    return (type(method).__name__ == "ModelOptFp8LinearMethod"
            and getattr(module, "input_scale", None) is not None
            and module.input_scale.numel() == 1
            and module.weight_scale.numel() == 1)


def is_nvfp4(module) -> bool:
    """Return whether a linear module runs unpadded SM100 ModelOpt NVFP4."""
    import torch

    method = getattr(module, "quant_method", None)
    return (type(method).__name__ == "ModelOptNvFp4LinearMethod"
            and getattr(module, "weights_padding_cols", 0) == 0
            and module.weight.is_cuda
            and torch.cuda.get_device_capability(module.weight.device)[0] == 10)


def inv_input_scale(module):
    """Return cached 1 / input_scale for a static per-tensor FP8 linear."""
    cached = getattr(module, "quail_inv_input_scale", None)
    if cached is None:
        cached = (1.0 / module.input_scale.float()).reshape(1).contiguous()
        module.quail_inv_input_scale = cached
    return cached


def _scale_1d(tensor):
    return tensor.reshape(1) if tensor.ndim == 0 else tensor


def fp8_linear(x_fp8, module, bias=None):
    """Run a static per-tensor FP8 GEMM through cuBLASLt (torch._scaled_mm).

    A chunk of more than 16,384 rows runs in tiles of ROW_TILE rows written
    into one output.
    """
    import torch

    rows = x_fp8.shape[0]
    scale_a = _scale_1d(module.input_scale)
    scale_b = _scale_1d(module.weight_scale)
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

    backend = ("cutlass" if x.shape[0] <= _FP4_CUTLASS_MAX_ROWS
               else "flashinfer-cutedsl")
    return scaled_fp4_quant(
        x, module.input_global_scale_inv, is_sf_swizzled_layout=True,
        backend=backend)


def nvfp4_linear(x_fp4, x_scale, module, bias=None, *, out=None):
    """Run an SM100 NVFP4 GEMM: vLLM CUTLASS at small M, CuTe-DSL at large M.

    Args:
        x_fp4: (rows, in / 2) packed NVFP4 activations.
        x_scale: Their swizzled block scales.
        module: The ModelOpt NVFP4 linear.
        bias: Optional bias added to the output.
        out: Optional (rows, out features) bf16 tensor to write.

    Returns:
        The (rows, out features) bf16 output.
    """
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        slice_nvfp4_output,
    )

    rows = x_fp4.shape[0]
    output_size = module.output_size_per_partition
    if rows == 0:
        return out if out is not None else torch.empty(
            (0, output_size), dtype=torch.bfloat16, device=x_fp4.device)
    if x_fp4.dtype != torch.uint8:
        x_fp4 = x_fp4.view(torch.uint8)
    if x_scale.dtype != torch.float8_e4m3fn:
        x_scale = x_scale.view(torch.float8_e4m3fn)
    unpadded = module.weight.shape[0] == output_size
    if rows > _FP4_CUTLASS_MAX_ROWS and unpadded and bias is None:
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
    if out is None:
        return result
    out.copy_(result)
    return out


def nvfp4_mlp(x, gate_up, down):
    """Run a SwiGLU MLP of two NVFP4 linears, in row tiles above 16,384 rows.

    Args:
        x: (rows, hidden) bf16 input.
        gate_up: The ModelOpt NVFP4 gate and up projection.
        down: The ModelOpt NVFP4 down projection.

    Returns:
        The (rows, hidden) bf16 output.
    """
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
        high = min(rows, low + ROW_TILE)
        tile(x[low:high], out[low:high])
    return out


def silu_mul_nvfp4(gate_up, module):
    """Fuse SwiGLU and swizzled NVFP4 quantization in one SM100 CuTe-DSL launch."""
    import flashinfer

    return flashinfer.silu_and_mul_nvfp4_quantize(
        gate_up, module.input_global_scale_inv, is_sf_swizzled_layout=True)


def gemma_add_norm(x, residual, weight, eps, *, inv_scale=None,
                   write_bf16=True, inplace_x=False):
    """Add x into residual in place, normalize with (1 + weight), and quantize.

    Args:
        x: (rows, width) block output.
        residual: (rows, width) running residual updated in place, or None.
        weight: (width,) raw norm weight; scale is 1 + weight.
        eps: Variance epsilon.
        inv_scale: Optional (1,) float32 1 / input_scale to also emit FP8 rows.
        write_bf16: Whether to return bf16 normalized rows.
        inplace_x: Write bf16 normalized rows into x when residual is not None.

    Returns:
        (fp8_rows_or_none, bf16_rows_or_none, residual).
    """
    import torch

    rows, width = x.shape
    has_residual = residual is not None
    new_residual = residual if has_residual else x
    write_fp8 = inv_scale is not None
    bf16_out = (None if not write_bf16
                else (x if inplace_x and has_residual else torch.empty_like(x)))
    fp8_out = (torch.empty((rows, width), dtype=torch.float8_e4m3fn,
                           device=x.device) if write_fp8 else None)
    if rows == 0:
        return fp8_out, bf16_out, new_residual
    dummy = x
    _kernels()["gemma_add_norm_fp8"][(rows,)](
        x, new_residual, weight, fp8_out if write_fp8 else dummy,
        bf16_out if write_bf16 else dummy, x.stride(0), new_residual.stride(0),
        fp8_out.stride(0) if write_fp8 else 0,
        bf16_out.stride(0) if write_bf16 else 0,
        inv_scale if write_fp8 else weight, eps, H=width,
        BLOCK=1 << (width - 1).bit_length(), HAS_RESIDUAL=has_residual,
        WRITE_FP8=write_fp8, WRITE_BF16=write_bf16, num_warps=8)
    return fp8_out, bf16_out, new_residual


def gdn_gated_norm(core, z, weight, eps, *, inv_scale=None, activation="silu"):
    """RMS-normalize per-head core outputs, gate by z, and optionally quantize."""
    import torch

    rows, heads, width = core.shape
    total_width = heads * width
    write_fp8 = inv_scale is not None
    dtype = torch.float8_e4m3fn if write_fp8 else core.dtype
    out = torch.empty((rows, total_width), dtype=dtype, device=core.device)
    if rows == 0:
        return out
    z_2d = z.view(rows, total_width)
    core_2d = core.reshape(rows, total_width)
    heads_per_block = 8 if heads >= 8 else (1 << (heads - 1).bit_length())
    grid = (rows, (heads + heads_per_block - 1) // heads_per_block)
    _kernels()["gdn_gated_norm"][grid](
        core_2d, z_2d, weight, out, inv_scale if write_fp8 else weight,
        core_2d.stride(0), z_2d.stride(0), out.stride(0), eps, HEADS=heads,
        D=width, HEADS_PER_BLOCK=heads_per_block,
        BLOCK=1 << (width - 1).bit_length(),
        SILU_GATE=activation != "sigmoid", WRITE_FP8=write_fp8, num_warps=4)
    return out


def qkv_gemma_norm_rope_gate(qkv, positions, attn, *, q_out=None, k_out=None,
                             v_out=None, gate_out=None):
    """Split qkv, apply Gemma Q/K RMSNorm and partial RoPE/MRoPE, and copy V/gate."""
    import torch

    rows = qkv.shape[0]
    num_q_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    head_dim = attn.head_dim
    rotary = attn.rotary_emb
    rotary_dim = rotary.rotary_dim
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    dtype, device = qkv.dtype, qkv.device
    q = (torch.empty((rows, q_width), dtype=dtype, device=device)
         if q_out is None else q_out)
    k = (torch.empty((rows, kv_width), dtype=dtype, device=device)
         if k_out is None else k_out)
    v = (torch.empty((rows, kv_width), dtype=dtype, device=device)
         if v_out is None else v_out)
    gate = (torch.empty((rows, q_width), dtype=dtype, device=device)
            if gate_out is None else gate_out)
    if rows == 0:
        return q, k, v, gate
    mrope_section = getattr(rotary, "mrope_section", None)
    if positions.ndim == 2 and not mrope_section:
        positions = positions[0]
    has_mrope = positions.ndim == 2
    if has_mrope:
        stride_pm, stride_pt = positions.stride()
        mrope_h, mrope_w = mrope_section[1], mrope_section[2]
    else:
        stride_pm, stride_pt = 0, positions.stride(0)
        mrope_h = mrope_w = 0
    half_rot = rotary_dim // 2
    head_block = 1 << (head_dim - 1).bit_length()
    rot_block = max(16, 1 << (half_rot - 1).bit_length())
    v_block = 512
    v_chunks = (kv_width + v_block - 1) // v_block
    grid = (rows, num_q_heads + num_kv_heads + v_chunks)
    _kernels()["qkv_gemma_norm_rope_gate"][grid](
        qkv, q, k, v, gate, attn.q_norm.weight, attn.k_norm.weight,
        rotary.cos_sin_cache, positions, qkv.stride(0), q.stride(0),
        k.stride(0), v.stride(0), gate.stride(0),
        rotary.cos_sin_cache.stride(0), stride_pm, stride_pt,
        attn.q_norm.variance_epsilon, QH=num_q_heads, KH=num_kv_heads,
        HD=head_dim, ROT=rotary_dim, HALF_ROT=half_rot, HEAD_BLOCK=head_block,
        ROT_BLOCK=rot_block, V_BLOCK=v_block, HAS_PASS=rotary_dim < head_dim,
        HAS_MROPE=has_mrope, MROPE_H=mrope_h, MROPE_W=mrope_w,
        num_warps=max(1, head_block // 128))
    return q, k, v, gate


def gather_gdn_inputs(qkv, a, b, rows):
    """Gather qkv, a, and b at selected row indices in one vectorized launch."""
    import torch

    count = rows.shape[0]
    conv_dim = qkv.shape[1]
    hv = a.shape[1]
    qkv_out = torch.empty((count, conv_dim), dtype=qkv.dtype, device=qkv.device)
    a_out = torch.empty((count, hv), dtype=a.dtype, device=a.device)
    b_out = torch.empty((count, hv), dtype=b.dtype, device=b.device)
    if count == 0:
        return qkv_out, a_out, b_out
    block = 1024
    qkv_blocks = (conv_dim + block - 1) // block
    hv_block = max(16, 1 << (hv - 1).bit_length())
    _kernels()["gather_gdn_inputs"][(count, qkv_blocks + 1)](
        qkv, a, b, rows, qkv_out, a_out, b_out, qkv.stride(0), a.stride(0),
        b.stride(0), qkv_out.stride(0), a_out.stride(0), b_out.stride(0),
        CONV_DIM=conv_dim, HV=hv, QKV_BLOCKS=qkv_blocks, BLOCK=block,
        HV_BLOCK=hv_block, num_warps=4)
    return qkv_out, a_out, b_out


def scatter_gdn_rows(core, rows, out):
    """Scatter 3D core recurrence outputs into out at selected row indices."""
    count = rows.shape[0]
    if count == 0:
        return out
    core_2d = core.reshape(count, -1)
    out_2d = out.view(out.shape[0], -1)
    width = core_2d.shape[1]
    block = 1024
    grid = (count, (width + block - 1) // block)
    _kernels()["scatter_gdn_rows"][grid](
        core_2d, rows, out_2d, core_2d.stride(0), out_2d.stride(0),
        WIDTH=width, BLOCK=block, num_warps=4)
    return out


def copy_slots(pool, src, dst):
    """Copy pool[src[i]] into pool[dst[i]]; no dst may also be a src.

    Args:
        pool: (slots, ...) tensor whose slots hold contiguous values; only
            dimension 0 may be strided.
        src: (pairs,) source slots.
        dst: (pairs,) destination slots.

    Returns:
        pool.
    """
    count = src.shape[0]
    if count == 0:
        return pool
    width = pool[0].numel()
    if not pool[0].is_contiguous():
        raise ValueError("each slot must hold contiguous values")
    block = 4096
    _kernels()["copy_slots"][(count, (width + block - 1) // block)](
        pool, src, dst, pool.stride(0), WIDTH=width, BLOCK=block, num_warps=8)
    return pool


def gdn_decode_step(conv_out, a, b, a_log, dt_bias, state_pool, state_rows, out,
                    *, num_key_heads, scale, out_rows=None, block_v=8,
                    num_warps=1):
    """Advance one-token Gated DeltaNet sequences and write their outputs.

    One launch reads the convolved q, k, and v columns in place, L2-normalizes
    q and k, computes the decay and write gates, updates each sequence's fp32
    state row in place, and writes its output row.

    Args:
        conv_out: (sequences, conv channels) convolution output, unit inner
            stride.
        a: (sequences, value heads) decay gate inputs.
        b: (sequences, value heads) write gate inputs.
        a_log: (value heads,) log decay rates.
        dt_bias: (value heads,) decay biases.
        state_pool: (state rows, value heads, value dim, key dim) fp32 states
            of one layer; only dimension 0 may be strided.
        state_rows: (sequences,) state row of each sequence.
        out: (rows, value heads, value dim) layer output.
        num_key_heads: Query and key heads.
        scale: Query scale.
        out_rows: Optional (sequences,) output row of each sequence; None
            writes sequence s to row s.
        block_v: Value dims each program updates.
        num_warps: Warps per program.

    Returns:
        out.
    """
    count = state_rows.shape[0]
    if count == 0:
        return out
    _, value_heads, value_dim, key_dim = state_pool.shape
    if state_pool.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise ValueError("each state row must hold contiguous heads")
    out_2d = out.view(out.shape[0], -1)
    grid = (value_dim // block_v, count * value_heads)
    _kernels()["gdn_decode_step"][grid](
        conv_out, a, b, a_log, dt_bias, state_pool, state_rows, out_2d,
        state_rows if out_rows is None else out_rows,
        conv_out.stride(0), a.stride(0), b.stride(0), state_pool.stride(0),
        out_2d.stride(0), scale, H=num_key_heads, HV=value_heads, K=key_dim,
        V=value_dim, BV=block_v, HAS_OUT_ROWS=out_rows is not None,
        num_warps=num_warps)
    return out


def sigmoid_gate(out, gate, *, inv_scale=None):
    """Scale attention output by sigmoid(gate) and optionally quantize to FP8."""
    import torch

    rows, width = out.shape
    write_fp8 = inv_scale is not None
    dst = (torch.empty((rows, width), dtype=torch.float8_e4m3fn,
                       device=out.device) if write_fp8 else out)
    if rows == 0:
        return dst
    block = 512
    grid = (rows, (width + block - 1) // block)
    _kernels()["sigmoid_gate"][grid](
        out, gate, dst, inv_scale if write_fp8 else out, out.stride(0),
        gate.stride(0), dst.stride(0), WIDTH=width, BLOCK=block,
        WRITE_FP8=write_fp8, num_warps=4)
    return dst
