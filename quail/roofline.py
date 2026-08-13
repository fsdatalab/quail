"""Component-level roofline: what the hardware allows, before any run.

The roofline question is which of two clocks a piece of work waits on.
A GPU does two things at once - move bytes and do math - so a kernel
takes

    max(bytes / memory bandwidth, FLOPs / peak math rate)

Divide those two hardware numbers and you get the RIDGE, the
arithmetic intensity (FLOPs per byte) where the two take equally long.
On an H100 SXM at fp8 that is 1,979 TFLOP/s / 3.35 TB/s = 591
FLOP/byte. Work above the ridge is compute-bound (the math is the
limit, faster memory would not help); work below it is memory-bound.

Each component of a transformer step sits somewhere different:

  Dense projections (QKV, output, MLP gate/up/down) load a weight
  matrix ONCE and apply it to every token in the step, so their
  arithmetic intensity climbs with the step's token count B and they
  cross the ridge at a specific B. That crossing is the minimum
  useful batch size - the reason a step budget of 20 tokens wastes
  the card and 400 does not.

  Attention loads no weights. It reads the KV cache, which grows with
  the CONTEXT length S, and does work proportional to B * S. So its
  intensity is governed by S and by how many query tokens share each
  cached context, not by B alone.

  Normalization, fp8 quantize/scale, and the activation function read
  and write activations with almost no math per byte - an intensity
  of about 1 to 3. No batch size moves them across the ridge. They
  are a fixed per-token tax.

Nothing here is measured; every number comes from a spec sheet and
the model's shape. That is the point: the analytical roofline says
where the knees are and which component owns each regime, and the
measured step model in quail/plan/cost.py says what the machine
actually delivers. The gap between them is the overhead multiplier
(about 0.49 for Qwen 4B on an H100), and knowing both is what makes
the gap a number instead of a mystery.

Run it for a table:
    python -m quail.roofline
"""

from dataclasses import dataclass

from .configs import H100_SXM, QWEN3_4B_FP8, DeviceConfig, ModelConfig


@dataclass(frozen=True)
class Precision:
    """Bytes per element for each tensor a step touches."""
    weight: int = 1      # fp8 weights
    act: int = 2         # bf16 activations
    kv: int = 1          # fp8 KV cache


FP8_W_BF16_ACT = Precision()


def ridge(device: DeviceConfig) -> float:
    """FLOP per byte where compute time equals memory time."""
    return device.R_D / device.BW


# ---- the dense projections -------------------------------------------

def _projection_shapes(model: ModelConfig):
    """(name, in_dim, out_dim) for every dense projection in a layer.
    Gate and up share an input, so vLLM fuses them into one GEMM."""
    qkv_out = (model.n_q + 2 * model.n_kv) * model.d_h
    inter = _intermediate(model)
    return (("qkv", model.h, qkv_out),
            ("output", model.n_q * model.d_h, model.h),
            ("gate_up", model.h, 2 * inter),
            ("down", inter, model.h))


def _intermediate(model: ModelConfig) -> int:
    """The MLP intermediate width, from the parameter budget.

    Per layer a transformer holds attention projections plus MLP:
        P/L = h*(n_q + 2*n_kv)*d_h + n_q*d_h*h + 3*h*inter
    so inter follows from P, L, and the attention shape. Qwen3-4B
    lands on 9,216 this way, which matches its config."""
    attn = (model.h * (model.n_q + 2 * model.n_kv) * model.d_h
            + model.n_q * model.d_h * model.h)
    per_layer = model.P / model.L
    return max(1, int(round((per_layer - attn) / (3 * model.h) / 256) * 256))


def projection_intensity(model: ModelConfig, B: int,
                         prec: Precision = FP8_W_BF16_ACT):
    """Arithmetic intensity of each dense projection at B tokens per
    step, plus the combined figure.

    FLOPs are 2 per weight per token (one multiply, one add). Bytes
    are the weight matrix once, plus each token's input read and
    output write. At small B the weight term dominates and intensity
    is low; at large B the activation term dominates and intensity
    saturates."""
    out = {}
    tot_f = tot_m = 0.0
    for name, din, dout in _projection_shapes(model):
        params = din * dout
        flops = 2.0 * params * B
        moved = params * prec.weight + B * (din + dout) * prec.act
        out[name] = flops / moved
        tot_f += flops
        tot_m += moved
    out["combined"] = tot_f / tot_m
    return out


def projection_knee(model: ModelConfig, device: DeviceConfig,
                    prec: Precision = FP8_W_BF16_ACT):
    """The B at which each projection crosses the ridge, and the
    combined crossing.

    Setting intensity = ridge and solving for B:
        2*P*B / (P*w + B*(din+dout)*a) = ridge
        B = ridge * P * w / (2*P - ridge*(din+dout)*a)
    A projection whose denominator is negative never crosses at any
    batch size (its activation traffic grows as fast as its math);
    that returns None."""
    r = ridge(device)
    knees = {}
    tot_p = tot_io = 0.0
    for name, din, dout in _projection_shapes(model):
        params = din * dout
        denom = 2.0 * params - r * (din + dout) * prec.act
        knees[name] = (r * params * prec.weight / denom
                       if denom > 0 else None)
        tot_p += params
        tot_io += (din + dout)
    denom = 2.0 * tot_p - r * tot_io * prec.act
    knees["combined"] = (r * tot_p * prec.weight / denom
                         if denom > 0 else None)
    return knees


# ---- attention --------------------------------------------------------

def attention_intensity(model: ModelConfig, B: int, S: int,
                        prec: Precision = FP8_W_BF16_ACT):
    """Arithmetic intensity of the attention kernel itself (QK^T and
    attn*V; the projections around it are dense GEMMs, priced above).

    FLOPs: 4 * B * S * n_q * d_h - two matmuls, each 2 FLOPs per
    element pair. Bytes: the KV cache read (2 * S * n_kv * d_h * kv
    bytes, shared by every query token in the step) plus each query
    token's Q read and O write.

    Two regimes fall out. With a long context and few query tokens
    the KV read dominates and intensity rises with B - more queries
    amortize one KV read, which is why batching decode helps. With
    many query tokens and a short context the Q/O traffic dominates
    and intensity is set by S alone."""
    flops = 4.0 * B * S * model.n_q * model.d_h
    kv_bytes = 2.0 * S * model.n_kv * model.d_h * prec.kv
    qo_bytes = 2.0 * B * model.n_q * model.d_h * prec.act
    return flops / (kv_bytes + qo_bytes)


def attention_crossover_S(model: ModelConfig, device: DeviceConfig,
                          B: int, prec: Precision = FP8_W_BF16_ACT):
    """The context length S at which attention costs as much time as
    the dense projections at the same step size. Below it the step is
    a GEMM problem; above it the KV read is the story, and the tricks
    that matter change (fewer, longer documents per step; KV
    compression; paging)."""
    t_dense = projection_time(model, device, B, prec)
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if attention_time(model, device, B, int(mid), prec) < t_dense:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---- ideal times ------------------------------------------------------

def projection_time(model: ModelConfig, device: DeviceConfig, B: int,
                    prec: Precision = FP8_W_BF16_ACT) -> float:
    """Ideal seconds for all dense projections in a step, all layers:
    each kernel takes max(its math, its memory), and they run in
    sequence."""
    t = 0.0
    for _name, din, dout in _projection_shapes(model):
        params = din * dout
        flops = 2.0 * params * B
        moved = params * prec.weight + B * (din + dout) * prec.act
        t += max(flops / device.R_D, moved / device.BW)
    return t * model.L


def attention_time(model: ModelConfig, device: DeviceConfig, B: int,
                   S: int, prec: Precision = FP8_W_BF16_ACT) -> float:
    """Ideal seconds for the attention kernels in a step, all layers."""
    flops = 4.0 * B * S * model.n_q * model.d_h
    moved = (2.0 * S * model.n_kv * model.d_h * prec.kv
             + 2.0 * B * model.n_q * model.d_h * prec.act)
    return max(flops / device.R_D, moved / device.BW) * model.L


def elementwise_time(model: ModelConfig, device: DeviceConfig, B: int,
                     prec: Precision = FP8_W_BF16_ACT) -> float:
    """Ideal seconds for the per-token tax: two RMSNorms, the fp8
    quantization before each GEMM, the SwiGLU activation, and two
    residual adds. All of it is memory-bound at every batch size, so
    the price is bytes over bandwidth."""
    inter = _intermediate(model)
    qkv_out = (model.n_q + 2 * model.n_kv) * model.d_h
    norm = 2 * (2 * model.h * prec.act)
    quant = (model.h + qkv_out + model.h + inter) * (prec.act + prec.weight)
    swiglu = 3 * inter * prec.act
    residual = 2 * (3 * model.h * prec.act)
    per_token = norm + quant + swiglu + residual
    return per_token * B * model.L / device.BW


def ideal_step_time(model: ModelConfig, device: DeviceConfig, B: int,
                    S: int, prec: Precision = FP8_W_BF16_ACT) -> dict:
    """The whole step, by component, in ideal seconds. Excludes kernel
    launch overhead, the scheduler, and anything else the host does -
    which is exactly why the measured step model needs its own fixed
    term."""
    d = projection_time(model, device, B, prec)
    a = attention_time(model, device, B, S, prec)
    e = elementwise_time(model, device, B, prec)
    return dict(dense=d, attention=a, elementwise=e, total=d + a + e)


def spec_ceiling_tokens_per_s(model: ModelConfig,
                              device: DeviceConfig) -> float:
    """The headline speed of light: tokens per second if every FLOP
    the card can do went into the forward pass and nothing else
    existed. 2P FLOPs per token against the peak rate."""
    return device.R_D / (2.0 * model.P)


def _main():
    m, d = QWEN3_4B_FP8, H100_SXM
    print(f"{m.name} on {d.name}")
    print(f"  peak {d.R_D / 1e12:,.0f} TFLOP/s, bandwidth "
          f"{d.BW / 1e12:.2f} TB/s, ridge {ridge(d):.0f} FLOP/byte")
    print(f"  MLP intermediate (derived) {_intermediate(m):,}")
    print(f"  speed-of-light prefill "
          f"{spec_ceiling_tokens_per_s(m, d):,.0f} tokens/s")

    print("\n  dense projections cross the ridge at:")
    for name, B in projection_knee(m, d).items():
        print(f"    {name:<10} {'never' if B is None else f'B = {B:,.0f}'}")

    print(f"\n  {'B':>7} {'dense_ms':>9} {'attn_ms':>8} {'elem_ms':>8} "
          f"{'total_ms':>9} {'tok/s':>10}")
    for B in (64, 256, 512, 1024, 4096, 16384, 25305):
        t = ideal_step_time(m, d, B, S=320)
        print(f"  {B:>7} {t['dense'] * 1e3:>9.2f} {t['attention'] * 1e3:>8.2f}"
              f" {t['elementwise'] * 1e3:>8.2f} {t['total'] * 1e3:>9.2f}"
              f" {B / t['total']:>10,.0f}")

    print(f"\n  attention overtakes the dense projections at S = "
          f"{attention_crossover_S(m, d, 25305):,.0f} tokens "
          f"(our documents are ~320)")


if __name__ == "__main__":
    _main()
