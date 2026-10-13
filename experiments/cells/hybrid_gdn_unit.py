"""Compare Quail's Gated DeltaNet reference maths with vLLM's kernels.

Random bf16 tensors with the Qwen3.5-4B geometry, sequences of mixed
lengths from one token to a few hundred. The convolution is compared
with and without a history, and the recurrence from zero and from a
random state. The recurrence also runs on q, k, and v that are column
slices of one wider tensor, as the layer passes them, and the
convolution runs over one sequence long enough that a row offset times
the channel count passes 2^31.

Prediction: the batched convolution matches the reference within 0.05
absolute and its tails exactly; the chunked recurrence matches the
reference within 0.05 absolute on outputs and 0.02 on states, for
contiguous and strided inputs alike; the long convolution matches the
reference on its last rows within 0.05; none produces NaN.

    uv run modal run experiments/cells/hybrid_gdn_unit.py \
        --backend flashinfer 2>&1 | tee /tmp/hybrid_gdn_unit.log

Record: /results/ablations/hybrid-gdn-unit-<backend>.json.
"""

import json
import os

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
results = modal.Volume.from_name("quail-results", create_if_missing=True)
LENGTHS = (1, 2, 3, 4, 5, 17, 64, 300)
TAIL_ROWS = 64


@app.function(image=image, gpu="H100!", timeout=1800,
              volumes={"/results": results})
def check(backend: str = "triton") -> dict:
    """Return the largest differences between the reference and the kernels."""
    os.environ["QUAIL_GDN_PREFILL"] = backend
    import torch

    from quail.backends.quail.executor import gdn
    from quail.specs import MODELS

    spec = MODELS["qwen3.5-4b-bf16"]

    torch.manual_seed(0)
    device = "cuda"
    channels = spec.conv_channels
    width = spec.conv_width
    count = len(LENGTHS)
    rows = sum(LENGTHS)
    cu = torch.tensor([0] + list(torch.tensor(LENGTHS).cumsum(0)),
                      dtype=torch.int32, device=device)
    out = {"lengths": LENGTHS, "backend": backend}

    x = torch.randn(rows, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, width, device=device,
                         dtype=torch.bfloat16) * 0.5
    bias = None
    phase = gdn.GdnPhase(rows=None, cu_seqlens=cu, lengths=LENGTHS)
    for name, history in (
            ("zero", None),
            ("history", torch.randn(count, width - 1, channels,
                                    device=device, dtype=torch.bfloat16))):
        ref, ref_tails = gdn.causal_conv_reference(
            x, weight, LENGTHS, history, bias)
        ker, ker_tails = gdn.causal_conv_kernel(x, weight, phase, history,
                                                bias)
        out[f"conv_{name}"] = {
            "nan_kernel": int(torch.isnan(ker).sum()),
            "max_abs_diff": float((ref.float() - ker.float()).abs().max()),
            "nan_tails": int(torch.isnan(ker_tails).sum()),
            "tails_max_abs_diff": float(
                (ref_tails.float() - ker_tails.float()).abs().max()),
            "nan_per_sequence": [
                int(torch.isnan(ker[int(cu[i]):int(cu[i + 1])]).sum())
                for i in range(count)],
        }

    hk, hv = spec.gdn_key_heads, spec.gdn_value_heads
    dk, dv = spec.gdn_key_dim, spec.gdn_value_dim
    q = torch.randn(rows, hk, dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(rows, hk, dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(rows, hv, dv, device=device, dtype=torch.bfloat16)
    g = -torch.rand(rows, hv, device=device) * 0.5
    beta = torch.rand(rows, hv, device=device)
    scale = dk ** -0.5
    # the layer's q, k, and v are column slices of one convolution output
    joined = torch.cat([q.reshape(rows, -1), k.reshape(rows, -1),
                        v.reshape(rows, -1)], dim=-1)
    key_width = hk * dk
    sq, sk, sv = joined.split(
        [key_width, key_width, joined.shape[1] - 2 * key_width], dim=-1)
    sq = sq.reshape(rows, hk, dk)
    sk = sk.reshape(rows, hk, dk)
    sv = sv.reshape(rows, hv, dv)
    out["strided_inputs_are_contiguous"] = bool(
        sq.is_contiguous() or sk.is_contiguous() or sv.is_contiguous())
    for name, state0 in (
            ("zero", torch.zeros(count, hv, dv, dk, device=device)),
            ("state", torch.randn(count, hv, dv, dk, device=device) * 0.1)):
        for layout, (a, b, c) in (("contiguous", (q, k, v)),
                                  ("strided", (sq, sk, sv))):
            s_ref, s_ker = state0.clone(), state0.clone()
            ref = gdn.gated_delta_reference(a, b, c, g, beta, s_ref, LENGTHS,
                                            scale)
            ker = gdn.gated_delta_kernel(a, b, c, g, beta, s_ker, cu, scale)
            out[f"recurrence_{name}_{layout}"] = {
                "nan_kernel": int(torch.isnan(ker).sum()),
                "max_abs_diff": float(
                    (ref.float() - ker.float()).abs().max()),
                "state_max_abs_diff": float((s_ref - s_ker).abs().max()),
            }

    # one sequence past the row offset where rows * channels overflows int32
    long_rows = 2 ** 31 // channels + 2048
    big = torch.randn(long_rows, channels, device=device,
                      dtype=torch.bfloat16)
    big_phase = gdn.GdnPhase(
        rows=None, cu_seqlens=torch.tensor([0, long_rows], dtype=torch.int32,
                                           device=device),
        lengths=(long_rows,))
    long_out, _ = gdn.causal_conv_kernel(big, weight, big_phase, None, bias)
    history = big[long_rows - TAIL_ROWS - (width - 1):
                  long_rows - TAIL_ROWS].unsqueeze(0)
    ref_tail, _ = gdn.causal_conv_reference(
        big[long_rows - TAIL_ROWS:], weight, (TAIL_ROWS,), history, bias)
    out["conv_long"] = {
        "rows": long_rows, "last_row_offset": long_rows * channels,
        "nan_kernel": int(torch.isnan(long_out[-TAIL_ROWS:]).sum()),
        "max_abs_diff": float((ref_tail.float()
                               - long_out[-TAIL_ROWS:].float()).abs().max()),
    }
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/hybrid-gdn-unit-{backend}.json", "w") as handle:
        json.dump(out, handle)
    results.commit()
    return out


@app.local_entrypoint()
def main(backend: str = "triton"):
    call = check.spawn(backend)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1))
