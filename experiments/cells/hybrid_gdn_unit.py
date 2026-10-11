"""Compare Quail's Gated DeltaNet reference maths with vLLM's kernels.

Random bf16 tensors with the Qwen3.5-4B geometry, sequences of mixed
lengths from one token to a few hundred. The convolution is compared
with and without a history, and the recurrence from zero and from a
random state.

Prediction: the batched convolution matches the reference within 0.05
absolute and its tails exactly; the chunked recurrence matches the
reference within 0.05 absolute on outputs and 0.02 on states; neither
produces NaN.

    uv run modal run experiments/cells/hybrid_gdn_unit.py \
        2>&1 | tee /tmp/hybrid_gdn_unit.log
"""

import json

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
LENGTHS = (1, 2, 3, 4, 5, 17, 64, 300)


@app.function(image=image, gpu="H100!", timeout=1800)
def check() -> dict:
    """Return the largest differences between the reference and the kernels."""
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
    out = {"lengths": LENGTHS}

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
    for name, state0 in (
            ("zero", torch.zeros(count, hv, dv, dk, device=device)),
            ("state", torch.randn(count, hv, dv, dk, device=device) * 0.1)):
        s_ref, s_ker = state0.clone(), state0.clone()
        ref = gdn.gated_delta_reference(q, k, v, g, beta, s_ref, LENGTHS,
                                        scale)
        ker = gdn.gated_delta_kernel(q, k, v, g, beta, s_ker, cu, scale)
        out[f"recurrence_{name}"] = {
            "nan_kernel": int(torch.isnan(ker).sum()),
            "max_abs_diff": float((ref.float() - ker.float()).abs().max()),
            "state_max_abs_diff": float((s_ref - s_ker).abs().max()),
        }
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=1))
