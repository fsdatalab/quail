"""Compare Quail's Gated DeltaNet reference maths with vLLM's kernels.

Random bf16 tensors with the Qwen3.5-4B geometry, sequences of mixed
lengths from one token to a few hundred. The convolution is compared
with and without a history, and the recurrence from zero and from a
random state. Fused preparation normalizes q and k and computes the
gates from contiguous and strided projection outputs. The convolution
runs over one sequence long enough that a row offset times the channel
count passes 2^31.

Prediction: the batched convolution matches the reference within 0.05
absolute and its tails exactly; the chunked recurrence matches the
reference within 0.05 absolute on outputs and 0.02 on states, for
contiguous and strided inputs alike; the long convolution matches the
reference on its last rows within 0.05; none produces NaN. One-token
decode steps that update arena rows in place match the chunked kernel
on gathered state within 0.05 on outputs and 0.02 on states, and leave
row 0 and the other layer untouched.

    uv run modal run experiments/cells/hybrid_gdn_unit.py \
        --backend flashinfer 2>&1 | tee /tmp/hybrid_gdn_unit.log

Record: /results/ablations/hybrid-gdn-unit-<backend>.json.
"""

import json
import os

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image().add_local_python_source("experiments")
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
    import torch.nn.functional as F
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

    from experiments.hybrid import causal_conv_reference, gated_delta_reference
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
    for name, history in (
            ("zero", None),
            ("history", torch.randn(count, width - 1, channels,
                                    device=device, dtype=torch.bfloat16))):
        phase = gdn.GdnPhase(rows=None, cu_seqlens=cu, lengths=LENGTHS)
        ref, ref_tails = causal_conv_reference(
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
    ba = torch.randn(rows, 2 * hv, device=device, dtype=torch.bfloat16)
    b, a = ba.chunk(2, dim=-1)
    a_log = torch.randn(hv, device=device) * 0.1
    dt_bias = torch.randn(hv, device=device)
    g = -a_log.exp() * F.softplus(a.float() + dt_bias)
    beta = b.float().sigmoid()
    scale = dk ** -0.5
    joined = torch.cat([q.reshape(rows, -1), k.reshape(rows, -1),
                        v.reshape(rows, -1)], dim=-1)
    strided = torch.cat([joined, joined], dim=-1)[:, :joined.shape[1]]
    out["strided_inputs_are_contiguous"] = strided.is_contiguous()
    for name, state0 in (
            ("zero", torch.zeros(count, hv, dv, dk, device=device)),
            ("state", torch.randn(count, hv, dv, dk, device=device) * 0.1)):
        for layout, mixed in (("contiguous", joined), ("strided", strided)):
            s_ref = state0.clone()
            ref = gated_delta_reference(q, k, v, g, beta, s_ref, LENGTHS, scale)
            pq, pk, pv, pg, pb = fused_post_conv_prep(
                conv_output=mixed, a=a, b=b, A_log=a_log, dt_bias=dt_bias,
                num_k_heads=hk, head_k_dim=dk, head_v_dim=dv,
                apply_l2norm=True, output_g_exp=False)
            ker, final = gdn.gated_delta_kernel(
                pq, pk, pv, pg, pb, state0, cu, scale, output_final_state=True)
            read_only, discarded = gdn.gated_delta_kernel(
                pq, pk, pv, pg, pb, state0, cu, scale, output_final_state=False)
            assert discarded is None
            torch.testing.assert_close(read_only, ker)
            out[f"recurrence_{name}_{layout}"] = {
                "nan_kernel": int(torch.isnan(ker).sum()),
                "max_abs_diff": float(
                    (ref.float() - ker.float()).abs().max()),
                "state_max_abs_diff": float((s_ref - final).abs().max()),
            }

    # one-token decode steps updating arena rows in place, against the
    # chunked kernel on gathered state: slot 0, a second layer, and a
    # step that moves its state to another slot first (slot 1 to 3, while
    # another step advances slot 1 itself)
    from types import SimpleNamespace

    reads, writes = [0, 2, 1, 1], [0, 2, 3, 1]
    layer = 1
    steps = len(reads)
    state_rows = torch.zeros(5, 2, hv, dv, dk, device=device)
    state_rows[1:] = torch.randn(4, 2, hv, dv, dk, device=device) * 0.1
    arena = SimpleNamespace(gdn_state_rows=state_rows,
                            gdn_state=state_rows[1:])
    gathered = arena.gdn_state[torch.tensor(reads, device=device), layer].clone()
    step_cu = torch.arange(steps + 1, dtype=torch.int32, device=device)
    pq, pk, pv, pg, pb = fused_post_conv_prep(
        conv_output=joined[:steps], a=a[:steps], b=b[:steps], A_log=a_log,
        dt_bias=dt_bias, num_k_heads=hk, head_k_dim=dk, head_v_dim=dv,
        apply_l2norm=True, output_g_exp=False)
    expected, final = gdn.gated_delta_kernel(
        pq, pk, pv, pg, pb, gathered, step_cu, scale, output_final_state=True)
    moved = [i for i in range(steps) if reads[i] != writes[i]]
    step = gdn.GdnPhase(
        rows=None, cu_seqlens=step_cu, lengths=(1,) * steps,
        state_rows=torch.tensor(writes, device=device) + 1,
        copy_reads=torch.tensor([reads[i] for i in moved], device=device),
        copy_writes=torch.tensor([writes[i] for i in moved], device=device))
    other_layer = state_rows[:, 0].clone()
    got = gdn._recurrent_decode(arena, step, layer, pq, pk, pv, pg, pb, scale)
    written = arena.gdn_state[torch.tensor(writes, device=device), layer]
    out["recurrent_decode"] = {
        "nan_kernel": int(torch.isnan(got).sum()),
        "max_abs_diff": float((expected.float() - got.float()).abs().max()),
        "state_max_abs_diff": float((final - written).abs().max()),
        "row_0_untouched": bool(not state_rows[0].any()),
        "other_layer_untouched": bool(torch.equal(other_layer,
                                                  state_rows[:, 0])),
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
    ref_tail, _ = causal_conv_reference(
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
