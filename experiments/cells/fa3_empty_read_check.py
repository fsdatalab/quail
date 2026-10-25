"""Check what FlashAttention 3 returns for a paged query that reads no keys.

A captured decode graph pads its split reads with sequences of zero
resident keys; merging their result into a real row is a no-op only
when the output is finite and the log-sum-exp is minus infinity.

    uv run modal run -m experiments.cells.fa3_empty_read_check
"""

from bench.quailb_parallel import app, image


@app.function(image=image, gpu="H100!", timeout=600)
def fa3_empty_read_check() -> str:
    """Return the output and log-sum-exp of empty and one-key paged reads."""
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    q = torch.randn(3, 32, 128, dtype=torch.bfloat16, device="cuda")
    kp = torch.randn(4, 16, 8, 128, dtype=torch.bfloat16, device="cuda")
    vp = torch.randn_like(kp)
    cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    table = torch.zeros((3, 2), dtype=torch.int32, device="cuda")
    used = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
    out, lse = flash_attn_varlen_func(
        q, kp, vp, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=32,
        cu_seqlens_k=None, block_table=table, seqused_k=used, causal=False,
        fa_version=3, return_softmax_lse=True)
    torch.cuda.synchronize()
    report = {
        "out_finite": [bool(torch.isfinite(out[i]).all()) for i in range(3)],
        "out_abs_max": [float(out[i].abs().max()) for i in range(3)],
        "lse": [[float(v) for v in lse[:4, i]] for i in range(3)],
    }
    print(report, flush=True)
    return str(report)


@app.local_entrypoint()
def launch():
    call = fa3_empty_read_check.spawn()
    print(f"function call id: {call.object_id}")
    print(call.get())
