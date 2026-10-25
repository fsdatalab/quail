"""Check the fused readout reduction against torch, and time both.

The map readout reads each row's full-vocabulary logits once in a
Triton kernel for the normalizer and the likeliest token. This cell
compares it with torch's logsumexp and max on random logits and times
the two over a pass's worth of rows.

    uv run modal run -m experiments.cells.readout_check
"""

import json

from bench.quailb_parallel import app, image

VOCABULARY = 151_936
HIDDEN = 2_560
REPEATS = 20


@app.function(image=image, gpu="H100!", timeout=900)
def readout_check() -> str:
    """Compare and time the fused and torch readouts; return JSON."""
    import torch

    from quail.backends.quail.executor.readout import row_top

    torch.manual_seed(0)
    results = {}
    for rows in (85, 760, 1024, 2048):
        normed = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        head = torch.randn(VOCABULARY, HIDDEN, dtype=torch.bfloat16,
                           device="cuda") * 0.02
        logits = torch.nn.functional.linear(normed, head)
        lse, top, value = row_top(torch, logits)
        norm = torch.logsumexp(logits.float(), dim=1)
        expected_value, expected_top = logits.max(dim=1)
        same_top = (top.long() == expected_top).float().mean().item()
        # a tie between two tokens lets either be the top; the values agree
        value_gap = (value - expected_value.float()).abs().max().item()
        lse_gap = (lse - norm).abs().max().item()

        def time(call):
            for _ in range(3):
                call()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPEATS):
                call()
            end.record()
            torch.cuda.synchronize()
            return round(start.elapsed_time(end) * 1000 / REPEATS, 1)

        def torch_path():
            norm = torch.logsumexp(logits, dim=1, keepdim=True).float()
            top, ids = logits.max(dim=1, keepdim=True)
            return top.float() - norm, ids

        results[rows] = dict(
            same_top=same_top, value_gap=value_gap, lse_gap=lse_gap,
            fused_us=time(lambda: row_top(torch, logits)),
            torch_us=time(torch_path),
            gemm_us=time(lambda: torch.nn.functional.linear(normed, head)))
        print(rows, json.dumps(results[rows]), flush=True)
    return json.dumps(results)


@app.local_entrypoint()
def launch():
    call = readout_check.spawn()
    print(f"function call id: {call.object_id}")
    print(call.get())
