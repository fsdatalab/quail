"""Time FlashAttention 3 at a small decode chunk's shapes.

A CUAD long map chunk at scale factor 0.1 holds about 17 documents
of about 830 resident tokens, feeding about 5 tokens each, under a
graph captured for 4,096-token queries, 65,536-token contexts, and
rows padded to a bucket. The bench times the resident-read call and
the fresh-rows call at those shapes while varying the padding, the
query limit, and the context limit, eager and replayed from a graph.

    uv run modal run experiments/cells/fa3_decode_bench.py
"""

import json

from bench.quailb_parallel import app, image

NUM_DOCUMENTS = 17
FED_TOKENS = 5
CONTEXT_TOKENS = 830
PAGE_TOKENS = 16
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
REPEATS = 50


def _case(torch, fa, *, padded_rows, max_query, max_context, graph):
    """Time one shape; return microseconds per call for the two calls."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = "cuda"
    rows = NUM_DOCUMENTS * FED_TOKENS
    total_rows = max(padded_rows, rows)
    q = torch.randn(total_rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    k = torch.randn(total_rows, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    v = torch.randn_like(k)
    # the real documents, then one-token padding sequences
    padding = total_rows - rows
    lengths = [FED_TOKENS] * NUM_DOCUMENTS + [1] * padding
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()],
                      dtype=torch.int32, device=device)
    sequences = len(lengths)
    pages_per_document = (CONTEXT_TOKENS + PAGE_TOKENS - 1) // PAGE_TOKENS
    table_width = (max_context + PAGE_TOKENS - 1) // PAGE_TOKENS
    num_pages = NUM_DOCUMENTS * pages_per_document + 1
    kp = torch.randn(num_pages, PAGE_TOKENS, NUM_KV_HEADS, HEAD_DIM,
                     dtype=torch.bfloat16, device=device)
    vp = torch.randn_like(kp)
    table = torch.zeros((sequences, table_width), dtype=torch.int32, device=device)
    for document in range(NUM_DOCUMENTS):
        table[document, :pages_per_document] = torch.arange(
            document * pages_per_document, (document + 1) * pages_per_document)
    used = torch.tensor([CONTEXT_TOKENS] * NUM_DOCUMENTS + [0] * padding,
                        dtype=torch.int32, device=device)

    def fresh():
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=max_query, cu_seqlens_q=cu, max_seqlen_k=max_query,
            cu_seqlens_k=cu, causal=True, fa_version=3, return_softmax_lse=True)

    def resident():
        return flash_attn_varlen_func(
            q, kp, vp, max_seqlen_q=max_query, cu_seqlens_q=cu,
            max_seqlen_k=max_context, cu_seqlens_k=None, block_table=table,
            seqused_k=used, causal=False, fa_version=3, return_softmax_lse=True)

    out = {}
    for name, call in (("fresh", fresh), ("resident", resident)):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                call()
            run = captured.replay
        else:
            run = call
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPEATS):
            run()
        end.record()
        torch.cuda.synchronize()
        out[name] = round(start.elapsed_time(end) * 1000 / REPEATS, 1)
    return out


@app.function(image=image, gpu="H100!", timeout=1800)
def fa3_decode_bench() -> str:
    """Run every case; return the timings as JSON."""
    import torch

    results = []
    for padded_rows in (NUM_DOCUMENTS * FED_TOKENS, 128, 1024, 4096):
        for max_query in (FED_TOKENS, 32, 4096):
            for max_context in (CONTEXT_TOKENS, 2048, 65536):
                for graph in (False, True):
                    case = dict(padded_rows=padded_rows, max_query=max_query,
                                max_context=max_context, graph=graph)
                    case.update(_case(torch, None, **case))
                    results.append(case)
                    print(json.dumps(case), flush=True)
    return json.dumps(results)


@app.local_entrypoint()
def launch():
    call = fa3_decode_bench.spawn()
    print(f"function call id: {call.object_id}")
    print(call.get())
