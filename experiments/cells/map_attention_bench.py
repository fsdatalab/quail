"""Time the attention calls of one decode pass of several maps on one GPU.

Shapes follow CUAD-MAP-2 on Qwen3 4B: documents of about 7,500 tokens,
three branches each, one decode row per branch, 8 KV heads of 128.
Each case runs one layer's call; the KV bytes it must read set its
memory-bound floor.

    uv run --no-sync modal run experiments/cells/map_attention_bench.py
"""

import json

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-map-attention-bench")


@app.function(image=gpu_image(), gpu="H100!", timeout=1200)
def bench(documents: int = 45, branches: int = 3, shared_tokens: int = 7488,
          own_tokens: int = 112) -> str:
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    page, heads, kv_heads, dim = 16, 32, 8, 128
    shared_pages = shared_tokens // page
    own_pages = -(-own_tokens // page)
    rows = documents * branches
    num_pages = documents * shared_pages + rows * own_pages + 1
    kp = torch.randn(num_pages, page, kv_heads, dim, device="cuda",
                     dtype=torch.bfloat16)
    vp = torch.randn_like(kp)
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    doc_pages = torch.arange(documents * shared_pages, device="cuda",
                             dtype=torch.int32).view(documents, shared_pages)
    own = (documents * shared_pages + torch.arange(
        rows * own_pages, device="cuda", dtype=torch.int32)).view(rows, own_pages)
    doc_of_row = torch.arange(rows, device="cuda") // branches
    whole = torch.cat([doc_pages[doc_of_row], own], dim=1)

    def ints(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    one_row = ints(list(range(rows + 1)))
    stacked = ints([d * branches for d in range(documents + 1)])
    cases = {
        # every branch reads the whole document and its own tokens
        "unified": dict(cu=one_row, max_q=1, table=whole,
                        used=ints([shared_tokens + own_tokens] * rows),
                        causal=True, tokens=rows * (shared_tokens + own_tokens)),
        # each branch reads only its own pages
        "own": dict(cu=one_row, max_q=1, table=own,
                    used=ints([own_tokens] * rows), causal=True,
                    tokens=rows * own_tokens),
        # a document's branches read its pages once, stacked
        "shared": dict(cu=stacked, max_q=branches, table=doc_pages,
                       used=ints([shared_tokens] * documents), causal=False,
                       tokens=documents * shared_tokens),
    }
    bytes_per_token = 2 * kv_heads * dim * 2
    results = {}
    for name, case in cases.items():
        query = q if name != "shared" else q
        for splits in (0, 1):
            def call():
                return flash_attn_varlen_func(
                    query, kp, vp, max_seqlen_q=case["max_q"],
                    cu_seqlens_q=case["cu"], max_seqlen_k=int(case["used"].max()),
                    seqused_k=case["used"], block_table=case["table"],
                    causal=case["causal"], fa_version=3,
                    return_softmax_lse=True, num_splits=splits)
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                call()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 100
            floor = case["tokens"] * bytes_per_token / 3.35e12 * 1e3
            results[f"{name} splits={splits}"] = dict(
                ms=round(ms, 4), floor_ms=round(floor, 4),
                tb_per_s=round(case["tokens"] * bytes_per_token / ms / 1e9, 2))
    return json.dumps(results, indent=1)


@app.local_entrypoint()
def main(documents: int = 45, branches: int = 3, shared_tokens: int = 7488,
         own_tokens: int = 112):
    print(bench.remote(documents, branches, shared_tokens, own_tokens))
