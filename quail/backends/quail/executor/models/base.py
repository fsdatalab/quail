"""The forward pass one model family runs over the shared engine.

A model pipeline owns the forward loop for one architecture: where the
modules sit in vLLM's loaded model, how the packed rows are embedded,
the per-layer block, and the final norm. It calls the Engine for the
kernels, quantization, attention paths, and arena access. The chunk
loop sees only this class.
"""

from contextlib import nullcontext


def overlaid_rows(torch, num_tokens, device, *layouts):
    """Row-major boundary tensors of mixer kinds that are never live at once.

    Each layout is a list of widths, laid out back to back from the
    start of one shared buffer as (num_tokens, width) tensors; the
    buffer fits the largest layout.

    Returns:
        One list of tensors per layout.
    """
    flat = torch.zeros(num_tokens * max(sum(layout) for layout in layouts),
                       dtype=torch.bfloat16, device=device)
    views = []
    for layout in layouts:
        start, tensors = 0, []
        for width in layout:
            end = start + num_tokens * width
            tensors.append(flat[start:end].view(num_tokens, width))
            start = end
        views.append(tensors)
    return views


class ModelPipeline:
    """The contract between the chunk loop and one model family.

    A subclass sets self.engine and self.max_chunk_tokens in __init__
    and implements forward_chunk; with gemm_warmup it also implements
    linears. tree_attention says the model's kernels can run the
    tree attention path (two calls, then a fused merge); a
    chunk asked for it on a model without it runs unified, the one
    causal call every model has. A diffusion model also sets
    canvas_ids: the token ids the loop packs after every suffix, with
    the answer at canvas_answer_row of them. needs_pages says every
    chunk must carry arena pages, for a model whose attention kernel
    reads paged KV only. warm_tokens are the chunk sizes the warmup
    builds one small chunk of: a gated sequence's trailing chunks are
    100-500 tokens, a shape the full-size warm chunks do not cover.
    """

    engine = None
    max_chunk_tokens = None
    warm_tokens = (64, 128, 256, 512, 1024, 2048)
    canvas_ids = ()
    canvas_answer_row = 0
    needs_pages = False
    tree_attention = True
    gemm_warmup = True
    warm_row_classes = True    # row-count classes keep one KV row per reader
    warm_joins = True          # join and classification chunks borrow prefixes
    piecewise_graphs = None
    # For a spec with an SAE, forward_chunk returns the residual stream
    # after the last loaded layer instead of the final-normed rows, and
    # forward_hook, when a query sets it, is called with (stream, chunk)
    # over every row of each chunk before the final_indices select.
    forward_hook = None

    def forward_chunk(self, chunk):
        """Return the final-normed hidden state of chunk.final_indices.

        For a spec with an SAE, return the residual stream of those rows
        instead, without the final norm. The loop owns the chunk's
        pages, temporaries included; this only computes.
        """
        raise NotImplementedError

    def linears(self):
        """One linear module per distinct GEMM shape, for kernel warm-up."""
        raise NotImplementedError

    def convolutions(self):
        """Return (weight, bias, activate) per convolution shape for warmup."""
        return ()

    def forward_context(self, num_tokens):
        """Return the context a forward pass, or a captured piece, runs under."""
        return nullcontext()

    # A family whose specs get CUDA graph sizes (graph_token_buckets)
    # splits its forward pass at its token mixers (executor/cuda_graphs.py):
    # piece 0 runs up to layer 0's mixer, piece i from layer i - 1's mixer
    # to layer i's, and the last piece writes the final-normed rows.

    def boundary(self, num_tokens, device) -> dict:
        """Row-major tensors the pieces and mixers pass to each other."""
        raise NotImplementedError

    def bucket_boundary(self, num_tokens, device) -> dict:
        """Boundary tensors each graph size allocates, not sliced by row."""
        return {}

    def piece(self, index, buffers):
        """Captured work before layer index's token mixer (see above)."""
        raise NotImplementedError

    def mixer(self, index, buffers, chunk):
        """Run layer index's token mixer eagerly over the chunk's real rows."""
        raise NotImplementedError
