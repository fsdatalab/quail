"""The forward pass one model family runs over the shared engine.

A model pipeline owns the forward loop for one architecture: where the
modules sit in vLLM's loaded model, how the packed rows are embedded,
the per-layer block, and the final norm. It calls the Engine for the
kernels, quantization, attention paths, and arena access. The chunk
loop sees only this class.
"""


class ModelPipeline:
    """The contract between the chunk loop and one model family.

    A subclass sets self.engine and self.max_chunk_tokens in __init__
    and implements forward_chunk and linears.
    """

    engine = None
    max_chunk_tokens = None

    def forward_chunk(self, chunk):
        """Return the final-normed hidden state of chunk.final_indices.

        The loop owns the chunk's pages, temporaries included; this
        only computes.
        """
        raise NotImplementedError

    def linears(self):
        """One linear module per distinct GEMM shape, for kernel warm-up."""
        raise NotImplementedError

    @property
    def is_fp8(self):
        return self.engine.is_fp8
