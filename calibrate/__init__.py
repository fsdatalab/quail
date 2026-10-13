"""Measured hardware ceilings for Quail's cost model.

Measures how fast a GPU really runs the work Quail prices, with vendor
libraries rather than Quail's own kernels, so the ceilings do not
depend on Quail's code:

  gemm       fp8 matmuls with stas00's mamf-finder (cuBLAS): the GPU's
             best sustained rate, and a model's projections over chunk
             sizes
  attention  FlashAttention 3 and FlashInfer at a model's head layout:
             whole causal documents, query rows reading cached KV, and
             a device-to-device copy for memory bandwidth
  to_spec    turns the two saved summaries into the DeviceSpec fields
             of quail/specs/<device>.py

The measurement functions run on Modal and never import quail; the
local entrypoints and to_spec read model and device specs from
quail.specs. Like bench/, this package is not in the quail wheel.
"""
