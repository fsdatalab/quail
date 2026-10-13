"""Measured hardware ceilings for Quail's cost model.

Measures how fast a GPU really runs the work Quail prices, with vendor
libraries rather than Quail's own kernels, so the ceilings do not
depend on Quail's code:

  gemm       fp8 matmuls with stas00's mamf-finder (cuBLAS): the GPU's
             best sustained rate, and optionally a model's projections
             over chunk sizes
  attention  FlashAttention 3 and FlashInfer at one head layout: whole
             causal documents, query rows reading cached KV, and a
             device-to-device copy for memory bandwidth
  to_spec    turns the two saved summaries into the DeviceSpec fields
             of quail/specs/<device>.py
  modal_app  runs gemm and attention on Modal

gemm and attention run on any machine with the GPU and write under the
results directory they are given. Like bench/, this package is not in
the quail wheel.
"""

from datetime import datetime, timezone


def utc_stamp() -> str:
    """Return the current UTC time as a run directory name."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
