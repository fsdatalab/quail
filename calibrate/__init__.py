"""Measured fp8 GEMM rate for Quail's cost model.

Measures how fast a GPU runs fp8 matmuls with a vendor library rather
than Quail's own kernels, so the rate does not depend on Quail's code:

  gemm       fp8 matmuls with stas00's mamf-finder (cuBLAS): the GPU's
             MAMF and MSMF
  to_spec    turns the saved summaries into the sustained_fp8_flops
             field of quail/specs/<device>.py
  modal_app  runs gemm on Modal

gemm runs on any machine with the GPU and writes under the results
directory it is given. Like bench/, this package is not in the quail
wheel.
"""

from datetime import datetime, timezone


def utc_stamp() -> str:
    """Return the current UTC time as a run directory name."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
