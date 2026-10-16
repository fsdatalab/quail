"""The GPUs calibrate runs on."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """How to run the measurements on one device.

    Attributes:
        device: The quail.specs device name the results are for.
        modal_gpu: The Modal GPU string; a trailing "!" stops Modal from
            substituting another GPU type.
        published_fp8: mamf-finder's published fp8 MAMF and MSMF for
            this GPU; its two shapes are timed to check the GPU against
            them. None runs mamf-finder's own shape search instead.
    """

    device: str
    modal_gpu: str
    published_fp8: dict | None = None


# H100 SXM rows of the comparison table and its note 17, README of
# stas00/ml-engineering compute/accelerator at 8d028c29.
H100_PUBLISHED_FP8 = {
    "mamf_tflops": 1550, "mamf_shape": "1280x16896x16384",
    "msmf_tflops": 1358, "msmf_shape": "1536x2816x16384",
    "msmf_watts": 674, "msmf_mhz": 1824,
    "stack": "NGC 26.04, torch 2.12.0a0, CUDA 13.2, 700 W, siblings idle",
}

TARGETS = {target.device: target for target in (
    Target("h100-sxm", "H100!", H100_PUBLISHED_FP8),
)}
