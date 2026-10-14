"""The GPUs calibrate runs on, and the model shapes and call sizes it measures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """How to run the measurements on one device.

    Attributes:
        device: The quail.specs device name the results are for.
        modal_gpu: The Modal GPU string; a trailing "!" stops Modal from
            substituting another GPU type.
        flashinfer_arch: FLASHINFER_CUDA_ARCH_LIST, the architecture
            FlashInfer compiles its kernels for.
        published_fp8: mamf-finder's published fp8 MAMF and MSMF for
            this GPU; its two shapes are timed to check the GPU against
            them. None runs mamf-finder's own shape search instead.
    """

    device: str
    modal_gpu: str
    flashinfer_arch: str
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
    Target("h100-sxm", "H100!", "9.0a", H100_PUBLISHED_FP8),
)}


def gemm_shapes(model) -> dict[str, tuple[int, int]]:
    """Return (N, K) of each dense projection in one layer of a model.

    Args:
        model: A quail.specs ModelSpec.

    Returns:
        qkv, o, gate_up, and down, each as (output width, input width).
    """
    return {
        "qkv": ((model.n_q + 2 * model.n_kv) * model.d_head, model.hidden),
        "o": (model.hidden, model.n_q * model.d_head),
        "gate_up": (model.ffn_width, model.hidden),
        "down": (model.hidden, model.intermediate),
    }



def measured_models():
    """Return the quail.specs models whose attention the cost model prices.

    The cost model uses measured attention rates only for a model with
    no sliding-window layers and one head layout on every layer.
    """
    from quail.specs import MODELS

    return [m for m in MODELS.values() if not m.sliding_layer_set
            and all(shape == (m.n_kv, m.d_head) for shape in m.kv_shapes)]


def head_layouts() -> list[tuple[int, int, int]]:
    """Return the distinct (query heads, KV heads, head dim) of measured_models()."""
    return sorted({(m.n_q, m.n_kv, m.d_head) for m in measured_models()})


def call_sizes(heads, device: str) -> tuple[int, int]:
    """Return the largest Quail chunk and KV arena of the models with a layout.

    Args:
        heads: (query heads, KV heads, head dimension).
        device: The quail.specs device name.

    Returns:
        (chunk tokens, KV arena tokens), each the largest over the
        measured_models() with this layout on this device.

    Raises:
        ValueError: If no measured model has the layout.
    """
    from quail.cost import budgets
    from quail.specs import DEVICES

    spec = DEVICES[device]
    sizes = []
    for model in measured_models():
        if (model.n_q, model.n_kv, model.d_head) != tuple(heads):
            continue
        chunk = budgets.chunk_budget(model, spec)
        pages, _ = budgets.arena_pages(model, spec, chunk)
        sizes.append((chunk, pages * budgets.PAGE_TOKENS))
    if not sizes:
        raise ValueError(f"no measured model has head layout {tuple(heads)}")
    return max(c for c, _ in sizes), max(k for _, k in sizes)
