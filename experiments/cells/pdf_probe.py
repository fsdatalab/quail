r"""Phase 0 probes for PDF image inputs (issue #148).

Two functions on the milestone app, one GPU and one CPU:

- probe_vision (H100): loads DiffusionGemma through Quail's own
  load_model, then reports what the FP8 checkpoint gives us for
  images: vision_config and use_bidirectional_attention in the
  config, whether vLLM built vision_tower and embed_vision, their
  parameter count and GPU bytes, whether the checkpoint carries a
  processor, the per-page soft token count the processor assigns to
  a US letter page at each budget, and the time embed_multimodal
  takes for 1 and 8 rendered pages.
- probe_render (CPU only): renders every page of one 15 page arXiv
  paper at the 280 budget target size with PDFium, once in this
  process and then through spawn pools of 2, 4, and 8 processes, and
  reports milliseconds per page and pages per second for each.

PREDICTION (before the run): the FP8 checkpoint keeps vision_config
and the vision weights, so vision_tower is built with about 550M
parameters in bf16 (about 1.1 GB). A letter page at budget 280
gets 266 soft tokens (672 x 912 px, 42 x 57 patches). One PDFium
process renders a text page at that size in 15 to 40 ms; pools
scale close to linearly up to the container's CPU count. The vision
encoder takes 5 to 10 ms per page on the H100.

Run from the repository root:

    uv run modal run -m experiments.cells.pdf_probe 2>&1 \\
        | tee /tmp/pdf_probe.log

Pass ``--only render`` or ``--only vision`` to run one probe.

Records land on the quail-results volume under
/results/ablations/pdf_probe_vision.json and pdf_probe_render.json.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import modal

app = modal.App("quail-milestone1")

PYPDFIUM2 = "pypdfium2==4.30.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow",
                 "sqlglot>=27.0", "gigatoken>=0.10.0", "datasets>=5.0.1",
                 PYPDFIUM2)
    .env({
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernels/torchinductor",
    })
    .add_local_python_source("quail", "experiments")
)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
volumes = {
    "/results": results_vol,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}

# "Attention Is All You Need": 15 US letter pages of text, tables,
# and one figure-heavy page, a fair document rendering workload.
PDF_URL = "https://arxiv.org/pdf/1706.03762v7"
LETTER_POINTS = (612.0, 792.0)
BUDGETS = (70, 140, 280, 560, 1120)
PATCH = 16
POOL = 3


def target_pixels(width: float, height: float, budget: int) -> tuple[int, int]:
    """The Gemma 4 processor's resize target for one page, as (h, w).

    Mirrors transformers' get_aspect_ratio_preserving_size: the
    largest size with at most budget * POOL**2 patches whose sides are
    multiples of POOL * PATCH.
    """
    max_patches = budget * POOL * POOL
    factor = math.sqrt(max_patches * PATCH * PATCH / (width * height))
    side = POOL * PATCH
    h = int(math.floor(factor * height / side)) * side
    w = int(math.floor(factor * width / side)) * side
    return max(h, side), max(w, side)


def soft_tokens(h: int, w: int) -> int:
    return (h // PATCH) * (w // PATCH) // (POOL * POOL)


def fetch_pdf(path: str) -> str:
    if not Path(path).exists():
        urllib.request.urlretrieve(PDF_URL, path)
    return path


def render_page(path: str, index: int, size_hw: tuple[int, int]):
    """Render one page to an RGB uint8 array of exactly size_hw."""
    import numpy as np
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    h, w = size_hw
    pdf = pdfium.PdfDocument(path)
    page = pdf[index]
    bitmap = pdfium.PdfBitmap.new_native(
        w, h, format=pdfium_c.FPDFBitmap_BGR, rev_byteorder=True)
    bitmap.fill_rect(0, 0, w, h, (255, 255, 255, 255))
    pdfium_c.FPDF_RenderPageBitmap(
        bitmap.raw, page.raw, 0, 0, w, h, 0,
        pdfium_c.FPDF_ANNOT | pdfium_c.FPDF_LCD_TEXT)
    array = np.array(bitmap.to_numpy(), copy=True)
    page.close()
    pdf.close()
    return array


def _render_bytes(args):
    path, index, size_hw = args
    started = time.perf_counter()
    array = render_page(path, index, size_hw)
    return array.tobytes(), time.perf_counter() - started


def _time_pool(path, n_pages, size_hw, processes):
    ctx = get_context("spawn")
    jobs = [(path, i, size_hw) for i in range(n_pages)] * 3
    with ProcessPoolExecutor(processes, mp_context=ctx) as pool:
        list(pool.map(_render_bytes, jobs[:processes]))  # warm the workers
        started = time.perf_counter()
        worker_s = [t for _, t in pool.map(_render_bytes, jobs)]
        wall = time.perf_counter() - started
    return {"processes": processes, "pages": len(jobs), "wall_s": wall,
            "pages_per_s": len(jobs) / wall,
            "worker_ms_per_page": 1000 * statistics.fmean(worker_s)}


@app.function(image=image, memory=16384, cpu=8.0, volumes=volumes, timeout=1200)
def probe_render():
    import pypdfium2 as pdfium

    path = fetch_pdf("/tmp/paper.pdf")
    pdf = pdfium.PdfDocument(path)
    n_pages = len(pdf)
    sizes = [pdf[i].get_size() for i in range(n_pages)]
    pdf.close()
    size_hw = target_pixels(*sizes[0], 280)
    record = {"cpu_count": os.cpu_count(), "pages": n_pages,
              "page_points": sizes[0], "render_hw": size_hw,
              "soft_tokens_280": soft_tokens(*size_hw),
              "bytes_per_page": size_hw[0] * size_hw[1] * 3}
    per_page = []
    for i in range(n_pages):
        started = time.perf_counter()
        render_page(path, i, size_hw)
        per_page.append(1000 * (time.perf_counter() - started))
    record["single_process_ms"] = {
        "mean": statistics.fmean(per_page), "median": statistics.median(per_page),
        "max": max(per_page), "per_page": per_page}
    record["pools"] = [_time_pool(path, n_pages, size_hw, n) for n in (2, 4, 8)]
    print(json.dumps(record, indent=1), flush=True)
    _save("pdf_probe_render.json", record)
    return record


def _save(name, record):
    destination = Path("/results/ablations") / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=1))
    results_vol.commit()
    print("saved", destination, flush=True)


@app.function(image=image, gpu="H100!", memory=98304, volumes=volumes,
              timeout=1800)
def probe_vision():
    import torch

    from quail.backends.quail.executor.model import load_model, resolve_model_path
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    spec = DIFFUSION_GEMMA_26B_FP8
    record = {"model": spec.hf_name}
    model_path = resolve_model_path(spec.hf_name, spec.revision or None)
    config = json.loads((Path(model_path) / "config.json").read_text())
    text_config = config.get("text_config", config)
    record["config"] = {
        "has_vision_config": "vision_config" in config,
        "use_bidirectional_attention": text_config.get(
            "use_bidirectional_attention"),
        "image_token_id": config.get("image_token_id"),
        "boi_token_id": config.get("boi_token_id"),
        "eoi_token_id": config.get("eoi_token_id"),
        "vision_config": config.get("vision_config"),
        "files": sorted(p.name for p in Path(model_path).iterdir()),
    }
    index = Path(model_path) / "model.safetensors.index.json"
    if index.exists():
        keys = json.loads(index.read_text())["weight_map"]
        record["config"]["vision_weight_keys"] = sum(
            "vision" in k for k in keys)
        record["config"]["weight_key_sample"] = sorted(
            k for k in keys if "vision" in k)[:5]

    before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    model = load_model(spec.hf_name, spec.revision or None,
                       moe_backend=spec.moe_backend)
    record["load_s"] = time.perf_counter() - started
    record["model_bytes"] = torch.cuda.memory_allocated() - before
    tower = getattr(model, "vision_tower", None)
    embed_vision = getattr(model, "embed_vision", None)
    record["vision_tower"] = None
    if tower is not None:
        params = list(tower.parameters()) + list(embed_vision.parameters())
        record["vision_tower"] = {
            "class": type(tower).__name__,
            "params": sum(p.numel() for p in params),
            "bytes": sum(p.numel() * p.element_size() for p in params),
            "dtypes": sorted({str(p.dtype) for p in params}),
        }
    print(json.dumps(record, indent=1, default=str), flush=True)

    record["processor"] = _probe_processor(model_path)
    print(json.dumps(record["processor"], indent=1, default=str), flush=True)

    if tower is not None and record["processor"].get("ok"):
        record["encoder"] = _time_encoder(torch, model, model_path)
        print(json.dumps(record["encoder"], indent=1, default=str), flush=True)
    _save("pdf_probe_vision.json", record)
    return record


def _probe_processor(model_path):
    """Load the checkpoint's processor and count soft tokens per budget."""
    import numpy as np
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    except Exception as error:  # noqa: BLE001 - the probe reports, not raises
        return {"ok": False, "error": repr(error)}
    out = {"ok": True, "class": type(processor).__name__,
           "image_processor": type(processor.image_processor).__name__,
           "image_seq_length": getattr(processor, "image_seq_length", None),
           "budgets": {}}
    for budget in BUDGETS:
        h, w = target_pixels(*LETTER_POINTS, budget)
        page = np.full((h, w, 3), 255, dtype=np.uint8)
        try:
            features = processor.image_processor(
                images=[page], max_soft_tokens=budget, return_tensors="pt")
            out["budgets"][budget] = {
                "render_hw": [h, w], "our_soft_tokens": soft_tokens(h, w),
                "processor_soft_tokens": [
                    int(n) for n in features["num_soft_tokens_per_image"]],
                "pixel_values_shape": list(features["pixel_values"].shape),
            }
        except Exception as error:  # noqa: BLE001
            out["budgets"][budget] = {"error": repr(error)}
    try:
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": "Is this a table?"}]}],
            tokenize=False, add_generation_prompt=True)
        out["chat_template_text"] = text
    except Exception as error:  # noqa: BLE001
        out["chat_template_error"] = repr(error)
    return out


def _time_encoder(torch, model, model_path):
    """Time embed_multimodal on 1 and 8 rendered pages at budget 280."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    path = fetch_pdf("/tmp/paper.pdf")
    size_hw = target_pixels(*LETTER_POINTS, 280)
    pages = [render_page(path, i, size_hw) for i in range(8)]
    out = {"render_hw": size_hw}
    for count in (1, 8):
        features = processor.image_processor(
            images=pages[:count], max_soft_tokens=280, return_tensors="pt")
        pixel_values = features["pixel_values"].to("cuda", torch.bfloat16)
        positions = features["image_position_ids"].to("cuda")
        timings = []
        # the loaded modules still require grad; without inference mode
        # autograd keeps every encoder layer's activations
        with torch.inference_mode():
            for _ in range(4):
                torch.cuda.synchronize()
                started = time.perf_counter()
                embeds = model.embed_multimodal(
                    pixel_values=pixel_values, pixel_position_ids=positions)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
        out[f"images_{count}"] = {
            "warm_ms": 1000 * min(timings[1:]),
            "first_ms": 1000 * timings[0],
            "embeds": len(embeds),
            "embed_shape": list(embeds[0].shape),
            "embed_dtype": str(embeds[0].dtype),
            "soft_tokens": [int(n) for n in features["num_soft_tokens_per_image"]],
        }
    out["peak_bytes"] = torch.cuda.max_memory_allocated()
    return out


@app.local_entrypoint()
def main(only: str = ""):
    """Run both probes, or one of them with --only render|vision."""
    calls = []
    if only in ("", "render"):
        calls.append(("probe_render", probe_render.spawn()))
    if only in ("", "vision"):
        calls.append(("probe_vision", probe_vision.spawn()))
    for name, call in calls:
        print(f"{name} function call id: {call.object_id}", flush=True)
    for _, call in calls:
        call.get()
