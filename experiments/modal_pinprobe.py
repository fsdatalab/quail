"""Platform probe: which pinned-memory paths work on Modal H100
containers, and what host-to-GPU bandwidth each path achieves.

Three paths:
  1. unpinned  - plain host tensor, driver bounce-buffers each chunk
  2. alloc     - torch pin_memory=True (cudaHostAlloc), the HiCache
                 style host pool
  3. register  - mmap a region, then cudaHostRegister it (the vLLM
                 tiering connector style); expected to fail with 304
                 in this sandbox, and the probe verifies the error
                 clears rather than killing the next kernel

Banks results/engine/pinprobe.json. Run:
    modal run experiments/modal_pinprobe.py
"""
import json

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0")   # same torch build the flights use
)

app = modal.App("docengine-pinprobe")
GB = 1 << 30


@app.function(image=image, gpu="H100!", timeout=900, memory=65536)
def probe(size_gb: int = 4) -> dict:
    import ctypes
    import mmap
    import time

    import torch

    n = size_gb * GB
    dev = torch.device("cuda")
    out = dict(size_gb=size_gb)

    def h2d_gbps(t):
        # one warm pass, then three timed
        g = torch.empty_like(t, device=dev)
        g.copy_(t)
        torch.cuda.synchronize()
        best = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            g.copy_(t)
            torch.cuda.synchronize()
            best = max(best, n / (time.perf_counter() - t0) / 1e9)
        return round(best, 2)

    src = torch.randint(0, 255, (n,), dtype=torch.uint8)
    out["unpinned_h2d_GBps"] = h2d_gbps(src)
    del src

    try:
        pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        pinned.random_(0, 255)
        out["alloc_pinned_ok"] = True
        out["alloc_pinned_h2d_GBps"] = h2d_gbps(pinned)
        del pinned
    except Exception as e:
        out["alloc_pinned_ok"] = False
        out["alloc_pinned_error"] = f"{type(e).__name__}: {e}"

    # a big pinned allocation, the persist32-scale check
    try:
        big = torch.empty(48 * GB, dtype=torch.uint8, pin_memory=True)
        out["alloc_pinned_48GB_ok"] = True
        del big
    except Exception as e:
        out["alloc_pinned_48GB_ok"] = False
        out["alloc_pinned_48GB_error"] = f"{type(e).__name__}: {e}"

    region = mmap.mmap(-1, 1 * GB)
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    rc = torch.cuda.cudart().cudaHostRegister(ptr, 1 * GB, 0)
    out["register_mmap_code"] = int(rc.value)
    if rc.value == 0:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    else:
        # verify the documented hazard: without a clear, the next
        # kernel inherits the error; with one, it does not
        torch.cuda.cudart().cudaGetLastError()
    try:
        torch.ones(8, device=dev).sum().item()
        out["kernel_after_register_ok"] = True
    except Exception as e:
        out["kernel_after_register_ok"] = False
        out["kernel_after_register_error"] = f"{type(e).__name__}: {e}"
    return out


@app.local_entrypoint()
def main(size_gb: int = 4):
    data = probe.remote(size_gb)
    print(json.dumps(data, indent=2))
    with open("results/engine/pinprobe.json", "w") as f:
        json.dump(data, f, indent=2)
    print("saved results/engine/pinprobe.json")
