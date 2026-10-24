"""A kernel moves scattered arena rows straight into pinned host memory.

Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword host_copy`.
"""

import time

import pytest

from quail.backends.quail.executor.host_copy import copy_rows

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

WIDTH = 8 * 128


def test_rows_gather_to_the_host_and_scatter_back_through_the_kernel():
    pool = torch.randn((65536, WIDTH), device="cuda", dtype=torch.bfloat16)
    rows = torch.randperm(65536, device="cuda")[:300 * 72]
    host = torch.empty((rows.numel(), WIDTH), dtype=torch.bfloat16,
                       pin_memory=True)
    copy_rows(pool, host, source_rows=rows)
    torch.cuda.synchronize()
    assert torch.equal(host, pool[rows].cpu())
    scattered = torch.zeros_like(pool)
    copy_rows(host, scattered, destination_rows=rows)
    torch.cuda.synchronize()
    assert torch.equal(scattered[rows], pool[rows])


def test_the_kernel_moves_a_document_at_the_link_speed():
    pool = torch.randn((65536, WIDTH), device="cuda", dtype=torch.bfloat16)
    count, repeats = 300 * 72, 50
    rows = torch.randperm(65536, device="cuda")[:count]
    host = torch.empty((repeats, count, WIDTH), dtype=torch.bfloat16,
                       pin_memory=True)
    copy_rows(pool, host[0], source_rows=rows)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for repeat in range(repeats):
        copy_rows(pool, host[repeat], source_rows=rows)
    host_s = time.perf_counter() - started
    torch.cuda.synchronize()
    total_s = time.perf_counter() - started
    moved = repeats * count * WIDTH * 2
    print(f"\n{repeats} gathers to the host: host {host_s * 1e3 / repeats:.3f} ms "
          f"per launch, {moved / total_s / 1e9:.1f} GB/s")
    started = time.perf_counter()
    for repeat in range(repeats):
        copy_rows(host[repeat], pool, destination_rows=rows)
    host_s = time.perf_counter() - started
    torch.cuda.synchronize()
    total_s = time.perf_counter() - started
    print(f"{repeats} scatters from the host: host {host_s * 1e3 / repeats:.3f} ms "
          f"per launch, {moved / total_s / 1e9:.1f} GB/s")
    # the copy engine on the same host, for the same bytes, contiguous
    staging = pool[:count].contiguous()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for repeat in range(repeats):
        host[repeat].copy_(staging, non_blocking=True)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - started
    print(f"copy engine to the host: {moved / total_s / 1e9:.1f} GB/s")
    started = time.perf_counter()
    for repeat in range(repeats):
        staging.copy_(host[repeat], non_blocking=True)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - started
    print(f"copy engine from the host: {moved / total_s / 1e9:.1f} GB/s")
    assert host_s / repeats < 0.001
