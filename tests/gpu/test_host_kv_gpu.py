"""The host store round-trips KV on a CUDA arena, and its host cost per save.

Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword host_kv`.
"""

import time

import pytest

from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.host_kv import HostKV

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

LAYERS, HEADS, DIM, PAGE = 36, 8, 128, 16


def make_arena(pages=4096):
    return KVArena(num_layers=LAYERS, num_pages=pages, page_tokens=PAGE,
                   num_kv_heads=HEADS, d_head=DIM, dtype=torch.bfloat16,
                   device="cuda")


def fill(arena, key, tokens, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = arena.capacity_rows(key)[:tokens].to("cuda")
    for layer in range(LAYERS):
        for tensor in arena.layer_kv(layer):
            tensor[rows] = torch.randn((tokens, HEADS, DIM), generator=generator,
                                       device="cuda").to(torch.bfloat16)


def snapshot(arena, key, tokens):
    rows = arena.capacity_rows(key)[:tokens].to("cuda")
    return [tensor[rows].clone() for layer in range(LAYERS)
            for tensor in arena.layer_kv(layer)]


def test_a_saved_prefix_restores_bit_for_bit_on_cuda():
    arena = make_arena()
    host = HostKV(arena, 2 * 2**30, torch)
    arena.host = host
    key, tokens = ("d", 0), 300
    digest = host.register(key, list(range(tokens)), tokens)
    assert arena.reserve(key, tokens, capacity_tokens=tokens + 64)
    fill(arena, key, tokens, seed=1)
    host.mark_written(key)
    before = snapshot(arena, key, tokens)
    arena.free_key(key)
    assert host.holds(digest)
    # the pages stay the key's until the save has read them
    assert arena.is_resident(key) and arena.reclaim(wait=True)
    assert not arena.is_resident(key)
    for layer in range(LAYERS):
        for tensor in arena.layer_kv(layer):
            tensor.zero_()
    assert arena.reserve(("e", 1), 100, capacity_tokens=100)
    assert arena.reserve(key, tokens, capacity_tokens=tokens + 64)
    assert host.restore(key, digest)
    torch.cuda.synchronize()
    after = snapshot(arena, key, tokens)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_a_save_costs_the_host_little_while_a_pass_runs():
    arena = make_arena(pages=8192)
    host = HostKV(arena, 16 * 2**30, torch)
    arena.host = host
    tokens, count = 300, 200
    keys = [("d", index) for index in range(count)]
    for index, key in enumerate(keys):
        host.register(key, [index] * tokens, tokens)
        assert arena.reserve(key, tokens, capacity_tokens=tokens + 64)
        host.mark_written(key)
    weight = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    activations = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    # a pass in flight on the compute stream: each save must queue
    # behind it without waiting for it
    for _ in range(20):
        activations = activations @ weight
    started = time.perf_counter()
    # a burst past the staging blocks skips, as a query's settles do
    for key in keys[:100]:
        arena.free_key(key)
    burst = host.saved_keys + host.busy_keys
    # at a query's end the frees wait for blocks instead
    with host.patient():
        for key in keys[100:]:
            arena.free_key(key)
    host_s = time.perf_counter() - started
    arena.reclaim(wait=True)
    total_s = time.perf_counter() - started
    print(f"\n{count} saves of {tokens} tokens: host {host_s * 1e3 / count:.2f} ms "
          f"per save, {total_s:.2f} s until the copies landed, "
          f"store host_s {host.host_s:.3f} s")
    assert burst == 100 and host.busy_keys > 0
    assert host.saved_keys + host.busy_keys == count and arena.free_pages == 8192
    saved = [key for key in keys if host.holds(host._keys[key][0])]
    assert len(saved) == host.saved_keys >= 100
    keys = saved
    count = len(keys)
    # restores: the same keys, pages reserved again
    for _ in range(20):
        activations = activations @ weight
    started = time.perf_counter()
    # a burst of restores past the blocks is a miss, computed instead;
    # patient, each waits for the block the compute stream has read
    with host.patient():
        for index, key in enumerate(keys):
            assert arena.reserve(key, tokens, capacity_tokens=tokens + 64)
            assert host.restore(key, host._keys[key][0])
    host_s = time.perf_counter() - started
    torch.cuda.synchronize()
    total_s = time.perf_counter() - started
    print(f"{count} restores: host {host_s * 1e3 / count:.2f} ms per restore, "
          f"{total_s:.2f} s until landed")
    assert activations.shape == weight.shape
    # a save is one launch per layer group and a restore the same; a
    # burst longer than the index slots is paced at the link's speed,
    # about 1.7 ms per 45 MB document
    assert host_s / count < 0.003
