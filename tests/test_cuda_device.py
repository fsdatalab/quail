"""The CUDA device implementation's pools and row selection, on the CPU."""

import numpy as np
import pytest
from fakes import cpu_implementation

from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.cuda_device import TorchKVPools
from quail.backends.quail.executor.device import DeviceImplementation

torch = pytest.importorskip("torch")


def test_torch_pools_are_built_and_rebuilt_by_the_arena():
    pools = TorchKVPools(torch, dtype=torch.float32, device="cpu")
    # layer 0 keeps every token, layer 1 slides and has wider KV
    arena = KVArena(n_layers=2, n_pages=64, page_tokens=16, n_kv=1, d_head=2,
                    pools=pools, layer_kv=[(1, 2), (3, 4)], sliding_layers=(1,),
                    sliding_window=32, n_sliding_pages=16)
    k, v = arena.layer_kv(0)
    assert k.shape == v.shape == (64 * 16, 1, 2) and k.dtype == torch.float32
    assert arena.layer_kv(1)[0].shape == (16 * 16, 3, 4)
    paged_k, paged_v = arena.paged_kv(1)
    assert paged_k.shape == paged_v.shape == (16, 16, 3, 4)
    # a paged pool is a view of the flat one
    paged_k[2, 5] = 7.0
    assert arena.layer_kv(1)[0][2 * 16 + 5].eq(7.0).all()
    arena.resize(32, 8)
    assert arena.layer_kv(0)[0].shape[0] == 32 * 16
    assert arena.layer_kv(1)[0].shape[0] == 8 * 16
    assert TorchKVPools(torch).dtype == torch.bfloat16


def test_rows_are_selected_by_a_host_index_or_a_staged_one():
    implementation = cpu_implementation()
    assert implementation.name == "cuda"
    rows = torch.arange(12).reshape(6, 2)
    assert implementation.select_rows(rows, np.array([4, 1])).tolist() == [
        [8, 9], [2, 3]]
    staged = implementation.stage([0, 5], np.int64)
    assert implementation.select_rows(rows, staged).tolist() == [[0, 1], [10, 11]]
    # a test model's list of answers is selected on the host
    assert implementation.select_rows([10, 11, 12], np.array([2, 0])) == [12, 10]


def test_the_base_implementation_keeps_arrays_on_the_host():
    host = DeviceImplementation()
    staged = host.stage([[1, 2], [3, 4]], np.int32)
    assert isinstance(staged, np.ndarray) and staged.dtype == np.int32
    ids = np.array([5, 6], dtype=np.int64)
    assert host.stage_tokens(ids) is ids
    assert host.select_rows(np.arange(6).reshape(3, 2), staged[0]).tolist() == [
        [2, 3], [4, 5]]
    assert host.input_staging() is None
    host.time_chunks(True)
    start = host.record_event()
    host.synchronize()
    assert host.elapsed_ms(start, host.record_event()) >= 0
    assert host.peak_memory_bytes() == 0
    with host.inference_mode():
        pass
    for readout in (lambda: host.decision_choices(None, (1, 0)),
                    lambda: host.scores(None),
                    lambda: host.label_readout(None, [1], rows=1, normalize=True)):
        with pytest.raises(NotImplementedError, match="host has no"):
            readout()
