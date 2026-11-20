"""Attention version and FP8 GEMM dispatch without a GPU."""

import sys
from types import ModuleType, SimpleNamespace

from test_kv_reads import named

from quail.backends.quail.executor.attention import (
    Engine,
    flash_attention_version,
)


def check_attention_dispatch_preserves_each_device_path(
        monkeypatch, capability, head_dim, max_q, backend, version):
    calls = []

    def fake_attention(*args, **kwargs):
        calls.append(("fa", kwargs))
        return "fa-out", None

    def fake_flashinfer(self, query, keys, values, metadata, **kwargs):
        calls.append(("flashinfer", metadata))
        return "flashinfer-out"

    def fake_triton(self, query, keys, values, cu_q, max_q, used, max_used,
                    table, **kwargs):
        calls.append(("triton", dict(block_table=table, seqused_k=used)))
        return "triton-out"

    flash = ModuleType("vllm.vllm_flash_attn")
    flash.flash_attn_varlen_func = fake_attention
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", flash)
    monkeypatch.setattr(Engine, "_flashinfer_paged", fake_flashinfer)
    monkeypatch.setattr(Engine, "triton_paged", fake_triton)
    engine = Engine.__new__(Engine)
    engine.fa_version = flash_attention_version(capability)
    metadata = object()
    query = SimpleNamespace(shape=(4, 16, head_dim))
    keys = SimpleNamespace(shape=(64, 128, 4, head_dim))
    result = engine._paged(
        query, keys, keys, "cu", max_q, "used", 32,
        SimpleNamespace(shape=(4, 1)), causal=True, softmax_scale=1.0,
        metadata=metadata)
    assert result == f"{backend}-out"
    actual, args = calls.pop()
    assert actual == backend
    if backend == "flashinfer":
        assert args is metadata
        return
    assert args["block_table"].shape == (4, 1)
    assert args["seqused_k"] == "used"
    if backend == "fa":
        assert args["fa_version"] == version


def test_attention_dispatch_preserves_each_device_path(monkeypatch):
    for capability, head_dim, max_q, backend, version in [
        ((9, 0), 128, 8, "fa", 3),
        ((10, 0), 512, 8, "flashinfer", None),
        ((10, 0), 256, 32, "fa", 4),
        ((10, 0), 256, 16, "triton", None),
        ((10, 0), 256, 1, "triton", None),
    ]:
        named(
            f"capability={capability} head_dim={head_dim} max_q={max_q} "
            f"backend={backend} version={version}",
            check_attention_dispatch_preserves_each_device_path,
            monkeypatch=monkeypatch, capability=capability, head_dim=head_dim,
            max_q=max_q, backend=backend, version=version)


def test_the_gated_delta_rule_passes_bounds_flashinfer_takes_on_each_gpu(
        monkeypatch):
    import torch

    from quail.backends.quail.executor.gdn import _flashinfer_delta_rule

    seen = []
    prefill = ModuleType("flashinfer.gdn_prefill")
    prefill.chunk_gated_delta_rule = (
        lambda **kwargs: seen.append(kwargs["cu_seqlens"].dtype) or "out")
    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", prefill)
    rows = torch.zeros(4, 1, 2)
    bounds = torch.tensor([0, 4], dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for capability in ((9, 0), (10, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability",
                            lambda *_, capability=capability: capability)
        _flashinfer_delta_rule(rows, rows, rows, rows[..., 0], rows[..., 0],
                               None, bounds, False)
    # H100's context-parallel kernel takes int64, B200's int32
    assert seen == [torch.int64, torch.int32]
