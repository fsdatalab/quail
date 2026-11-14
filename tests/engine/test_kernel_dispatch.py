"""Attention version and FP8 GEMM dispatch without a GPU."""

import sys
from types import ModuleType, SimpleNamespace

from test_kv_reads import named

from quail.backends.quail.executor.attention import (
    Engine,
    flash_attention_version,
)


def check_attention_dispatch_preserves_each_device_path(
        monkeypatch, capability, head_dim, backend, version):
    calls = []

    def fake_attention(*args, **kwargs):
        calls.append(("fa", kwargs))
        return "fa-out", None

    def fake_flashinfer(self, query, keys, values, metadata, **kwargs):
        calls.append(("flashinfer", metadata))
        return "flashinfer-out"

    flash = ModuleType("vllm.vllm_flash_attn")
    flash.flash_attn_varlen_func = fake_attention
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", flash)
    monkeypatch.setattr(Engine, "_flashinfer_paged", fake_flashinfer)
    engine = Engine.__new__(Engine)
    engine.fa_version = flash_attention_version(capability)
    metadata = object()
    query = SimpleNamespace(shape=(4, 16, head_dim))
    result = engine._paged(
        query, "k", "v", "cu", 8, "used", 32, "table",
        causal=True, softmax_scale=1.0, metadata=metadata)
    assert result == f"{backend}-out"
    actual, args = calls.pop()
    assert actual == backend
    if backend == "flashinfer":
        assert args is metadata
    else:
        assert args["fa_version"] == version
        assert args["block_table"] == "table"
        assert args["seqused_k"] == "used"


def test_attention_dispatch_preserves_each_device_path(monkeypatch):
    for capability, head_dim, backend, version in [
        ((9, 0), 128, "fa", 3),
        ((10, 0), 512, "flashinfer", None),
    ]:
        named(
            f"capability={capability} head_dim={head_dim} "
            f"backend={backend} version={version}",
            check_attention_dispatch_preserves_each_device_path,
            monkeypatch=monkeypatch, capability=capability, head_dim=head_dim,
            backend=backend, version=version)
