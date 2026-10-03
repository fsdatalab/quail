"""Attention version and FP8 GEMM dispatch without a GPU."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from quail.backends.quail.executor.attention import (
    GROUP,
    Engine,
    deep_gemm_supported,
    flash_attention_version,
)


@pytest.mark.parametrize(
    ("capability", "version"),
    [((8, 9), 2), ((9, 0), 3), ((10, 0), 4), ((12, 0), 2)],
)
def test_flash_attention_version(capability, version):
    assert flash_attention_version(capability) == version


def test_flash_attention_version_rejects_unknown():
    with pytest.raises(ValueError, match=r"\(7, 5\)"):
        flash_attention_version((7, 5))


def test_deep_gemm_supported_matches_vllm_allow_list():
    assert deep_gemm_supported((9, 0))
    assert deep_gemm_supported((10, 0))
    assert deep_gemm_supported((12, 0))
    assert not deep_gemm_supported((8, 9))


def test_attention_dispatch_covers_l40s_and_b200(monkeypatch):
    for capability, version in [((8, 9), 2), ((10, 0), 4)]:
        recorded = {}
        expected = object()

        def fake_attention(*args, **kwargs):
            recorded.update(kwargs)
            return expected

        module = ModuleType("vllm.vllm_flash_attn")
        module.flash_attn_varlen_func = fake_attention
        monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)
        pipeline = Engine.__new__(Engine)
        pipeline.fa_version = flash_attention_version(capability)
        table, lengths = object(), object()
        result = pipeline._fa(
            None, None, None, None, None, 16, 32, True,
            block_table=table, seqused_k=lengths,
        )
        assert result is expected
        assert recorded["fa_version"] == version
        assert recorded["block_table"] is table
        assert recorded["seqused_k"] is lengths


def test_fp8_gemm_uses_triton_block_mm_without_deep_gemm(monkeypatch):
    recorded = {}

    def fake_mm(a, b, as_, bs, block_size, output_dtype=None):
        recorded.update(a=a, b=b, scales=as_, weight_scale=bs,
                        block_size=block_size, output_dtype=output_dtype)
        return "triton"

    module = ModuleType(
        "vllm.model_executor.layers.quantization.utils.fp8_utils")
    module.w8a8_triton_block_scaled_mm = fake_mm
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        module)
    engine = Engine.__new__(Engine)
    engine.is_fp8 = True
    engine.use_deep_gemm = False
    engine.torch = SimpleNamespace(bfloat16="bf16")
    q_input = SimpleNamespace(shape=(4, 256))
    scales = SimpleNamespace(shape=(4, 256 // GROUP))
    linear = SimpleNamespace(weight="W", weight_scale="Ws")
    assert engine.gemm(q_input, scales, linear) == "triton"
    assert recorded["a"] is q_input
    assert recorded["b"] == "W"
    assert recorded["scales"] is scales
    assert recorded["weight_scale"] == "Ws"
    assert recorded["block_size"] == [GROUP, GROUP]
    assert recorded["output_dtype"] == "bf16"


def test_fp8_gemm_transposes_group_major_scales_for_triton(monkeypatch):
    recorded = {}

    def fake_mm(a, b, as_, bs, block_size, output_dtype=None):
        recorded["scales"] = as_
        return "triton"

    module = ModuleType(
        "vllm.model_executor.layers.quantization.utils.fp8_utils")
    module.w8a8_triton_block_scaled_mm = fake_mm
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        module)
    engine = Engine.__new__(Engine)
    engine.is_fp8 = True
    engine.use_deep_gemm = False
    engine.torch = SimpleNamespace(bfloat16="bf16")
    q_input = SimpleNamespace(shape=(4, 256))
    scales = SimpleNamespace(shape=(256 // GROUP, 4), transpose=lambda a, b: "T")
    linear = SimpleNamespace(weight="W", weight_scale="Ws")
    assert engine.gemm(q_input, scales, linear) == "triton"
    assert recorded["scales"] == "T"


def test_fp8_gemm_keeps_deep_gemm_when_supported(monkeypatch):
    recorded = {}

    def fake_gemm_nt(inp, weight, out, is_deep_gemm_e8m0_used=False):
        recorded["inp"] = inp
        recorded["weight"] = weight
        recorded["ue8m0"] = is_deep_gemm_e8m0_used
        return None

    module = ModuleType("vllm.utils.deep_gemm")
    module.fp8_gemm_nt = fake_gemm_nt
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", module)
    engine = Engine.__new__(Engine)
    engine.is_fp8 = True
    engine.use_deep_gemm = True
    engine.use_ue8m0 = True
    out = object()
    engine.torch = SimpleNamespace(
        bfloat16="bf16",
        empty=lambda *args, **kwargs: out)
    q_input = SimpleNamespace(shape=(4, 256), device="cuda")
    linear = SimpleNamespace(weight=SimpleNamespace(shape=(128, 256)),
                             weight_scale="Ws")
    assert engine.gemm(q_input, "As", linear) is out
    assert recorded["inp"] == (q_input, "As")
    assert recorded["weight"] == (linear.weight, "Ws")
    assert recorded["ue8m0"] is True
