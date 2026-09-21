"""The Qwen3 pipeline picks its attention paths from the weight dtype."""

from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline


def _model(torch, dtype):
    weight = SimpleNamespace(dtype=dtype, shape=(4096, 1024))
    attn = SimpleNamespace(num_heads=16, num_kv_heads=8, head_dim=128,
                           rotary_emb=None, qkv_proj=SimpleNamespace(weight=weight))
    mlp = SimpleNamespace(gate_up_proj=SimpleNamespace(weight=weight))
    layer = SimpleNamespace(self_attn=attn, mlp=mlp)
    return SimpleNamespace(model=SimpleNamespace(
        layers=[layer], embed_tokens=None, norm=None))


@pytest.mark.parametrize("dtype_name, expected", [
    ("float8_e4m3fn", "merge_quant"), ("bfloat16", "unified")])
def test_join_attention_follows_the_weight_precision(dtype_name, expected):
    torch = pytest.importorskip("torch")
    engines = []

    def engine(arena, **kwargs):
        engines.append(kwargs)
        return SimpleNamespace(**kwargs)

    pipeline = Qwen3Pipeline(_model(torch, getattr(torch, dtype_name)), None,
                             spec=None, engine_class=engine)
    assert engines[0]["fp8"] is (expected == "merge_quant")
    assert pipeline.join_attention == expected
    assert pipeline.max_chunk_tokens == (2**31 - 1) // 4096
