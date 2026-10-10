"""The Qwen3.8 27B spec: a hybrid model the stock vLLM backend runs."""

from types import SimpleNamespace

from quail.backends.quail.backend import QuailBackend
from quail.backends.request import RequestBackend
from quail.cost import budgets
from quail.cost.budgets import tree_attention_allowed
from quail.cost.dense_decoder_cost import dense_params
from quail.specs import H100_SXM, MODELS, QWEN3_8_27B_FP8, QWEN3_32B_FP8

SPEC = QWEN3_8_27B_FP8


def test_only_the_full_attention_layers_keep_kv():
    shapes = SPEC.kv_shapes
    assert len(shapes) == 64
    full = [i for i, s in enumerate(shapes) if s == (4, 256)]
    assert full == list(range(3, 64, 4))
    assert all(s == (0, 0) for i, s in enumerate(shapes) if i not in full)
    assert [i for i in range(64) if SPEC.is_linear_layer(i)] == \
        [i for i in range(64) if i not in full]
    assert SPEC.sliding_layer_set == frozenset()
    # 16 layers x 2 x 4 heads x 256 x bf16
    assert SPEC.kappa == 65_536.0
    assert SPEC.kappa_full == SPEC.kappa
    assert SPEC.widest_projection == 34_816


def test_params_follow_the_checkpoint_dimensions():
    hidden, intermediate = 5120, 17408
    full = (hidden * (2 * 24 + 2 * 4) * 256 + 24 * 256 * hidden)
    mlp = 3 * hidden * intermediate
    expected = 48 * (SPEC.linear_attention_params + mlp) + 16 * (full + mlp)
    assert SPEC.params == 24.35e9
    assert abs(SPEC.params - expected) / expected < 0.01
    # the cost model leaves out the full layers' output gate
    assert abs(dense_params(SPEC) - expected) / expected < 0.03
    assert SPEC.head_mem_bytes == 248_320 * 5120 * 2
    assert SPEC.W_mem == 29.4e9
    assert budgets.minimum_weight_gpus(SPEC, H100_SXM) == 1
    # a quarter of Qwen3 32B's KV per token, so more than three times
    # the resident document tokens on one H100
    assert budgets.arena_tokens(SPEC, H100_SXM) > \
        3 * budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM)
    assert not tree_attention_allowed(SPEC)


def test_quail_refuses_and_stock_vllm_accepts():
    assert MODELS["qwen3.8-27b-fp8"] is SPEC
    support = QuailBackend().supports(SPEC, H100_SXM, 1)
    assert not support.supported
    assert "qwen3_5" in support.reason
    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine,
                             filter_submission="operator")
    assert backend.supports(SPEC, H100_SXM, 1).supported
