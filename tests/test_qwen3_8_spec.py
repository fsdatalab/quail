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
    assert not tree_attention_allowed(SPEC)


def test_state_per_sequence_and_the_pool_split():
    assert SPEC.linear_layer_set == frozenset(
        i for i in range(64) if (i + 1) % 4)
    assert SPEC.state_shape == (48, 128, 128)
    assert SPEC.conv_shape == (10_240, 3)
    # 48 layers of a 3,145,728-byte state and a 61,440-byte window
    assert SPEC.state_bytes == 48 * (48 * 128 * 128 * 4 + 10_240 * 3 * 2)
    assert QWEN3_32B_FP8.state_bytes == 0 and QWEN3_32B_FP8.state_shape is None
    chunk = budgets.chunk_budget(SPEC, H100_SXM)
    slots = budgets.state_slots(SPEC, H100_SXM, chunk)
    assert budgets.state_slots(QWEN3_32B_FP8, H100_SXM, chunk) == 0
    # the pool and the pages hold the same documents of the mean
    # length, and together they take the free memory to within a slot
    documents = (slots - 1) // budgets.STATE_SLOTS_PER_DOCUMENT
    assert documents >= 64
    tokens = budgets.arena_tokens(SPEC, H100_SXM, chunk)
    assert abs(tokens - documents * budgets.STATE_MEAN_DOC_TOKENS) \
        < SPEC.state_bytes / SPEC.kappa + budgets.PAGE_TOKENS
    free = budgets.arena_bytes(SPEC, H100_SXM, chunk)
    assert free - (tokens * SPEC.kappa + slots * SPEC.state_bytes) \
        < SPEC.state_bytes + SPEC.kappa * budgets.PAGE_TOKENS
    # a longer mean document shifts memory from slots to pages
    longer = budgets.state_slots(SPEC, H100_SXM, chunk, mean_doc_tokens=4096)
    assert longer < slots
    assert budgets.arena_tokens(SPEC, H100_SXM, chunk, mean_doc_tokens=4096) \
        > tokens


def test_both_backends_accept_the_model():
    assert MODELS["qwen3.8-27b-fp8"] is SPEC
    assert QuailBackend().supports(SPEC, H100_SXM, 1).supported
    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine,
                             filter_submission="operator")
    assert backend.supports(SPEC, H100_SXM, 1).supported
