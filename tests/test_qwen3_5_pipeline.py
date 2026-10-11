"""The Qwen3.5-family pipeline over fakes: layout, dispatch, gating, segments."""

import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.models.qwen3_5 import Qwen35Pipeline
from quail.specs import QWEN3_8_27B_FP8

HIDDEN = 4
HEADS, KV_HEADS, DIM = 2, 1, 2          # the full-attention layers
K_HEADS, V_HEADS, HEAD = 1, 2, 2        # the linear-attention layers
KEY_DIM, VALUE_DIM = K_HEADS * HEAD, V_HEADS * HEAD
CONV_DIM, KERNEL = 2 * KEY_DIM + VALUE_DIM, 4
LAYERS = 8


class _Norm:
    variance_epsilon = 1e-6

    def __init__(self, torch, weight=1.0):
        self.weight = torch.tensor(weight)

    def __call__(self, x, residual=None):
        if residual is None:
            return x * self.weight
        residual = residual + x
        return residual * self.weight, residual


class _Linear:
    def __init__(self, weight):
        self.weight = weight
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return x @ self.weight.T, None


def _attention(torch, gate_rows):
    """A gated attention module whose q is 1 and whose gate is `gate_rows`."""
    rows = []
    for head in range(HEADS):
        rows += [[0.25] * HIDDEN] * DIM                    # q of the head
        rows += [[gate_rows[head] / HIDDEN] * HIDDEN] * DIM   # its gate
    rows += [[0.25] * HIDDEN] * (2 * KV_HEADS * DIM)         # k and v
    return SimpleNamespace(
        num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=DIM,
        q_size=HEADS * DIM, kv_size=KV_HEADS * DIM, attn_output_gate=True,
        qkv_proj=_Linear(torch.tensor(rows)),
        o_proj=_Linear(torch.eye(HIDDEN, HEADS * DIM)),
        q_norm=_Norm(torch), k_norm=_Norm(torch),
        rotary_emb=lambda positions, q, k: (q, k))


def _linear_attention(torch):
    weight = torch.zeros(2 * KEY_DIM + 2 * VALUE_DIM, HIDDEN)
    # v comes out as the row's first hidden value, repeated
    weight[2 * KEY_DIM:2 * KEY_DIM + VALUE_DIM, 0] = 1.0
    return SimpleNamespace(
        key_dim=KEY_DIM, value_dim=VALUE_DIM, num_k_heads=K_HEADS,
        num_v_heads=V_HEADS, head_k_dim=HEAD, head_v_dim=HEAD,
        activation="silu",
        in_proj_qkvz=_Linear(weight),
        in_proj_ba=_Linear(torch.zeros(2 * V_HEADS, HIDDEN)),
        conv1d=SimpleNamespace(weight=torch.ones(CONV_DIM, 1, KERNEL), bias=None),
        A_log=torch.zeros(V_HEADS), dt_bias=torch.zeros(V_HEADS),
        norm=lambda x, z: x,
        out_proj=_Linear(torch.eye(HIDDEN, VALUE_DIM)))


def _layer(torch, index, gate_rows=(0.0, 0.0)):
    full = (index + 1) % 4 == 0
    return SimpleNamespace(
        layer_type="full_attention" if full else "linear_attention",
        layer_scale=False,
        self_attn=_attention(torch, gate_rows) if full else None,
        linear_attn=None if full else _linear_attention(torch),
        input_layernorm=_Norm(torch), post_attention_layernorm=_Norm(torch),
        mlp=lambda x: x)


def _model(torch, layers=None):
    layers = layers or [_layer(torch, i) for i in range(LAYERS)]
    backbone = SimpleNamespace(
        layers=layers,
        embed_tokens=lambda ids: torch.ones(ids.shape[0], HIDDEN),
        norm=_Norm(torch, 0.5))
    return SimpleNamespace(language_model=SimpleNamespace(model=backbone))


class _Engine:
    def __init__(self, arena, **kwargs):
        self.torch = sys.modules["torch"]
        self.calls = []
        self.geometry = (kwargs["n_q"], kwargs["n_kv"], kwargs["head_dim"],
                         kwargs["fp8"])

    def attention_unified(self, q3, k3, v3, meta, *, softmax_scale=None,
                          window=None):
        self.calls.append((meta["layer"], tuple(q3.shape), tuple(k3.shape),
                           tuple(v3.shape)))
        self.q = q3.reshape(q3.shape[0], -1)
        meta["layer"] += 1
        return self.q


def _kernel_stubs(monkeypatch, torch):
    """Stand in for vLLM's conv and delta-rule kernels; record their calls."""
    seen = {"conv": [], "delta": []}
    conv = types.ModuleType("vllm.model_executor.layers.mamba.ops.causal_conv1d")

    def causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc,
                         cache_indices, has_initial_state, activation):
        seen["conv"].append(dict(
            x=tuple(x.shape), weight=tuple(weight.shape), bias=bias,
            states=tuple(conv_states.shape), cu=query_start_loc.tolist(),
            slots=cache_indices.tolist(), initial=has_initial_state.tolist(),
            activation=activation))
        return x

    conv.causal_conv1d_fn = causal_conv1d_fn
    ops = types.ModuleType("vllm.third_party.flash_linear_attention.ops")

    def fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, num_k_heads,  # noqa: N803
                             head_k_dim, head_v_dim, apply_l2norm,
                             output_g_exp):
        rows = conv_output.shape[0]
        q, k, v = conv_output.split(
            [num_k_heads * head_k_dim, num_k_heads * head_k_dim,
             conv_output.shape[1] - 2 * num_k_heads * head_k_dim], dim=-1)
        return (q.reshape(rows, num_k_heads, head_k_dim),
                k.reshape(rows, num_k_heads, head_k_dim),
                v.reshape(rows, -1, head_v_dim), a, b)

    def chunk_gated_delta_rule(q, k, v, g, beta, initial_state,
                               output_final_state, cu_seqlens,
                               use_qk_l2norm_in_kernel):
        seen["delta"].append(dict(
            q=tuple(q.shape), v=tuple(v.shape), initial=initial_state,
            final=output_final_state, cu=cu_seqlens.tolist(),
            l2norm=use_qk_l2norm_in_kernel))
        return v, None

    ops.fused_post_conv_prep = fused_post_conv_prep
    ops.chunk_gated_delta_rule = chunk_gated_delta_rule
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.mamba",
                 "vllm.model_executor.layers.mamba.ops", "vllm.third_party",
                 "vllm.third_party.flash_linear_attention"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.ops.causal_conv1d", conv)
    monkeypatch.setitem(
        sys.modules, "vllm.third_party.flash_linear_attention.ops", ops)
    return seen


def _spec():
    return replace(QWEN3_8_27B_FP8, layers=LAYERS, hidden=HIDDEN, n_q=HEADS,
                   n_kv=KV_HEADS, d_head=DIM, ffn_width=8)


def _chunk(torch, rows=5):
    return SimpleNamespace(
        input_ids=torch.arange(rows), positions=torch.arange(rows),
        final_indices=torch.tensor([2, rows - 1]),
        meta={"layer": 0, "unified": {"cu_q": torch.tensor([0, 3, rows])}})


def test_layers_dispatch_by_type_and_segments_start_from_zero(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = _kernel_stubs(monkeypatch, torch)
    pipeline = Qwen35Pipeline(_model(torch), arena=None, spec=_spec(),
                               engine_class=_Engine)
    assert pipeline.engine.geometry == (HEADS, KV_HEADS, DIM, False)
    assert (pipeline.needs_pages, pipeline.tree_attention,
            pipeline.gemm_warmup) == (True, False, False)
    chunk = _chunk(torch)
    out = pipeline.forward_chunk(chunk)
    # the full layers address the arena by their own index
    assert [call[0] for call in pipeline.engine.calls] == [3, 7]
    assert pipeline.engine.calls[0][1:] == (
        (5, HEADS, DIM), (5, KV_HEADS, DIM), (5, KV_HEADS, DIM))
    # six linear layers, each over the chunk's two sequences
    assert len(seen["conv"]) == len(seen["delta"]) == 6
    conv = seen["conv"][0]
    assert conv["x"] == (CONV_DIM, 5) and conv["weight"] == (CONV_DIM, KERNEL)
    assert conv["cu"] == [0, 3, 5] and conv["slots"] == [1, 2]
    assert conv["initial"] == [False, False]
    # slot 0 is the kernels' null slot; the windows start at slot 1
    assert conv["states"] == (3, CONV_DIM, KERNEL - 1)
    assert conv["activation"] == "silu"
    delta = seen["delta"][0]
    assert delta["q"] == (1, 5, K_HEADS, HEAD) and delta["v"] == (1, 5, V_HEADS, HEAD)
    assert delta["initial"] is None and delta["final"] is False
    assert delta["cu"] == [0, 3, 5] and delta["l2norm"] is False
    assert out.shape == (2, HIDDEN)


def test_gate_scales_each_head_of_the_attention_output(monkeypatch):
    torch = pytest.importorskip("torch")
    _kernel_stubs(monkeypatch, torch)
    # head 0's gate is far negative (sigmoid 0); head 1's is 0 (sigmoid 0.5)
    layers = [_layer(torch, i, gate_rows=(-100.0, 0.0)) for i in range(4)]
    pipeline = Qwen35Pipeline(_model(torch, layers), arena=None,
                               spec=replace(_spec(), layers=4),
                               engine_class=_Engine)
    pipeline.forward_chunk(_chunk(torch, rows=3))
    attn = layers[3].self_attn
    gated = attn.o_proj.inputs[0]
    assert gated.shape == (3, HEADS * DIM)
    q = pipeline.engine.q
    assert torch.all(q > 0)
    assert torch.allclose(gated, q * torch.tensor([0.0, 0.0, 0.5, 0.5]))


def test_layer_output_is_feedforward_plus_residual(monkeypatch):
    torch = pytest.importorskip("torch")
    _kernel_stubs(monkeypatch, torch)
    layers = [_layer(torch, i) for i in range(4)]
    pipeline = Qwen35Pipeline(_model(torch, layers), arena=None,
                              spec=replace(_spec(), layers=4),
                              engine_class=_Engine)
    chunk = _chunk(torch, rows=4)
    hidden, residual = pipeline.backbone_rows(chunk)
    # Each linear layer's v is the row's first hidden value, so it
    # returns its input, and the residual doubles: 2, 8, 32. The full
    # layer's q is its input and the gate halves it: 64 + 32 = 96.
    assert torch.equal(residual, torch.full((4, HIDDEN), 96.0))
    assert torch.equal(hidden, torch.full((4, HIDDEN), 96.0))
    # the final norm, weight 0.5, over the two final rows' sums
    normed = pipeline.forward_chunk(chunk)
    assert torch.equal(normed, torch.full((2, HIDDEN), 96.0))


def test_geometry_mismatches_raise(monkeypatch):
    torch = pytest.importorskip("torch")
    _kernel_stubs(monkeypatch, torch)
    spec = _spec()
    with pytest.raises(ValueError, match="full-attention layers"):
        Qwen35Pipeline(_model(torch), arena=None,
                        spec=replace(spec, full_attention_period=2),
                        engine_class=_Engine)
    with pytest.raises(ValueError, match="attention geometry"):
        Qwen35Pipeline(_model(torch), arena=None, spec=replace(spec, n_q=4),
                        engine_class=_Engine)
    layers = [_layer(torch, i) for i in range(LAYERS)]
    layers[3].self_attn.attn_output_gate = False
    with pytest.raises(ValueError, match="gated attention"):
        Qwen35Pipeline(_model(torch, layers), arena=None, spec=spec,
                        engine_class=_Engine)
    layers = [_layer(torch, i) for i in range(LAYERS)]
    layers[0].layer_scale = True
    with pytest.raises(ValueError, match="layer scales"):
        Qwen35Pipeline(_model(torch, layers), arena=None, spec=spec,
                        engine_class=_Engine)
