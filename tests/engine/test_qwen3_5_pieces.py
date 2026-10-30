"""Qwen3.5's captured pieces and eager mixers match its eager forward pass."""

from functools import partial
from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.cuda_graphs import CapturedPieces, PiecewiseBuffers
from quail.backends.quail.executor.models import qwen3_5
from quail.backends.quail.executor.models.qwen3_5 import Qwen35Pipeline

torch = pytest.importorskip("torch")

HIDDEN, HEADS, VALUE_DIM, KEY_WIDTH = 8, 2, 4, 4
Q_HEADS, KV_HEADS, HEAD_DIM = 2, 1, 4
VALUE_WIDTH = HEADS * VALUE_DIM
CONV_DIM = 2 * KEY_WIDTH + VALUE_WIDTH


def linear(inputs, outputs):
    module = torch.nn.Linear(inputs, outputs, bias=False).to(torch.bfloat16)
    module.quant_method = SimpleNamespace(
        apply=lambda m, x, bias: torch.nn.functional.linear(x, m.weight, bias))
    return module


def norm():
    return SimpleNamespace(weight=torch.randn(HIDDEN) * 0.1, variance_epsilon=1e-6)


def gemma_add_rms_norm(x, residual, weight, eps):
    total = x.float() if residual is None else x.float() + residual.float()
    normed = total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps)
    new_residual = x if residual is None else total.to(x.dtype)
    return (normed * (1 + weight)).to(x.dtype), new_residual


class Engine:
    """Fake attention that reads every real row, and recorded KV writes."""

    num_kv_heads, head_dim = KV_HEADS, HEAD_DIM

    def __init__(self):
        self.torch = torch
        self.arena = SimpleNamespace(k=[], page_tokens=16, has_sliding=False)
        self.kv = {}

    gemma_add_rms_norm = staticmethod(gemma_add_rms_norm)

    def write_kv(self, k3, v3, meta, *, copy_tail=True):
        if meta.get("kv_written"):
            return
        pool = meta["unified"]
        live = pool["dst"] >= 0
        self.kv[meta["layer"]] = (k3[pool["src"][live]].clone(),
                                  v3[pool["src"][live]].clone())

    def attention_unified(self, q3, k3, v3, meta, *, output=None):
        self.write_kv(k3, v3, meta)
        rows = torch.arange(1, q3.shape[0] + 1).view(-1, 1, 1)
        mean = v3.float().cumsum(0) / rows
        out = (q3.float() * mean.repeat_interleave(Q_HEADS // KV_HEADS, 1)
               ).to(q3.dtype)
        if output is not None:
            output.copy_(out)
            out = output
        meta["layer"] += 1
        return out.reshape(q3.shape[0], -1)


def fake_gdn_core(spec, tensors, phases, arena, slot, *, output=None):
    """A recurrence over every real row: a running sum, scaled by b and a."""
    values = tensors["qkv"][:, 2 * KEY_WIDTH:].float().cumsum(0)
    gate = torch.sigmoid(tensors["b"].float() + tensors["a"].float() + slot)
    core = (values.view(-1, HEADS, VALUE_DIM) * gate.unsqueeze(-1)).to(
        tensors["qkv"].dtype)
    if output is None:
        return core
    return output.copy_(core)


def gdn_mixer():
    weight = torch.randn(CONV_DIM, 1, 4, dtype=torch.bfloat16)
    return SimpleNamespace(
        in_proj_qkvz=linear(HIDDEN, CONV_DIM + VALUE_WIDTH),
        in_proj_ba=linear(HIDDEN, 2 * HEADS),
        conv1d=SimpleNamespace(weight=weight, bias=None), A_log=None, dt_bias=None,
        norm=lambda core, z: (core.float() * torch.sigmoid(z.float())).to(core.dtype),
        out_proj=linear(VALUE_WIDTH, HIDDEN),
        conv_dim=CONV_DIM, value_dim=VALUE_WIDTH)


def attention():
    q_size, kv_size = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

    def rotary(positions, q, k):
        turn = positions.view(-1, 1).to(q.dtype) * 0.25
        return q + turn, k - turn

    return SimpleNamespace(
        qkv_proj=linear(HIDDEN, 2 * q_size + 2 * kv_size), q_size=q_size,
        kv_size=kv_size, num_heads=Q_HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, q_norm=lambda x: x * 0.5, k_norm=lambda x: x * 2,
        rotary_emb=rotary, o_proj=linear(q_size, HIDDEN))


def layer(kind):
    mlp = SimpleNamespace(gate_up_proj=linear(HIDDEN, 2 * HIDDEN),
                          down_proj=linear(HIDDEN, HIDDEN),
                          act_fn=lambda x: torch.nn.functional.silu(
                              x[:, :HIDDEN]) * x[:, HIDDEN:])
    mixer = {"linear_attn": gdn_mixer()} if kind == "gdn" else {
        "self_attn": attention()}
    return SimpleNamespace(input_layernorm=norm(), post_attention_layernorm=norm(),
                           mlp=mlp, **mixer)


def pipeline(monkeypatch):
    monkeypatch.setattr(qwen3_5, "gdn_core", fake_gdn_core)
    torch.manual_seed(0)
    model = Qwen35Pipeline.__new__(Qwen35Pipeline)
    model.spec = SimpleNamespace(gdn_value_heads=HEADS, gdn_value_dim=VALUE_DIM,
                                 experts=0, sae=None)
    model.vllm_config = None
    model.arena = None
    model.engine = Engine()
    model.layers = [layer(kind) for kind in ("gdn", "gdn", "attention", "gdn")]
    model.embed = torch.nn.Embedding(32, HIDDEN).to(torch.bfloat16)
    model.final_norm = norm()
    model.attention_layer, model.gdn_layer = model.layers[2], model.layers[0]
    model.gdn_slots = {0: 0, 1: 1, 3: 2}
    model.feed_forwards = [partial(model._mlp, each.mlp) for each in model.layers]
    return model


def chunk(num_tokens):
    rows = torch.arange(num_tokens)
    return SimpleNamespace(
        input_ids=rows + 3, positions=rows, tokens=num_tokens,
        final_indices=torch.tensor([1, num_tokens - 1]), attention_mode="unified",
        meta={"layer": 0, "gdn": [], "unified": dict(
            src=rows, dst=rows + 16, tail_src=None, tail_dst=None)})


def test_pieces_and_mixers_match_the_eager_forward_with_padding(monkeypatch):
    # the worker captures and replays under inference mode
    with torch.inference_mode():
        model = pipeline(monkeypatch)
        expected = model.forward_chunk(chunk(5))
        eager_kv = dict(model.engine.kv)
        model.engine.kv.clear()
        buffers = PiecewiseBuffers(model, 8, device="cpu")
        for tensor in buffers.rows.values():
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
        pieces = [SimpleNamespace(replay=partial(buffers.piece, model, index))
                  for index in range(len(model.layers) + 1)]
        source = chunk(5)
        actual = CapturedPieces(buffers, pieces).forward(model, source)
        assert torch.equal(actual, expected)
        assert eager_kv.keys() == model.engine.kv.keys() == {2}
        for got, want in zip(model.engine.kv[2], eager_kv[2]):
            assert torch.equal(got, want)
        assert "kv_written" not in source.meta


def test_boundary_holds_each_mixer_kinds_inputs_and_output(monkeypatch):
    model = pipeline(monkeypatch)
    rows = model.boundary(3, "cpu")
    assert {name: tuple(tensor.shape) for name, tensor in rows.items()} == {
        "qkv": (3, CONV_DIM), "z": (3, VALUE_WIDTH), "b": (3, HEADS),
        "a": (3, HEADS), "core": (3, HEADS, VALUE_DIM),
        "q": (3, Q_HEADS * HEAD_DIM), "k": (3, KV_HEADS * HEAD_DIM),
        "v": (3, KV_HEADS * HEAD_DIM), "gate": (3, Q_HEADS * HEAD_DIM),
        "attention": (3, Q_HEADS * HEAD_DIM)}
    # a pass runs one mixer at a time, so the two kinds share storage
    for gdn, attention in (("qkv", "q"), ("z", "gate"), ("core", "attention")):
        assert rows[gdn].data_ptr() == rows[attention].data_ptr()
    assert rows["k"].data_ptr() == rows["q"][-1, -1].data_ptr() + 2
