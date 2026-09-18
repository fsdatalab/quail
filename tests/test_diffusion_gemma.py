"""DiffusionGemma: spec geometry, chat-turn text, canvas rows, and the layer loop."""

import contextlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import cpu_arena, fake_torch

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor import loop
from quail.backends.quail.executor.models import supported_archs
from quail.backends.quail.executor.models.diffusion_gemma import (
    DiffusionGemmaPipeline,
    canvas_token_ids,
)
from quail.backends.quail.executor.pack import FilterAdmission, JoinAdmission
from quail.cost import budgets
from quail.cost.dense_decoder_cost import mlp_params, mlp_weight_params
from quail.logical.prompts import (
    ANSWER_CUE,
    SHARED_PRE,
    bind_join_prompt,
    bind_prompt,
    render_filter_prompt_ids,
    render_join_prompt_ids,
)
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

SPEC = DIFFUSION_GEMMA_26B_FP8


def _tok(text):
    return [ord(c) for c in text]


# ------------------------------------------------------------- spec


def test_spec_geometry_and_registration():
    shapes = SPEC.kv_shapes
    assert len(shapes) == 30
    assert [i for i, s in enumerate(shapes) if s == (2, 512)] == [5, 11, 17, 23, 29]
    assert all(s == (8, 256) for i, s in enumerate(shapes) if (i + 1) % 6)
    # 25 sliding layers at 8 x 256 and 5 full layers at 2 x 512, K and V
    assert SPEC.kv_elements_per_token == 2 * (25 * 2048 + 5 * 1024)
    assert SPEC.kappa == 225_280.0
    assert QWEN3_4B_FP8.kv_shapes == ((8, 128),) * 36
    assert QWEN3_4B_FP8.kappa == 147_456.0
    assert SPEC.arch in supported_archs()
    assert QuailBackend().supports(SPEC, H100_SXM, 1).supported
    assert SPEC.canvas_tokens == 256
    assert SPEC.turn == (SPEC.turn_prefix, SPEC.turn_suffix)


def test_spec_budgets_and_moe_costs():
    assert budgets.chunk_budget(SPEC, H100_SXM) == SPEC.chunk_cap_tokens
    assert budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM) == \
        budgets.kernel_index_cap(QWEN3_4B_FP8)
    assert budgets.arena_tokens(SPEC, H100_SXM) > 100_000
    # a chunk reads every expert but a token multiplies eight of them
    assert mlp_weight_params(SPEC) > 10 * mlp_params(SPEC)
    assert mlp_weight_params(QWEN3_4B_FP8) == mlp_params(QWEN3_4B_FP8)
    with pytest.raises(ValueError, match="layer_kv"):
        _ = SPEC.__class__(**{**SPEC.__dict__, "layer_kv": ((8, 256),)}).kv_shapes


# ------------------------------------------------------ chat turns


def test_turn_text_wraps_filter_and_join_prompts():
    turn = ("<bos><|turn>user\n", "<turn|>\n<|turn>model\n")
    ref = SimpleNamespace(alias="d")
    prompt = bind_prompt("Is {0} about food?", (ref,), _tok, turn=turn)
    assert prompt.preamble == turn[0] + SHARED_PRE
    assert prompt.tail.endswith(ANSWER_CUE + turn[1])
    doc = [7, 8, 9]
    ids = render_filter_prompt_ids(prompt, doc, _tok)
    assert ids[:len(_tok(prompt.preamble))] == _tok(prompt.preamble)
    assert ids[-len(_tok(turn[1])):] == _tok(turn[1])
    assert ids == (_tok(prompt.preamble) + doc
                   + _tok(prompt.tail.replace("{0}", "", 1)))
    assert tuple(_tok(prompt.preamble)) == prompt.preamble_token_ids

    left, right = SimpleNamespace(alias="a"), SimpleNamespace(alias="b")
    join = bind_join_prompt("Same topic in {0} and {1}?", (left, right), _tok,
                            turn=turn)
    assert join.preamble == turn[0] + SHARED_PRE
    assert join.tail == ANSWER_CUE + turn[1]
    assert join.preamble_token_ids == tuple(_tok(join.preamble))
    assert join.tail_token_ids == tuple(_tok(join.tail))
    ids = render_join_prompt_ids(join, [[1, 2], [3]], 0, _tok)
    assert ids[-len(_tok(join.tail)):] == _tok(join.tail)

    plain = bind_prompt("Is {0} about food?", (ref,), _tok)
    assert plain.preamble == SHARED_PRE
    assert plain.tail.endswith(ANSWER_CUE)


# ---------------------------------------------------- canvas rows


def test_canvas_ids_are_fixed_and_in_vocabulary():
    ids = canvas_token_ids(1000, 16)
    assert ids == canvas_token_ids(1000, 16)
    assert len(ids) == 16 and all(0 <= i < 1000 for i in ids)
    assert canvas_token_ids(1000, 0) == ()


def test_join_admission_charges_canvas_rows():
    sched = JoinAdmission([10], [[5, 5]], 100, arena_pages=100, page_tokens=16,
                          frame_tokens=[3], canvas_tokens=4)
    assert sched.stages == [[9, 9]]
    assert sched.frames == [3]        # kept in the anchor's KV
    assert sched.frame_rows == [7]    # packed: frame plus its canvas
    assert sched._first_cost(0, 0, 0) == 16
    assert sched._extra == 7
    plain = JoinAdmission([10], [[5, 5]], 100, arena_pages=100, page_tokens=16,
                          frame_tokens=[3])
    assert plain.frame_rows == [3] and plain.stages == [[5, 5]]
    empty = JoinAdmission([10], [[5]], 100, arena_pages=100, page_tokens=16,
                          canvas_tokens=4)
    assert empty.frame_rows == [0]


def test_filter_stream_charges_canvas_rows():
    pipeline = SimpleNamespace(is_fp8=False, canvas_ids=(1, 2, 3),
                               forward_chunk=None)
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    docs = [[5] * 20, [6] * 30]
    questions = [[40, 41, 42], [40, 41, 43, 44]]
    stream = loop.FilterStream(
        fake_torch(), cpu_arena(64), pipeline, answers, docs, questions, 200,
        arena_writes=True, arena_keys=[("d", 0), ("d", 1)])
    assert stream.preamble == 2
    assert stream.sched.stage_tokens == [3 + 3, 2 + 3]
    assert stream.capacity_extra == 2 + 2 + 3
    assert stream.canvas == (1, 2, 3)

    paged = SimpleNamespace(is_fp8=False, canvas_ids=(), needs_pages=True,
                            forward_chunk=None)
    stream = loop.FilterStream(
        fake_torch(), cpu_arena(64), paged, answers, docs[:1], questions[:1],
        200, arena_writes=False, arena_keys=[("d", 0)])
    assert stream.arena_writes
    assert stream.sched.free_pages is not None


def _cpu_staging(monkeypatch):
    import torch

    def staged(torch_, data, dtype, pinned=True):
        if isinstance(data, np.ndarray) or torch.is_tensor(data):
            return torch.as_tensor(data, dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def token_parts(torch_, sequences, total, pinned=True, staging=None):
        ids = [int(t) for seq in sequences for part in loop._token_parts(seq)
               for t in part]
        assert len(ids) == total
        return torch.tensor(ids, dtype=torch.int64)

    monkeypatch.setattr(loop, "_staged", staged)
    monkeypatch.setattr(loop, "_staged_token_parts", token_parts)
    return torch


def test_pack_chunk_appends_canvas_rows_after_each_suffix(monkeypatch):
    torch = _cpu_staging(monkeypatch)
    arena = cpu_arena(64)
    canvas = (90, 91, 92, 93)
    groups = [dict(key=("d", 0), prefix=[1, 2, 3], f=3, suffixes=[[10, 11]]),
              dict(key=("d", 1), prefix=[4, 5], f=2, suffixes=[[12]])]
    chunk = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas)
    assert chunk.tokens == 3 + 2 + 4 + 2 + 1 + 4
    assert chunk.input_ids.tolist() == [1, 2, 3, 10, 11, *canvas, 4, 5, 12, *canvas]
    assert chunk.positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                        0, 1, 2, 3, 4, 5, 6]
    # the answer row is the first canvas row of each group
    assert chunk.final_indices.tolist() == [5, 12]
    meta = chunk.meta
    assert meta["cu_a"].tolist() == [0, 9, 16]
    assert meta["unified"] is None
    assert meta["canvas"]["rows"].tolist() == [5, 6, 7, 8, 12, 13, 14, 15]
    assert meta["canvas"]["cu_q"].tolist() == [0, 4, 8]
    assert meta["canvas"]["max_q"] == 4

    plain = loop.pack_chunk(torch, arena, groups, attention_mode="unified")
    assert plain.meta["canvas"] is None
    assert plain.final_indices.tolist() == [4, 7]
    with pytest.raises(ValueError, match="unified"):
        loop.pack_chunk(torch, arena, groups, attention_mode="merge_quant",
                        canvas=canvas)


# -------------------------------------------------- the layer loop


class _Norm:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


class _Linear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return x @ self.weight.T, None


class _Attention:
    def __init__(self, torch, hidden, heads, kv_heads, dim, sliding):
        self.num_heads, self.num_kv_heads, self.head_dim = heads, kv_heads, dim
        self.q_size, self.kv_size = heads * dim, kv_heads * dim
        width = self.q_size + 2 * self.kv_size
        self.qkv_proj = _Linear(torch.ones(width, hidden))
        self.o_proj = _Linear(torch.ones(hidden, self.q_size))
        self.q_norm = _Norm(1.0)
        self.k_norm = _Norm(1.0)
        self.v_norm = _Norm(1.0)
        self.rotary_emb = lambda positions, q, k: (q, k)
        self.is_sliding = sliding


class _Layer:
    def __init__(self, torch, hidden, sliding, moe):
        heads, kv, dim = (2, 1, 4) if sliding else (2, 1, 8)
        self.self_attn = _Attention(torch, hidden, heads, kv, dim, sliding)
        self.input_layernorm = _Norm(1.0)
        self.post_attention_layernorm = _Norm(0.0)
        self.pre_feedforward_layernorm = _Norm(1.0)
        self.post_feedforward_layernorm = _Norm(1.0)
        self.mlp = lambda x: x
        self.enable_moe_block = moe
        self.post_feedforward_layernorm_1 = _Norm(1.0)
        self.pre_feedforward_layernorm_2 = _Norm(1.0)
        self.post_feedforward_layernorm_2 = _Norm(1.0)
        self.router = lambda x: x
        self.moe = lambda x, logits: x
        self.layer_scalar = torch.tensor([0.5])


class _Engine:
    """Records each attention call; returns zeros at the layer's q width."""

    def __init__(self, arena, **kwargs):
        self.calls = []
        self.torch = sys.modules["torch"]
        self.is_fp8 = kwargs["fp8"]

    def attention_unified(self, q3, k3, v3, meta, *, softmax_scale=None,
                          window=None):
        self.calls.append((meta["layer"], q3.shape, k3.shape, softmax_scale,
                           window))
        meta["layer"] += 1
        return self.torch.zeros(q3.shape[0], q3.shape[1] * q3.shape[2])


def _fake_model(torch, hidden=4):
    layers = [_Layer(torch, hidden, sliding=True, moe=True),
              _Layer(torch, hidden, sliding=False, moe=False)]
    backbone = SimpleNamespace(
        layers=layers,
        embed_tokens=lambda ids: torch.ones(ids.shape[0], hidden),
        normalizer=torch.tensor(2.0),
        norm=_Norm(1.0),
        config=SimpleNamespace(sliding_window=1024))
    return SimpleNamespace(
        model=backbone,
        self_conditioning=SimpleNamespace(post_norm=_Norm(0.25)),
        quail_vllm_config="config")


def test_pipeline_runs_the_gemma4_layer_order(monkeypatch):
    torch = pytest.importorskip("torch")
    context = types.ModuleType("vllm.forward_context")
    seen = {}

    @contextlib.contextmanager
    def set_forward_context(attn_metadata, vllm_config, num_tokens=None):
        seen["context"] = (vllm_config, num_tokens)
        yield

    context.set_forward_context = set_forward_context
    workspace = types.ModuleType("vllm.v1.worker.workspace")
    workspace.is_workspace_manager_initialized = lambda: False
    workspace.init_workspace_manager = lambda device: seen.setdefault(
        "workspace", str(device))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker",
                        types.ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace)

    spec = SimpleNamespace(vocab=50, canvas_tokens=3)
    pipeline = DiffusionGemmaPipeline(_fake_model(torch), None, spec=spec,
                                      engine_class=_Engine)
    assert len(pipeline.canvas_ids) == 3
    assert seen["workspace"] == "cuda"
    assert not pipeline.is_fp8
    assert pipeline.max_chunk_tokens == (2**31 - 1) // 32

    rows = torch.tensor([3, 4, 5])
    chunk = SimpleNamespace(
        input_ids=torch.zeros(6, dtype=torch.int64),
        positions=torch.arange(6),
        final_indices=torch.tensor([0, 3]),
        meta={"layer": 0, "canvas": {"rows": rows}})
    out = pipeline.forward_chunk(chunk)
    assert seen["context"] == ("config", 6)
    calls = pipeline.engine.calls
    # sliding layer: 2 x 4 heads with the window; full layer: 2 x 8, no window
    assert calls == [(0, (6, 2, 4), (6, 1, 4), 1.0, 1024),
                     (1, (6, 2, 8), (6, 1, 8), 1.0, None)]
    # A prompt row embeds to 1 x 2 = 2; a canvas row also passes the
    # post-norm (x 0.25) and starts at 0.5. Attention adds 0 (its
    # post-norm is x 0). The MoE layer adds the residual twice (dense
    # MLP and experts, both identity) and halves: 2 -> 3, 0.5 -> 0.75.
    # The dense layer adds it once and halves: 3 -> 3, 0.75 -> 0.75.
    assert out.shape == (2, 4)
    assert out.tolist() == [[3.0] * 4, [0.75] * 4]


def test_filter_admission_takes_canvas_in_stage_tokens():
    sched = FilterAdmission([10, 10], [3 + 4, 2 + 4], 40,
                            arena_pages=100, page_tokens=16,
                            kept_extra_tokens=2 + 2 + 4)
    first = sched.next_chunk()
    assert first == [(0, 0, True), (1, 0, True)]
