"""DiffusionGemma: spec geometry, chat-turn text, canvas rows, and the layer loop."""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
from fakes import cpu_arena, cpu_staging, fake_pipeline, fake_torch

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
    # the widest GEMM is the full layers' qkv projection
    assert SPEC.widest_projection == 16 * 512 + 2 * 2 * 512
    # 25 sliding layers at 8 x 256 and 5 full layers at 2 x 512, K and V
    assert SPEC.kv_elements_per_token == 2 * (25 * 2048 + 5 * 1024)
    assert SPEC.kappa == 225_280.0
    assert QWEN3_4B_FP8.kv_shapes == ((8, 128),) * 36
    assert QWEN3_4B_FP8.kappa == 147_456.0
    assert SPEC.arch in supported_archs()
    assert QuailBackend().supports(SPEC, H100_SXM, 1).supported
    assert SPEC.canvas_tokens == 1
    assert SPEC.turn == ("<bos><|turn>user\n",
                         "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    # the model opens its turn with a four-token empty thinking channel
    assert SPEC.canvas_answer_row == 0


def test_spec_budgets_and_moe_costs():
    assert budgets.chunk_budget(SPEC, H100_SXM) == SPEC.chunk_cap_tokens
    assert budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM) == \
        budgets.kernel_index_cap(QWEN3_4B_FP8)
    assert budgets.arena_tokens(SPEC, H100_SXM) > 100_000
    # a chunk reads every expert but a token multiplies eight of them
    assert mlp_weight_params(SPEC) > 10 * mlp_params(SPEC)
    assert mlp_weight_params(QWEN3_4B_FP8) == mlp_params(QWEN3_4B_FP8)
    # the dense MLP and eight of the 128 experts per layer
    assert mlp_params(SPEC) == 30 * 3 * 2816 * (2112 + 8 * 704)
    assert mlp_weight_params(SPEC) == 30 * 3 * 2816 * (2112 + 128 * 704)


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
    pipeline = fake_pipeline(canvas_ids=(1, 2, 3))
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

    paged = fake_pipeline(needs_pages=True)
    stream = loop.FilterStream(
        fake_torch(), cpu_arena(64), paged, answers, docs[:1], questions[:1],
        200, arena_writes=False, arena_keys=[("d", 0)])
    assert stream.arena_writes
    assert stream.sched.free_pages is not None


class _ImageSource:
    """An image source that records the chain's calls."""

    def __init__(self):
        self.opened = 0
        self.taken = []
        self.closed = False

    def open(self):
        self.opened += 1

    def take(self, doc):
        self.taken.append(doc)
        return (("block", f"page{doc}"),)

    def metrics(self):
        return {"pages_rendered": len(self.taken)}

    def close(self):
        self.closed = True


def test_filter_stream_hands_images_to_fresh_documents_only(monkeypatch):
    from fakes import FakeModel, fake_pack, fake_torch

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    truth = [[1, 1], [1, 0], [1, 1]]
    model = FakeModel(truth, {})
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk,
                             takes_images=True)
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    docs = [[5] * 20, [6] * 30, [7] * 10]
    questions = [[2000, 41], [2001, 41]]
    source = _ImageSource()
    stream = loop.FilterStream(
        fake_torch(), cpu_arena(64), pipeline, answers, docs, questions, 200,
        arena_writes=True, arena_keys=[("d", d) for d in range(3)],
        images=source)
    assert source.opened == 1 and stream.image_metrics == {}
    loop.run_stream(stream)
    # every document was rendered once, for its fresh pass
    assert sorted(source.taken) == [0, 1, 2]
    fresh_specs = [spec for _, specs in model.launched for spec in specs
                   if spec["prefix"] is not None]
    kept_specs = [spec for _, specs in model.launched for spec in specs
                  if spec["prefix"] is None]
    assert all(spec["images"] == (("block", f"page{spec['key'][1]}"),)
               for spec in fresh_specs)
    assert kept_specs and all("images" not in spec for spec in kept_specs)
    assert source.closed and stream.image_metrics == {"pages_rendered": 3}
    assert stream.answers == {0: [1, 1], 1: [1, 0], 2: [1, 1]}

    text_pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    with pytest.raises(ValueError, match="does not embed images"):
        loop.FilterStream(
            fake_torch(), cpu_arena(64), text_pipeline, answers, docs,
            questions, 200, arena_writes=True, images=_ImageSource())
    with pytest.raises(ValueError, match="paged unified path"):
        loop.FilterStream(
            fake_torch(), cpu_arena(64), pipeline, answers, docs[:1],
            questions[:1], 200, arena_writes=False, images=_ImageSource())


def test_pack_chunk_appends_canvas_rows_after_each_suffix(monkeypatch):
    torch = cpu_staging(monkeypatch)
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

    later = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas, answer_row=2)
    assert later.final_indices.tolist() == [7, 14]
    with pytest.raises(ValueError, match="answer_row"):
        loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                        canvas=canvas, answer_row=4)

    plain = loop.pack_chunk(torch, arena, groups, attention_mode="unified")
    assert plain.meta["canvas"] is None
    assert plain.final_indices.tolist() == [4, 7]
    with pytest.raises(ValueError, match="unified"):
        loop.pack_chunk(torch, arena, groups, attention_mode="merge_quant",
                        canvas=canvas)


# -------------------------------------------------- the layer loop


class _Norm:
    """A norm module as the pipeline reads it; the fake engine applies it."""

    hidden_size = 4
    variance_epsilon = 1e-6

    def __init__(self, scale):
        self.scale = scale
        self.has_weight = True
        self.weight = scale


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
        self.v_norm.has_weight = False
        self.v_norm.hidden_size = dim
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(2, dim), is_neox_style=True)
        self.is_sliding = sliding


class _Layer:
    def __init__(self, torch, hidden, sliding, moe, layer_scalar=0.5):
        heads, kv, dim = (2, 1, 4) if sliding else (2, 1, 8)
        self.self_attn = _Attention(torch, hidden, heads, kv, dim, sliding)
        self.input_layernorm = _Norm(torch.tensor(1.0))
        self.post_attention_layernorm = _Norm(torch.tensor(0.0))
        self.pre_feedforward_layernorm = _Norm(torch.tensor(1.0))
        self.post_feedforward_layernorm = _Norm(torch.tensor(1.0))
        self.mlp = _Mlp(torch, hidden)
        self.enable_moe_block = moe
        self.post_feedforward_layernorm_1 = _Norm(torch.tensor(1.0))
        self.pre_feedforward_layernorm_2 = _Norm(torch.tensor(1.0))
        self.post_feedforward_layernorm_2 = _Norm(torch.tensor(1.0))
        self.router = SimpleNamespace(
            norm=_Norm(torch.tensor(1.0)), root_size=torch.tensor(1.0),
            scale=torch.ones(hidden), proj=lambda x: (x, None))
        self.moe = SimpleNamespace(experts=lambda x, logits: x)
        self.layer_scalar = torch.tensor([layer_scalar])


class _Mlp:
    """An identity feedforward with the linears the fused path calls."""

    def __init__(self, torch, hidden):
        self.gate_up_proj = _Linear(torch.eye(hidden))
        self.down_proj = _Linear(torch.eye(hidden))
        self.act_fn = lambda x: x

    def __call__(self, x):
        return x


class _Engine:
    """Records each attention call; returns zeros at the layer's q width."""

    def __init__(self, arena, **kwargs):
        self.calls = []
        self.torch = sys.modules["torch"]
        self.is_fp8 = kwargs["fp8"]

    def attention_unified(self, q3, k3, v3, meta, *, softmax_scale=None,
                          window=None, bidirectional_blocks=None):
        self.calls.append((meta["layer"], q3.shape, k3.shape, softmax_scale,
                           window))
        self.blocks = getattr(self, "blocks", [])
        self.blocks.append(bidirectional_blocks)
        meta["layer"] += 1
        return self.torch.zeros(q3.shape[0], q3.shape[1] * q3.shape[2])

    def norm_rows(self, x, weight, eps):
        # the fakes' norms scale by their weight and nothing else
        return x * weight

    # the fused primitives, with the same fake norm (x times its
    # weight) and no quantization
    def norm_quant_rows(self, x, weight, eps, residual=None):
        if residual is not None:
            residual.add_(x)
            x = residual
        return x * weight, None

    def fp8_linear(self, module, x_q, x_s):
        return module(x_q)[0]

    def fused_add_rms_norm(self, hidden, residual, norm):
        residual.add_(hidden)
        hidden.copy_(residual * norm.weight)
        return hidden, residual

    def qkv_norm_rope_heads(self, qkv, positions, *, n_q, n_kv, head_dim,
                            q_weight, k_weight, eps, cos_sin_cache):
        q = qkv[:, :n_q * head_dim] * q_weight
        k = qkv[:, n_q * head_dim:(n_q + n_kv) * head_dim] * k_weight
        return (q.contiguous(), k.contiguous(),
                qkv[:, (n_q + n_kv) * head_dim:].contiguous())

    def gelu_mul_quant(self, gate_up):
        # the fake feedforward's activation is the identity
        return gate_up, None

    def scale_add_norm_quant(self, x, residual, scale, weight, eps):
        residual.mul_(scale).add_(x)
        return residual * weight, None

    def norm_router_quant(self, x, weight1, weight2, eps):
        return x * weight1, None, x * weight2


def _spec_for(model, **overrides):
    """A spec fake whose geometry matches the fake model's."""
    attn = model.model.layers[0].self_attn
    fields = dict(vocab=50, canvas_tokens=3, canvas_answer_row=1,
                  sliding_window=model.model.config.sliding_window,
                  n_q=attn.num_heads, n_kv=attn.num_kv_heads,
                  head_dim=attn.head_dim, d_head=attn.head_dim,
                  widest_projection=32)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _fake_model(torch, hidden=4, layer_scalar=0.5):
    layers = [_Layer(torch, hidden, sliding=True, moe=True,
                     layer_scalar=layer_scalar),
              _Layer(torch, hidden, sliding=False, moe=False,
                     layer_scalar=layer_scalar)]
    backbone = SimpleNamespace(
        layers=layers,
        embed_tokens=lambda ids: torch.ones(ids.shape[0], hidden),
        normalizer=torch.tensor(2.0),
        norm=_Norm(1.0),
        config=SimpleNamespace(sliding_window=1024))
    post_norm = _Norm(0.25)
    post_norm.has_weight = False
    return SimpleNamespace(
        model=backbone,
        self_conditioning=SimpleNamespace(post_norm=post_norm),
        quail_vllm_config="config")


def test_pipeline_runs_the_gemma4_layer_order(monkeypatch):
    """The layer order and the residual arithmetic, on identity fakes.

    The fused kernels are stand-ins here (a norm is x times its weight,
    quantization is skipped), so this checks which kernel runs when and
    what it is handed, not the kernels; tests/gpu compares the real
    kernels against stock vLLM layer by layer.
    """
    torch = pytest.importorskip("torch")
    seen = _vllm_stubs(monkeypatch, workspace_ready=False)
    spec = _spec_for(_fake_model(torch), canvas_tokens=3, canvas_answer_row=1)
    pipeline = DiffusionGemmaPipeline(_fake_model(torch), None, spec=spec,
                                      engine_class=_Engine)
    assert len(pipeline.canvas_ids) == 3
    assert pipeline.canvas_answer_row == 1
    with pytest.raises(ValueError, match="geometry"):
        DiffusionGemmaPipeline(
            _fake_model(torch), None, engine_class=_Engine,
            spec=_spec_for(_fake_model(torch), sliding_window=512))
    assert seen["workspace"] == "cuda"
    assert pipeline.join_attention == "unified" and not pipeline.gemm_warmup
    assert pipeline.max_chunk_tokens == (2**31 - 1) // 32

    rows = torch.tensor([3, 4, 5])
    chunk = SimpleNamespace(
        input_ids=torch.zeros(6, dtype=torch.int64),
        positions=torch.arange(6),
        final_indices=torch.tensor([0, 3]),
        meta={"layer": 0, "canvas": {"rows": rows}}, images=())
    assert pipeline.vision is None and not pipeline.takes_images
    out = pipeline.forward_chunk(chunk)
    assert seen["context"] == ("config", 6)
    calls = pipeline.engine.calls
    # sliding layer: 2 x 4 heads with the window; full layer: 2 x 8, no window
    assert calls == [(0, (6, 2, 4), (6, 1, 4), 1.0, 1024),
                     (1, (6, 2, 8), (6, 1, 8), 1.0, None)]
    assert pipeline.engine.blocks == [None, None]
    # A prompt row embeds to 1 x 2 = 2; a canvas row passes the
    # weightless post-norm, which the fake engine applies as x 1, so it
    # starts at 2 too. Attention adds 0 (its post-norm is x 0). The MoE
    # layer adds the residual twice (dense MLP and experts, both
    # identity) and halves: 2 -> 3. The dense layer adds it once and
    # halves: 3 -> 3.
    assert out.shape == (2, 4)
    assert out.tolist() == [[3.0] * 4, [3.0] * 4]


def test_pipeline_embeds_images_and_runs_blocks_on_sliding_layers(monkeypatch):
    """Soft token rows take the tower's output; only sliding layers see blocks."""
    import numpy as np
    torch = pytest.importorskip("torch")

    from quail.backends.quail.executor.attention import ChunkImage
    from quail.backends.quail.executor.models.vision import VisionEmbedder
    from quail.pdf.prefetch import RenderedPage

    _vllm_stubs(monkeypatch)
    model = _fake_model(torch)
    seen = {}

    def embed_multimodal(*, pixel_values, pixel_position_ids):
        seen["pixel_values"] = pixel_values
        seen["positions"] = pixel_position_ids
        # two soft tokens per page, each row the page id plus 10
        return [torch.full((2, 4), float(10 + i))
                for i in range(len(pixel_values))]

    model.vision_tower = object()
    model.embed_multimodal = embed_multimodal
    pipeline = DiffusionGemmaPipeline(
        model, None, spec=_spec_for(model), engine_class=_Engine,
        vision_class=lambda torch_, m: VisionEmbedder(torch_, m, device="cpu"))
    assert pipeline.takes_images
    # rows: [start, soft, soft, end, question] for one document
    patches = np.full((36, 3 * 16 * 16), 51, dtype=np.uint8)
    page = RenderedPage(0, (6, 6), patches, 0.0)
    chunk = SimpleNamespace(
        input_ids=torch.zeros(5, dtype=torch.int64),
        positions=torch.arange(5),
        final_indices=torch.tensor([4]),
        meta={"layer": 0, "blocks": {"rows": torch.tensor([1, 2])}},
        images=(ChunkImage(1, 2, page),))
    rows = {}
    original = pipeline._layers

    def layers(hidden, positions, meta):
        rows["hidden"] = hidden.clone()
        return original(hidden, positions, meta)

    monkeypatch.setattr(pipeline, "_layers", layers)
    pipeline.forward_chunk(chunk)
    hidden = rows["hidden"]
    # text rows embed to 1 x 2; the soft rows are the tower's output
    assert hidden[0].tolist() == [2.0] * 4 and hidden[4].tolist() == [2.0] * 4
    assert hidden[1].tolist() == [10.0] * 4 and hidden[2].tolist() == [10.0] * 4
    # pixels reach the tower rescaled to [0, 1] with (column, row) positions
    assert seen["pixel_values"][0].shape == (36, 768)
    assert torch.allclose(seen["pixel_values"][0],
                          torch.full((36, 768), 51 / 255))
    assert seen["positions"][0][7].tolist() == [1, 1]
    blocks = pipeline.engine.blocks
    assert blocks[0] is chunk.meta["blocks"] and blocks[1] is None
    assert pipeline.vision.images_embedded == 1

    # a page whose soft token count differs from its rows is an error
    chunk.images = (ChunkImage(1, 3, page),)
    with pytest.raises(RuntimeError, match="reserved 3"):
        pipeline.forward_chunk(chunk)


def _vllm_stubs(monkeypatch, workspace_ready=True):
    """Stub the vLLM modules the pipeline imports; returns what they saw."""
    monkeypatch.setattr(
        "quail.backends.quail.executor.models.diffusion_gemma.FP8Experts",
        lambda module, engine: lambda x, scale, logits: module(x, logits))
    seen = {}
    context = types.ModuleType("vllm.forward_context")

    @contextlib.contextmanager
    def set_forward_context(attn_metadata, vllm_config, num_tokens=None):
        seen["context"] = (vllm_config, num_tokens)
        yield

    context.set_forward_context = set_forward_context
    workspace = types.ModuleType("vllm.v1.worker.workspace")
    workspace.is_workspace_manager_initialized = lambda: workspace_ready
    workspace.init_workspace_manager = lambda device: seen.setdefault(
        "workspace", str(device))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker",
                        types.ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace)
    return seen


def test_layer_scalars_fold_into_the_post_feedforward_norms(monkeypatch):
    torch = pytest.importorskip("torch")
    _vllm_stubs(monkeypatch)
    model = _fake_model(torch, layer_scalar=0.5)
    spec = _spec_for(model)
    DiffusionGemmaPipeline(model, None, spec=spec, engine_class=_Engine)
    for layer in model.model.layers:
        assert layer.post_feedforward_layernorm.weight.item() == 0.5
        assert layer.post_attention_layernorm.weight.item() == 0.0
        assert layer.input_layernorm.weight.item() == 1.0
    assert model.quail_scalars_folded
    # building again does not fold twice
    DiffusionGemmaPipeline(model, None, spec=spec, engine_class=_Engine)
    assert model.model.layers[1].post_feedforward_layernorm.weight.item() == 0.5


def test_filter_admission_takes_canvas_in_stage_tokens():
    sched = FilterAdmission([10, 10], [3 + 4, 2 + 4], 40,
                            arena_pages=100, page_tokens=16,
                            kept_extra_tokens=2 + 2 + 4)
    first = sched.next_chunk()
    assert first == [(0, 0, True), (1, 0, True)]


def test_tuned_moe_configs_preserve_upstream_settings(tmp_path):
    import json

    from quail.backends.quail.executor.moe_configs import TUNED, write_configs

    base = tmp_path / "vllm"
    base.mkdir()
    upstream = {"16384": {"BLOCK_SIZE_M": 64}, "32768": {"BLOCK_SIZE_M": 16}}
    for name in TUNED:
        (base / name).write_text(json.dumps(upstream))
    folder = write_configs(tmp_path / "configs", base)
    for name, overrides in TUNED.items():
        table = json.loads((folder / name).read_text())
        assert table["16384"] == upstream["16384"]
        assert table["32768"] == overrides["32768"]
        assert table["65536"] == overrides["65536"]


def test_tuned_moe_configs_fall_back_without_a_matching_upstream_table(tmp_path):
    from quail.backends.quail.executor.moe_configs import TUNED, write_configs

    folder = tmp_path / "configs"
    folder.mkdir()
    for name in TUNED:
        (folder / name).write_text("{}")
    write_configs(folder, tmp_path / "missing")
    assert list(folder.glob("*.json")) == []


@pytest.mark.parametrize("canvas, calls", [((90,), 1), ((90, 91), 2)])
def test_one_row_canvas_skips_the_second_attention_call(
        monkeypatch, canvas, calls):
    """A one-row canvas is its segment's last causal row: one call covers it."""
    import torch

    from quail.backends.quail.executor.attention import Engine

    cpu_staging(monkeypatch)
    arena = cpu_arena(64)
    groups = [dict(key=("d", 0), prefix=[1, 2, 3], f=3, suffixes=[[10, 11]])]
    chunk = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas)
    engine = Engine.__new__(Engine)
    engine.arena = arena
    engine.torch = torch
    seen = []

    def fa(q, k, v, cu_q, cu_k, max_q, max_k, causal, **kwargs):
        seen.append((q.shape[0], causal))
        return torch.zeros(q.shape[0], 2, 4), None

    monkeypatch.setattr(engine, "_fa", fa)
    rows = chunk.tokens
    q3 = torch.zeros(rows, 2, 4)
    meta = dict(chunk.meta, layer=0)
    out = engine.attention_unified(q3, q3, q3, meta)
    assert out.shape == (rows, 8)
    assert len(seen) == calls
    assert seen[0] == (rows, True)
    if calls == 2:
        assert seen[1] == (len(canvas), False)
