"""Decision 2.0 models: prompt layout, readout rows, head, planning, and loading."""

import json
import math
import re
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import cpu_arena, fake_pipeline, fake_torch

import quail
from quail.backends.quail.executor import loop, model
from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline
from quail.backends.quail.executor.readout import DecisionHead
from quail.backends.request import RequestBackend
from quail.catalog import Catalog, DocumentProvider
from quail.logical import (
    ColumnRef,
    answer_row_offsets,
    bind_join_prompt,
    bind_prompt,
    render_filter_prompt_ids,
    render_join_prompt_ids,
    render_join_prompt_text,
)
from quail.physical import AiFilter, AiJoin, AiScore
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import DECISION_2_KAI_0_6B_BF16, H100_SXM

LAYOUT = "decision2-noul"
OPTIONS = ('\n<option>\n{"description":"No","key":"false"}\n</option>'
           '\n<option>\n{"description":"Yes","key":"true"}\n</option>')
CLOSING = ("\n\nSelect the single option best supported by the context "
           "and instructions.\nDecision:")
REF = ColumnRef(alias="d", provider="documents", column="body")
PAIR = (ColumnRef(alias="q", provider="queries", column="text"), REF)


def _tokens(text):
    """Word tokens that keep trailing newlines, so segment edges change them."""
    return re.findall(r"\S+\n*|\n+|[ \t]+", text)


def test_filter_prompt_is_the_decision2_noul_format_tokenized_by_segment():
    prompt = bind_prompt("Does the customer have a receipt? {0}", (REF,),
                         _tokens, layout=LAYOUT)
    document = "The order arrived damaged."
    # decision2 segments(): prefix, one block per option, then the closing
    assert prompt.preamble + document + prompt.tail[len("{0}"):] == (
        "Context:\nThe order arrived damaged.\n\nTask type: noul\n"
        "Question:\nDoes the customer have a receipt?\nOptions:"
        + OPTIONS + CLOSING)
    segments = prompt.tail_segments
    assert len(segments) == 4
    assert prompt.tail_token_ids == tuple(
        t for segment in segments for t in _tokens(segment))
    assert prompt.tail_token_ids != tuple(_tokens("".join(segments)))

    ids = render_filter_prompt_ids(prompt, _tokens(document), _tokens)
    last = len(ids) - 1
    no, yes, query = (ids[last - o] for o in answer_row_offsets(LAYOUT, _tokens))
    assert (no, yes) == ("</option>", "</option>")
    assert query == "Decision:"
    assert ids[last - answer_row_offsets(LAYOUT, _tokens)[0] - 1] == (
        '{"description":"No","key":"false"}\n')


def test_join_prompt_keeps_the_question_in_the_anchor_frame():
    prompt = bind_join_prompt("Does {1} answer {0}?", PAIR, _tokens,
                              layout=LAYOUT)
    assert render_join_prompt_text(prompt, ["Where is it?", "In the box."], 0) == (
        "Context:\nWhere is it?\n\n(The document above is DOCUMENT {0}.)"
        "\n\nTask type: noul\nQuestion:\nDoes {1} answer {0}?"
        "\n\nDOCUMENT {1}:\nIn the box.\nOptions:" + OPTIONS + CLOSING)
    frames = {alias: frame for alias, _, frame in prompt.label_token_ids}
    assert "".join(frames["q"]) == (
        "\n\n(The document above is DOCUMENT {0}.)"
        "\n\nTask type: noul\nQuestion:\nDoes {1} answer {0}?")
    ids = render_join_prompt_ids(prompt, [_tokens("Where is it?"),
                                          _tokens("In the box.")], 0, _tokens)
    assert tuple(ids[-len(prompt.tail_token_ids):]) == prompt.tail_token_ids


def test_ai_if_layout_keeps_one_tail_segment():
    prompt = bind_prompt("Is it positive? {0}", (REF,), _tokens)
    assert prompt.preamble == "DOCUMENT:\n"
    assert prompt.tail_segments == (
        "\n\nEvaluate TRUE or FALSE for the following question: "
        "Is it positive?\nANSWER:",)
    assert answer_row_offsets("ai-if", _tokens) == (0,)


def test_pipeline_reads_each_answer_offset_and_clamps_at_the_chunk_start():
    torch = pytest.importorskip("torch")
    weight = SimpleNamespace(dtype=torch.bfloat16, shape=(4096, 1024))
    attn = SimpleNamespace(num_heads=16, num_kv_heads=8, head_dim=128,
                           rotary_emb=None, qkv_proj=SimpleNamespace(weight=weight))
    layer = SimpleNamespace(self_attn=attn,
                            mlp=SimpleNamespace(gate_up_proj=SimpleNamespace(
                                weight=weight)))
    embed = SimpleNamespace(weight=torch.zeros(1))
    fake = SimpleNamespace(model=SimpleNamespace(
        layers=[layer], embed_tokens=embed, norm=None))
    pipeline = Qwen3Pipeline(fake, None, spec=None,
                             engine_class=lambda arena, **kw: None,
                             answer_offsets=(5, 2, 0))
    final = torch.tensor([3, 20])
    assert pipeline.answer_indices(final).tolist() == [0, 1, 3, 15, 18, 20]
    plain = Qwen3Pipeline(fake, None, spec=None,
                          engine_class=lambda arena, **kw: None)
    assert plain.answer_indices(final) is final


def test_streams_refuse_suffixes_the_readout_reads_past():
    pipeline = fake_pipeline(answer_offsets=(4, 2, 0))
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    with pytest.raises(ValueError, match="reads 4 rows back"):
        loop.FilterStream(fake_torch(), cpu_arena(64), pipeline, answers,
                          [[5] * 20], [[40, 41, 42]], 200, arena_writes=True,
                          arena_keys=[("d", 0)])
    with pytest.raises(ValueError, match="4 rows the readout reads back"):
        loop.run_join(fake_torch(), cpu_arena(64), pipeline, answers,
                      [[5] * 20], [[[1, 2, 3, 4]]], 200)


def _reference_scores(torch, weights, options, last):
    """The CandidateHead forward of Decision 2.0's decision_model.py."""
    nn, F = torch.nn, torch.nn.functional
    hidden, head_dim = options.shape[-1], weights["key.weight"].shape[0]
    head = nn.Module()
    head.candidate_norm = nn.LayerNorm(hidden)
    head.query_norm = nn.LayerNorm(hidden)
    head.key = nn.Linear(hidden, head_dim, bias=False)
    head.query = nn.Linear(hidden, head_dim, bias=False)
    head.candidate_mlp = nn.Linear(hidden, head_dim)
    head.query_mlp = nn.Linear(hidden, head_dim, bias=False)
    head.scalar = nn.Linear(head_dim, 1, bias=False)
    head.load_state_dict(weights)
    candidate = head.candidate_norm(options.float())
    query = head.query_norm(last.float())
    bilinear = (head.key(candidate) * head.query(query)[:, None, :]).sum(-1) \
        / math.sqrt(head_dim)
    nonlinear = head.scalar(F.gelu(
        head.candidate_mlp(candidate) + head.query_mlp(query)[:, None, :]
    )).squeeze(-1)
    return bilinear + nonlinear


def test_decision_head_matches_the_reference_module():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(0)
    hidden, head_dim = 32, 8

    def tensor(*shape):
        return torch.randn(*shape, generator=generator)

    weights = {
        "candidate_norm.weight": tensor(hidden),
        "candidate_norm.bias": tensor(hidden),
        "query_norm.weight": tensor(hidden),
        "query_norm.bias": tensor(hidden),
        "key.weight": tensor(head_dim, hidden),
        "query.weight": tensor(head_dim, hidden),
        "candidate_mlp.weight": tensor(head_dim, hidden),
        "candidate_mlp.bias": tensor(head_dim),
        "query_mlp.weight": tensor(head_dim, hidden),
        "scalar.weight": tensor(1, head_dim),
    }
    options = tensor(5, 2, hidden).to(torch.bfloat16)
    last = tensor(5, hidden).to(torch.bfloat16)
    head = DecisionHead(torch, torch.nn.functional, weights)
    with torch.no_grad():
        expected = _reference_scores(torch, weights, options, last)
        got = head.scores(options, last)
    assert got.dtype == torch.float32
    assert torch.allclose(got, expected, atol=1e-5)


@pytest.fixture()
def session(tmp_path):
    catalog = Catalog()
    for name, column, values in (("documents", "body", ["refund please", "ok"]),
                                 ("queries", "text", ["refund", "shipping"])):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table({"id": [1, 2], column: values}), path)
        catalog.register(name, DocumentProvider.from_parquet(str(path), id_col="id"))
    value = quail.Session(
        EngineConfig(model=DECISION_2_KAI_0_6B_BF16.name, device="h100-sxm"),
        tokenizer=_tokens)
    for name in ("documents", "queries"):
        value.register(name, catalog.get(name))
    yield value
    value.close()


def test_plans_carry_the_decision_layout(session):
    plan = session.sql(
        "SELECT d.id FROM documents d "
        "WHERE AI_FILTER(PROMPT('Asks for a refund? {0}', d.body))").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiFilter)]
    (question,) = node.question_token_ids
    assert "".join(question).endswith(OPTIONS + CLOSING)
    assert "".join(question).startswith("\n\nTask type: noul\nQuestion:\n")
    assert plan.settings["pre_ids"] == _tokens("Context:\n")

    plan = session.sql(
        "SELECT q.id, d.id FROM queries q JOIN documents d "
        "ON AI_FILTER(PROMPT('Does {1} answer {0}?', q.text, d.body))").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    (stage,) = node.stages
    assert "".join(stage.tail_token_ids).endswith(OPTIONS + CLOSING)
    assert "".join(stage.frame_token_ids).endswith("Does {1} answer {0}?")
    assert "".join(stage.tail_token_ids).startswith("\nOptions:")

    plan = session.sql(
        "SELECT d.id, AI.SCORE(PROMPT('Refund? {0}', d.body)) AS s "
        "FROM documents d").plan()
    assert not isinstance(plan, Refusal)
    assert plan.settings["score_normalization"] == "decision_head_softmax"
    (score,) = [n for n in plan.nodes if isinstance(n, AiScore)]
    preamble, tail = score.spec.prompt_token_parts
    assert "".join(preamble) == "Context:\n"
    assert "".join(tail).endswith(OPTIONS + CLOSING)

    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine,
                             filter_submission="operator")
    assert not backend.supports(DECISION_2_KAI_0_6B_BF16, H100_SXM, 1).supported


def _package(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    src = tmp_path / "models--org--decision" / "snapshots" / "abc123"
    (src / "backbone").mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({"model_type": "decision2"}))
    (src / "backbone" / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3Model"], "model_type": "qwen3",
        "dtype": "float32"}))
    save_file({"embed_tokens.weight": torch.ones(4, 2),
               "layers.0.mlp.down_proj.weight": torch.ones(2, 2,
                                                           dtype=torch.bfloat16)},
              str(src / "backbone" / "model.safetensors"))
    for name in model.DECISION2_COPIED:
        (src / name).write_text(name)
    return src


def test_decision2_package_converts_once_to_a_bf16_qwen3_checkpoint(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    src = _package(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(model, "resolve_model_path", lambda name, rev: str(src))
    model.checkpoint_path.cache_clear()
    try:
        dest = model.checkpoint_path("org/decision", "abc123")
    finally:
        model.checkpoint_path.cache_clear()
    assert dest == str(tmp_path / "hf" / "quail-checkpoints"
                       / model.DECISION2_FORMAT / "models--org--decision"
                       / "abc123")
    config = json.loads((tmp_path / dest / "config.json").read_text())
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["torch_dtype"] == config["dtype"] == "bfloat16"
    tensors = load_file(f"{dest}/model.safetensors")
    assert sorted(tensors) == ["model.embed_tokens.weight",
                               "model.layers.0.mlp.down_proj.weight"]
    assert {t.dtype for t in tensors.values()} == {torch.bfloat16}
    for name, copy in model.DECISION2_COPIED.items():
        assert (tmp_path / dest / copy).read_text() == name
    # vLLM loads every top-level safetensors file as model weights
    assert sorted(p.name for p in (tmp_path / dest).glob("*.safetensors")) == [
        "model.safetensors"]
    assert not model.is_decision2(dest)
