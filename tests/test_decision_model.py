"""Decision 2.0 models: prompt layout, readout rows, head, planning, and loading."""

import json
import math
import re
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.backends.quail.executor import model
from quail.backends.quail.executor.readout import DecisionHead, DecisionRows
from quail.backends.quail.executor.stages import Stage
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
from quail.physical import AiClassify, AiFilter, AiJoin, AiScore, RequestExecution
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


def test_decision_rows_read_offsets_from_each_answers_trailing_rows():
    torch = pytest.importorskip("torch")
    seen = []

    def scores(options, last):
        seen.append((options[..., 0].tolist(), last[..., 0].tolist()))
        return torch.zeros(options.shape[:2])

    rows = DecisionRows(torch, SimpleNamespace(scores=scores), (5, 2, 0))
    assert rows.trailing_rows == 6
    # two answers of six rows, then a frame entry's one row
    normed = torch.arange(13, dtype=torch.float32)[:, None]
    rows.scores(normed, rows_per_answer=[6, 6, 1])
    assert seen == [([[0.0, 3.0], [6.0, 9.0], [12.0, 12.0]],
                     [5.0, 11.0, 12.0])]
    # a stage over this readout reads its trailing rows from every suffix
    stage = Stage(suffixes=[[1] * 9, [2] * 7], readout=rows)
    assert stage.read_rows == [6, 6] and stage.read_all_rows


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

    plan = session.sql(
        "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'other']) AS c "
        "FROM documents d").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    spec = node.spec
    assert spec.scoring == "decision_choice"
    _, tail = spec.prompt_token_parts
    frame, request = "".join(tail[:spec.frame_tokens]), "".join(
        tail[spec.frame_tokens:])
    assert frame == ("\n\nTask type: choice\nQuestion:\nWhich option best "
                     "fits the context?\nOptions:")
    assert ["".join(block) for block in spec.label_token_ids] == [
        '\n<option>\n{"description":null,"key":"refund"}\n</option>',
        '\n<option>\n{"description":null,"key":"other"}\n</option>']
    assert request == "".join("".join(b) for b in spec.label_token_ids) + CLOSING

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


def test_a_decision_classification_is_reestimated(session):
    from quail.planner.classify import _Table

    plan = session.sql(
        "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['x', 'y']) AS c "
        "FROM documents d").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    table = _Table(alias="d", mean=100, longest=100, budget=65_536,
                   chunk=65_536, backend_name="quail",
                   model=DECISION_2_KAI_0_6B_BF16, device=H100_SXM,
                   tokenizer=_tokens, capacity=1_000_000,
                   lengths=(100,) * 50, shared=(0,) + (60,) * 49)
    spec = table.reestimate(node.spec, resident=True)
    assert spec.scoring == "decision_choice" and spec.estimated_seconds > 0


def test_decision_readouts_return_their_host_values():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.readout import (
        AsyncDecisionChoices,
        AsyncDecisionScores,
    )

    event = SimpleNamespace(synchronize=lambda: None)
    values = torch.tensor([[0.5, 1.5]])
    choices = AsyncDecisionChoices(torch, None, (4, 2, 0))
    assert choices.dtype == np.dtype((np.float32, (2,)))
    assert choices.result((event, values)).tolist() == [[0.5, 1.5]]
    scores = AsyncDecisionScores(torch, None, (2, 0, 0))
    assert scores.result((event, values[0])).tolist() == [0.5, 1.5]


def test_choice_stage_reads_each_option_end_and_the_last_row(session):
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.classify import ClassifyStages

    plan = session.sql(
        "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['x', 'yy', 'zzz']) AS c "
        "FROM documents d").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    spec = node.spec
    state = SimpleNamespace(torch=torch, loaded_model=SimpleNamespace(
        decision_head=object(), model_spec=None))
    labeled = []
    stages = ClassifyStages(state, spec, 2, lambda key: key,
                            on_label=lambda anchor, label: labeled.append(label))
    (stage,) = stages.stages
    _, tail = spec.prompt_token_parts
    request = list(tail[spec.frame_tokens:])
    assert stage.frame == list(tail[:spec.frame_tokens])
    assert stage.suffixes == [request]
    # every row from the first option's end to the last row is read
    offsets = stages.readout.offsets.tolist()
    blocks = [len(block) for block in spec.label_token_ids]
    assert offsets == [len(request) - sum(blocks[:i + 1]) for i in range(3)] + [0]
    assert stage.read_rows == [max(offsets) + 1]
    assert all("".join(request[len(request) - 1 - o]) == "</option>"
               for o in offsets[:-1])
    stage.decide(0, np.array([0.1, 2.0, 0.3], dtype=np.float32))
    assert labeled == ["yy"]


def test_stock_vllm_plans_decision_readouts_from_the_layout(session):
    vocab = {}

    def ids(text):
        return [vocab.setdefault(token, len(vocab)) for token in _tokens(text)]

    words = {}

    def text(token_ids):
        if not words:
            words.update((i, token) for token, i in vocab.items())
        return "".join(words[i] for i in token_ids)

    vllm = quail.Session(
        EngineConfig(model=DECISION_2_KAI_0_6B_BF16.name, device="h100-sxm",
                     backend="stock_vllm"), tokenizer=ids)
    vllm.register("documents", session.catalog.get("documents"))
    plan = vllm.sql(
        "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'other']) AS c "
        "FROM documents d WHERE AI_FILTER(PROMPT('Refund? {0}', d.body))").plan()
    (request,) = [n for n in plan.nodes if isinstance(n, RequestExecution)]
    offsets = plan.settings["decision_offsets"]
    assert offsets == list(answer_row_offsets(LAYOUT, ids))
    (question,) = request.filters[0].question_token_ids
    ends = [text(question[:len(question) - offset]) for offset in offsets]
    assert ends[0].endswith('"false"}\n</option>')
    assert ends[1].endswith('"true"}\n</option>')
    assert ends[2].endswith("Decision:")
    (classify,) = request.classifies
    tail = classify.tail_token_ids
    ends = [text(tail[:len(tail) - offset]) for offset in classify.option_offsets]
    assert ends[0].endswith('"key":"refund"}\n</option>')
    assert ends[1].endswith('"key":"other"}\n</option>')
    assert ends[2] == text(tail)
    vllm.close()


def test_vllm_decision_head_scores_the_rows_at_each_request_offsets():
    torch = pytest.importorskip("torch")
    from quail.backends.vllm_decision import score_requests

    head = SimpleNamespace(scores=lambda options, last: options[..., 0]
                           * 100 + last[:, None, 0])
    rows = torch.arange(10.0)[:, None].repeat(1, 2)

    def params(offsets):
        return SimpleNamespace(extra_kwargs={"decision_offsets": offsets})

    # request 1 is still prefilling; request 3 is vLLM's profiling run
    partial, done, short, probe = score_requests(
        torch, head, [None, rows[:8], rows[:4], rows[:6]],
        [params([3, 1, 0]), params([3, 1, 0]), params([2, 0]),
         SimpleNamespace(extra_kwargs=None)])
    assert partial is None
    assert done.tolist() == [4 * 100 + 7, 6 * 100 + 7]
    assert short.tolist() == [1 * 100 + 3]
    assert probe.tolist() == [5.0]
