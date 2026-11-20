"""Base Labs mammoth-1 and elephant-1: prompt layout, mask, readout, and loading."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pytest
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.api.prompts import bind_classify_prompt, bind_prompt
from quail.backends.quail.executor import model_loading
from quail.backends.quail.executor.models import PIPELINES, build_pipeline
from quail.backends.quail.executor.models.base import pipeline_mask
from quail.backends.quail.executor.readout import DecisionRows, LetterReadout
from quail.baselines.vllm import stock_vllm_backend
from quail.catalog import DocumentProvider
from quail.common.expressions import ColumnRef, CompileError
from quail.common.tokens import close_documents
from quail.physical import AiClassify, AiFilter, decode_plan_value
from quail.physical.plan import Refusal
from quail.specs import H100_SXM, MODELS, QWEN3_4B_FP8

MAMMOTH = MODELS["mammoth-1-bf16"]
ELEPHANT = MODELS["elephant-1-bf16"]
REF = ColumnRef(alias="d", provider="documents", column="body")
DOCUMENT = "The order arrived damaged yesterday."
CLOSE = "\n</state>\n\n"
# OneJev's build_messages through the checkpoint's chat template, thinking
# off, at 81ce62f
TURN_OPEN_TEXT = (
    "<|im_start|>system\nApply the question to the state. Choose exactly one "
    "of the listed options. Respond with only its uppercase letter, with no "
    "explanation or reasoning.<|im_end|>\n<|im_start|>user\n")
TURN_CLOSE_TEXT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
NOUL_OPTIONS = (
    "\n\nOptions:\nA. yes: the statement is true / the answer is yes"
    "\nB. no: the statement is false / the answer is no"
    "\n\nAnswer with one letter: A, B.")
TURN = (MAMMOTH.turn_prefix, MAMMOTH.turn_suffix)


def test_a_mammoth_filter_prompt_is_the_published_noul_format():
    prompt = bind_prompt("Does the customer have a receipt? {0}", (REF,),
                         turn=TURN, layout=MAMMOTH.prompt_layout)
    assert prompt.preamble + DOCUMENT + CLOSE + prompt.tail[len("{0}"):] == (
        TURN_OPEN_TEXT + "<state>\n" + DOCUMENT + "\n</state>\n\n"
        "Question: Does the customer have a receipt?" + NOUL_OPTIONS
        + TURN_CLOSE_TEXT)


def test_a_mammoth_classification_lists_each_label_under_a_letter():
    prompt = bind_classify_prompt(
        "Which team should handle this request? {0}", (REF,),
        ("returns", "billing", "technical"),
        ("Refunds and damaged deliveries", "", "Product faults"),
        turn=TURN, layout="mammoth")
    assert prompt.preamble + DOCUMENT + CLOSE + prompt.tail[len("{0}"):] == (
        TURN_OPEN_TEXT + "<state>\n" + DOCUMENT + "\n</state>\n\n"
        "Question: Which team should handle this request?\n\nOptions:"
        "\nA. returns: Refunds and damaged deliveries\nB. billing"
        "\nC. technical: Product faults"
        "\n\nAnswer with one letter: A, B, C." + TURN_CLOSE_TEXT)


def test_a_mammoth_classification_past_z_asks_for_one_label():
    prompt = bind_classify_prompt("{0} Which one?", (REF,),
                                  tuple(f"x{index}" for index in range(27)),
                                  turn=TURN, layout="mammoth")
    assert "\nZ. x25\nAA. x26\n\nAnswer with one label: A, B, " in prompt.tail
    assert prompt.tail.endswith("Y, Z, AA." + TURN_CLOSE_TEXT)
    with pytest.raises(CompileError, match="at most 255 options"):
        bind_classify_prompt("{0} Which one?", (REF,),
                             tuple(f"x{index}" for index in range(256)),
                             turn=TURN, layout="mammoth")


def test_elephant_runs_mammoths_prompt_mask_and_readout_on_the_9b():
    shared = ("prompt_layout", "turn_prefix", "turn_suffix", "prompt_format",
              "readout", "role", "attention_mask")
    for field in shared:
        assert getattr(ELEPHANT, field) == getattr(MAMMOTH, field)
    base = MODELS["qwen3.5-9b-bf16"]
    assert (ELEPHANT.layers, ELEPHANT.hidden) == (base.layers, base.hidden)
    assert pipeline_mask(MAMMOTH) == pipeline_mask(ELEPHANT) == "document_isolated"
    assert pipeline_mask(base) == "causal"


def test_a_causal_forward_pass_refuses_a_bidirectional_model(monkeypatch):
    def causal(model, arena, spec, **kwargs):
        return SimpleNamespace(mask="causal")

    monkeypatch.setitem(PIPELINES, "qwen3", causal)
    spec = replace(QWEN3_4B_FP8, attention_mask="document_isolated")
    with pytest.raises(ValueError, match="runs the causal mask"):
        build_pipeline(spec, model=None, arena=None)
    assert build_pipeline(QWEN3_4B_FP8, model=None, arena=None).mask == "causal"


def test_each_document_ends_with_the_close_tokens():
    documents = close_documents({"d": [[1, 2], [3]]}, [8, 9])
    assert [list(tokens) for tokens in documents["d"]] == [[1, 2, 8, 9],
                                                           [3, 8, 9]]
    plain = {"d": [[1, 2]]}
    assert close_documents(plain, ()) is plain


def test_vllm_loads_mammoth_in_its_text_dtype(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "dtype": "float32",
        "text_config": {"dtype": "bfloat16"}}))
    assert model_loading.engine_args(tmp_path) == {
        "model": str(tmp_path), "dtype": "bfloat16"}


def _checkpoint(path, torch):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    letters = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    vocab = {"[UNK]": 0, **{letter: 2 * index + 1
                            for index, letter in enumerate(letters)}}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    head = torch.randn(64, 8, generator=torch.Generator().manual_seed(0))
    save_file({"lm_head.weight": head}, str(path / "model-00002.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"lm_head.weight": "model-00002.safetensors"}}))
    return head, [vocab[letter] for letter in letters]


def test_the_readout_reads_an_unsharded_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    head, ids = _checkpoint(tmp_path, torch)
    (tmp_path / "model.safetensors.index.json").unlink()
    (tmp_path / "model-00002.safetensors").rename(
        tmp_path / "model.safetensors")
    readout = LetterReadout.load(torch, torch.nn.functional, tmp_path,
                               ["A", "B"], device="cpu")
    assert torch.equal(readout.weight, head[ids[:2]])


def test_the_mammoth_readout_scores_each_option_by_its_letter_row(tmp_path):
    torch = pytest.importorskip("torch")
    head, ids = _checkpoint(tmp_path, torch)
    letters = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    readout = LetterReadout.load(torch, torch.nn.functional, tmp_path, letters,
                               device="cpu")
    normed = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
    rows = DecisionRows(torch, readout, [0, 0, 0, 0])
    assert torch.allclose(rows.scores(normed, [1, 1]),
                          normed @ head[ids[:3]].T)
    # a filter lists yes (A) then no (B); the yes/no readers take no first
    yes_no = DecisionRows(torch, readout.reordered((1, 0)), [0, 0, 0])
    assert torch.allclose(yes_no.scores(normed, [1, 1]),
                          normed @ head[[ids[1], ids[0]]].T)


def _text(session, ids) -> str:
    return session.tokenizer.decode(list(ids)).decode()


@pytest.fixture()
def session():
    value = quail.Session(EngineConfig(model=MAMMOTH.name, device="h100-sxm"),
                          tokenizer=TOKENIZER)
    value.register("documents", DocumentProvider.from_table(
        pa.table({"id": [1, 2], "body": ["refund please", "refund now"]}),
        id_col="id"))
    value.register("others", DocumentProvider.from_table(
        pa.table({"id": [1], "body": ["shoes"]}), id_col="id"))
    yield value
    value.close()


def test_a_mammoth_plan_sends_the_close_and_shares_no_prefix(session):
    plan = session.sql(
        "SELECT d.id FROM documents d "
        "WHERE AI_FILTER(PROMPT('Asks for a refund? {0}', d.body))").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiFilter)]
    assert _text(session, plan.settings["document_close"]) == CLOSE
    assert _text(session, node.question_token_ids) == (
        "Question: Asks for a refund?" + NOUL_OPTIONS + TURN_CLOSE_TEXT)
    # both documents start "refund", but neither borrows the other's KV
    assert not node.share_prefixes


def test_a_mammoth_classification_reads_every_label_at_the_last_row(session):
    plan = session.sql(
        "SELECT d.id, AI.CLASSIFY(d.body, ['refund', 'other']) AS c "
        "FROM documents d").plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert node.spec.scoring == "decision_choice"
    assert not node.spec.label_token_ids
    _, tail = node.spec.prompt_token_parts
    assert _text(session, tail) == (
        "Question: Which option applies to the state?\n\nOptions:"
        "\nA. refund\nB. other\n\nAnswer with one letter: A, B."
        + TURN_CLOSE_TEXT)


def test_a_mammoth_refuses_a_join_and_a_device_it_was_not_checked_on(session):
    joined = session.sql(
        "SELECT d.id, o.id FROM documents d JOIN others o "
        "ON AI_FILTER(PROMPT('Does {0} mention {1}?', d.body, o.body))").plan()
    assert isinstance(joined, Refusal)
    assert joined.constraint == "unsupported_bidirectional_query"
    config = EngineConfig(model=MAMMOTH.name,
                          device="rtx-pro-6000-blackwell-server")
    with quail.Session(config, tokenizer=TOKENIZER) as other:
        other.register("documents", DocumentProvider.from_table(
            pa.table({"id": [1], "body": ["refund"]}), id_col="id"))
        refused = other.sql(
            "SELECT d.id FROM documents d "
            "WHERE AI_FILTER(PROMPT('Asks for a refund? {0}', d.body))").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "bidirectional_device_unchecked"


def test_the_speed_of_light_counts_the_close_in_each_document(session):
    query = session.sql(
        "SELECT d.id FROM documents d "
        "WHERE AI_FILTER(PROMPT('Asks for a refund? {0}', d.body))")
    prompt = decode_plan_value(json.dumps(
        query.facts()["prompts"]["filters"]["d"][0]))
    lowered = quail.lower_prompt(prompt, session.tokenizer)
    estimate = quail.speed_of_light_estimate(query, lambda *_: True)
    # each document is computed once, its close in its own block, then asked
    close = len(session.tokenizer(CLOSE))
    assert estimate.fresh_tokens == sum(
        lowered.preamble_tokens + len(session.tokenizer(body)) + close
        + lowered.tail_tokens for body in ("refund please", "refund now"))


def test_a_mammoth_document_longer_than_a_chunk_is_refused():
    # the byte-level test tokenizer gives one token per character
    config = EngineConfig(model=MAMMOTH.name, device="h100-sxm")
    with quail.Session(config, tokenizer=TOKENIZER) as session:
        session.register("documents", DocumentProvider.from_table(
            pa.table({"id": [1], "body": ["x" * 40_000]}), id_col="id"))
        refused = session.sql(
            "SELECT d.id FROM documents d "
            "WHERE AI_FILTER(PROMPT('Asks for a refund? {0}', d.body))").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "bidirectional_document_exceeds_chunk"


def test_stock_vllm_refuses_mammoth_which_it_would_run_causal():
    support = stock_vllm_backend().supports(MAMMOTH, H100_SXM, 1)
    assert not support.supported
    assert "document_isolated" in support.reason
