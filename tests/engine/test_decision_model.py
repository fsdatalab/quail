"""Decision 2.0 models: prompt layout, readout rows, head, planning, and loading."""

import json
import re
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.catalog import Catalog, DocumentProvider
from quail.logical import (
    ColumnRef,
    bind_prompt,
)
from quail.planner.plan import EngineConfig
from quail.planner.prompt_lowering import (
    answer_row_offsets,
    lower_prompt,
    render_filter_prompt_ids,
)
from quail.specs import DECISION_2_KAI_0_6B_BF16

LAYOUT = "decision2-noul"
OPTIONS = ('\n<option>\n{"description":"No","key":"false"}\n</option>'
           '\n<option>\n{"description":"Yes","key":"true"}\n</option>')
CLOSING = ("\n\nSelect the single option best supported by the context "
           "and instructions.\nDecision:")
REF = ColumnRef(alias="d", provider="documents", column="body")


def _tokens(text):
    """Word tokens that keep trailing newlines, so segment edges change them."""
    return re.findall(r"\S+\n*|\n+|[ \t]+", text)


def test_filter_prompt_is_the_decision2_noul_format_tokenized_by_segment():
    prompt = bind_prompt("Does the customer have a receipt? {0}", (REF,),
                         layout=LAYOUT)
    lowered = lower_prompt(prompt, _tokens)
    document = "The order arrived damaged."
    # decision2 segments(): prefix, one block per option, then the closing
    assert prompt.preamble + document + prompt.tail[len("{0}"):] == (
        "Context:\nThe order arrived damaged.\n\nTask type: noul\n"
        "Question:\nDoes the customer have a receipt?\nOptions:"
        + OPTIONS + CLOSING)
    segments = prompt.tail_segments
    assert len(segments) == 4
    assert lowered.tail_token_ids == tuple(
        t for segment in segments for t in _tokens(segment))
    assert lowered.tail_token_ids != tuple(_tokens("".join(segments)))

    ids = render_filter_prompt_ids(lowered, _tokens(document))
    last = len(ids) - 1
    no, yes, query = (ids[last - o] for o in answer_row_offsets(LAYOUT, _tokens))
    assert (no, yes) == ("</option>", "</option>")
    assert query == "Decision:"
    assert ids[last - answer_row_offsets(LAYOUT, _tokens)[0] - 1] == (
        '{"description":"No","key":"false"}\n')


def test_worker_answer_offsets_match_session_tokens(tmp_path):
    from gigatoken import Tokenizer

    from quail.backends.quail.worker import decision_offsets

    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [byte for byte in range(256) if byte not in visible]
    alphabet = [chr(byte) for byte in visible] + [
        chr(256 + index) for index in range(len(missing))]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False,
                  "trim_offsets": True, "use_regex": True}
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "version": "1.0", "added_tokens": [], "normalizer": None,
        "pre_tokenizer": byte_level, "post_processor": None,
        "decoder": byte_level, "model": {"type": "BPE", "merges": [],
        "vocab": {char: index for index, char in enumerate(alphabet)}}}))
    spec = replace(DECISION_2_KAI_0_6B_BF16, name="local-decision",
                   hf_name=str(tmp_path), turn_suffix="\nEND")
    registry = quail.ExtensionRegistry.with_built_ins()
    registry.register_model(spec)
    with quail.Session(EngineConfig(model=spec.name, device="h100-sxm"),
                       registry=registry) as session:
        offsets = decision_offsets(spec, str(tmp_path))
        prompt = lower_prompt(bind_prompt("Refund? {0}", (REF,), layout=LAYOUT,
                                          turn=spec.turn), session.tokenizer)
        ids = render_filter_prompt_ids(
            prompt, session.tokenizer("Please refund this."))
        assert offsets == answer_row_offsets(
            LAYOUT, session.tokenizer, spec.turn_suffix)
        tokenizer = Tokenizer(tmp_path).as_hf()
        ends = [tokenizer.decode(ids[:len(ids) - offset]) for offset in offsets]
        assert ends[0].endswith('"false"}\n</option>')
        assert ends[1].endswith('"true"}\n</option>')
        assert ends[2].endswith("Decision:\nEND")


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


def test_a_decision_head_needs_exactly_its_tensors():
    import torch

    from quail.backends.quail.executor.readout import DecisionHead

    weights = {name: torch.zeros(4) for name in DecisionHead.TENSORS}
    weights["key.weight"] = torch.zeros(2, 4)

    assert DecisionHead(torch, torch.nn.functional, weights).head_dim == 2
    with pytest.raises(ValueError, match="missing"):
        DecisionHead(torch, torch.nn.functional,
                     {k: v for k, v in weights.items() if k != "scalar.weight"})
    with pytest.raises(ValueError, match="does not use"):
        DecisionHead(torch, torch.nn.functional,
                     {**weights, "key.bias": torch.zeros(2)})
