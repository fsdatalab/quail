"""The Quail executor's AI.MAP: its decode rounds on a scripted model."""

from types import SimpleNamespace

import pytest
from helpers.extract_fakes import HEAD, LEASE, TAIL, make_vocabulary
from helpers.fakes import fake_pack

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.map.stages import MapStages
from quail.backends.quail.executor.operators.map import MapPart
from quail.backends.quail.executor.stages import run_stages
from quail.physical import AiMap, MapSpec
from tests.engine.test_extract_executor import ScriptedModel, _fake_readout, _state


def make_map_spec(tables, **settings) -> MapSpec:
    return MapSpec(
        name="summary", aliases=("c",), query_template="{0}",
        arguments=(("c", "text"),), expected_rows=3, estimated_seconds=0.0,
        prompt_token_parts=(tuple(tables.encode_ids(HEAD)),
                            tuple(tables.encode_ids(TAIL))),
        instruction="Summarize the document.", **settings)


def _map_one(monkeypatch, text, gold, *extra, recurrent=False, **settings):
    """Put one instruction to one document; return its settled answer."""
    tables = make_vocabulary(text, *([gold] if gold else []), *extra)
    spec = make_map_spec(tables, **settings)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model, recurrent=recurrent)
    state.loaded_model.vocabulary = tables
    stages = MapStages(state, [spec], tables, [0], [tables.encode_ids(text)])
    key = stages.documents[0].key
    model.documents[key] = text
    model.gold[key] = gold
    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys,
               appended_tokens=stages.appended_tokens)
    stages.finish()
    return stages.documents[0].answers[0], model, key


@pytest.mark.parametrize("gold,draft_tokens,recurrent,limit", [
    ("The lease ends March 3", 15, False, 256),
    ("March 3, 2027 at noon", 15, False, 256),
    ("March 3, 2027 at noon", 0, False, 256),
    (None, 15, False, 256),
    ("March 3, 2027", 15, True, 256),
    ("March 3, 2027 at noon", 0, False, 3),
])
def test_maps_decode_greedily_until_stop_or_limit(
        monkeypatch, gold, draft_tokens, recurrent, limit):
    asked, model, _ = _map_one(monkeypatch, LEASE, gold,
                               draft_tokens=draft_tokens, recurrent=recurrent,
                               max_answer_tokens=limit)
    expected = model.tables.encode_ids(gold)[:limit] if gold else []
    assert len(asked.written) <= limit
    assert asked.answer == model.tables.tokenizer.decode(expected).strip()
    assert asked.done


def test_a_rejected_draft_is_replaced_by_the_next_round(monkeypatch):
    # " five" occurs twice; the draft copies " days" after the first,
    # and the model chooses " years": the round ends there, and the
    # next round feeds " years" in the draft's place
    text = "or five days or five years"
    asked, model, key = _map_one(monkeypatch, text, "five years")
    assert asked.answer == "five years"
    tables = model.tables
    five, days, years = (tables.encode_ids(w)[0] for w in (" five", " days", " years"))
    fed = [suffix for k, suffix, _, _, _ in model.packed if k == key]
    assert fed[1][:2] == [five, days]
    assert fed[2][:1] == [years]


@pytest.mark.parametrize("limit,keep_kv", [(None, False), (1, False), (None, True)])
def test_two_maps_share_document_reads_and_return_finished_answers(
        monkeypatch, limit, keep_kv):
    # two instructions over one table run as one stage list: the
    # second rewinds past the first's answer and reads the resident KV
    text = LEASE
    gold = "March 3, 2027"
    tables = make_vocabulary(text, gold)
    spec = make_map_spec(tables)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    seen = {}
    original = MapStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for document in self.documents:
            model.documents[document.key] = text
            model.gold[document.key] = gold if document.key[1] else "March"
        seen["stages"] = self

    monkeypatch.setattr(MapStages, "__init__", init)

    def node(name):
        from dataclasses import replace
        return AiMap(node_id=f"ai-map:{name}", backend_name="quail",
                     model="qwen3-4b-fp8", spec=replace(spec, name=name),
                     keep_kv=keep_kv)

    inputs = {"document_ids": [0, 1, 2],
              "documents": {"c": [tables.encode_ids(text)] * 3},
              "texts": {"c": [text] * 3}, "limit": limit}
    results = MapPart.execute_group(state, [node("first"), node("second")], inputs)
    stages = seen["stages"]
    first = results["ai-map:first"].outputs["rows:c"]
    second = results["ai-map:second"].outputs["rows:c"]
    expected = ["March"] if limit else ["March", gold, gold]
    assert first.column("first").to_pylist() == expected
    assert second.column("second").to_pylist() == expected
    # the document's prefix was packed once for both instructions
    assert model.reads[stages.documents[0].key] == 1


def test_greedy_decoding_does_not_sum_stop_probabilities():
    from quail.backends.quail.executor.map.decoder import MapAnswer, MapDecoder

    tables = make_vocabulary()
    token = tables.encode_ids("March")[0]
    document = SimpleNamespace(answers=[MapAnswer()])
    decoder = MapDecoder(tables, 1, make_map_spec(tables))
    decoder.choose(document, 0, [-0.5, token, -0.7])
    assert document.answers[0].written == [token]
    assert not document.answers[0].done
