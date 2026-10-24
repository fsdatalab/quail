"""The Quail executor's AI.MAP: its decode rounds on a scripted model."""

from types import SimpleNamespace

import numpy as np
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


def _map_one(monkeypatch, text, gold, *extra, recurrent=False, on_state=None,
             **settings):
    """Put one instruction to one document; return its settled answer.

    on_state, when given, is called with the state before the run.
    """
    tables = make_vocabulary(text, *([gold] if gold else []), *extra)
    spec = make_map_spec(tables, **settings)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model, recurrent=recurrent)
    state.loaded_model.vocabulary = tables
    if on_state is not None:
        on_state(state)
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


def test_a_map_writes_text_the_document_does_not_contain(monkeypatch):
    # the answer mixes document words with new ones; every round feeds
    # the chosen token, with a draft only where the document continues it
    gold = "The lease ends March 3"
    asked, _, _ = _map_one(monkeypatch, LEASE, gold)
    assert asked.answer == gold
    assert asked.done


def test_a_draft_copied_from_the_document_settles_in_few_rounds(monkeypatch):
    # the answer is a run of document text: after its first token the
    # draft copies the rest, so one round verifies the whole run
    gold = "March 3, 2027 at noon"
    asked, _, _ = _map_one(monkeypatch, LEASE, gold, draft_tokens=15,
                           draft_policy="fixed")
    assert asked.answer == gold
    written = len(asked.written)
    # the cue chooses the first token; one round copies the rest; one
    # round reads the stop
    assert asked.steps <= 3, (asked.steps, written)


def test_without_a_draft_each_round_writes_one_token(monkeypatch):
    gold = "March 3, 2027 at noon"
    asked, _, _ = _map_one(monkeypatch, LEASE, gold, draft_tokens=0)
    assert asked.answer == gold
    assert asked.steps == len(asked.written) + 1


def test_an_immediate_turn_end_writes_an_empty_answer(monkeypatch):
    asked, _, _ = _map_one(monkeypatch, LEASE, None)
    assert asked.answer == ""
    assert asked.written == []


def test_a_rejected_draft_is_replaced_by_the_next_round(monkeypatch):
    # " five" occurs twice; the draft copies " days" after the first,
    # and the model chooses " years": the round ends there, and the
    # next round feeds " years" in the draft's place
    text = "or five days or five years"
    asked, model, key = _map_one(monkeypatch, text, "five years",
                                 draft_policy="fixed")
    assert asked.answer == "five years"
    tables = model.tables
    five, days, years = (tables.encode_ids(w)[0] for w in (" five", " days", " years"))
    fed = [suffix for k, suffix, _, _, _ in model.packed if k == key]
    assert fed[1][:2] == [five, days]
    assert fed[2][:1] == [years]


def test_the_answer_stops_at_the_limit(monkeypatch):
    gold = "March 3, 2027 at noon"
    asked, _, _ = _map_one(monkeypatch, LEASE, gold, max_answer_tokens=3,
                           draft_tokens=0)
    assert len(asked.written) == 3
    assert asked.done


def test_a_recurrent_model_replays_accepted_tokens(monkeypatch):
    gold = "March 3, 2027"
    asked, _, _ = _map_one(monkeypatch, LEASE, gold, recurrent=True)
    assert asked.answer == gold


def test_two_maps_share_one_document_read(monkeypatch):
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
            model.gold[document.key] = gold
        seen["stages"] = self

    monkeypatch.setattr(MapStages, "__init__", init)

    def node(name):
        from dataclasses import replace
        return AiMap(node_id=f"ai-map:{name}", backend_name="quail",
                     model="qwen3-4b-fp8", spec=replace(spec, name=name))

    inputs = {"document_ids": [0], "documents": {"c": [tables.encode_ids(text)]},
              "texts": {"c": [text]}}
    results = MapPart.execute_group(state, [node("first"), node("second")], inputs)
    stages = seen["stages"]
    first = results["ai-map:first"].outputs["rows:c"]
    second = results["ai-map:second"].outputs["rows:c"]
    assert first.column("first").to_pylist() == [gold]
    assert second.column("second").to_pylist() == [gold]
    # the document's prefix was packed once for both instructions
    assert model.reads[stages.documents[0].key] == 1


def test_map_answers_need_the_document_tokens_or_text():
    tables = make_vocabulary()
    spec = make_map_spec(tables)
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    try:
        MapStages(state, [spec], tables, [0], None)
    except ValueError as error:
        assert "tokens or the text" in str(error)
    else:
        raise AssertionError("MapStages accepted a document without tokens")


def test_readout_rows_unwrap_object_arrays():
    from quail.backends.quail.executor.readout import readout_rows

    values = np.empty(1, dtype=object)
    values[0] = np.arange(18, dtype=np.float32)
    assert readout_rows(values, 9).shape == (2, 9)


def test_an_adaptive_copy_draft_runs_long_below_the_ridge(monkeypatch):
    # with the pass below the ridge, a marginal draft token is free:
    # the policy copies the whole run and the answer settles in a few
    # rounds, as the fixed policy would
    gold = "March 3, 2027 at noon"
    asked, _, _ = _map_one(monkeypatch, LEASE, gold, draft_tokens=15,
                           ridge_tokens=4096.0)
    assert asked.answer == gold
    assert asked.steps <= 3


def test_a_sibling_answer_seeds_the_cache_draft(monkeypatch):
    # two documents share one gold answer the second document's text
    # does not contain: the first answer fills the cache, and the
    # second drafts from it
    gold = "March 3, 2027 at noon"
    text_without_gold = "No dates here."
    tables = make_vocabulary(LEASE, text_without_gold, gold)
    spec = make_map_spec(tables, draft_tokens=15, ridge_tokens=4096.0)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model, chunk_tokens=120)
    state.loaded_model.vocabulary = tables
    stages = MapStages(state, [spec], tables, [0, 1],
                       [tables.encode_ids(LEASE),
                        tables.encode_ids(text_without_gold)])
    for document, text in zip(stages.documents, (LEASE, text_without_gold)):
        model.documents[document.key] = text
        model.gold[document.key] = gold
    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys,
               appended_tokens=stages.appended_tokens)
    stages.finish()
    answers = [document.answers[0] for document in stages.documents]
    assert [answer.answer for answer in answers] == [gold, gold]
    assert len(stages.decoder.cache)
    # the cache drafts were verified, so the policy recorded trials
    assert stages.decoder.policies[0].trials.sum() > 0


def test_rejected_copies_teach_the_policy_to_stop_drafting(monkeypatch):
    # every written token is in the document, but the model never
    # follows the document's continuation: the copy rate falls and the
    # rounds stop paying for long drafts
    text = "red one red two red three red four red five red six"
    gold = "red red red red red red red red red red red red"
    asked, _, _ = _map_one(monkeypatch, text, gold, draft_tokens=15,
                           max_answer_tokens=24)
    # each round feeds the chosen token and at most a few draft tokens
    assert asked.fed_tokens <= 3 * asked.steps + 24


def test_a_document_reserves_kv_for_the_whole_answer_and_a_draft(monkeypatch):
    # the rounds repeat one stage, so the reservation comes from the
    # answer cap, not from the stages' suffixes summed
    reserved = {}

    def record(state):
        arena = state.loaded_model.arena
        reserve = arena.reserve

        def recording(key, tokens, *, capacity_tokens, **kw):
            reserved[key] = (tokens, capacity_tokens)
            return reserve(key, tokens, capacity_tokens=capacity_tokens, **kw)

        arena.reserve = recording

    gold = "The lease ends March 3 2027 at noon and rent is due"
    asked, _, key = _map_one(monkeypatch, LEASE, gold, on_state=record,
                             max_answer_tokens=12, draft_tokens=3)
    assert asked.answer and len(asked.written) == 12
    tokens, capacity = reserved[key]
    # the prompt, then the tail's frame and cue, the answer, and a draft
    assert capacity - tokens >= len(TAIL.split()) + 12 + 3
