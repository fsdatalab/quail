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
    # the cue chooses the first token; the round launched before that
    # choice is decided feeds it alone; one round copies the rest; one
    # round reads the stop
    assert asked.steps <= 4, (asked.steps, written)


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
    # " five" occurs twice. The round launched before the cue's choice
    # is decided feeds that choice, " five", alone. The round after is
    # drafted once " five" is known: the document continues it with
    # " days or five years", the GPU's choice is expected to be
    # " days", and the draft is what follows. The model chooses
    # " years" instead, so the round feeds it with a draft the choice
    # refutes, and the answer ends at its row.
    text = "or five days or five years"
    asked, model, key = _map_one(monkeypatch, text, "five years",
                                 draft_policy="fixed")
    assert asked.answer == "five years"
    tables = model.tables
    five, or_, years = (tables.encode_ids(w)[0] for w in (" five", " or", " years"))
    fed = [suffix for k, suffix, _, _, _ in model.packed if k == key]
    assert fed[1] == [five]
    assert fed[2][:2] == [years, or_]
    # one more round was launched before the end was decided; its
    # rows decide nothing
    assert len(fed) == 4 and len(fed[3]) == 1


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
    # two instructions over one table are put to the document at once:
    # the first branch computes the document, and the second borrows
    # its whole pages and computes only the tokens past them; the plan
    # put the branches on the tree path
    text = LEASE
    gold = "March 3, 2027"
    tables = make_vocabulary(text, gold)
    spec = make_map_spec(tables)
    packed = []

    def recording(torch, arena, specs, **kw):
        packed.extend(specs)
        return fake_pack(torch, arena, specs, **kw)

    monkeypatch.setattr(chunk_mod, "pack_chunk", recording)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    seen = {}
    original = MapStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for key in self.keys:
            model.documents[key] = text
            model.gold[key] = gold
        seen["stages"] = self

    monkeypatch.setattr(MapStages, "__init__", init)

    def node(name):
        from dataclasses import replace
        return AiMap(node_id=f"ai-map:{name}", backend_name="quail",
                     model="qwen3-4b-fp8", spec=replace(spec, name=name),
                     attention="tree")

    inputs = {"document_ids": [0], "documents": {"c": [tables.encode_ids(text)]},
              "texts": {"c": [text]}}
    results = MapPart.execute_group(state, [node("first"), node("second")], inputs)
    stages = seen["stages"]
    first = results["ai-map:first"].outputs["rows:c"]
    second = results["ai-map:second"].outputs["rows:c"]
    assert first.column("first").to_pylist() == [gold]
    assert second.column("second").to_pylist() == [gold]
    assert stages.branches
    length = len(stages.documents[0].prefix)
    shared = stages.prefix_tree.shared[1]
    assert shared and shared % 16 == 0 and shared < length
    prefix_tokens = {}
    for packed_spec in packed:
        for key in packed_spec.get("keys", [packed_spec.get("key")]):
            if packed_spec.get("prefix") is not None:
                prefix_tokens[key] = (prefix_tokens.get(key, 0)
                                      + len(packed_spec["prefix"]))
    # the document is computed once; the second branch computes only
    # the tokens past the pages it borrows
    assert prefix_tokens[stages.keys[0]] == length
    assert prefix_tokens[stages.keys[1]] == length - shared
    # on the tree path the two branches' rounds read the borrowed pages
    # once, as one group, and each reads only its own pages beside it
    together = [packed_spec["document_reads"] for packed_spec in packed
                if len(packed_spec.get("keys", ())) == 2]
    assert together
    for groups, tokens in together:
        assert groups.tolist() == [0, 0] and tokens.tolist() == [shared, shared]


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
    assert asked.steps <= 4


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


def test_a_long_document_is_prefilled_in_slices_and_answers_the_same(monkeypatch):
    # with room for 24 tokens a pass and slices of 8, the lease is
    # packed over several passes before its first round; the answer
    # is the one a whole prefill gives
    from quail.backends.quail.executor import admission as admission_mod

    gold = "The lease ends March 3"
    whole, _, _ = _map_one(monkeypatch, LEASE, gold)
    monkeypatch.setattr(admission_mod, "SLICE_MIN_TOKENS", 8)
    packed = []
    original = fake_pack

    def recording(torch, arena, specs, **kw):
        packed.extend(specs)
        return original(torch, arena, specs, **kw)

    def budget(state):
        state.chunk_tokens = 24

    # _map_one installs this module's fake_pack as the packer
    monkeypatch.setitem(globals(), "fake_pack", recording)
    sliced, _, _ = _map_one(monkeypatch, LEASE, gold, on_state=budget)
    assert sliced.answer == whole.answer == gold
    slices = [spec for spec in packed
              if spec.get("prefix") is not None and not len(spec["suffixes"])]
    assert slices
    assert all(spec["start"] == spec["f"] for spec in slices)
    starts = [spec["start"] for spec in slices]
    assert starts == [sum(len(s["prefix"]) for s in slices[:i])
                      for i in range(len(slices))]
    # the slice that completes the prefix carries the cue
    final = [spec for spec in packed
             if spec.get("prefix") is not None and len(spec["suffixes"])]
    assert final and final[0]["start"] == sum(len(s["prefix"]) for s in slices)


def test_the_round_feed_applies_the_decoders_rule_on_the_gpu():
    # two answers: the first fed a chosen token and a two-token draft,
    # and its rows choose the first draft token, then another token;
    # the second's one row prefers the stop. The feed says how many
    # draft tokens each accepted and the token its next row chose.
    import torch

    from quail.backends.quail.executor.readout import round_feed

    stop_mask = torch.zeros(100, dtype=torch.bool)
    stop_mask[99] = True
    # each row: the stop value, then its likeliest id and value (k = 1)
    values = torch.tensor([[-5.0, 11.0, -0.1],
                           [-5.0, 30.0, -0.2],
                           [-5.0, 12.0, -0.3],
                           [-0.1, 99.0, -0.1]])
    following = np.array([11, 12, -1, -1])
    feed = round_feed(torch, values, following, [3, 1], stop_mask, 1, 1)
    assert feed.accepted.tolist() == [1, 0]
    assert feed.chosen.tolist() == [30, 99]


def test_a_live_document_runs_in_every_pass(monkeypatch):
    # two documents decoding answers of different lengths: once a
    # document's tail is packed, it is in every pass until the pass
    # after its answer ends, as its next round is launched before the
    # one before it is decided
    from helpers.extract_fakes import TERM

    golds = ["March 3, 2027 at noon", "five years"]
    tables = make_vocabulary(LEASE, TERM, *golds)
    spec = make_map_spec(tables, draft_tokens=0)
    passes = []

    def recording(torch, arena, specs, **kw):
        # rounds are packed together, under their documents' keys
        passes.append([key for spec in specs
                       for key in spec.get("keys", [spec.get("key")])])
        return fake_pack(torch, arena, specs, **kw)

    monkeypatch.setattr(chunk_mod, "pack_chunk", recording)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    stages = MapStages(state, [spec], tables, [0, 1],
                       [tables.encode_ids(LEASE), tables.encode_ids(TERM)])
    for document, text, gold in zip(stages.documents, (LEASE, TERM), golds):
        model.documents[document.key] = text
        model.gold[document.key] = gold
    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys,
               appended_tokens=stages.appended_tokens)
    stages.finish()
    answers = [document.answers[0] for document in stages.documents]
    assert [answer.answer for answer in answers] == golds
    for document, answer in zip(stages.documents, answers):
        in_passes = [index for index, keys in enumerate(passes)
                     if document.key in keys]
        # the tail, one round per answer token (the last reads the
        # stop), and the round launched before the end was decided
        assert len(in_passes) == len(answer.written) + 2
        assert in_passes == list(range(in_passes[0], in_passes[-1] + 1))


def test_a_small_pool_runs_more_documents_when_pages_grow_as_they_write(monkeypatch):
    # twenty-four documents in a pool that holds six of them reserved
    # for the whole answer limit: each gets pages for its prompt and
    # first rounds, and more as it writes. Once short answers settle,
    # the admission expects the rest to be short, so more run at once,
    # and all finish
    from helpers.fakes import cpu_arena

    gold = "March 3"
    tables = make_vocabulary(LEASE, gold)
    spec = make_map_spec(tables, max_answer_tokens=48, draft_tokens=2,
                         draft_policy="fixed", expected_answer_tokens=4)
    passes = []

    def recording(torch, arena, specs, **kw):
        passes.append({key for packed_spec in specs
                       for key in packed_spec.get("keys", [packed_spec.get("key")])})
        return fake_pack(torch, arena, specs, **kw)

    monkeypatch.setattr(chunk_mod, "pack_chunk", recording)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    tokens = tables.encode_ids(LEASE)
    head, tail = spec.prompt_token_parts
    whole = len(head) + len(tokens) + len(tail) - 1 + 1 + 48 + 2 * 2
    state.loaded_model.arena = cpu_arena(6 * -(-whole // 16))
    original = MapStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for key in self.keys:
            model.documents[key] = LEASE
            model.gold[key] = gold

    monkeypatch.setattr(MapStages, "__init__", init)
    node = AiMap(node_id="ai-map:summary", backend_name="quail",
                 model="qwen3-4b-fp8", spec=spec)
    count = 24
    inputs = {"document_ids": list(range(count)), "documents": {"c": [tokens] * count},
              "texts": {"c": [LEASE] * count}}
    result = MapPart.execute_group(state, [node], inputs)["ai-map:summary"]
    assert result.outputs["rows:c"].column("summary").to_pylist() == [gold] * count
    assert max(len(keys) for keys in passes) > 6
    assert result.metrics.extension["kv_growth"]["max_running"] > 6


def test_answers_still_writing_keep_long_lengths_in_the_estimate(monkeypatch):
    # one answer settled after 3 tokens while two are still writing at
    # 40: the estimate counts the two as lasting at least 40, so it
    # expects them to go on rather than to end like the short one
    tables = make_vocabulary(LEASE)
    spec = make_map_spec(tables, max_answer_tokens=64)
    monkeypatch.setattr(MapStages, "_readout",
                        lambda self, state: _fake_readout(tables, 4))
    model = ScriptedModel(tables, {}, {}, 4)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    tokens = tables.encode_ids(LEASE)
    stages = MapStages(state, [spec], tables, [0, 1, 2], [tokens] * 3)
    decoder = stages.decoder
    answers = decoder.states[0]
    answers.asked[:] = True
    answers.num_written[:] = [3, 40, 40]
    decoder.settle(stages.documents[0], 0)
    left = decoder.remaining(0, np.array([1, 2]), 0.9)
    assert left.tolist() == [24, 24]


def test_maps_with_different_answer_limits_run_as_branches(monkeypatch):
    # two instructions over one table with limits of 2 and 5 tokens:
    # they share one pipeline, and each answer stops at its own limit
    gold = "March 3, 2027 at noon"
    tables = make_vocabulary(LEASE, gold)
    short = make_map_spec(tables, max_answer_tokens=2, draft_tokens=0)
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
        for key in self.keys:
            model.documents[key] = LEASE
            model.gold[key] = gold
        seen["stages"] = self

    monkeypatch.setattr(MapStages, "__init__", init)
    from dataclasses import replace
    longer = replace(short, name="longer", max_answer_tokens=5,
                     expected_answer_tokens=5)
    assert short.same_decoding(longer)
    nodes = [AiMap(node_id=f"ai-map:{spec.name}", backend_name="quail",
                   model="qwen3-4b-fp8", spec=spec) for spec in (short, longer)]
    inputs = {"document_ids": [0], "documents": {"c": [tables.encode_ids(LEASE)]},
              "texts": {"c": [LEASE]}}
    MapPart.execute_group(state, nodes, inputs)
    stages = seen["stages"]
    assert stages.branches
    written = [len(document_answers.written)
               for document_answers in stages.documents[0].answers]
    assert written == [2, 5]


def _filter_then_map(monkeypatch, *, pool_documents=None, filter_limit=None,
                     first_gold=None):
    """Run a filter then a map over twenty-four leases in one pipeline.

    Every fourth lease fails the filter. Returns the map's rows, the
    leases each pass packed, the arena, and the answer.
    """
    from helpers.fakes import cpu_arena, fake_pipeline, fake_readout, fake_torch

    from quail.backends.quail import QuailModelExecution
    from quail.backends.quail.graph import execute_single_graph
    from quail.builtins import built_in_registry
    from quail.physical import AiFilter, PhysicalGraph, PortRef, Scan
    from quail.physical.base import input_ports
    from quail.specs import DEVICES, MODELS

    gold = "March 3"
    tables = make_vocabulary(LEASE, gold, *([first_gold] if first_gold else []))
    spec = make_map_spec(tables, max_answer_tokens=48, draft_tokens=2,
                         draft_policy="fixed", expected_answer_tokens=4)
    head, tail = spec.prompt_token_parts
    passes = []

    def recording(torch, arena, specs, **kw):
        passes.append({key for packed_spec in specs
                       for key in packed_spec.get("keys", [packed_spec.get("key")])})
        return fake_pack(torch, arena, specs, **kw)

    monkeypatch.setattr(chunk_mod, "pack_chunk", recording)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    count = 24
    for index in range(count):
        model.documents[("c", index)] = LEASE
        model.gold[("c", index)] = gold
        model.filter_truth[("c", index)] = index % 4 != 3
    if first_gold:
        model.gold[("c", 0)] = first_gold
    tokens = tables.encode_ids(LEASE)
    question = tables.encode_ids(" question")
    graph = PhysicalGraph((
        Scan(node_id="input:c", alias="c", input_id="c"),
        AiFilter(node_id="ai_filter:c",
                 inputs=input_ports((PortRef("input:c", "rows:c"),)),
                 alias="c", arena_writes=True,
                 question_token_ids=tuple(question), written_position=0,
                 question_tokens=len(question), preamble_tokens=len(head),
                 document_head=tuple(head), selectivity=0.75,
                 expected_rows=18.0),
        AiMap(node_id="ai-map:summary",
              inputs=input_ports((PortRef("ai_filter:c", "rows:c"),)),
              backend_name="quail", model="qwen3-4b-fp8", spec=spec),
    ), PortRef("ai-map:summary", "rows:c"))
    whole = len(head) + len(tokens) + len(tail) + 48 + 2 * 2
    torch = fake_torch()
    arena = cpu_arena(256 if pool_documents is None
                      else pool_documents * -(-whole // 16))
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
    execution.bind_query(
        torch=torch, answer_rows=object(), chunk_tokens=400,
        async_answers=fake_readout(
            submit=lambda rows: [float(np.asarray(r).ravel()[0]) for r in rows],
            result=lambda v: v, dtype=None))
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution,
                                    spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": 400,
             "documents": {"c": [tokens] * count}, "texts": {"c": [LEASE] * count}}
    report = execute_single_graph(
        state, {"filter_limit": filter_limit, "retention": {}}, graph)
    table = report["_outputs"][PortRef("ai-map:summary", "rows:c")].to_pydict()
    return table, passes, arena, gold


def test_a_map_after_a_filter_grows_its_pages_in_a_small_pool(monkeypatch):
    # a pool that holds six leases reserved for the whole answer limit:
    # the map grows each survivor's pages as it writes, so more run at once
    table, passes, arena, gold = _filter_then_map(monkeypatch, pool_documents=6)
    survivors = [index for index in range(24) if index % 4 != 3]
    assert table["c"] == survivors
    assert table["summary"] == [gold] * len(survivors)
    assert max(len(keys) for keys in passes) > 6
    assert not arena.accounting.owned


def test_a_limit_after_a_filter_returns_only_whole_answers(monkeypatch):
    # the first lease's answer is longer, so it is still being written
    # when two others end and the limit stops the run
    longer = "March 3, 2027 at noon or the first business day after it"
    table, _, arena, gold = _filter_then_map(monkeypatch, filter_limit=2,
                                             first_gold=longer)
    answers = dict(zip(table["c"], table["summary"]))
    assert len(answers) >= 2
    assert answers.pop(0, longer) == longer
    assert set(answers.values()) == {gold}
    assert not arena.accounting.owned


def _run_maps(monkeypatch, count, num_maps, *, limit=None, keep_kv=False,
              pool_documents=None, chunk_tokens=400, pack=fake_pack):
    """Put num_maps instructions to count copies of a lease; return each map's rows."""
    from dataclasses import replace

    from helpers.fakes import cpu_arena

    gold = "March 3, 2027 at noon"
    tables = make_vocabulary(LEASE, gold)
    spec = make_map_spec(tables, max_answer_tokens=48, draft_tokens=0,
                         draft_policy="fixed", expected_answer_tokens=4)
    monkeypatch.setattr(chunk_mod, "pack_chunk", pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(MapStages, "_readout", lambda self, state: readout)
    state = _state(model, chunk_tokens=chunk_tokens)
    state.loaded_model.vocabulary = tables
    tokens = tables.encode_ids(LEASE)
    if pool_documents is not None:
        head, tail = spec.prompt_token_parts
        whole = len(head) + len(tokens) + len(tail) + 48
        state.loaded_model.arena = cpu_arena(pool_documents * -(-whole // 16))
    original = MapStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for key in self.keys:
            model.documents[key] = LEASE
            model.gold[key] = gold

    monkeypatch.setattr(MapStages, "__init__", init)
    specs = [spec] + [replace(spec, name=f"map{index}") for index in range(1, num_maps)]
    nodes = [AiMap(node_id=f"ai-map:{each.name}", backend_name="quail",
                   model="qwen3-4b-fp8", spec=each, keep_kv=keep_kv)
             for each in specs]
    inputs = {"document_ids": list(range(count)), "documents": {"c": [tokens] * count},
              "texts": {"c": [LEASE] * count}, "limit": limit}
    results = MapPart.execute_group(state, nodes, inputs)
    rows = {each.name: results[f"ai-map:{each.name}"].outputs["rows:c"].to_pydict()
            for each in specs}
    return rows, gold


def test_a_limit_over_several_maps_counts_documents(monkeypatch):
    rows, gold = _run_maps(monkeypatch, 8, 2, limit=3)
    assert len(rows["summary"]["c"]) >= 3
    assert rows["summary"]["c"] == rows["map1"]["c"]
    assert set(rows["summary"]["summary"]) == set(rows["map1"]["map1"]) == {gold}


def test_branches_that_keep_kv_finish_in_a_small_pool(monkeypatch):
    rows, gold = _run_maps(monkeypatch, 60, 2, keep_kv=True, pool_documents=5,
                           chunk_tokens=160)
    assert rows["summary"]["summary"] == rows["map1"]["map1"] == [gold] * 60


def test_maps_the_plan_left_unified_read_no_document_apart(monkeypatch):
    packed = []

    def recording(torch, arena, specs, **kw):
        packed.append(kw.get("attention_mode"))
        packed.extend(spec for spec in specs if "rounds" in spec)
        return fake_pack(torch, arena, specs, **kw)

    rows, gold = _run_maps(monkeypatch, 4, 3, pack=recording)
    assert rows["summary"]["summary"] == [gold] * 4
    assert set(mode for mode in packed if isinstance(mode, str)) == {"unified"}
    assert not any("document_reads" in spec for spec in packed
                   if isinstance(spec, dict))
