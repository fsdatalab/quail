"""Joins over pairs: each anchor streams only the partners its equality allows."""

import random
from types import SimpleNamespace

import pyarrow as pa
from test_request_scheduling import _ParityClient
from test_streamed_loop import (
    DOC,
    FRAME,
    PARTNER,
    QUESTION,
    FakeModel,
    cpu_arena,
    expected_filter_rows,
    fake_pack,
    fake_torch,
    run_streamed,
)

import quail
from quail.backends.quail.coordinator import (
    join_group_payloads,
    merge_join_round,
    thin_survivors,
)
from quail.backends.request_scheduling import (
    join_cache_accounting,
    run_join_grouped,
)
from quail.execution import join_answer_cells
from quail.executor import loop
from quail.executor.attention import JOIN_ATTENTION
from quail.physical import AiJoin, PortRef
from quail.planner.plan import EngineConfig
from quail.runtime.pairs import pair_fraction, pair_table, partner_map


def test_pair_table_lists_equal_keys_once_each():
    pairs = pair_table(
        "c", [pa.array(["u1", "u2", None, "u1"])],
        "e", [pa.array(["u2", "u1", "u1"])])
    assert pairs.to_pydict() == {"c": [0, 0, 1, 3, 3], "e": [1, 2, 0, 1, 2]}
    assert pair_fraction(pairs, 4, 3) == 5 / 12
    assert partner_map(pairs, "e", "c") == {0: [1], 1: [0, 3], 2: [0, 3]}
    # two equalities: both key columns must match; integer keys on one
    # side are cast to the other side's type
    pairs = pair_table(
        "c", [pa.array(["u1", "u1"]), pa.array([1, 2])],
        "e", [pa.array(["u1", "u1"]), pa.array([2, 2], type=pa.int8())])
    assert pairs.to_pydict() == {"c": [1, 1], "e": [0, 1]}


def test_streamed_join_over_pairs_packs_only_allowed_partners(monkeypatch):
    rng = random.Random(11)
    n_docs, n_partners = 30, 6
    doc_lengths = [rng.randrange(8, 30) for _ in range(n_docs)]
    filter_truth = [[1 if rng.random() < 0.8 else 0,
                     1 if rng.random() < 0.7 else 0] for _ in range(n_docs)]
    partner_lengths = [rng.randrange(5, 20) for _ in range(n_partners)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)] for d in range(n_docs)}
    # document d pairs with the partners i where d + i is a multiple
    # of 4; every seventh document pairs with none
    allowed = {d: [i for i in range(n_partners)
                   if (d + i) % 4 == 0 and d % 7 != 3]
               for d in range(n_docs)}
    run = run_streamed(
        monkeypatch, doc_lengths=doc_lengths, filter_truth=filter_truth,
        partner_lengths=partner_lengths, join_truth=join_truth,
        budget=400, pages=40,
        anchor_partners=lambda key: [allowed[key[1]]])
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    assert sorted(key[1] for key in run["anchor_keys"]) == survivors
    assert run["stream"].answers == expected_filter_rows(filter_truth)
    for local, key in enumerate(run["anchor_keys"]):
        expected = [join_truth[key][i] for i in allowed[key[1]]]
        assert run["settled"][key] == expected
        assert run["join_answers"][0].get(local, []) == expected
    # every streamed anchor packs the frame plus its own partners only;
    # an anchor with no partner packs nothing and is settled at once
    assert run["join_tokens"] == sum(
        3 + sum(partner_lengths[i] for i in allowed[d])
        for d in survivors if allowed[d])
    assert not run["arena"].accounting.owned
    assert any(not allowed[d] for d in survivors)


def _graph(pin_survivors, equalities):
    from quail.physical import (
        AiFilter,
        FilterStage,
        JoinStage,
        PhysicalGraph,
        Scan,
    )
    from quail.physical.base import input_ports

    scan_r = Scan(node_id="input:r", alias="r", input_id="r")
    scan_p = Scan(node_id="input:p", alias="p", input_id="p")
    chain = AiFilter(
        node_id="filter:r",
        inputs=input_ports((PortRef("input:r", "ids:r"),)),
        alias="r", arena_writes=True, pin_survivors=pin_survivors,
        keep_kv=not pin_survivors, hold_tokens=1 if pin_survivors else 0,
        stages=(FilterStage(0, 1, 0, 0.8, 14),),
        question_token_ids=((QUESTION,),))
    join = AiJoin(
        node_id="group:0", anchor="r",
        anchor_resident="filter" if pin_survivors else "none",
        inputs=input_ports((PortRef("filter:r", "ids:r"),
                            PortRef("input:p", "ids:p"))),
        stages=(JoinStage(
            written_pos=0, exec_idx=0, anchor="r", partners=("p",),
            semantics="full", selectivity=0.5, expected_tuples=1,
            anchor_frame_tokens=1, pair_tail_tokens=0,
            anchor_resident="filter" if pin_survivors else "none",
            tuple_tokens=0, equalities=equalities,
            frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
            tail_token_ids=()),))
    return PhysicalGraph((scan_r, chain, scan_p, join),
                         PortRef("group:0", "ids:r"))


def _run_graph(monkeypatch, pin_survivors, pairs):
    from quail.backends.quail import QuailModelExecution
    from quail.backends.quail.graph import execute_single_graph
    from quail.builtins import built_in_registry
    from quail.specs import DEVICES, MODELS

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    rng = random.Random(5)
    n_docs, n_partners = 14, 4
    docs = {
        "r": [[DOC + d] * rng.randrange(10, 40) for d in range(n_docs)],
        "p": [[PARTNER + i] * 20 for i in range(n_partners)],
    }
    filter_truth = [[1 if rng.random() < 0.8 else 0] for _ in range(n_docs)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)] for d in range(n_docs)}
    model = FakeModel(filter_truth, join_truth)
    torch = SimpleNamespace(
        inference_mode=fake_torch().inference_mode,
        cuda=SimpleNamespace(
            Event=lambda **kw: SimpleNamespace(record=lambda: None),
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0))
    arena = cpu_arena(64)
    pipeline = SimpleNamespace(attention_mode=JOIN_ATTENTION,
                               forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(model=object(), arena=arena,
                                pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v),
        chunk_tokens=120)
    registry = built_in_registry()
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution, "runtimes": registry.runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs,
        "pairs": {} if pairs is None else {0: pairs},
    }
    equalities = () if pairs is None else (("r", "key", "p", "key"),)
    result = execute_single_graph(
        state, {"filter_limit": None, "pre_ids": [], "retention": {}},
        _graph(pin_survivors, equalities))
    assert not arena.accounting.owned
    return result, filter_truth, join_truth


def test_pair_join_runs_through_the_quail_graph(monkeypatch):
    # document d pairs with partner d % 4 and, for even d, with 3 too
    rows = [(d, d % 4) for d in range(14)] + [(d, 3) for d in range(0, 14, 2)
                                              if d % 4 != 3]
    pairs = pa.table({"r": pa.array([r for r, _ in rows], pa.int32()),
                      "p": pa.array([p for _, p in rows], pa.int32())})
    allowed = partner_map(pairs, "r", "p")
    for pin_survivors in (True, False):
        result, filter_truth, join_truth = _run_graph(
            monkeypatch, pin_survivors, pairs)
        cross, _, _ = _run_graph(monkeypatch, pin_survivors, None)
        survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
        stage = result["joins"][0]
        assert sorted(stage["anchor_index"]) == survivors
        members = stage["anchor_partners"]
        for local, document in enumerate(stage["anchor_index"]):
            mine = sorted(allowed[document])
            assert members[local] == mine
            assert stage["rows"][local] == [join_truth[("r", document)][i]
                                            for i in mine]
        # the exported answer table holds exactly the evaluated pairs
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert sorted(zip(table.column("r").to_pylist(),
                          table.column("p").to_pylist())) == sorted(
            (d, i) for d in survivors for i in allowed[d])
        assert table.column("answer").to_pylist() == [
            bool(join_truth[("r", d)][i])
            for d, i in zip(table.column("r").to_pylist(),
                            table.column("p").to_pylist())]
        # fewer pairs, fewer fresh tokens than the cross join
        metrics = result["node_metrics"]["group:0"]
        assert metrics["evaluated_document_pairs"] == sum(
            len(allowed[d]) for d in survivors)
        assert metrics["fresh_tokens"] < (
            cross["node_metrics"]["group:0"]["fresh_tokens"])
        assert result["regret_tokens"] == 0
        # anchors whose pairs all answered FALSE are gone; the rest
        # survive with the same rule as a cross join
        kept = [d for d in survivors
                if any(join_truth[("r", d)][i] for i in allowed[d])]
        root = result["_outputs"][PortRef("group:0", "ids:r")]
        assert sorted(root.column("r").to_pylist()) == kept


def test_coordinator_ships_and_merges_per_anchor_partner_lists():
    # merged rows keep each anchor's own member list
    outs = [
        dict(joins=[dict(rows={0: [1], 1: [0, 1]}, anchor_index=[0, 4],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [2], 1: [0, 1]})],
             fresh_tokens=1, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 0]}, anchor_index=[3],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [1, 2]})],
             fresh_tokens=1, wall_s=1.0),
    ]
    merged = merge_join_round(outs)[0]
    assert merged["anchor_partners"] == {0: [2], 1: [0, 1], 2: [1, 2]}
    assert list(join_answer_cells(merged)) == [
        (0, 2, 1), (1, 0, 0), (1, 1, 1), (2, 1, 1), (2, 2, 0)]
    # thinning reads the member list: anchor 0 matched partner 3
    # (member 2), anchor 4 matched partner 2, anchor 3 matched partner 2
    merged.update(anchor="r", partners=["p"])
    survivors = {"r": [0, 3, 4], "p": [1, 2, 3]}
    thin_survivors([merged], survivors)
    assert survivors == {"r": [0, 3, 4], "p": [2, 3]}

    # the payload of a pair stage carries each worker's anchors with
    # their live partner rows only
    pairs = pa.table({"r": pa.array([0, 0, 2, 3, 5], pa.int32()),
                      "p": pa.array([0, 3, 1, 2, 3], pa.int32())})
    payload = dict(
        model="qwen3-4b-fp8", chunk_tokens=1000, true_ids=[1],
        false_ids=[2], pre_ids=[9], filter_limit=None,
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]}, physical_plan={}, pairs={0: pairs},
        shards={"r": ((0, 2, 4), (1, 3, 5))})
    group = [dict(anchor="r", partners=["p"], semantics="full",
                  written_pos=0, equalities=[["r", "k", "p", "k"]],
                  labels={"p": [2]}, frame=[8], tail=[3])]
    subs = join_group_payloads(payload, 2, {"r": [0, 2, 3, 5], "p": [1, 3]},
                               group)
    assert [sub["anchor_index"] for sub in subs] == [[0, 2], [3, 5]]
    assert [sub["pairs"] for sub in subs] == [
        {0: {0: [3], 2: [1]}}, {0: {3: [], 5: [3]}}]


def test_request_backend_evaluates_listed_pairs_only():
    prefixes = [[100 + i] * (4 + i) for i in range(3)]
    suffixes = [[200 + j] * 3 for j in range(4)]
    pairs = [(0, 1), (0, 3), (2, 0), (2, 2), (2, 3)]
    for submission in ("anchor-major", "suffix-major"):
        client = _ParityClient()
        result = run_join_grouped(client, object(), prefixes, suffixes, {1},
                                  submission=submission, pairs=pairs)
        heads = [(p["prompt_token_ids"][0] - 100, p["prompt_token_ids"][-1] - 200)
                 for p in client.calls[0]]
        assert sorted(heads) == pairs
        if submission == "suffix-major":
            assert heads == sorted(pairs, key=lambda pair: pair[::-1])
        # answers and cache counts come back in the listed order
        assert result["answers"] == [
            1 if (100 + a + 200 + s) % 2 == 0 else 0 for a, s in pairs]
        assert result["cached_per_request"] == [(100 + a) % 7 for a, _ in pairs]
    # cache accounting takes one request count per anchor
    accounting = join_cache_accounting(
        [[7] * 40, [8] * 40, [9] * 40], [2, 0, 3], [0, 32, 16, 16, 40],
        [40, 0, 40], 16)
    assert accounting["regret_tokens"] == (32 - 0) + (32 - 16) + (32 - 16)


def _register(session):
    session.register("claims", quail.DocumentProvider.from_table(pa.table({
        "id": [f"c{i}" for i in range(4)],
        "claim": [f"{i} " + "word " * 20 for i in range(4)],
        "url": ["u0", "u1", "u1", "u9"],
    }), id_col="id"))
    session.register("evidence", quail.DocumentProvider.from_table(pa.table({
        "id": [f"e{i}" for i in range(3)],
        "text": [f"{i} " + "word " * 100 for i in range(3)],
        "url": ["u1", "u0", "u1"],
    }), id_col="id"))


def _pair_query(session, on):
    query = session.docs("claims").alias("c")
    partner = session.docs("evidence").alias("e")
    if on:
        query = query.join(partner, on=quail.col("c.url") == quail.col("e.url"))
    else:
        query = query.join(partner)
    return query.ai_filter(
        quail.prompt("Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")),
        selectivity=0.5).select("c.id", "e.id")


def test_session_plans_prices_and_ships_the_pair_table():
    with quail.Session(EngineConfig(),
                       tokenizer=lambda text: list(text.encode())) as session:
        _register(session)
        paired = _pair_query(session, on=True)
        cross = _pair_query(session, on=False)
        paired_plan, cross_plan = paired.plan(), cross.plan()
        paired_stage = paired_plan.graph.nodes_by_type(
            AiJoin.type_name)[0].stages[0]
        cross_stage = cross_plan.graph.nodes_by_type(
            AiJoin.type_name)[0].stages[0]
        # c1 and c2 pair with e0 and e2, c0 with e1: 5 of 12 pairs
        assert paired_stage.equalities == (("c", "url", "e", "url"),)
        assert paired_stage.pair_fraction == 5 / 12
        assert cross_stage.pair_fraction == 1.0
        assert paired_stage.expected_tuples == round(
            cross_stage.expected_tuples * 5 / 12, 1)
        assert paired_plan.estimated_seconds < cross_plan.estimated_seconds
        assert "on c.url = e.url" in paired.explain()
        request = paired._prepare_physical()
        assert request.pair_tables()[0].to_pydict() == {
            "c": [0, 1, 1, 2, 2], "e": [1, 0, 2, 0, 2]}
        assert cross._prepare_physical().relations == {}

        def answer(prompt, assignment):
            return (assignment["c"] + assignment["e"]) % 2 == 0

        estimate = quail.speed_of_light_estimate(paired, answer)
        assert estimate.join_pair_evaluations == 5
        assert estimate.join_stages[0]["passing_pairs"] == 2
        assert quail.speed_of_light_estimate(
            cross, answer).join_pair_evaluations == 12
