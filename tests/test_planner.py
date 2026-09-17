"""Planner ordering, anchor selection, sharding, break-evens, and refusals."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import keep_even, register_claims_evidence, two_alias_graph

import quail
from quail.backends.quail import expected_join_stages
from quail.catalog import Catalog, DocumentProvider
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, ask, scan, triangle
from quail.frontend.builder import col, docs, prompt
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Foreign,
    GraphValidationError,
    PhysicalGraph,
    PortRef,
    Scan,
    validate_streams,
)
from quail.physical.base import input_ports
from quail.planner.decide import explain, filter_cost, order_filters, plan_query
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8


def filter_chain(plan, alias=None):
    return next(n for n in plan.nodes if isinstance(n, AiFilter)
                and (alias is None or n.alias == alias))


def join_stages(plan):
    """Return every expected join stage in execution order."""
    return [stage.to_dict() for stage in expected_join_stages(plan)]


def node_kinds(plan):
    return [type(node).__name__ for node in plan.nodes]


def _parquet(path, columns):
    pq.write_table(pa.table({c: ["x"] for c in columns}), str(path))
    return str(path)


@pytest.fixture()
def catalog(tmp_path):
    cat = Catalog()
    cat.register("reviews", DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", ["id", "review"]), id_col="id"))
    cat.register("products", DocumentProvider.from_parquet(
        _parquet(tmp_path / "p.parquet", ["asin", "description"]),
        id_col="asin"))
    cat.register("threads", DocumentProvider.from_parquet(
        _parquet(tmp_path / "t.parquet", ["id", "thread"]), id_col="id"))
    return cat


def tok(text):
    return text.split()


def _cell(text, title):
    """Return the column cells of the explain row that starts with title."""
    for line in text.splitlines():
        if line.strip().startswith(title):
            return line.split(title, 1)[1].split()
    raise AssertionError(f"no row starts with {title!r}")


def _five_filter_plan(catalog, sels):
    q = docs(catalog, "reviews", tok).alias("r")
    for j, s in enumerate(sels):
        q = q.ai_filter(prompt(f"flag {j} of: {{0}}", col("r.review")),
                        selectivity=s)
    return q.select("r.id")


def test_gpu_copies_and_memory_refusals(catalog):
    logical = _five_filter_plan(catalog, (0.5,))
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        for gpus in (1, 2, 4, 8):
            plan = plan_query(
                logical,
                model=model,
                device=H100_SXM,
                doc_tokens={"r": [100] * 8},
                gpus=gpus,
            )

            assert plan.model == model.name
            assert plan.workers == gpus
            scan_node = next(
                node for node in plan.nodes
                if isinstance(node, Scan)
            )
            assert len(scan_node.shards) == gpus

    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    logical = _five_filter_plan(catalog, (0.9,))
    for gpus in (1, 8):
        result = plan_query(
            logical,
            model=big,
            device=H100_SXM,
            doc_tokens={"r": [100] * 10},
            gpus=gpus,
        )
        assert isinstance(result, Refusal)
        assert result.constraint == "weights_need_more_cards"
        assert result.needed > result.available

    logical = _five_filter_plan(catalog, (0.9,))
    r = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                   doc_tokens={"r": [150_000]})
    assert isinstance(r, Refusal)
    assert r.constraint == "suffix_over_chunk"


def test_filter_ordering_and_kv_writes(catalog):
    # the B3 shape: a 0.2-selectivity filter written third among five
    sels = (0.9, 0.9, 0.2, 0.9, 0.9)
    logical = _five_filter_plan(catalog, sels)
    toks = {"r": [400] * 100}

    by_cost = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                         doc_tokens=toks)
    chain = filter_chain(by_cost)
    assert by_cost.settings["order_rule"] == "by_cost"
    assert chain.stages[0].selectivity == 0.2
    # the other four only see the 20% it passes
    assert chain.stages[1].expected_docs == pytest.approx(20.0)

    as_written = plan_query(logical, model=QWEN3_4B_FP8,
                            device=H100_SXM, doc_tokens=toks,
                            order="as_written")
    chain = filter_chain(as_written)
    assert [stage.selectivity for stage in chain.stages] == list(sels)

    class Predicate:
        def __init__(self, tail, selectivity):
            self.prompt = type("Prompt", (), {
                "tail_tokens": tail, "preamble_tokens": 0})()
            self.selectivity = selectivity

    first, second = Predicate(10, 0.5), Predicate(10, 0.5)
    kwargs = dict(prefix_tokens=400, model=QWEN3_4B_FP8,
                  device=H100_SXM, chunk_tokens=110_376)
    assert order_filters([first, second], "by_cost", **kwargs) == \
        [first, second]
    never_kills = Predicate(1, 1.0)
    assert order_filters([never_kills, first], "by_cost", **kwargs) == \
        [first, never_kills]

    class Predicate:
        def __init__(self, tail, selectivity):
            self.prompt = type("Prompt", (), {
                "tail_tokens": tail, "preamble_tokens": 0})()
            self.selectivity = selectivity

    predicates = [
        Predicate(12, 0.8),
        Predicate(50, 0.1),
        Predicate(8, 1.0),
        Predicate(25, 0.0),
    ]
    kwargs = dict(prefix_tokens=400, model=QWEN3_4B_FP8,
                  device=H100_SXM, chunk_tokens=110_376)
    ask_costs = [filter_cost(predicate, first=False, **kwargs)
                 for predicate in predicates]
    scan_costs = [filter_cost(predicate, first=True, **kwargs)
                  for predicate in predicates]

    candidates = []
    for first in range(len(predicates)):
        def score(index):
            rejected = 1.0 - predicates[index].selectivity
            return ask_costs[index] / rejected if rejected else float("inf")

        remaining = sorted(
            (index for index in range(len(predicates)) if index != first),
            key=score)
        order = [first, *remaining]
        expected = 0.0
        live = 1.0
        for position, index in enumerate(order):
            expected += live * (scan_costs[index] if position == 0
                                else ask_costs[index])
            live *= predicates[index].selectivity
        candidates.append((expected, order))

    expected_order = min(candidates)[1]
    actual = order_filters(predicates, "by_cost", **kwargs)
    assert actual == [predicates[index] for index in expected_order]

    logical = docs(catalog, "reviews", tok).alias("r")
    logical = logical.ai_filter(
        prompt(" ".join(["long"] * 100) + " {0}", col("r.review")),
        selectivity=0.1)
    logical = logical.ai_filter(
        prompt(" ".join(["short"] * 10) + " {0}", col("r.review")),
        selectivity=0.8435).select("r.id")
    plan = plan_query(
        logical, model=QWEN3_4B_FP8, device=H100_SXM,
        doc_tokens={"r": [400] * 100})
    assert [stage.written_pos
            for stage in filter_chain(plan).stages] == [1, 0]

    # one stage: nothing reads the KV again - writes off, visible in
    # the operator, the remark, and explain()
    single = _five_filter_plan(catalog, (0.9,))
    plan = plan_query(single, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    chain = filter_chain(plan)
    assert chain.arena_writes is False
    assert any("arena writes off" in r for r in plan.remarks)
    assert "arena_writes=False" in explain(single, plan, verbose=True)

    # a second stage re-reads survivors' KV
    plan = plan_query(_five_filter_plan(catalog, (0.9, 0.9)),
                      model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    chain = filter_chain(plan)
    assert chain.arena_writes is True
    assert not any("arena writes off" in r for r in plan.remarks)

    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_filter(prompt("a: {0}", col("r.review")))
               .ai_filter(prompt("b: {0}", col("r.review")),
                          selectivity=0.5)
               .select("r.id"))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [100] * 10})
    assert plan.settings["order_rule"] == "by_cost"
    assert "selectivity 0.2" in plan.settings["order_source"]


def test_component_costs_and_model_weights():
    result = speed_of_light(
        ask(400, 50) * 1000, QWEN3_4B_FP8, H100_SXM, 110_376)
    assert result.bound_by == "mixed"
    assert [c.name for c in result.components] == [
        "attn_proj", "mlp", "attention"]
    assert result.seconds == pytest.approx(sum(
        max(c.compute_seconds, c.memory_seconds)
        for c in result.components))
    assert result.component("mlp") is result.components[1]
    assert result.seconds > max(result.compute, result.memory)

    from quail.cost.qwen3_cost import dense_params

    work = ask(400, 50) * 1000
    result = speed_of_light(
        work, QWEN3_4B_FP8, H100_SXM, 110_376)
    dense_bytes = sum(
        c.bytes_moved for c in result.components
        if c.name != "attention")
    assert dense_bytes == dense_params(QWEN3_4B_FP8) * result.passes

    larger_resident_copy = replace(
        QWEN3_4B_FP8, w_mem_bytes=150e9)
    assert speed_of_light(
        work, larger_resident_copy, H100_SXM, 110_376).seconds \
        == pytest.approx(result.seconds)


def test_join_anchors_groups_and_forced_order(catalog):
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join([docs(catalog, "products", tok).alias("p"),
                         docs(catalog, "threads", tok).alias("t")],
                        prompt("m {0} {1} {2}", col("r.review"),
                               col("p.description"), col("t.thread")),
                        selectivity=0.02)
               .select("r.id"))
    toks = {"r": [3000] * 20, "p": [100] * 30, "t": [50] * 40}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    stage = join_stages(plan)[0]
    # every tuple of the cross product is one evaluation
    assert stage["expected_tuples"] == 20 * 30 * 40
    # the longest side anchors; partners keep placeholder order
    assert stage["anchor"] == "r"
    assert stage["partners"] == ["p", "t"]

    # r's documents are 3,000 tokens against t's 50 and p's 100:
    # streaming r as a partner would pay its tokens once per pair, so
    # the cheapest plan anchors r for stage 1 and p for stage 2 - two
    # groups with a barrier between them, not one shared-anchor group
    toks = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    plan = plan_query(_chain(catalog), model=QWEN3_4B_FP8,
                      device=H100_SXM, doc_tokens=toks,
                      order="as_written")
    stages = join_stages(plan)
    assert [s["anchor"] for s in stages] == ["r", "p"]
    assert [s["written_pos"] for s in stages] == [0, 1]
    assert stages[0]["partners"] == ["t"]
    assert stages[1]["partners"] == ["t"]
    assert stages[0]["expected_tuples"] == 10 * 8
    # the second stage sees the gate-thinned live counts
    assert stages[1]["expected_tuples"] < 8 * 6
    kinds = node_kinds(plan)
    assert kinds.count("AiJoin") == 2
    assert kinds.count("Barrier") == 1
    barrier = plan.graph.nodes_by_type(Barrier.type_name)[0]
    assert barrier.next_anchor == "p"
    assert set(barrier.aliases) == {"r", "t", "p"}
    # the barrier's outputs feed the second group's inputs
    group2 = plan.graph.nodes_by_type(AiJoin.type_name)[1]
    assert all(input_port.source.node_id == barrier.node_id
               for input_port in group2.inputs)

    # t (the shared table) has the long documents: anchoring it once
    # and keeping its KV across both stages is the cheapest plan -
    # one group, no barrier
    toks = {"r": [50] * 10, "t": [3000] * 8, "p": [50] * 6}
    plan = plan_query(_chain(catalog), model=QWEN3_4B_FP8,
                      device=H100_SXM, doc_tokens=toks,
                      order="as_written")
    stages = join_stages(plan)
    assert [s["anchor"] for s in stages] == ["t", "t"]
    kinds = node_kinds(plan)
    assert kinds.count("AiJoin") == 1
    assert kinds.count("Barrier") == 0

    # stage 1 forced onto t (the short side) is honored; stage 2 is
    # free and switches to p. The remark names the cheaper free plan.
    toks = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    plan = plan_query(_chain(catalog, anchors=("t", None)),
                      model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks, order="as_written")
    stages = join_stages(plan)
    assert stages[0]["anchor"] == "t"
    assert any("prices lower" in r for r in plan.remarks)

    # a forced anchor that IS part of the free optimum needs no remark
    same = plan_query(_chain(catalog, anchors=("r", None)),
                      model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks, order="as_written")
    assert join_stages(same)[0]["anchor"] == "r"
    assert not any("prices lower" in r for r in same.remarks)


def _chain(catalog, sels=(0.1, 0.1), anchors=(None, None)):
    """Build a two-join chain sharing table t: ai(r, t), ai(t, p)."""
    return (docs(catalog, "reviews", tok).alias("r")
            .ai_join(docs(catalog, "threads", tok).alias("t"),
                     prompt("m1 {0} {1}", col("r.review"),
                            col("t.thread")),
                     selectivity=sels[0], anchor=anchors[0])
            .ai_join(docs(catalog, "products", tok).alias("p"),
                     prompt("m2 {0} {1}", col("t.thread"),
                            col("p.description")),
                     selectivity=sels[1], anchor=anchors[1])
            .select("r.id", "t.id", "p.asin"))


def test_selective_gates_and_later_kv_reuse(catalog):
    # an expensive .9 full join and a cheap .01 exists gate on the
    # same table: by_cost runs the gate first so the join sees few
    # surviving anchors; as_written keeps the written order
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.9)
               .ai_join(docs(catalog, "threads", tok).alias("t"),
                        prompt("m {0} {1}", col("r.review"),
                               col("t.thread")),
                        selectivity=0.01, semantics="exists")
               .select("r.id"))
    toks = {"r": [400] * 100, "p": [100] * 100, "t": [100] * 100}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    stages = join_stages(plan)
    assert stages[0]["selectivity"] == 0.01

    as_written = plan_query(logical, model=QWEN3_4B_FP8,
                            device=H100_SXM, doc_tokens=toks,
                            order="as_written")
    stages = join_stages(as_written)
    assert stages[0]["selectivity"] == 0.9

    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.9)
               .ai_join(docs(catalog, "threads", tok).alias("t"),
                        prompt("m {0} {1}", col("r.review"),
                               col("t.thread")),
                        selectivity=0.01, semantics="exists")
               .select("r.id"))
    toks = {"r": [400] * 100, "p": [100] * 100, "t": [100] * 100}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert len(groups) == 2
    assert [group.anchor for group in groups] == ["r", "r"]
    assert groups[0].keep_anchor_kv is True
    assert groups[1].keep_anchor_kv is False
    # the gate's anchor KV is kept for the second group: a repeat
    # anchor use pays no prefix under unlimited KV pricing
    assert groups[1].anchor_resident == "kept"


def test_join_token_costs_and_retention(catalog):
    # the complete question frame is written into kept KV once per
    # anchor document; a partner's block label and the answer cue
    # ride in every tuple's suffix
    from quail.logical import SHARED_PRE, join_label, render_join_frame

    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.5, anchor="r")
               .select("r.id"))
    toks = {"r": [100] * 4, "p": [10] * 3}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    stage = join_stages(plan)[0]
    pre = len(tok(SHARED_PRE))
    # r is placeholder 0 (the anchor frame), p is placeholder
    # 1 (its block label)
    label = len(tok(join_label(1)))
    pred = logical.root.input.prompt
    frame = len(tok(render_join_frame(pred.template, 0)))
    tail = len(tok(pred.tail))
    expect = 4 * (100 + pre + frame) + 12 * (10 + label + tail)
    assert stage["tuple_tokens"] == pytest.approx(expect)
    assert stage["anchor_frame_tokens"] == frame
    assert stage["pair_tail_tokens"] == tail
    assert stage["expected_tuples"] == 12

    from quail.logical import join_label, render_join_frame

    logical = _filtered_join(catalog)
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [300] * 50, "p": [5] * 20})
    chain = filter_chain(plan)
    # one stage would normally turn arena writes off; the join reads
    # the KV. The chain streams its survivors into the join with their
    # KV pinned, so it retains nothing in the pool
    assert chain.arena_writes is True
    assert chain.keep_kv is False
    assert chain.pin_survivors is True
    assert chain.hold_tokens > 0
    group = plan.graph.nodes_by_type(AiJoin.type_name)[0]
    assert group.anchor == "r"
    assert group.anchor_resident == "filter"
    assert group.keep_anchor_kv is False    # nothing consumes r later

    # resident anchors pay the frame only - no preamble, no document
    pred = logical.root.input.prompt
    frame = len(tok(render_join_frame(pred.template, 0)))
    label = len(tok(join_label(1)))
    tail = len(tok(pred.tail))
    live = 50 * 0.5
    expect = live * frame + live * 20 * (5 + label + tail)
    stage = group.stages[0]
    assert stage.anchor_resident == "filter"
    assert stage.tuple_tokens == pytest.approx(expect)
    assert any("streams its survivors" in r for r in plan.remarks)
    assert not any("shared KV on 'r'" in r for r in plan.remarks)

    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.1)
               .select("r.id"))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [300] * 50, "p": [5] * 20})
    group = plan.graph.nodes_by_type(AiJoin.type_name)[0]
    assert group.anchor_resident == "none"
    assert not any("keep KV" in r for r in plan.remarks)

    # A filtered alias whose first use is as the anchor of a later
    # group has its chain emitted right before that group, after the
    # barrier, and streams into it: the pool is never involved.
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_filter(prompt("about food: {0}", col("r.review")),
                          selectivity=0.5)
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.5)
               .ai_join(docs(catalog, "threads", tok).alias("t")
                        .ai_filter(prompt("g {0}", col("t.thread")),
                                   selectivity=1.0),
                        prompt("m {0} {1}", col("p.description"),
                               col("t.thread")),
                        selectivity=0.5)
               .select("r.id"))
    toks = {"r": [300] * 50, "p": [5] * 20,
            "t": [3000] * 40 + [1000] * 100}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert [group.anchor for group in groups] == ["r", "t"]
    assert groups[1].anchor_resident == "filter"
    for alias in ("r", "t"):
        assert filter_chain(plan, alias).keep_kv is False
        assert filter_chain(plan, alias).pin_survivors is True
    order = [node.node_id for node in plan.nodes]
    assert order.index("barrier:t") < order.index("ai_filter:t") \
        < order.index("ai_join:t")
    assert not any("shared KV" in r for r in plan.remarks)

    # An alias that is a partner before it anchors must finish its
    # chain before that earlier group, so its survivors go through the
    # retention pool and its later anchor use is not streamed.
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_filter(prompt("about food: {0}", col("r.review")),
                          selectivity=0.5)
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.5, anchor="p")
               .ai_join(docs(catalog, "threads", tok).alias("t"),
                        prompt("m {0} {1}", col("r.review"),
                               col("t.thread")),
                        selectivity=0.5, anchor="r")
               .select("r.id"))
    toks = {"r": [300] * 50, "p": [500] * 20, "t": [5] * 20}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks, order="as_written")
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert [group.anchor for group in groups] == ["p", "r"]
    assert groups[1].anchor_resident == "filter"
    chain = filter_chain(plan, "r")
    assert chain.keep_kv is True
    assert chain.pin_survivors is False
    assert chain.arena_writes is True
    order = [node.node_id for node in plan.nodes]
    assert order.index("ai_filter:r") < order.index("ai_join:r")


# ------------------------------------------------ KV keep (residency)

def _filtered_join(catalog, doc_sel=0.5, join_sel=0.1):
    return (docs(catalog, "reviews", tok).alias("r")
            .ai_filter(prompt("about food: {0}", col("r.review")),
                       selectivity=doc_sel)
            .ai_join(docs(catalog, "products", tok).alias("p"),
                     prompt("m {0} {1}", col("r.review"),
                            col("p.description")),
                     selectivity=join_sel)
            .select("r.id"))


def test_search_matches_complete_left_deep_enumeration(catalog, tmp_path):
    import itertools as it

    from quail.planner.decide import join_specs
    from quail.planner.joins import _feasible_anchors, search_joins, walk

    catalog.register("tags", DocumentProvider.from_parquet(
        _parquet(tmp_path / "g.parquet", ["id", "tag"]), id_col="id"))
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "threads", tok).alias("t"),
                        prompt("m1 {0} {1}", col("r.review"),
                               col("t.thread")), selectivity=0.2)
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m2 {0} {1}", col("t.thread"),
                               col("p.description")), selectivity=0.05)
               .ai_join(docs(catalog, "tags", tok).alias("g"),
                        prompt("m3 {0} {1}", col("p.description"),
                               col("g.tag")), selectivity=0.5)
               .select("r.id"))
    toks = {"r": [900] * 6, "t": [40] * 8, "p": [200] * 5,
            "g": [30] * 9}
    joins = logical.operators().joins
    specs = join_specs(joins)
    live0 = {a: float(len(t)) for a, t in toks.items()}
    pre = 1
    chunk = 10_000

    def key(work):
        seconds = speed_of_light(work, QWEN3_4B_FP8, H100_SXM,
                                 chunk).seconds
        return (seconds, work.tokens, work.pairs, work.kv_written,
                work.kv_read)

    found = search_joins(specs, live0, toks, {}, pre, chunk,
                         QWEN3_4B_FP8, H100_SXM)

    # complete enumeration of the same space under the same cost
    # convention: live counts come from the applied edge set, thinned
    # in written order, exactly as the search prices them
    edge_aliases = [frozenset(s["aliases"]) for s in specs]
    best = []

    def visit_alias(order, pos, applied, sequence):
        if pos == len(order):
            if len(applied) == len(specs):
                work, _ = walk(
                    sequence, live0, toks, {}, pre,
                    QWEN3_4B_FP8, H100_SXM)
                best.append(work)
            return
        added = order[pos]
        crossing = [i for i, ends in enumerate(edge_aliases)
                    if added in ends and i not in applied
                    and ends <= set(order[:pos + 1])]
        if not crossing:
            return
        for edge_order in it.permutations(crossing):
            def visit_edge(k, applied_now, sequence_now):
                if k == len(edge_order):
                    visit_alias(order, pos + 1, applied_now,
                                sequence_now)
                    return
                i = edge_order[k]
                spec = specs[i]
                for anchor in _feasible_anchors(spec, True, toks, pre,
                                                chunk):
                    visit_edge(k + 1, applied_now | {i},
                               sequence_now + [(spec, anchor)])
            visit_edge(0, set(applied), list(sequence))

    aliases = sorted({a for ends in edge_aliases for a in ends})
    for order in it.permutations(aliases):
        visit_alias(list(order), 1, set(), [])

    assert best, "enumeration found no connected left deep plan"
    assert key(found["work"]) == min(key(w) for w in best)


def _join_search_spec(position, aliases, anchor):
    return dict(
        written_pos=position,
        aliases=list(aliases),
        anchor=anchor,
        anchor_free=False,
        semantics="full",
        selectivity=None,
        frame_tokens={alias: 5 for alias in aliases},
        label_tokens={alias: 4 for alias in aliases},
        tail_tokens=6,
    )


def test_join_search_residency_and_replanning():
    from quail.planner.joins import search_joins

    specs = [
        _join_search_spec(0, ("a", "b"), "a"),
        _join_search_spec(1, ("b", "c"), "b"),
        _join_search_spec(2, ("a", "c"), "a"),
    ]
    live = {"a": 3.0, "b": 3.0, "c": 2.0}
    lengths = {"a": [90, 100, 110], "b": [400] * 3,
               "c": [50] * 2}

    current = search_joins(
        specs, live, lengths, {"a"}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM, fixed_order=True)
    # a's filter computed its prefixes: its first anchor use pays no
    # prefix, and neither does its later use (kept after group 0)
    assert current["records"][0]["resident"] == "filter"
    assert current["records"][0]["resident_docs"] == 3
    assert current["records"][1]["resident"] == "none"
    assert current["records"][2]["resident"] == "kept"
    assert current["records"][2]["resident_docs"] == 3

    from quail.planner.joins import search_joins

    specs = [
        _join_search_spec(0, ("a", "b"), "b"),
        _join_search_spec(2, ("c", "d"), "c"),
    ]
    live = {alias: 2.0 for alias in "abcd"}
    lengths = {alias: [100, 120] for alias in "abcd"}

    assert search_joins(
        specs, live, lengths, {}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM) is None

    found = search_joins(
        specs, live, lengths, {}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM,
        already_joined={"b", "c"})

    assert found is not None
    assert {position for position, _ in found["seq"]} == {0, 2}

    from quail.planner.joins import AliasStats, search_joins

    million = AliasStats(
        count=1_000_000,
        total=100_000_000,
        squared=10_000_000_000,
        maximum=100,
    )
    stats = {alias: million for alias in "abcd"}
    live = {alias: 1_000_000.0 for alias in "abcd"}
    specs = [
        _join_search_spec(0, ("a", "b"), "a"),
        _join_search_spec(1, ("b", "c"), "b"),
        _join_search_spec(2, ("c", "d"), "c"),
    ]

    found = search_joins(
        specs, live, stats, {}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM)

    assert found is not None
    assert len(found["seq"]) == 3
    assert all("resident_positions" not in record
               for record in found["records"])


def test_aggregate_join_work_matches_per_document_sum():
    import pytest

    from quail.planner.joins import stage_work, summarize_alias
    spec = _join_search_spec(0, ("a", "b"), "a")
    live = {"a": 2.25, "b": 3.5}
    raw = {"a": [90, 100, 110], "b": [30, 50, 70, 90]}
    stats = {
        "a": summarize_alias(raw["a"]),
        "b": summarize_alias(raw["b"]),
    }

    def per_document(resident):
        n = live["a"]
        tuples = live["a"] * live["b"]
        suffix = (spec["tail_tokens"] + spec["label_tokens"]["b"]
                  + sum(raw["b"]) / len(raw["b"]))
        frame = spec["frame_tokens"]["a"]
        per_anchor = tuples / n
        fraction = n / len(raw["a"])
        total = Work()
        for position, document in enumerate(raw["a"]):
            prefix = 10 + document
            start = (ask(prefix, frame) if resident
                     else scan(prefix, frame))
            stream = Work(
                tokens=per_anchor * suffix,
                pairs=per_anchor * (
                    suffix * (prefix + frame) + triangle(suffix)),
                kv_written=per_anchor * suffix,
                kv_read=prefix + frame,
            )
            total = total + (start + stream) * fraction
        return total

    for resident in (False, True):
        actual = stage_work(spec, "a", live, stats, 10, resident=resident)
        expected = per_document(resident)
        assert actual.tokens == pytest.approx(expected.tokens)
        assert actual.pairs == pytest.approx(expected.pairs)
        assert actual.kv_written == pytest.approx(expected.kv_written)
        assert actual.kv_read == pytest.approx(expected.kv_read)


def test_explain_estimates_and_limits(catalog):
    for sels, expected in [
    ((0.5, 0.25), "12.5"), ((1.0, 0.0), "0"),
    ((None, 0.5), "10"), ((None, 0.0), "0"),
]:
        logical = _five_filter_plan(catalog, sels)
        plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                          doc_tokens={"r": [400] * 100})
        text = explain(logical, plan)
        physical = text.split("physical:", 1)[1]
        assert _cell(physical, "Project: r.id") == [expected]
        assert _cell(physical, "AiFilter: r")[0] == expected
        assert _cell(physical, "AiFilter: r")[-1] in ("ms", "s")
        assert _cell(physical, "Scan reviews as r") == ["100"]
        assert "tokens=40,000" in physical
        assert "node_id=" not in text
        assert "arena_writes" not in text
        assert "admission_tokens" not in text
        assert "KV=bf16, chunk budget=" in text
        assert "admission budget=" in text
        assert "KV rewind=on" in text
        assert "joins follow" not in text
        assert "stage {" not in text
        for stage in filter_chain(plan).stages:
            template = logical.root.input.predicates[stage.written_pos].prompt.template
            assert (f"predicate {stage.written_pos + 1}  PROMPT({template!r})"
                    in physical)
        verbose = explain(logical, plan, verbose=True)
        assert "node_id=ai_filter:r" in verbose
        assert "admission_tokens=" in verbose
        assert "expected_docs=" in verbose

    logical = _five_filter_plan(catalog, (0.25,))
    logical = replace(logical, root=replace(logical.root, limit=10))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    physical = explain(logical, plan).split("physical:", 1)[1]
    assert _cell(physical, "Limit: 10") == ["10"]
    assert _cell(physical, "Project: r.id") == ["25"]
    assert "KV: not stored" in physical

    logical = _filtered_join(catalog)
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100, "p": [40] * 10})
    physical = explain(logical, plan).split("physical:", 1)[1]
    assert "AiJoin: anchor=" in physical
    assert "KV: anchor=" in physical
    # the join row has no row estimate, only a time; its stage row has
    # the expected pairs and pass rate
    assert _cell(physical, "AiJoin: anchor=")[-1] in ("ms", "s")
    assert len(_cell(physical, "AiJoin: anchor=")[1:]) == 2
    assert _cell(physical, "join 1 full (")[-1] == "10%"
    assert "expected_tuples=" not in physical
    assert "survivors stream into the join with KV pinned" in physical
    assert "KV: anchor=streamed from its filter" in physical
    for node in plan.nodes:
        if isinstance(node, AiJoin):
            assert f"anchor={node.anchor}" in physical
            for stage in node.stages:
                assert f"{stage.semantics} ({stage.anchor}, " in physical


# ------------------------------------------------ Foreign nodes and streams


def _apply_query(session, kind):
    claims = (session.docs("claims").alias("c")
              .ai_filter(prompt("about a person: {0}", col("c.claim")),
                         selectivity=0.5)
              .apply(keep_even, columns=[col("c.url")], kind=kind))
    return (claims.join(session.docs("evidence").alias("e"))
            .ai_filter(prompt("{1} supports {0}", col("c.claim"),
                              col("e.text")), selectivity=0.5)
            .select("c.id", "e.id"))


def test_planner_places_foreign_nodes_and_keeps_or_drops_the_stream():
    with quail.Session(EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    ),
                       tokenizer=lambda text: list(text.encode())) as session:
        # claims are the long side, so the planner anchors on them
        register_claims_evidence(session, claim_words=200, text_words=10)
        per_batch = _apply_query(session, "per_batch").plan()
        chain = per_batch.graph.node("ai_filter:c")
        foreign = per_batch.graph.node("apply:keep_even")
        join = per_batch.graph.nodes_by_type(AiJoin.type_name)[0]
        assert chain.pin_survivors
        assert isinstance(foreign, Foreign) and foreign.kind == "per_batch"
        assert foreign.inputs[0].source == PortRef("ai_filter:c", "ids:c")
        assert PortRef("apply:keep_even", "ids:c") in {
            port.source for port in join.inputs}
        assert "keep_even" in session.registry.functions
        text = per_batch.graph.explain()
        assert "Foreign: keep_even (per_batch, drop) on c" in text
        # a barrier needs every survivor at once: the chain materializes
        barrier = _apply_query(session, "barrier").plan()
        assert not barrier.graph.node("ai_filter:c").pin_survivors
        assert barrier.graph.node("apply:keep_even").kind == "barrier"
        request = _apply_query(session, "barrier")._prepare_physical()
        assert request.column_tables()["c"].column_names == ["c", "url"]
        assert request.column_tables()["c"].column("url").to_pylist()[:2] == [
            "u0", "u1"]

    with quail.Session(EngineConfig(
        gpus=2,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    ),
                       tokenizer=lambda text: list(text.encode())) as session:
        register_claims_evidence(session, claim_words=200, text_words=10)
        plan = _apply_query(session, "per_batch").plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "per_batch_apply_needs_one_gpu"
        assert not isinstance(_apply_query(session, "barrier").plan(), Refusal)
    with quail.Session(EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="stock_vllm",
        device="h100-sxm",
    ),
                       tokenizer=lambda text: list(text.encode())) as session:
        register_claims_evidence(session, claim_words=200, text_words=10)
        plan = _apply_query(session, "barrier").plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "apply_needs_quail_backend"


def test_stream_validator_refuses_a_barrier_on_a_pinned_edge():
    validate_streams(two_alias_graph(True, foreign=("per_batch", "drop")))
    validate_streams(two_alias_graph(True, foreign=("per_batch", "pairs")))
    with pytest.raises(GraphValidationError, match="per-batch apply"):
        validate_streams(two_alias_graph(True, foreign=("barrier", "drop")))
    with pytest.raises(GraphValidationError, match="per-batch apply"):
        validate_streams(two_alias_graph(True, foreign=("barrier", "pairs")))
    # a pinned chain that no join consumes is refused too
    graph = two_alias_graph(True)
    orphan = PhysicalGraph(
        tuple(node for node in graph.nodes if node.node_id != "group:0")
        + (graph.node("group:0").with_inputs(input_ports(
            (PortRef("input:r", "ids:r"), PortRef("input:p", "ids:p")))),),
        graph.root)
    with pytest.raises(GraphValidationError, match="no join anchored"):
        validate_streams(orphan)


# ------------------------------------------------ per-node estimates and plan edits


def _big_plan(catalog):
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_filter(prompt("negative: {0}", col("r.review")),
                          selectivity=0.5)
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("about {0} {1}", col("r.review"),
                               col("p.description")), selectivity=0.1)
               .select("r.id", "p.asin"))
    return logical, plan_query(
        logical, model=QWEN3_4B_FP8, device=H100_SXM,
        doc_tokens={"r": [400] * 20000, "p": [20] * 10})


def test_node_ids_estimates_and_the_recompute_column(catalog):
    logical, plan = _big_plan(catalog)
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:p", "ai_filter:r", "ai_join:r", "project"]
    chain = plan.graph.node("ai_filter:r")
    assert chain.pin_survivors
    seconds = {node_id: entry["seconds"]
               for node_id, entry in plan.estimates.items()
               if "seconds" in entry}
    assert set(seconds) == {"ai_filter:r", "ai_join:r"}
    # each node priced alone: the parts add up to at least the packed
    # whole, and to no more than three times it
    assert plan.estimated_seconds <= sum(seconds.values()) \
        <= 3 * plan.estimated_seconds
    # 10,000 expected survivors of 400 tokens do not fit the retention
    # pool, so releasing the chain's KV would recompute most of them
    recompute = plan.estimates["ai_filter:r"]
    assert recompute["release_recompute_tokens"] > 0.9 * 10000 * 401
    assert recompute["release_recompute_seconds"] > 0
    text = explain(logical, plan)
    assert "est. time" in text
    assert "if the KV were released here instead of pinned" in text
    assert "do not add up to the plan estimate" in text

    # a Barrier on the pinned edge turns the pin off; the chain keeps
    # its survivors in the pool and the recompute becomes expected
    edited = plan.insert(
        Barrier(node_id="barrier:r", next_anchor="r", aliases=("r",)),
        between=("ai_filter:r", "ai_join:r"))
    assert [node.node_id for node in edited.nodes] == [
        "scan:r", "scan:p", "ai_filter:r", "barrier:r", "ai_join:r", "project"]
    new_chain = edited.graph.node("ai_filter:r")
    assert not new_chain.pin_survivors and new_chain.keep_kv
    assert new_chain.hold_tokens == 0
    assert edited.graph.node("barrier:r").inputs[0].source.node_id == "ai_filter:r"
    assert [port.source.node_id for port in edited.graph.node("ai_join:r").inputs] == [
        "barrier:r", "scan:p"]
    # unpinned, a survivor holds no frame room, so a few more fit
    assert edited.estimates["ai_filter:r"]["release_recompute_tokens"] == \
        pytest.approx(recompute["release_recompute_tokens"], rel=0.01)
    assert "expected recompute at the join" in explain(logical, edited)
    # the edited plan's total carries the recompute the edit causes
    assert edited.estimated_seconds == pytest.approx(
        plan.estimated_seconds
        + edited.estimates["ai_filter:r"]["release_recompute_seconds"])
    # the input plan is untouched, and remove gives the plan back
    assert plan.graph.node("ai_filter:r").pin_survivors
    assert edited.remove("barrier:r") == plan
    moved = edited.move("barrier:r", between=("scan:r", "ai_filter:r"))
    assert [node.node_id for node in moved.nodes][:4] == [
        "scan:r", "scan:p", "barrier:r", "ai_filter:r"]
    assert moved.graph.node("ai_filter:r").pin_survivors
