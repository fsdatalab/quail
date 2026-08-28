"""Planner ordering, anchor selection, sharding, break-evens, and refusals."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quail.builder import col, docs, prompt
from quail.catalog import Catalog, DocumentProvider
from quail.planner.decide import explain, order_filters, plan_query
from quail.planner.sol import ask, speed_of_light
from quail.planner.plan import PhysicalPlan, Refusal, resolve_model
from quail.specs import H100_SXM, QWEN3_4B_FP8


def filter_chain(plan, alias=None):
    return next(n for n in plan.nodes if n["op"] == "FilterChain"
                and (alias is None or n["alias"] == alias))


def join_stages(plan):
    """Return every join stage in execution order across JoinGroup nodes."""
    return [st for n in plan.nodes if n["op"] == "JoinGroup"
            for st in n["stages"]]


def node_kinds(plan):
    return [n["op"] for n in plan.nodes]


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


def _five_filter_plan(catalog, sels):
    q = docs(catalog, "reviews", tok).alias("r")
    for j, s in enumerate(sels):
        q = q.ai_filter(prompt(f"flag {j} of: {{0}}", col("r.review")),
                        selectivity=s)
    return q.select("r.id")


def test_b3_ordering_by_cost_vs_as_written(catalog):
    # the B3 shape: a 0.2-selectivity filter written third among five
    sels = (0.9, 0.9, 0.2, 0.9, 0.9)
    logical = _five_filter_plan(catalog, sels)
    toks = {"r": [400] * 100}

    by_cost = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                         doc_tokens=toks)
    chain = filter_chain(by_cost)
    assert by_cost.order_rule == "by_cost"     # every predicate has a sel
    assert chain["stages"][0]["selectivity"] == 0.2
    # the other four only see the 20% it passes
    assert chain["stages"][1]["expected_docs"] == pytest.approx(20.0)

    as_written = plan_query(logical, model=QWEN3_4B_FP8,
                            device=H100_SXM, doc_tokens=toks,
                            order="as_written")
    chain = filter_chain(as_written)
    assert [s["selectivity"] for s in chain["stages"]] == list(sels)

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


def test_filter_order_uses_dense_and_attention_rooflines():
    class Predicate:
        def __init__(self, tail, selectivity):
            self.prompt = type("Prompt", (), {
                "tail_tokens": tail, "preamble_tokens": 0})()
            self.selectivity = selectivity

    long_selective = Predicate(100, 0.1)
    short_weak = Predicate(10, 0.9101)
    ordered = order_filters(
        [long_selective, short_weak], "by_cost", prefix_tokens=400,
        model=QWEN3_4B_FP8, device=H100_SXM, chunk_tokens=110_376)
    assert ordered == [short_weak, long_selective]


def test_sol_adds_separate_dense_and_attention_rooflines():
    result = speed_of_light(
        ask(400, 50) * 1000, QWEN3_4B_FP8, H100_SXM, 110_376)
    assert result.bound_by == "mixed"
    assert result.seconds == pytest.approx(
        max(result.dense, result.dense_memory)
        + max(result.attention, result.attention_memory))
    assert result.seconds > max(result.compute, result.memory)


def test_plan_uses_roofline_filter_order(catalog):
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
    assert [stage["written_pos"]
            for stage in filter_chain(plan)["stages"]] == [1, 0]


def test_filter_arena_writes_decision(catalog):
    # one stage: nothing reads the KV again - writes off, visible in
    # the operator, the remark, and explain()
    single = _five_filter_plan(catalog, (0.9,))
    plan = plan_query(single, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    chain = filter_chain(plan)
    assert chain["arena_writes"] is False
    assert any("arena writes off" in r for r in plan.remarks)
    assert "arena_writes=False" in explain(single, plan)

    # a second stage re-reads survivors' KV
    plan = plan_query(_five_filter_plan(catalog, (0.9, 0.9)),
                      model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    chain = filter_chain(plan)
    assert chain["arena_writes"] is True
    assert not any("arena writes off" in r for r in plan.remarks)


def test_default_rule_falls_back_without_selectivity(catalog):
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_filter(prompt("a: {0}", col("r.review")))
               .ai_filter(prompt("b: {0}", col("r.review")),
                          selectivity=0.5)
               .select("r.id"))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [100] * 10})
    assert plan.order_rule == "as_written"
    assert "no selectivity" in plan.order_source


def test_anchor_longer_side_and_override(catalog):
    def joined(anchor):
        return (docs(catalog, "reviews", tok).alias("r")
                .ai_join(docs(catalog, "products", tok).alias("p"),
                         prompt("m {0} {1}", col("r.review"),
                                col("p.description")),
                         selectivity=0.1, anchor=anchor)
                .select("r.id"))

    toks = {"r": [3000] * 50, "p": [100] * 500}
    plan = plan_query(joined(None), model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    stage = join_stages(plan)[0]
    assert stage["anchor"] == "r"      # the longer side anchors
    assert stage["partners"] == ["p"]

    forced = plan_query(joined("p"), model=QWEN3_4B_FP8,
                        device=H100_SXM, doc_tokens=toks)
    stage = join_stages(forced)[0]
    assert stage["anchor"] == "p"
    assert stage["partners"] == ["r"]
    assert any("prices lower" in r for r in forced.remarks)


def test_three_way_anchor_and_tuple_count(catalog):
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


def test_chain_splits_into_groups_when_the_long_side_anchors(catalog):
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
    assert kinds.count("JoinGroup") == 2
    assert kinds.count("Barrier") == 1
    barrier = plan.nodes_by_op("Barrier")[0]
    assert barrier["next_anchor"] == "p"
    assert set(barrier["thins"]) == {"t", "p"}
    # the barrier's outputs feed the second group's inputs
    group2 = plan.nodes_by_op("JoinGroup")[1]
    assert all(src == barrier["id"] for src, _ in group2["inputs"])


def test_chain_shares_one_anchor_when_the_shared_table_is_longest(
        catalog):
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
    assert kinds.count("JoinGroup") == 1
    assert kinds.count("Barrier") == 0


def test_forced_anchor_is_honored_with_a_remark_when_it_prices_worse(
        catalog):
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


def test_three_join_chain_plans_with_barriers(catalog, tmp_path):
    # ai(r,t), ai(t,p), ai(p,g): no table appears in all three
    # predicates, so no single anchor exists - the plan splits into
    # anchor groups with barriers between them
    catalog.register("tags", DocumentProvider.from_parquet(
        _parquet(tmp_path / "g.parquet", ["id", "tag"]), id_col="id"))
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "threads", tok).alias("t"),
                        prompt("m1 {0} {1}", col("r.review"),
                               col("t.thread")))
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m2 {0} {1}", col("t.thread"),
                               col("p.description")))
               .ai_join(docs(catalog, "tags", tok).alias("g"),
                        prompt("m3 {0} {1}", col("p.description"),
                               col("g.tag")))
               .select("r.id"))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [100] * 4, "t": [100] * 4,
                                  "p": [100] * 4, "g": [100] * 4},
                      order="as_written")
    stages = join_stages(plan)
    assert len(stages) == 3
    # every stage anchors on one of its own tables
    aliases = [("r", "t"), ("t", "p"), ("p", "g")]
    for st, tabs in zip(stages, aliases):
        assert st["anchor"] in tabs
    kinds = node_kinds(plan)
    # equal lengths and counts: one anchor switch is optimal (any
    # zero-switch plan would need a table in all three predicates)
    assert kinds.count("JoinGroup") == 2
    assert kinds.count("Barrier") == 1
    # recombination reads every stage's pairs
    rec = plan.nodes_by_op("Recombine")[0]
    pair_ports = [port for _, port in rec["inputs"]
                  if port.startswith("pairs:")]
    assert len(pair_ports) == 3


def test_join_order_runs_selective_gate_first(catalog):
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


def test_refusal_weights_need_more_cards(catalog):
    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    logical = _five_filter_plan(catalog, (0.9,))
    r = plan_query(logical, model=big, device=H100_SXM,
                   doc_tokens={"r": [100] * 10}, gpus=1)
    assert isinstance(r, Refusal)
    assert r.constraint == "weights_need_more_cards"
    assert r.needed > r.available


def test_preamble_counted_once_per_document(catalog):
    from quail.logical import SHARED_PRE
    logical = _five_filter_plan(catalog, (0.5,))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 10})
    chain = filter_chain(plan)
    st = chain["stages"][0]
    # the shared preamble is per document (stage 0), never per stage
    assert st["preamble_tokens"] == len(tok(SHARED_PRE))
    assert st["question_tokens"] == len(tok(
        "Evaluate TRUE or FALSE for the following question: "
        "flag 0 of: ANSWER:"))


def test_join_tokens_frame_per_anchor_labels_per_tuple(catalog):
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
    pred = logical.root.input.predicate
    frame = len(tok(render_join_frame(pred.template, 0)))
    tail = len(tok(pred.tail))
    expect = 4 * (100 + pre + frame) + 12 * (10 + label + tail)
    assert stage["tuple_tokens"] == pytest.approx(expect)
    assert stage["anchor_frame_tokens"] == frame
    assert stage["pair_tail_tokens"] == tail
    assert stage["expected_tuples"] == 12


def test_refusal_suffix_over_chunk(catalog):
    logical = _five_filter_plan(catalog, (0.9,))
    r = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                   doc_tokens={"r": [150_000]})
    assert isinstance(r, Refusal)
    assert r.constraint == "suffix_over_chunk"


def test_refusal_unknown_model():
    unknown = resolve_model("qwen9-13b")
    assert isinstance(unknown, Refusal)
    assert unknown.constraint == "unknown_model"


def test_kv_is_always_bf16(catalog):
    logical = _five_filter_plan(catalog, (0.9,))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    assert plan.kv_dtype == "bf16"
    assert any("kv_dtype=bf16 (always)" in r for r in plan.remarks)
    from quail.planner import budgets
    assert plan.admission_tokens == budgets.arena_tokens(
        QWEN3_4B_FP8, H100_SXM, plan.chunk_tokens)


def test_explain_prints_tree_settings_and_source(catalog):
    logical = _five_filter_plan(catalog, (0.9, 0.8))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    text = explain(logical, plan)
    assert "Scan reviews as r" in text
    assert "order=by_cost" in text
    assert "FilterChain" in text
    assert isinstance(plan, PhysicalPlan)
    assert plan.to_json()      # JSON-able

    refusal = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                         doc_tokens={"r": [150_000]})
    assert "refusal: suffix_over_chunk" in explain(logical, refusal)


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


def test_filter_keep_makes_the_join_anchor_resident(catalog):
    from quail.logical import join_label, render_join_frame

    logical = _filtered_join(catalog)
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [300] * 50, "p": [5] * 20})
    chain = filter_chain(plan)
    # one stage would normally turn arena writes off; the keep needs
    # them
    assert chain["arena_writes"] is True
    assert chain["keep_kv"] is True
    assert chain["keep_min_doc_tokens"] == 1     # everything fits
    group = plan.nodes_by_op("JoinGroup")[0]
    assert group["anchor"] == "r"
    assert group["anchor_resident"] == "filter"
    assert group["keep_anchor_kv"] is False    # nothing consumes r later

    # resident anchors pay the frame only - no preamble, no document
    pred = logical.root.input.predicate
    frame = len(tok(render_join_frame(pred.template, 0)))
    label = len(tok(join_label(1)))
    tail = len(tok(pred.tail))
    live = 50 * 0.5
    expect = live * frame + live * 20 * (5 + label + tail)
    stage = group["stages"][0]
    assert stage["anchor_resident"] == "filter"
    assert stage["tuple_tokens"] == pytest.approx(expect)
    assert any("keep KV on 'r'" in r for r in plan.remarks)


def test_unfiltered_anchor_is_not_resident(catalog):
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.1)
               .select("r.id"))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [300] * 50, "p": [5] * 20})
    group = plan.nodes_by_op("JoinGroup")[0]
    assert group["anchor_resident"] == "none"
    assert not any("keep KV" in r for r in plan.remarks)


def test_keep_split_keeps_longest_documents_that_fit():
    from quail.planner.decide import keep_split

    docs_tok = [100, 200, 300, 400]
    # page-rounded costs at overhead 4, 16-token pages: 416 + 304
    split = keep_split(docs_tok, 720, 1.0, overhead=4, page_tokens=16)
    assert split["min_doc_tokens"] == 300
    assert split["kept_expected_tokens"] == 720

    # survival halves the expected mass, so half the budget keeps the
    # same documents
    half = keep_split(docs_tok, 360, 0.5, overhead=4, page_tokens=16)
    assert half["min_doc_tokens"] == 300
    assert half["kept_expected_tokens"] == 360

    # everything fits -> threshold 1; nothing fits -> None
    assert keep_split(docs_tok, 1e9, 1.0, 4, 16)["min_doc_tokens"] == 1
    assert keep_split(docs_tok, 100, 1.0, 4, 16) is None


def test_keep_capped_by_arena_length_threshold(catalog):
    # 3,000-token documents fill the arena; only the long half of the
    # survivors stays resident and the remark names the threshold
    toks = {"r": [3000] * 100 + [1000] * 100, "p": [5] * 20}
    plan = plan_query(_filtered_join(catalog, doc_sel=1.0),
                      model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    chain = filter_chain(plan)
    assert chain["keep_kv"] is True
    assert chain["keep_min_doc_tokens"] == 3000
    assert any("survivors of 3000+ tokens" in r for r in plan.remarks)
    group = plan.nodes_by_op("JoinGroup")[0]
    assert group["anchor_resident"] == "filter"


def test_gate_group_retains_anchor_for_runtime_replan(catalog):
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
    groups = plan.nodes_by_op("JoinGroup")
    assert len(groups) == 2
    assert [g["anchor"] for g in groups] == ["r", "r"]
    assert groups[0]["keep_anchor_kv"] is True
    assert groups[1]["keep_anchor_kv"] is False
    assert groups[1]["anchor_resident"] == "none"


def test_search_matches_complete_left_deep_enumeration(catalog, tmp_path):
    import itertools as it

    from quail.planner import sol
    from quail.planner.decide import _collect, join_specs
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
    _, _, joins = _collect(logical)
    specs = join_specs(joins)
    live0 = {a: float(len(t)) for a, t in toks.items()}
    pre = 1
    chunk = 10_000
    arena = 2_500

    def key(work):
        seconds = sol.speed_of_light(work, QWEN3_4B_FP8, H100_SXM,
                                     chunk).seconds
        return (seconds, work.tokens, work.pairs, work.kv_written,
                work.kv_read)

    found = search_joins(specs, live0, toks, {}, pre, chunk,
                         QWEN3_4B_FP8, H100_SXM,
                         arena_tokens=arena, page_tokens=16)

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
                    QWEN3_4B_FP8, H100_SXM,
                    arena_tokens=arena, page_tokens=16)
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


def test_join_search_prices_current_partial_document_residency():
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
        specs, live, lengths, {"a": {1, 2}}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM, fixed_order=True)
    with_capacity = search_joins(
        specs, live, lengths, {"a": {1, 2}}, 10, 100_000,
        QWEN3_4B_FP8, H100_SXM, fixed_order=True,
        arena_tokens=41 * 16, page_tokens=16)

    assert current["records"][0]["resident_docs"] == 2
    assert current["records"][2]["resident_docs"] == 0
    assert with_capacity["records"] == current["records"]
    assert with_capacity["work"] == current["work"]


def test_join_replan_starts_from_already_joined_aliases():
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


def test_aggregate_join_work_matches_per_document_sum():
    import pytest

    from quail.planner import sol
    from quail.planner.joins import stage_work, summarize_alias
    from quail.planner.sol import Work

    spec = _join_search_spec(0, ("a", "b"), "a")
    live = {"a": 2.25, "b": 3.5}
    raw = {"a": [90, 100, 110], "b": [30, 50, 70, 90]}
    stats = {
        "a": summarize_alias(raw["a"], {1, 2}),
        "b": summarize_alias(raw["b"]),
    }

    def per_document(same_group):
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
            start = (sol.ask(prefix, frame)
                     if same_group or position in {1, 2}
                     else sol.scan(prefix, frame))
            stream = Work(
                tokens=per_anchor * suffix,
                pairs=per_anchor * (
                    suffix * (prefix + frame) + sol.triangle(suffix)),
                kv_written=per_anchor * suffix,
                kv_read=prefix + frame,
            )
            total = total + (start + stream) * fraction
        return total

    for same_group in (False, True):
        actual = stage_work(
            spec, "a", live, stats, 10, resident_at_start=True,
            same_group=same_group)
        expected = per_document(same_group)
        assert actual.tokens == pytest.approx(expected.tokens)
        assert actual.pairs == pytest.approx(expected.pairs)
        assert actual.kv_written == pytest.approx(expected.kv_written)
        assert actual.kv_read == pytest.approx(expected.kv_read)


def test_join_search_accepts_million_document_summaries():
    from quail.planner.joins import AliasStats, search_joins

    million = AliasStats(
        count=1_000_000,
        total=100_000_000,
        squared=10_000_000_000,
        maximum=100,
        resident_count=500_000,
        resident_total=50_000_000,
        resident_squared=5_000_000_000,
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
