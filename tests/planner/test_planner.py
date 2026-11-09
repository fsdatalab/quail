"""Planner ordering, anchor selection, sharding, break-evens, and refusals."""

import json
from dataclasses import replace

import _quail_native
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.planning import _plan, compile_sql, node_kinds, read_operators

from quail.catalog import Catalog, DocumentProvider
from quail.explain import explain as explain_plan
from quail.extensions import LogicalRule, PhysicalRule
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
)
from quail.physical.plan import Refusal
from quail.specs import QWEN3_4B_FP8, QWEN3_32B_FP8


def filter_sequence(plan, alias=None):
    return next(n for n in plan.nodes if isinstance(n, AiFilter)
                and (alias is None or n.alias == alias))


def join_stages(plan):
    return [json.loads(node.to_json())["attributes"]
            for node in plan.nodes if isinstance(node, AiJoin)]


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


def _cell(text, title):
    for line in text.splitlines():
        if line.strip().startswith(title):
            return line.split(title, 1)[1].split()
    raise AssertionError(f"no row starts with {title!r}")


def explain(operators, plan, **options):
    """Return the explain text of a region's operators and physical plan."""
    return explain_plan("", operators, plan, **options)


def _five_filter_plan(catalog, sels):
    filters = " AND ".join(
        f"AI_FILTER(PROMPT('flag {j} of: {{0}}', r.review)"
        + ("" if s is None else f", {{'selectivity': {s}}}") + ")"
        for j, s in enumerate(sels))
    return compile_sql(f"SELECT r.id FROM reviews r WHERE {filters}", catalog)


def test_gpu_copies_and_memory_refusals(catalog):
    operators = _five_filter_plan(catalog, (0.5,))
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        for gpus in (1, 2, 4, 8):
            plan = _plan(operators, {"r": [100] * 8}, model=model, gpus=gpus)
            assert plan.model == model.name
            assert plan.gpus == gpus

    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    operators = _five_filter_plan(catalog, (0.9,))
    for gpus in (1, 8):
        result = _plan(operators, {"r": [100] * 10}, model=big, gpus=gpus)
        assert isinstance(result, Refusal)
        assert result.constraint == "weights_need_more_cards"
        assert result.needed > result.available

    # a document too long for one pass is skipped at execution, not refused
    assert not isinstance(_plan(operators, {"r": [150_000]}), Refusal)


def test_filter_ordering_and_kv_writes(catalog):
    # the B3 shape: a 0.2-selectivity filter written third among five
    sels = (0.9, 0.9, 0.2, 0.9, 0.9)
    operators = _five_filter_plan(catalog, sels)
    tokens = {"r": [400] * 100}

    by_cost = _plan(operators, tokens)
    sequence = filter_sequence(by_cost)
    assert by_cost.settings["order_rule"] == "by_cost"
    assert sequence.selectivity == 0.2
    second = by_cost.graph.nodes_by_type(AiFilter.type_name)[1]
    assert second.expected_rows == pytest.approx(20.0)
    # the physical planner chooses the order; the operators keep the
    # predicates as written
    assert sequence.written_position == 2
    assert "predicate 3" in explain(operators, by_cost)

    as_written = _plan(operators, tokens, order="as_written")
    assert [node.selectivity
            for node in as_written.graph.nodes_by_type(AiFilter.type_name)] \
        == list(sels)

    operators = compile_sql(
        "SELECT r.id FROM reviews r "
        f"WHERE AI_FILTER(PROMPT('{'long ' * 100}{{0}}', r.review), "
        "{'selectivity': 0.1}) "
        f"AND AI_FILTER(PROMPT('{'short ' * 10}{{0}}', r.review), "
        "{'selectivity': 0.5})", catalog)
    plan = _plan(operators, {"r": [400] * 100})
    assert [node.written_position
            for node in plan.graph.nodes_by_type(AiFilter.type_name)] == [1, 0]

    # one stage: nothing reads the KV again
    single = _five_filter_plan(catalog, (0.9,))
    plan = _plan(single, {"r": [400] * 100})
    sequence = filter_sequence(plan)
    assert sequence.arena_writes is False
    assert "arena_writes=False" in explain(single, plan, verbose=True)

    plan = _plan(_five_filter_plan(catalog, (0.9, 0.9)), {"r": [400] * 100})
    sequence = filter_sequence(plan)
    assert sequence.arena_writes is True

    operators = compile_sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('a: {0}', r.review)) "
        "AND AI_FILTER(PROMPT('b: {0}', r.review), {'selectivity': 0.5})",
        catalog)
    plan = _plan(operators, {"r": [100] * 10})
    assert plan.settings["order_rule"] == "by_cost"
    assert "selectivity 0.2" in plan.settings["order_source"]


ROW_ESTIMATES = [
    ((0.5, 0.25), "12.5"), ((1.0, 0.0), "0"),
    ((None, 0.5), "10"), ((None, 0.0), "0"),
]


def test_explain_row_estimates_limits_and_verbose_fields(catalog):
    for sels, expected in ROW_ESTIMATES:
        operators = _five_filter_plan(catalog, sels)
        plan = _plan(operators, {"r": [400] * 100})
        text = explain(operators, plan)
        physical = text.split("physical:", 1)[1]
        assert _cell(physical, "AiFilter: r")[0] == expected, sels
        assert _cell(physical, "AiFilter: r")[-1] in ("ms", "s"), sels
        assert _cell(physical, "Scan reviews as r") == ["100"], sels
        assert "tokens=40,000" in physical, sels
        assert "node_id=" not in text and "arena_writes" not in text, sels
        assert "KV=bf16, chunk budget=" in text and "KV rewind=on" in text, sels
        prompts = json.loads(_quail_native.region_facts(operators))["prompts"]
        for stage in plan.graph.nodes_by_type(AiFilter.type_name):
            prompt = prompts["filters"]["r"][stage.written_position]
            assert (f"predicate {stage.written_position + 1}  "
                    f"PROMPT({prompt['template']!r})") in physical, sels
        verbose = explain(operators, plan, verbose=True)
        assert "node_id=ai_filter:r" in verbose, sels
        assert "admission_tokens=" in verbose and "expected_rows=" in verbose, sels

    operators = _five_filter_plan(catalog, (0.25,))
    operators = _quail_native.limited_operators(operators, 10)
    plan = _plan(operators, {"r": [400] * 100})
    physical = explain(operators, plan).split("physical:", 1)[1]
    assert _cell(physical, "Limit: 10") == ["10"]
    assert "KV: not stored" in physical

    rules = (LogicalRule("eliminate_distinct"),
             PhysicalRule("should_push_down_limit"), PhysicalRule("should_retain_kv"))
    assert "rules" not in explain(operators, plan, rules=rules)
    text = explain(operators, plan, verbose=True, rules=rules)
    assert "  logical: eliminate_distinct" in text
    assert "  physical: should_push_down_limit, should_retain_kv" in text
    assert ("refine each graph in order: should_push_down_limit, "
            "should_retain_kv") in text


def _join_sequence(catalog, sels=(0.1, 0.1), anchors=(None, None)):
    def options(selectivity, anchor):
        anchored = "" if anchor is None else f", 'anchor': '{anchor}'"
        return f"{{'selectivity': {selectivity}{anchored}}}"

    return compile_sql(f"""
        SELECT r.id, t.id, p.asin FROM reviews r
        JOIN threads t ON AI_FILTER(PROMPT('m1 {{0}} {{1}}', r.review, t.thread),
                                    {options(sels[0], anchors[0])})
        JOIN products p ON AI_FILTER(PROMPT('m2 {{0}} {{1}}', t.thread,
                                            p.description),
                                     {options(sels[1], anchors[1])})
    """, catalog)


def test_join_anchors_groups_forced_order_and_later_kv_reuse(catalog):
    operators = compile_sql("""
        SELECT r.id FROM reviews r CROSS JOIN products p JOIN threads t
        ON AI_FILTER(PROMPT('m {0} {1} {2}', r.review, p.description, t.thread),
                     {'selectivity': 0.02})
    """, catalog)
    tokens = {"r": [3000] * 20, "p": [100] * 30, "t": [50] * 40}
    plan = _plan(operators, tokens)
    stage = join_stages(plan)[0]
    assert stage["expected_rows"] == 20 * 30 * 40
    assert stage["anchor"] == "r"
    assert stage["partners"] == ["p", "t"]

    # streaming r's 3,000-token documents as partners would pay them once
    # per pair, so r anchors stage 1 and p stage 2, with a barrier between
    tokens = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    plan = _plan(_join_sequence(catalog), tokens, order="as_written")
    stages = join_stages(plan)
    assert [s["anchor"] for s in stages] == ["r", "p"]
    assert [s["written_position"] for s in stages] == [0, 1]
    # the physical planner chooses the stage order and anchors; the
    # operators keep the joins as written
    written = _join_sequence(catalog)
    assert [join.anchor for join in read_operators(written).joins] == [None, None]
    assert [(s["exec_index"], s["anchor"]) for s in stages] == [(0, "r"), (1, "p")]
    text = explain(written, plan)
    assert "AiJoin: anchor=r" in text and "AiJoin: anchor=p" in text
    assert stages[0]["partners"] == stages[1]["partners"] == ["t"]
    assert stages[0]["expected_rows"] == 10 * 8
    assert stages[1]["expected_rows"] < 8 * 6
    kinds = node_kinds(plan)
    assert (kinds.count("AiJoin"), kinds.count("Barrier")) == (2, 1)
    barrier = plan.graph.nodes_by_type(Barrier.type_name)[0]
    assert barrier.next_anchor == "p"
    assert set(barrier.aliases) == {"r", "t", "p"}
    group2 = plan.graph.nodes_by_type(AiJoin.type_name)[1]
    assert all(input_port.source.node_id == barrier.node_id
               for input_port in group2.inputs)

    # t, shared by both joins, has the long documents: one group anchors it
    tokens = {"r": [50] * 10, "t": [3000] * 8, "p": [50] * 6}
    plan = _plan(_join_sequence(catalog), tokens, order="as_written")
    assert [s["anchor"] for s in join_stages(plan)] == ["t", "t"]
    kinds = node_kinds(plan)
    assert (kinds.count("AiJoin"), kinds.count("Barrier")) == (2, 0)

    # a forced anchor is honored
    tokens = {"r": [3000] * 10, "t": [50] * 8, "p": [100] * 6}
    plan = _plan(
        _join_sequence(catalog, anchors=("t", None)), tokens, order="as_written"
    )
    assert join_stages(plan)[0]["anchor"] == "t"
    same = _plan(
        _join_sequence(catalog, anchors=("r", None)), tokens, order="as_written"
    )
    assert join_stages(same)[0]["anchor"] == "r"

    # by_cost runs the cheap .01 exists gate before the .9 full join
    operators = compile_sql("""
        SELECT r.id FROM reviews r
        JOIN products p ON AI_FILTER(PROMPT('m {0} {1}', r.review, p.description),
                                     {'selectivity': 0.9})
        WHERE EXISTS (SELECT 1 FROM threads t
                      WHERE AI_FILTER(PROMPT('m {0} {1}', r.review, t.thread),
                                      {'selectivity': 0.01}))
    """, catalog)
    tokens = {"r": [400] * 100, "p": [100] * 100, "t": [100] * 100}
    plan = _plan(operators, tokens)
    assert [(s["written_position"], s["exec_index"], s["anchor"])
            for s in join_stages(plan)] == [(1, 0, "r"), (0, 1, "r")]

    as_written = _plan(operators, tokens, order="as_written")
    assert join_stages(as_written)[0]["selectivity"] == 0.9

    assert not [node for node in plan.nodes if node.type_name == "quail.recombine"]
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert len(groups) == 2
    assert [group.anchor for group in groups] == ["r", "r"]
    assert groups[0].keep_kv is True
    assert groups[1].keep_kv is False
    assert groups[1].anchor_resident == "kept"
