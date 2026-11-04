"""Planner ordering, anchor selection, sharding, break-evens, and refusals."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.planning import _optimize, _plan, compile_sql, node_kinds

from quail.catalog import Catalog, DocumentProvider
from quail.common.sol import speed_of_light
from quail.common.work import ask, scan, stream
from quail.explain import explain
from quail.logical import SemanticJoin
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Project,
    Scan,
)
from quail.physical.codec import encode_fields
from quail.planner.optimizer.rules import LogicalRule, PhysicalRule
from quail.planner.plan import Refusal
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_32B_FP8


def filter_sequence(plan, alias=None):
    return next(n for n in plan.nodes if isinstance(n, AiFilter)
                and (alias is None or n.alias == alias))


def join_stages(plan):
    return [encode_fields(node, exclude=("node_id", "inputs"))
            for node in plan.nodes if isinstance(node, AiJoin)]


def _sink_source(plan):
    sink = next(node for node in plan.nodes if isinstance(node, Project))
    return sink.inputs[0].source


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


def _five_filter_plan(catalog, sels):
    filters = " AND ".join(
        f"AI_FILTER(PROMPT('flag {j} of: {{0}}', r.review)"
        + ("" if s is None else f", {{'selectivity': {s}}}") + ")"
        for j, s in enumerate(sels))
    return compile_sql(f"SELECT r.id FROM reviews r WHERE {filters}", catalog)


def test_gpu_copies_and_memory_refusals(catalog):
    from quail.common.shards import contiguous_shards

    assert contiguous_shards([2, 2, 2], 2) == (((0, 1), (1, 3)), [2, 4])
    assert contiguous_shards([1, 0, 9], 2) == (((0, 1), (1, 3)), [1, 9])
    assert contiguous_shards([0, 2, 0, 4], 3) == (
        ((0, 2), (2, 4), (4, 4)), [2, 4, 0])
    assert contiguous_shards([], 2) == (((0, 0), (0, 0)), [0, 0])
    logical = _five_filter_plan(catalog, (0.5,))
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        for gpus in (1, 2, 4, 8):
            plan = _plan(logical, {"r": [100] * 8}, model=model, gpus=gpus)
            assert plan.model == model.name
            assert plan.gpus == gpus
            scan_node = next(node for node in plan.nodes if isinstance(node, Scan))
            assert len(scan_node.shards) == gpus

    big = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    logical = _five_filter_plan(catalog, (0.9,))
    for gpus in (1, 8):
        result = _plan(logical, {"r": [100] * 10}, model=big, gpus=gpus)
        assert isinstance(result, Refusal)
        assert result.constraint == "weights_need_more_cards"
        assert result.needed > result.available

    # a document too long for one pass is skipped at execution, not refused
    assert not isinstance(_plan(logical, {"r": [150_000]}), Refusal)


def test_filter_ordering_and_kv_writes(catalog):
    # the B3 shape: a 0.2-selectivity filter written third among five
    sels = (0.9, 0.9, 0.2, 0.9, 0.9)
    logical = _five_filter_plan(catalog, sels)
    tokens = {"r": [400] * 100}

    by_cost = _plan(logical, tokens)
    sequence = filter_sequence(by_cost)
    assert by_cost.settings["order_rule"] == "by_cost"
    assert sequence.selectivity == 0.2
    second = by_cost.graph.nodes_by_type(AiFilter.type_name)[1]
    assert second.expected_rows == pytest.approx(20.0)
    # the physical planner chooses the order; the logical plan keeps
    # the predicates as written
    optimized, _ = _optimize(logical, tokens)
    assert sequence.written_position == 2
    assert "predicate 3" in explain(optimized, by_cost)

    as_written = _plan(logical, tokens, order="as_written")
    assert [node.selectivity
            for node in as_written.graph.nodes_by_type(AiFilter.type_name)] \
        == list(sels)

    logical = compile_sql(
        "SELECT r.id FROM reviews r "
        f"WHERE AI_FILTER(PROMPT('{'long ' * 100}{{0}}', r.review), "
        "{'selectivity': 0.1}) "
        f"AND AI_FILTER(PROMPT('{'short ' * 10}{{0}}', r.review), "
        "{'selectivity': 0.5})", catalog)
    plan = _plan(logical, {"r": [400] * 100})
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

    logical = compile_sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('a: {0}', r.review)) "
        "AND AI_FILTER(PROMPT('b: {0}', r.review), {'selectivity': 0.5})",
        catalog)
    plan = _plan(logical, {"r": [100] * 10})
    assert plan.settings["order_rule"] == "by_cost"
    assert "selectivity 0.2" in plan.settings["order_source"]


ROW_ESTIMATES = [
    ((0.5, 0.25), "12.5"), ((1.0, 0.0), "0"),
    ((None, 0.5), "10"), ((None, 0.0), "0"),
]


def test_explain_row_estimates_limits_and_verbose_fields(catalog):
    for sels, expected in ROW_ESTIMATES:
        logical = _five_filter_plan(catalog, sels)
        plan = _plan(logical, {"r": [400] * 100})
        text = explain(logical, plan)
        physical = text.split("physical:", 1)[1]
        assert _cell(physical, "Project: r.id") == [expected], sels
        assert _cell(physical, "AiFilter: r")[0] == expected, sels
        assert _cell(physical, "AiFilter: r")[-1] in ("ms", "s"), sels
        assert _cell(physical, "Scan reviews as r") == ["100"], sels
        assert "tokens=40,000" in physical, sels
        assert "node_id=" not in text and "arena_writes" not in text, sels
        assert "KV=bf16, chunk budget=" in text and "KV rewind=on" in text, sels
        for stage in plan.graph.nodes_by_type(AiFilter.type_name):
            predicate = logical.root.input.predicates[stage.written_position]
            assert (f"predicate {stage.written_position + 1}  "
                    f"PROMPT({predicate.prompt.template!r})") in physical, sels
        verbose = explain(logical, plan, verbose=True)
        assert "node_id=ai_filter:r" in verbose, sels
        assert "admission_tokens=" in verbose and "expected_rows=" in verbose, sels

    logical = _five_filter_plan(catalog, (0.25,))
    logical = replace(logical, root=replace(logical.result, limit=10))
    plan = _plan(logical, {"r": [400] * 100})
    physical = explain(logical, plan).split("physical:", 1)[1]
    assert _cell(physical, "Limit: 10") == ["10"]
    assert _cell(physical, "Project: r.id") == ["25"]
    assert "KV: not stored" in physical

    rules = (LogicalRule("should_eliminate_distinct"),
             PhysicalRule("should_push_down_limit"), PhysicalRule("should_retain_kv"))
    assert "rules" not in explain(logical, plan, rules=rules)
    text = explain(logical, plan, verbose=True, rules=rules)
    assert "  logical: should_eliminate_distinct" in text
    assert "  physical: should_push_down_limit, should_retain_kv" in text
    assert ("refine each graph in order: should_push_down_limit, "
            "should_retain_kv") in text


def mask_pairs(prefix, suffix, window):
    """Count allowed keys for every new query position."""
    return sum(1 for q in range(prefix, prefix + suffix) for k in range(q + 1)
               if not window or q - k < window)


def test_work_matches_attention_masks_and_component_costs():
    from quail.common.dense_decoder_cost import (
        dense_decoder_components,
        dense_params,
    )
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    for window, prefix, suffix in ((0, 2, 4), (1, 9, 3), (4, 0, 3),
                                  (4, 2, 4), (4, 4, 1), (4, 9, 3)):
        first = scan(prefix, suffix, window=window)
        continuation = ask(prefix, suffix, window=window)
        assert first.pairs == mask_pairs(0, prefix + suffix, 0)
        assert continuation.pairs == mask_pairs(prefix, suffix, 0)
        assert first.sliding_pairs == (
            mask_pairs(0, prefix + suffix, window) if window else 0)
        assert continuation.sliding_pairs == (
            mask_pairs(prefix, suffix, window) if window else 0)
        assert continuation.sliding_kv_read == (
            min(prefix, window - 1) if window else 0)
        branches = stream(prefix, [suffix, suffix + 1], window=window)
        assert branches.sliding_pairs == (
            sum(mask_pairs(prefix, s, window) for s in (suffix, suffix + 1))
            if window else 0)
        assert branches.sliding_kv_read == continuation.sliding_kv_read

    work = ask(400, 50) * 1000
    result = speed_of_light(work, QWEN3_4B_FP8, H100_SXM, 110_376)
    assert result.bound_by == "mixed"
    assert [c.name for c in result.components] == [
        "attn_proj", "mlp", "attention"]
    assert result.seconds == pytest.approx(sum(
        max(c.compute_seconds, c.memory_seconds)
        for c in result.components))
    assert result.component("mlp") is result.components[1]
    assert result.seconds > max(result.compute, result.memory)
    dense_bytes = sum(
        c.bytes_moved for c in result.components if c.name != "attention")
    assert dense_bytes == dense_params(QWEN3_4B_FP8) * result.passes
    larger_resident_copy = replace(QWEN3_4B_FP8, w_mem_bytes=150e9)
    assert speed_of_light(
        work, larger_resident_copy, H100_SXM, 110_376).seconds \
        == pytest.approx(result.seconds)

    model = DIFFUSION_GEMMA_26B_FP8
    work = ask(4096, 16, window=model.sliding_window)
    attention = dense_decoder_components(work, model, 1)[2]
    assert attention.flops == 4 * 16 * (
        5 * 512 * work.pairs + 25 * 256 * work.sliding_pairs)
    assert attention.bytes_moved == (
        5 * 2 * 2 * 512 * 2 * (16 + 4096)
        + 25 * 2 * 8 * 256 * 2 * (16 + 1023))
    assert work + work == work * 2
    assert not work.dominates(replace(work, sliding_pairs=work.sliding_pairs - 1))
    assert not work.dominates(replace(work, sliding_kv_read=1022))


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
    logical = compile_sql("""
        SELECT r.id FROM reviews r JOIN products p JOIN threads t
        ON AI_FILTER(PROMPT('m {0} {1} {2}', r.review, p.description, t.thread),
                     {'selectivity': 0.02})
    """, catalog)
    tokens = {"r": [3000] * 20, "p": [100] * 30, "t": [50] * 40}
    plan = _plan(logical, tokens)
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
    # logical plan keeps the joins as written
    optimized, _ = _optimize(_join_sequence(catalog), tokens, order="as_written")
    joins = [node for node in optimized.walk() if isinstance(node, SemanticJoin)]
    assert [join.anchor for join in joins] == [None, None]
    assert [(s["exec_index"], s["anchor"]) for s in stages] == [(0, "r"), (1, "p")]
    text = explain(optimized, plan)
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
    logical = compile_sql("""
        SELECT r.id FROM reviews r
        JOIN products p ON AI_FILTER(PROMPT('m {0} {1}', r.review, p.description),
                                     {'selectivity': 0.9})
        WHERE EXISTS (SELECT 1 FROM threads t
                      WHERE AI_FILTER(PROMPT('m {0} {1}', r.review, t.thread),
                                      {'selectivity': 0.01}))
    """, catalog)
    tokens = {"r": [400] * 100, "p": [100] * 100, "t": [100] * 100}
    plan = _plan(logical, tokens)
    assert [(s["written_position"], s["exec_index"], s["anchor"])
            for s in join_stages(plan)] == [(1, 0, "r"), (0, 1, "r")]

    as_written = _plan(logical, tokens, order="as_written")
    assert join_stages(as_written)[0]["selectivity"] == 0.9

    assert _sink_source(plan).node_id == "recombine"
    groups = plan.graph.nodes_by_type(AiJoin.type_name)
    assert len(groups) == 2
    assert [group.anchor for group in groups] == ["r", "r"]
    assert groups[0].keep_kv is True
    assert groups[1].keep_kv is False
    assert groups[1].anchor_resident == "kept"
