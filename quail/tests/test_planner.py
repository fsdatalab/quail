"""The planner's decisions: ordering, anchor, sharding, the two
break-evens, and the refusals - all CPU, no GPU anywhere."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quail.builder import col, docs, prompt
from quail.catalog import Catalog, DocumentProvider
from quail.planner.calibration import load_calibration
from quail.planner.decide import (balanced_shards, choose_kv_dtype,
                                  explain, order_filters, plan_query,
                                  restore_crossover_tokens)
from quail.planner.plan import (EngineConfig, PhysicalPlan, Refusal,
                                StoreSpec, resolve_model)
from quail.specs import H100_SXM, QWEN3_4B_FP8


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


CAL = load_calibration(QWEN3_4B_FP8, H100_SXM)


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
    chain = next(op for op in by_cost.operators
                 if op["op"] == "FilterChain")
    assert by_cost.order_rule == "by_cost"     # every predicate has a sel
    assert chain["stages"][0]["selectivity"] == 0.2
    # the other four only see the 20% it passes
    assert chain["stages"][1]["expected_docs"] == pytest.approx(20.0)

    as_written = plan_query(logical, model=QWEN3_4B_FP8,
                            device=H100_SXM, doc_tokens=toks,
                            order="as_written")
    chain = next(op for op in as_written.operators
                 if op["op"] == "FilterChain")
    assert [s["selectivity"] for s in chain["stages"]] == list(sels)


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


def test_order_filters_ties_keep_written_order():
    class P:
        def __init__(self, tail, sel):
            self.prompt = type("Pr", (), {"tail_tokens": tail,
                                          "preamble_tokens": 0})()
            self.selectivity = sel
    a, b = P(10, 0.5), P(10, 0.5)
    assert order_filters([a, b], "by_cost") == [a, b]
    # selectivity 1 kills nothing: last
    c = P(1, 1.0)
    assert order_filters([c, a], "by_cost") == [a, c]


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
    stage = next(op for op in plan.operators if op["op"] == "JoinStage")
    assert stage["anchor"] == "r"      # the longer side anchors
    assert stage["partners"] == ["p"]

    forced = plan_query(joined("p"), model=QWEN3_4B_FP8,
                        device=H100_SXM, doc_tokens=toks)
    stage = next(op for op in forced.operators
                 if op["op"] == "JoinStage")
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
    stage = next(op for op in plan.operators if op["op"] == "JoinStage")
    # every tuple of the cross product is one evaluation
    assert stage["expected_tuples"] == 20 * 30 * 40
    # the longest side anchors; partners keep placeholder order
    assert stage["anchor"] == "r"
    assert stage["partners"] == ["p", "t"]


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
    stages = [op for op in plan.operators if op["op"] == "JoinStage"]
    assert stages[0]["selectivity"] == 0.01

    as_written = plan_query(logical, model=QWEN3_4B_FP8,
                            device=H100_SXM, doc_tokens=toks,
                            order="as_written")
    stages = [op for op in as_written.operators
              if op["op"] == "JoinStage"]
    assert stages[0]["selectivity"] == 0.9


def test_balanced_shards():
    shards, loads = balanced_shards([100, 900, 500, 500], 2)
    assert sorted(loads) == [1000, 1000]
    assert sorted(i for s in shards for i in s) == [0, 1, 2, 3]


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
    chain = next(op for op in plan.operators
                 if op["op"] == "FilterChain")
    st = chain["stages"][0]
    # the shared preamble is per document (stage 0), never per stage
    assert st["preamble_tokens"] == len(tok(SHARED_PRE))
    assert st["question_tokens"] == len(tok(
        "Evaluate TRUE or FALSE for the following statement: "
        "flag 0 of: ANSWER:"))


def test_store_threshold_includes_preamble(catalog):
    # stored extents are [preamble + document] rows; a capacity that
    # holds exactly the two longest documents plus their preambles
    # yields a pre-inclusive threshold
    from quail.logical import SHARED_PRE
    logical = _five_filter_plan(catalog, (0.5,))
    pre = len(tok(SHARED_PRE))
    kappa = QWEN3_4B_FP8.with_kv_bytes(2.0).kappa
    cap = (700 + 2 * pre + 0.25) * kappa
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [100, 200, 300, 400]},
                      store=StoreSpec(read_bw=55e9, warm=False,
                                      capacity_bytes=cap))
    assert plan.store_min_doc_tokens == 300 + pre


def test_join_tokens_note_per_anchor_labels_per_tuple(catalog):
    # the anchor's naming line is written into kept KV once per
    # anchor document; a partner's block label and the rendered
    # question ride in every tuple's suffix
    from quail.logical import (SHARED_PRE, join_anchor_note,
                               join_label, render_join_question)

    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.5, anchor="r")
               .select("r.id"))
    toks = {"r": [100] * 4, "p": [10] * 3}
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks)
    stage = next(op for op in plan.operators if op["op"] == "JoinStage")
    pre = len(tok(SHARED_PRE))
    # r is placeholder 0 (the anchor's naming line), p is placeholder
    # 1 (its block label)
    note = len(tok(join_anchor_note(0)))
    label = len(tok(join_label(1)))
    pred = logical.root.input.predicate
    question = len(tok(render_join_question(pred.template)))
    expect = 4 * (100 + pre + note) + 12 * (10 + label + question)
    assert stage["tuple_tokens"] == pytest.approx(expect)
    assert stage["expected_tuples"] == 12


def test_refusal_suffix_over_chunk(catalog):
    logical = _five_filter_plan(catalog, (0.9,))
    r = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                   doc_tokens={"r": [150_000]})
    assert isinstance(r, Refusal)
    assert r.constraint == "suffix_over_chunk"


def test_refusal_store_needed_but_disabled(catalog):
    # fits a chunk but not the arena: only a store could hold it
    small_arena = replace(H100_SXM, mem_bytes=6.5e9)
    logical = _five_filter_plan(catalog, (0.9,))
    r = plan_query(logical, model=QWEN3_4B_FP8, device=small_arena,
                   doc_tokens={"r": [8_000]})
    assert isinstance(r, Refusal)
    assert r.constraint == "store_needed_but_disabled"


def test_refusal_unknown_model():
    r = resolve_model("qwen9-13b")
    assert isinstance(r, Refusal)
    assert r.constraint == "unknown_model"


def test_access_read_when_cold_restore_when_warm(catalog):
    logical = _five_filter_plan(catalog, (0.9,))
    toks = {"r": [400] * 100}
    cold = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks,
                      store=StoreSpec(read_bw=55e9, warm=False))
    scan = next(op for op in cold.operators if op["op"] == "DocScan")
    assert scan["access"] == "read"

    warm = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens=toks,
                      store=StoreSpec(read_bw=55e9, warm=True))
    scan = next(op for op in warm.operators if op["op"] == "DocScan")
    assert scan["access"] == "restore"     # 55 GB/s beats the break-even


def test_restore_crossover_zero_for_pinned():
    # pinned 55 GB/s beats kappa/a at every length; a slow volume
    # crosses only for long documents
    assert restore_crossover_tokens(QWEN3_4B_FP8, CAL, 55e9) == 0.0
    assert restore_crossover_tokens(QWEN3_4B_FP8, CAL, 3e9) > 10_000


def test_kv_dtype_cold_is_bf16_heavy_restore_is_fp8():
    dtype, tax, saving = choose_kv_dtype(QWEN3_4B_FP8, CAL,
                                         fresh_tokens=4e6,
                                         restored_tokens=0,
                                         store_bw=55e9)
    assert dtype == "bf16" and saving == 0.0
    dtype, tax, saving = choose_kv_dtype(QWEN3_4B_FP8, CAL,
                                         fresh_tokens=1e5,
                                         restored_tokens=50e6,
                                         store_bw=55e9)
    assert dtype == "fp8" and saving > tax


def test_admission_uses_chosen_dtype(catalog):
    logical = _five_filter_plan(catalog, (0.9,))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100}, kv_dtype="fp8")
    bf16 = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100}, kv_dtype="bf16")
    assert plan.kv_dtype == "fp8"
    assert plan.admission_tokens == pytest.approx(
        2 * bf16.admission_tokens, rel=0.01)


def test_explain_prints_tree_settings_and_source(catalog):
    logical = _five_filter_plan(catalog, (0.9, 0.8))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [400] * 100})
    text = explain(logical, plan)
    assert "Scan reviews as r" in text
    assert "order=by_cost" in text
    assert "calibration: calibrated" in text
    assert "FilterChain" in text
    assert isinstance(plan, PhysicalPlan)
    assert plan.to_json()      # JSON-able

    refusal = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                         doc_tokens={"r": [150_000]})
    assert "refusal: suffix_over_chunk" in explain(logical, refusal)


def test_engine_config_defaults():
    cfg = EngineConfig()
    assert (cfg.gpus, cfg.cpu_memory_gb, cfg.model) == \
        (1, 64, "qwen3-4b-fp8")
