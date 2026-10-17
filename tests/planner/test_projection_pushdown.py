"""Projection and filter pushdown rule tests."""

from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.sessions import _run, fake_tok, make_executor

import quail
from quail.catalog import Catalog
from quail.frontend.builder import docs
from quail.frontend.expressions import col, prompt
from quail.logical import (
    Apply,
    ColumnRef,
    Filter,
    LogicalPlan,
    Scan,
    SemanticClassify,
    SemanticJoin,
)
from quail.physical import Recombine
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.tree_rewrites import logical_alternatives
from quail.planner.plan import EngineConfig
from quail.planner.rules import built_in_rules
from quail.planner.rules.tree_transforms import (
    push_down_filters,
    push_down_projection,
)


def _scans(plan):
    return {node.alias: node for node in plan.walk()
            if isinstance(node, Scan)}


def _session(tmp_path, tokenizer=fake_tok):
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                     device="h100-sxm"),
        tokenizer=tokenizer,
    )
    path = tmp_path / "reviews.parquet"
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2"],
        "review": ["good one", "bad one", "fine one"],
        "stars": [5, 1, 3],
        "wide": ["x" * 50, "y" * 50, "z" * 50],
    }), str(path))
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(path), id_col="id"))
    return session


@pytest.mark.parametrize("kind", ["per_batch", "barrier"])
@pytest.mark.parametrize("nested", [False, True])
def test_apply_retains_document_values_without_returning_them(kind, nested):
    seen = []
    source = "payload" if nested else "body"
    expression = col(f"r.{source}")
    if nested:
        expression = expression.field("body")
    value_name = "payload.body" if nested else "body"

    def deduplicate(rows):
        values = rows["r"].column(value_name)
        seen.extend(values.to_pylist())
        return pa.table({"id": rows["r"].ids["r"], "text": values}).group_by(
            "text").aggregate([("id", "min")]).column("id_min")

    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"), tokenizer=fake_tok,
    ) as session:
        texts = ["one", "one", "two"]
        values = [{"body": text} for text in texts] if nested else texts
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [10, 20, 30], source: values,
        }), id_col="id"))
        query = (session.docs("docs").alias("r")
                 .apply(deduplicate, columns=[expression], kind=kind)
                 .ai_filter(prompt("q: {0}", expression)).select("r.id"))
        result = _run(query, make_executor(
            {"r": {"q:": [1, 1, 1]}}, functions=session.registry.functions))
        assert seen == texts
        assert sorted(result.to_rows()) == [(10,), (30,)]
        assert _scans(query.logical)["r"].columns == ("id", source)


@pytest.mark.parametrize("length", [False, True])
@pytest.mark.parametrize("left_count, right_count", [(0, 0), (0, 3), (3, 0), (3, 3)])
def test_join_keys_retain_document_values_without_returning_them(
    length, left_count, right_count,
):
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"), tokenizer=fake_tok,
    ) as session:
        texts = {"a": ["one", "one", "three"][:left_count],
                 "b": ["one", "two", "three"][:right_count]}
        for alias, values in texts.items():
            session.register(alias, quail.DocumentProvider.from_table(pa.table({
                "id": pa.array(range(len(values)), pa.int64()),
                "body": pa.array(values, pa.string()),
            }), id_col="id"))
        left, right = col("a.body"), col("b.body")
        keys = (left.length(), right.length()) if length else (left, right)
        query = (session.docs("a").alias("a")
                 .join(session.docs("b").alias("b"), on=keys[0] == keys[1])
                 .ai_filter(prompt("Matches? {0} {1}", left, right))
                 .select("a.id", "b.id"))
        result = _run(query, make_executor({}, {
            ("a", "b"): lambda a, b: True, ("b", "a"): lambda b, a: True,
        }))
        expected = [(a, b) for a, first in enumerate(texts["a"])
                    for b, second in enumerate(texts["b"])
                    if (len(first) == len(second) if length else first == second)]
        assert sorted(result.to_rows()) == expected
        assert all(scan.columns == ("id", "body")
                   for scan in _scans(query.logical).values())


def test_projected_results_token_reuse_and_join_projection(tmp_path):
    reviews = ["good one", "bad one", "fine one"]
    calls = []

    def counting_tok(text):
        # Prompts are tokenized on every query; count documents only.
        if text in reviews:
            calls.append(text)
        return fake_tok(text)

    session = _session(tmp_path, tokenizer=counting_tok)
    truth = {"r": {"q:": [1, 0, 1]}}
    query = session.sql(
        "SELECT r.stars, r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")

    result = _run(query, make_executor(truth))

    assert sorted(result.to_rows()) == [(3, "r2"), (5, "r0")]
    scan = _scans(query.logical)["r"]
    assert scan.columns == ("stars", "id")
    assert push_down_projection(query.logical.root) is query.logical.root
    store = query._token_inputs["r"]
    assert store.projected_columns == ("stars", "id")
    tokenized = len(calls)
    assert tokenized > 0

    second = _run(session.sql(
        "SELECT r.stars, r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert sorted(second.to_rows()) == [(3, "fine one"), (5, "good one")]
    assert len(calls) == tokenized
    assert len(session.inputs._token_stores) == 1
    assert sorted(name for _, name in session.inputs._column_stores) == [
        "id", "review", "stars"]

    truth = {"r": {"q:": [0, 1, 0]}}

    text = _run(session.sql(
        "SELECT r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert text.to_rows() == [("bad one",)]

    star = _run(session.sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert star.columns == ["r.id", "r.review", "r.stars", "r.wide"]
    assert star.to_rows() == [("r1", "bad one", 1, "y" * 50)]
    assert len(calls) == tokenized
    session.close()

    session = _session(tmp_path)
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["product 0", "product 1"],
    }), id_col="asin"))
    truth = {"r": {"q:": [1, 0, 1]}}
    # the planner picks the anchor; the rule is symmetric so either works
    even_sum = lambda a, b: (a + b) % 2 == 0  # noqa: E731
    join_truth = {("r", "p"): even_sum, ("p", "r"): even_sum}
    query = session.sql(
        "SELECT r.id, p.asin FROM reviews r JOIN products p ON "
        "AI_FILTER(PROMPT('m {0} {1}', r.review, p.description)) "
        "WHERE AI_FILTER(PROMPT('q: {0}', r.review))")
    result = _run(query, make_executor(truth, join_truth))
    assert not [n for n in result.plan.nodes if isinstance(n, Recombine)]
    assert sorted(result.to_rows()) == [("r0", "p0"), ("r2", "p0")]
    assert result.count() == 2
    session.close()


class _UnstableProvider:
    id_col = "id"
    columns = ("id", "body", "stars")

    def __init__(self):
        self.scans = 0

    def schema(self):
        return pa.schema({"id": pa.string(), "body": pa.string(),
                          "stars": pa.int64()})

    def content_identity(self):
        return "unstable"

    def statistics(self):
        from quail.catalog import TableStatistics
        return TableStatistics(row_count=3)

    def scan(self, request):
        self.scans += 1
        rows = 3 if self.scans == 1 else 2
        table = pa.table({
            "id": [f"r{i}" for i in range(rows)],
            "body": ["one two"] * rows,
            "stars": list(range(rows)),
        }).select(list(request.columns))
        return table.to_reader()


def test_failed_scans_and_tokenization_leave_no_partial_cache(tmp_path):
    session = _session(tmp_path)
    session.register("docs", _UnstableProvider())
    session.inputs.tokenize("docs", "body", ("id",))

    with pytest.raises(RuntimeError, match="stable order"):
        session.inputs.tokenize("docs", "body", ("stars",))
    with pytest.raises(RuntimeError, match="stable order"):
        session.inputs.tokenize("docs", "body", ("stars",))

    assert ("unstable", "stars") not in session.inputs._column_stores
    assert len(list(Path(session.inputs._token_directory.name).iterdir())) == 2
    session.close()

    def broken(text):
        raise ValueError("tokenizer failed")

    session = _session(tmp_path)
    session._tok = broken

    with pytest.raises(ValueError, match="tokenizer failed"):
        session.inputs.tokenize("reviews", "review", ("id",))

    assert session.inputs._token_stores == {}
    assert session.inputs._column_stores == {}
    assert session.inputs._token_directory is None or not list(
        Path(session.inputs._token_directory.name).iterdir())
    session.close()


def _pair_catalog():
    catalog = Catalog()
    catalog.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": ["r0", "r1"], "review": ["good one", "bad one"],
                  "stars": [5, 1]}), id_col="id"))
    catalog.register("products", quail.DocumentProvider.from_table(
        pa.table({"asin": ["p0"], "description": ["a thing"]}),
        id_col="asin"))
    return catalog


def _joined(catalog, semantics="full"):
    return (docs(catalog, "reviews").alias("r")
            .ai_join(docs(catalog, "products").alias("p"),
                     prompt("same {0} {1}", col("r.review"),
                            col("p.description")),
                     semantics=semantics)
            .select("r.id", "p.asin"))


def _stars_apply(node):
    return Apply(node, function="keep", kind="per_batch", ids="drop",
                 columns=(ColumnRef("r", "reviews", "stars"),),
                 aliases=("r",))


def test_filter_pushdown_moves_cheap_one_table_filters_below_joins():
    catalog = _pair_catalog()
    plan = _joined(catalog)
    joined = plan.root.input
    assert isinstance(joined, SemanticJoin)

    above = replace(plan.root, input=_stars_apply(joined))
    assert push_down_filters(above) is None
    LogicalPlan(above).validate()
    candidate, = logical_alternatives(
        LogicalPlan(above), built_in_rules(),
        PlanningContext(catalog=catalog))
    optimized, changed = candidate.plan, candidate.rules
    assert changed == ("should_push_down_projection",)
    assert optimized.root == push_down_projection(above)

    # both front ends place the filter on its table, so the rule
    # leaves their plans alone
    assert push_down_filters(plan.root) is None

    # a filter on a label returns to its table, above the
    # classification that computes the label
    labelled = (docs(catalog, "reviews").alias("r")
                .ai_classify(prompt("topic of {0}", col("r.review")),
                             ["a", "b"], name="topic")
                .label_in("topic", ["a"])
                .ai_join(docs(catalog, "products").alias("p"),
                         prompt("same {0} {1}", col("r.review"),
                                col("p.description")))
                .select("r.id", "p.asin", "topic"))
    joined = labelled.root.input
    label_filter = joined.input.left
    assert isinstance(label_filter, Filter)
    assert isinstance(label_filter.input, SemanticClassify)
    lifted = replace(labelled.root, input=replace(
        label_filter, input=replace(joined, input=replace(
            joined.input, left=label_filter.input))))
    LogicalPlan(lifted).validate()
    assert push_down_filters(lifted) == labelled.root

    # a gate's inner table and an apply returning pairs keep their
    # filters above them
    gate = _joined(catalog, semantics="exists")
    above = replace(gate.root, input=_stars_apply(gate.root.input))
    assert push_down_filters(above) is None
    pairs = replace(joined, input=Apply(
        joined.input, function="pairs", kind="barrier", ids="pairs",
        columns=(), aliases=("r", "p"), written_pos=0))
    above = replace(labelled.root, input=_stars_apply(pairs))
    assert push_down_filters(above) is None
