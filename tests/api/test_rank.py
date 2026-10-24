"""AI.RANK: pairwise top-k over a planted comparison order."""

import pyarrow as pa
import pyarrow.parquet as pq
from helpers.oracle import Oracle, finish

import quail
from quail import ai_rank, col, prompt
from quail.catalog import Catalog, DocumentProvider
from quail.planner.plan import EngineConfig, Refusal

CRITERION = "Which row is better? {0}"


def _provider(path, columns):
    pq.write_table(pa.table(columns), path)
    return DocumentProvider.from_parquet(str(path), id_col="id")


def _session(path, ids, **columns):
    body = columns.pop("body", [str(item) for item in ids])
    stored = {"id": list(ids), "body": list(body), **columns}
    catalog = Catalog()
    catalog.register("documents", _provider(path, stored))
    session = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=str.split,
    )
    session.register("documents", catalog.get("documents"))
    return session


def _prefer_higher(ids):
    """Return a comparison that prefers the larger planted id."""

    def comparisons(spec, first, second):
        return 0.9 if ids[first] > ids[second] else 0.1

    return comparisons


def _sql(k, *, where="", partition=""):
    pred = f"WHERE {where} " if where else ""
    over = f"PARTITION BY {partition}" if partition else ""
    return (
        "SELECT d.id FROM documents d "
        f"{pred}QUALIFY AI.RANK(PROMPT('{CRITERION}', d.body)) "
        f"OVER ({over}) <= {k}"
    )


def _ids(session, sql, comparisons):
    query = session.sql(sql)
    result = finish(query, Oracle(comparisons=comparisons))
    return result.collect().column("d.id").to_pylist(), query


def test_rank_returns_the_k_best_rows_of_a_planted_order_in_rank_order(tmp_path):
    ids = [1, 2, 3, 4, 5, 6]
    with _session(tmp_path / "documents.parquet", ids) as session:
        found, query = _ids(session, _sql(3), _prefer_higher(ids))
        plan = query.plan()
    assert found == [6, 5, 4]
    assert any(node.type_name == "quail.ai_top_k" for node in plan.nodes)
    assert "comparisons" in query.explain()


def test_rank_returns_the_same_rows_when_the_input_is_shuffled(tmp_path):
    def run(order):
        with _session(tmp_path / f"{order[0]}.parquet", order) as session:
            found, _ = _ids(session, _sql(2), _prefer_higher(order))
        return found

    assert run([1, 2, 3, 4]) == run([3, 1, 4, 2]) == [4, 3]


def test_rank_never_returns_a_row_the_where_clause_removed(tmp_path):
    ids = [1, 2, 3, 4, 5]
    with _session(tmp_path / "documents.parquet", ids) as session:
        found, _ = _ids(
            session, _sql(2, where="d.id <> 5"), _prefer_higher(ids))
    assert found == [4, 3]
    assert 5 not in found


def test_rank_returns_each_partitions_k_best_rows(tmp_path):
    ids = [1, 2, 3, 4, 5, 6]
    with _session(
            tmp_path / "documents.parquet", ids,
            kind=["a", "a", "a", "b", "b", "b"]) as session:
        found, _ = _ids(
            session, _sql(2, partition="d.kind"), _prefer_higher(ids))
    assert found == [3, 2, 6, 5]


def test_rank_returns_every_row_when_the_input_has_fewer_than_k(tmp_path):
    ids = [1, 2, 3]
    with _session(tmp_path / "documents.parquet", ids) as session:
        found, _ = _ids(session, _sql(25), _prefer_higher(ids))
    assert found == [3, 2, 1]


def test_rank_breaks_a_tied_comparison_by_row_id(tmp_path):
    ids = [10, 3, 7]

    def comparisons(spec, first, second):
        return 0.5

    with _session(tmp_path / "documents.parquet", ids) as session:
        found, _ = _ids(session, _sql(3), comparisons)
    assert found == [3, 7, 10]


def test_rank_returns_the_planted_order_when_the_model_prefers_the_first_row(
        tmp_path):
    ids = [1, 2, 3]
    quality = {0: 3, 1: 2, 2: 1}

    def comparisons(spec, first, second):
        # The planted order is 0.6 or 0.4, then 0.2 toward the row shown
        # first. A forward look is outside the margin and is trusted; a
        # reversed look is inside it and is averaged with the forward one.
        base = 0.6 if quality[first] > quality[second] else 0.4
        return base + 0.2

    with _session(tmp_path / "documents.parquet", ids) as session:
        found, _ = _ids(session, _sql(3), comparisons)
    assert found == [1, 2, 3]


def test_rank_returns_the_same_rows_on_two_runs_when_answers_are_inconsistent(
        tmp_path):
    ids = [1, 2, 3]
    wins = {(0, 1), (1, 2), (2, 0)}

    def comparisons(spec, first, second):
        return 0.9 if (first, second) in wins else 0.1

    with _session(tmp_path / "documents.parquet", ids) as session:
        query = session.sql(_sql(2))
        oracle = Oracle(comparisons=comparisons)
        first = finish(query, oracle).collect().column("d.id").to_pylist()
        second = finish(query, oracle).collect().column("d.id").to_pylist()
    assert first == second
    assert len(first) == 2


def test_rank_without_a_rank_filter_is_refused(tmp_path):
    sql = (
        "SELECT d.id, AI.RANK(PROMPT('" + CRITERION + "', d.body)) "
        "OVER () AS r FROM documents d"
    )
    with _session(tmp_path / "documents.parquet", [1, 2, 3]) as session:
        plan = session.sql(sql).plan()
        built = (
            session.docs("documents").alias("d")
            .window(r=ai_rank(prompt(CRITERION, col("d.body"))).over())
            .select("d.id", "r")
            .plan()
        )
    assert isinstance(plan, Refusal)
    assert "filter" in plan.reasons[0]
    assert isinstance(built, Refusal)
    assert "filter" in built.reasons[0]


def test_rank_in_sql_and_the_builder_return_the_same_rows(tmp_path):
    ids = [1, 2, 3, 4]
    criterion = prompt(CRITERION, col("d.body"))
    with _session(tmp_path / "documents.parquet", ids) as session:
        sql_ids, _ = _ids(session, _sql(2), _prefer_higher(ids))
        built = (
            session.docs("documents").alias("d")
            .window(r=ai_rank(criterion).over())
            .where(col("r") <= 2)
            .select("d.id")
        )
        shortcut = (
            session.docs("documents").alias("d")
            .ai_topk(criterion, k=2)
            .select("d.id")
        )
        oracle = _prefer_higher(ids)
        built_ids = finish(
            built, Oracle(comparisons=oracle)).collect().column(
                "d.id").to_pylist()
        shortcut_ids = finish(
            shortcut, Oracle(comparisons=oracle)).collect().column(
                "d.id").to_pylist()
    assert sql_ids == built_ids == shortcut_ids == [4, 3]
