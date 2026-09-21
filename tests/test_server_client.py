"""Remote Session, submit, get_run, watch, result, and cancel end to end."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import server_fakes

import quail
from quail.planner.plan import EngineConfig
from quail.server.records import QueryFailedError, ServerError

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from quail.server.app import ServerSettings, create_app  # noqa: E402

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")


@pytest.fixture()
def endpoint(tmp_path):
    servers = []

    def start(**overrides):
        settings = ServerSettings(
            data_dir=tmp_path / f"data{len(servers)}", models=("qwen3-4b-fp8",),
            device="h100-sxm", in_process=True,
            **{"hooks": server_fakes.hooks, **overrides})
        url, stop = server_fakes.start_server(create_app(settings))
        servers.append(stop)
        return url

    yield start
    for stop in servers:
        stop()


def register_reviews(session, tmp_path):
    pq.write_table(server_fakes.reviews_table(), tmp_path / "reviews.parquet")
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))


def test_local_session_reports_what_needs_an_endpoint(tmp_path):
    with quail.Session(CONFIG, tokenizer=server_fakes.fake_tok) as session:
        assert session.endpoint is None
        register_reviews(session, tmp_path)
        query = session.sql(server_fakes.FILTER_SQL)
        with pytest.raises(RuntimeError, match="needs a Session with an endpoint"):
            query.submit()
        with pytest.raises(RuntimeError, match="needs a Session with an endpoint"):
            session.get_run("abc")


def test_submit_close_reattach_watch_and_collect(endpoint, tmp_path):
    url = endpoint()
    with quail.Session(CONFIG, endpoint=url) as session:
        assert session.endpoint == url
        register_reviews(session, tmp_path)
        session.register("aspects", quail.DocumentProvider.from_table(
            pa.table({"id": ["a0"], "aspect": ["acting"]}), id_col="id"))
        query = session.sql(server_fakes.FILTER_SQL)
        with pytest.raises(RuntimeError, match="explain"):
            query.explain()
        with pytest.raises(RuntimeError, match="plan"):
            query.plan()
        run = query.submit(query_id="demo-1")
        query_id = run.id
        assert query_id == "demo-1"
        assert query.submit(query_id="demo-1").id == query_id
        assert run.status().session_id == session.session_id
        with pytest.raises(RuntimeError, match="builder API"):
            session.docs("reviews")
    # closing the client session leaves the accepted query on the service

    with quail.Session(CONFIG, endpoint=url) as session:
        run = session.get_run(query_id)
        assert repr(run) == f"QueryRun({query_id!r})"
        snapshots = list(run.watch(poll_s=2.0))
        revisions = [status.revision for status in snapshots]
        assert revisions == sorted(revisions) and len(set(revisions)) == len(
            revisions)
        final = snapshots[-1]
        assert final.done and final.state == "succeeded"
        assert final.plan["backend"] == "quail" and "Scan" in final.plan["text"]
        assert final.progress["done"] == 6
        assert set(final.inputs) == {"reviews", "aspects"}
        assert final.inputs["reviews"]["kind"] == "snapshot"
        assert run.status() == final
        result = run.result()
        assert isinstance(result, quail.QueryResult)
        assert sorted(result.to_rows()) == [("r0",), ("r3",)]
        assert result.collect().num_rows == 2
        assert set(result.answer_tables["filters"]) == {("r", 0), ("r", 1)}
        assert result.report["fresh_tokens"] > 0
        assert "Scan" in result.explain()
        # result() again is a fresh fetch of the same saved files
        assert run.result().count() == 2
        # the rows can also be read as a stream of Arrow batches
        reader = run.stream_result()
        assert isinstance(reader, pa.RecordBatchReader)
        assert sorted(reader.read_all().column("r.id").to_pylist()) == ["r0", "r3"]


def test_stream_answers_yields_each_anchor_and_pages_over_http(
        endpoint, tmp_path):
    url = endpoint()
    with quail.Session(CONFIG, endpoint=url) as session:
        register_reviews(session, tmp_path)
        session.register("products", quail.DocumentProvider.from_table(
            server_fakes.products_table(), id_col="asin"))
        run = session.sql(server_fakes.JOIN_SQL).submit()
        streamed = list(run.stream_answers(poll_s=1.0))
        assert sorted(entry["document"] for entry in streamed) == list(range(6))
        assert all(entry["anchor"] == "r" and entry["partners"] == ["p"]
                   and entry["asked"] == 4 for entry in streamed)
        # r0 matches p0 and p2 under the fake rule
        first = next(entry for entry in streamed if entry["document"] == 0)
        assert first["matches"] == [[0], [2]]

        page = run.answers(after=4, limit=1)
        assert page["next"] == 5 and page["done"] is True
        assert page["answers"] == [streamed[4]]
        assert run.answers(after=6)["answers"] == []
        assert run.status().progress["answers_saved"] == 6
        assert sorted(run.result().to_rows()) == sorted(
            (f"r{r}", f"p{p}") for r in range(6) for p in range(4)
            if (r + p) % 2 == 0)


def test_run_collect_and_execute_stream_match_the_local_query_api(
        endpoint, tmp_path):
    from quail.execution.session import Query, QueryLike

    url = endpoint()
    with quail.Session(CONFIG, endpoint=url) as session:
        register_reviews(session, tmp_path)
        query = session.sql(server_fakes.FILTER_SQL)
        assert isinstance(query, QueryLike)
        assert all(hasattr(Query, name) for name in
                   ("run", "submit", "execute_stream", "collect", "explain",
                    "plan"))
        rows = query.collect()
        assert sorted(rows.column("r.id").to_pylist()) == ["r0", "r3"]
        result = session.sql(server_fakes.FILTER_SQL).run()
        assert result.count() == 2
        streamed = session.sql(server_fakes.FILTER_SQL).execute_stream()
        assert isinstance(streamed, pa.RecordBatchReader)
        assert streamed.read_all().num_rows == 2
        one = session.sql(server_fakes.FILTER_SQL).execute_stream(limit=1)
        assert one.read_all().num_rows == 1
        with pytest.raises(RuntimeError, match="edited plans"):
            session.sql(server_fakes.FILTER_SQL).run(plan=object())


def test_unsupported_providers_and_bad_queries_fail_clearly(endpoint, tmp_path):
    url = endpoint()

    class Custom:
        id_col = "id"

        @property
        def columns(self):
            return ("id",)

        def schema(self):
            return pa.schema([("id", pa.string())])

    with quail.Session(CONFIG, endpoint=url) as session:
        with pytest.raises(TypeError, match="Custom cannot be sent"):
            session.register("custom", Custom())
        assert "custom" not in session.catalog
        register_reviews(session, tmp_path)
        with pytest.raises(ServerError, match="CompileError") as info:
            session.sql("SELECT r.nothing FROM reviews r").submit()
        assert info.value.status == 400
        with pytest.raises(ServerError, match="does not run model"):
            with quail.Session(EngineConfig(model="qwen3-32b-fp8",
                                            device="h100-sxm"),
                               endpoint=url) as other:
                register_reviews(other, tmp_path)
                other.sql(server_fakes.FILTER_SQL).submit()
    with pytest.raises(ServerError, match="cannot reach"):
        with quail.Session(CONFIG, endpoint="http://127.0.0.1:9") as session:
            session.get_run("x").status()
    with quail.Session(CONFIG, endpoint=url) as session:
        with pytest.raises(ServerError, match="unknown query") as info:
            session.get_run("missing").status()
        assert info.value.status == 404


def test_cancel_and_timeout_raise_from_result(endpoint, tmp_path):
    url = endpoint(hooks=server_fakes.sleeping_hooks)
    with quail.Session(CONFIG, endpoint=url) as session:
        register_reviews(session, tmp_path)
        query = session.sql(server_fakes.FILTER_SQL)
        first = query.submit()
        second = query.submit(timeout_s=0.5)
        cancelled = second.cancel()
        assert cancelled.state == "cancelled"
        with pytest.raises(QueryFailedError, match="cancelled") as info:
            second.result()
        assert info.value.status.state == "cancelled"
        for status in first.watch(poll_s=1.0):
            if status.state == "running":
                break
        flagged = first.cancel()
        assert flagged.cancel_requested
        with pytest.raises(QueryFailedError, match="cancelled"):
            first.result(poll_s=1.0)

        timed = query.submit(timeout_s=0.5)
        with pytest.raises(QueryFailedError, match="timeout") as info:
            timed.result(poll_s=1.0)
        assert info.value.status.error["type"] == "TimeoutError"
