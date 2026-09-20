"""Remote Session, submit, get_run, watch, result, and cancel end to end."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import service_fakes

import quail
from quail.planner.plan import EngineConfig
from quail.service.records import QueryFailedError, ServiceError

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from quail.service.app import ServiceSettings, create_app  # noqa: E402

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")


@pytest.fixture()
def endpoint(tmp_path):
    servers = []

    def start(**overrides):
        settings = ServiceSettings(
            data_dir=tmp_path / f"data{len(servers)}", models=("qwen3-4b-fp8",),
            device="h100-sxm", in_process=True,
            **{"hooks": service_fakes.hooks, **overrides})
        url, stop = service_fakes.start_server(create_app(settings))
        servers.append(stop)
        return url

    yield start
    for stop in servers:
        stop()


def register_reviews(session, tmp_path):
    pq.write_table(service_fakes.reviews_table(), tmp_path / "reviews.parquet")
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))


def test_local_session_reports_what_needs_an_endpoint(tmp_path):
    with quail.Session(CONFIG, tokenizer=service_fakes.fake_tok) as session:
        assert session.endpoint is None
        register_reviews(session, tmp_path)
        query = session.sql(service_fakes.FILTER_SQL)
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
        query = session.sql(service_fakes.FILTER_SQL)
        with pytest.raises(RuntimeError, match="explain"):
            query.explain()
        with pytest.raises(RuntimeError, match="plan"):
            query.plan()
        run = query.submit(request_key="demo-1")
        query_id = run.id
        assert query.submit(request_key="demo-1").id == query_id
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


def test_run_and_collect_are_submit_plus_result(endpoint, tmp_path):
    url = endpoint()
    with quail.Session(CONFIG, endpoint=url) as session:
        register_reviews(session, tmp_path)
        rows = session.sql(service_fakes.FILTER_SQL).collect()
        assert sorted(rows.column("r.id").to_pylist()) == ["r0", "r3"]
        result = session.sql(service_fakes.FILTER_SQL).run()
        assert result.count() == 2
        with pytest.raises(RuntimeError, match="edited plans"):
            session.sql(service_fakes.FILTER_SQL).run(plan=object())


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
        with pytest.raises(ServiceError, match="CompileError") as info:
            session.sql("SELECT r.nothing FROM reviews r").submit()
        assert info.value.status == 400
        with pytest.raises(ServiceError, match="does not run model"):
            with quail.Session(EngineConfig(model="qwen3-32b-fp8",
                                            device="h100-sxm"),
                               endpoint=url) as other:
                register_reviews(other, tmp_path)
                other.sql(service_fakes.FILTER_SQL).submit()
    with pytest.raises(ServiceError, match="cannot reach"):
        with quail.Session(CONFIG, endpoint="http://127.0.0.1:9") as session:
            session.get_run("x").status()
    with quail.Session(CONFIG, endpoint=url) as session:
        with pytest.raises(ServiceError, match="unknown query") as info:
            session.get_run("missing").status()
        assert info.value.status == 404


def test_cancel_and_timeout_raise_from_result(endpoint, tmp_path):
    url = endpoint(hooks=service_fakes.sleeping_hooks)
    with quail.Session(CONFIG, endpoint=url) as session:
        register_reviews(session, tmp_path)
        query = session.sql(service_fakes.FILTER_SQL)
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
