"""Recombine placement and the direct join answers result path."""

import pyarrow as pa
import pyarrow.parquet as pq
from test_planner import catalog, tok  # noqa: F401
from test_session import _run, fake_tok, make_executor

import quail
from quail.builder import col, docs, prompt
from quail.physical import Project, Recombine
from quail.planner.decide import plan_query
from quail.planner.plan import EngineConfig
from quail.specs import H100_SXM, QWEN3_4B_FP8


def _byte_tokens(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _sink_source(plan):
    sink = next(node for node in plan.nodes if isinstance(node, Project))
    return sink.inputs[0].source


def test_join_recombination_plans_and_results(catalog, tmp_path):  # noqa: F811
    logical = (docs(catalog, "reviews", tok).alias("r")
               .ai_join(docs(catalog, "threads", tok).alias("t"),
                        prompt("g {0} {1}", col("r.review"),
                               col("t.thread")),
                        selectivity=0.5, semantics="exists")
               .ai_join(docs(catalog, "products", tok).alias("p"),
                        prompt("m {0} {1}", col("r.review"),
                               col("p.description")),
                        selectivity=0.1)
               .select("r.id"))
    plan = plan_query(logical, model=QWEN3_4B_FP8, device=H100_SXM,
                      doc_tokens={"r": [300] * 5, "t": [50] * 5,
                                  "p": [100] * 5})

    assert [n for n in plan.nodes if isinstance(n, Recombine)]
    assert _sink_source(plan).node_id == "recombine"

    table = pa.table({
        "id": ["a", "b"], "body": ["one", "two"],
    })
    for backend in ("stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        with quail.Session(EngineConfig(
            gpus=1,
            model="qwen3-4b-fp8",
            backend=backend,
            device="h100-sxm",
        ),
                           tokenizer=_byte_tokens) as session:
            session.register("left", quail.DocumentProvider.from_table(
                table, id_col="id"))
            session.register("right", quail.DocumentProvider.from_table(
                table, id_col="id"))
            plan = (session.docs("left").alias("l")
                    .ai_join(session.docs("right").alias("x"),
                             quail.prompt("m {0} {1}", quail.col("l.body"),
                                          quail.col("x.body")))
                    .select("l.id", "x.id")).plan()
        assert not [n for n in plan.nodes if isinstance(n, Recombine)]
        assert _sink_source(plan).port == "join_answers:0"

    session = quail.Session(EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    ), tokenizer=fake_tok)
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2"],
        "review": ["review 0 " + "pad " * 20, "review 1 " + "pad " * 20,
                   "review 2 " + "pad " * 20],
    }), str(tmp_path / "r.parquet"))
    pq.write_table(pa.table({
        "asin": ["p0", "p1"],
        "description": ["product 0", "product 1"],
    }), str(tmp_path / "p.parquet"))
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "r.parquet"), id_col="id"))
    session.register("products", quail.DocumentProvider.from_parquet(
        str(tmp_path / "p.parquet"), id_col="asin"))
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
