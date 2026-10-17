"""Distributed predicate execution with fixed model answers on the CPU."""

import pyarrow as pa
import pytest
from helpers.fakes import letter_tokens

import quail
from quail.api.execute import execute_query
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.executor.operators.filter import stop_groups
from quail.backends.quail.worker import quail_runtime_payload
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import extract_answers
from quail.execution.rows import Rows
from quail.execution.types import PhysicalResponse
from quail.physical import AiExtract, AiFilter, decode_graph
from quail.planner.plan import EngineConfig

TEXTS = ["No date", "June", "No date", "August"]


@pytest.fixture(params=[2, 4])
def session(request):
    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=request.param),
        tokenizer=letter_tokens,
    ) as session:
        session.register("contracts", quail.DocumentProvider.from_table(
            pa.table({"id": range(len(TEXTS)), "text": TEXTS,
                      "group": ["lease"] * len(TEXTS)}), id_col="id"))
        yield session


class FixedWorkers:
    """Return fixed answers through the coordinator's worker protocol."""

    def __init__(self, session):
        self.session = session
        self.filter_limits = []
        self.filtered = []
        self.extracted = {}

    def execute(self, request):
        registry = self.session.registry
        graph = decode_graph(request.plan["graph"], registry.codecs)
        report = execute_distributed_graph(
            quail_runtime_payload(request, graph), graph, request.gpu_count,
            self.round, self.session.device,
            registry.runtimes, registry)
        return PhysicalResponse(report.pop("_outputs"), report)

    def round(self, kind, subs):
        assert kind == "execute"
        outputs = []
        for sub in subs:
            results = {}
            for node in sub["nodes"]:
                inputs = dict(sub["inputs"][node.node_id])
                for port in node.inputs:
                    if port.source.node_id in results:
                        inputs[port.name] = results[port.source.node_id].outputs[
                            port.source.port]
                if isinstance(node, (AiFilter, AiExtract)):
                    source = next(iter(inputs.values()))
                    ids = id_positions(source.ids["c"] if isinstance(source, Rows)
                                       else source).tolist()
                if isinstance(node, AiFilter):
                    self.filter_limits.append(sub["limit"])
                    groups = stop_groups(node, sub.get("query", {}).get(
                        "columns", {}).get("c"), ids)
                    if groups is not None:
                        firsts = {}
                        for document, group in zip(ids, groups):
                            firsts.setdefault(group, document)
                        ids = list(firsts.values())
                    if all(isinstance(member, AiFilter) for member in sub["nodes"]):
                        ids = ids[:sub["limit"]]
                    self.filtered.extend(ids)
                    result = NodeResult({
                        "ids:c": pa.array(ids, pa.int32()),
                        "filter_answers:c": {document: [True] for document in ids},
                    }, NodeMetrics(input_rows=len(ids), output_rows=len(ids)))
                elif isinstance(node, AiExtract):
                    self.extracted.setdefault(node.spec.name, []).extend(ids)
                    answers = [None if TEXTS[i] == "No date" else TEXTS[i] for i in ids]
                    rows = extract_answers(
                        "c", node.spec.name, node.spec.span_name, ids, answers,
                        [None if answer is None else (0, len(answer))
                         for answer in answers])
                    result = NodeResult(
                        {"rows": rows, "ids:c": rows.ids["c"]},
                        NodeMetrics(input_rows=len(ids), output_rows=len(ids)))
                else:
                    result = self.session.registry.runtimes[node.runtime_key].execute(
                        node, inputs, None)
                results[node.node_id] = result
            outputs.append({"results": results, "kv_manager": {"evicted_keys": 2},
                            "peak_gib": 1.0})
        return outputs


@pytest.mark.parametrize("condition,expected", [
    ("IS NOT NULL", [1, 3]), ("= 'absent'", []),
])
def test_extraction_after_a_predicate_receives_surviving_ids(
        session, condition, expected):
    query = session.sql(
        "SELECT c.id, AI.EXTRACT(c.text, "
        "{'ends': 'When does it end?', 'starts': 'When does it start?'}) AS lease "
        f"FROM contracts c WHERE lease.ends {condition}", order="as_written")
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute).collect()
    assert result.column("c.id").to_pylist() == expected
    assert result.column("lease").to_pylist() == [
        {"ends": TEXTS[i], "starts": TEXTS[i]} for i in expected]
    assert sorted(workers.extracted["lease.starts"]) == expected


@pytest.mark.parametrize("limit", [1, 3, 10])
def test_limit_waits_for_downstream_extraction_predicates(session, limit):
    query = session.sql(
        "SELECT c.id, AI.EXTRACT(c.text, {'ends': 'When does it end?'}) AS lease "
        "FROM contracts c WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text)) "
        f"AND lease.ends IS NOT NULL LIMIT {limit}", order="as_written")
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute).collect()
    assert result.column("c.id").to_pylist() == [1, 3][:limit]
    assert workers.filter_limits == [limit] * session.config.gpus


def test_filter_only_query_keeps_early_limit(session):
    query = session.sql(
        "SELECT c.id FROM contracts c "
        "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text)) LIMIT 1")
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute).collect()
    assert result.num_rows == 1
    assert workers.filter_limits == [1] * session.config.gpus


@pytest.mark.parametrize("order", ["as_written", "by_cost"])
def test_distinct_waits_for_downstream_extraction_predicates(session, order):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} relevant?", quail.col("c.text")))
             .ai_extract("c.text", {"ends": "When does it end?"}, name="lease")
             .where(quail.col("lease.ends").is_not_null())
             .distinct().select("c.group", order=order))
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute)
    assert result.to_rows() == [("lease",)]
    assert sorted(workers.extracted["lease.ends"]) == list(range(len(TEXTS)))


def test_distinct_filter_only_query_keeps_per_key_stopping(session):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} relevant?", quail.col("c.text")))
             .distinct().select("c.group"))
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute)
    assert result.to_rows() == [("lease",)]
    assert len(workers.filtered) == 1


def test_filter_reports_every_workers_measurements(session):
    query = (session.docs("contracts").alias("c")
             .ai_filter(quail.prompt("Is {0} relevant?", quail.col("c.text")))
             .select("c.id"))
    workers = FixedWorkers(session)
    result = execute_query(query, physical_executor=workers.execute)
    assert result.report["kv_manager"]["evicted_keys"] == 2 * session.config.gpus
    assert result.report["peak_gib"] == 1.0
