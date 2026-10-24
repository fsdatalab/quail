"""Plan and execute dependent query stages through the physical graph runner."""

import time
from contextlib import closing

import pyarrow as pa

from quail.api.execute import execute_query
from quail.catalog import Catalog, MemoryTableProvider
from quail.catalog.tokens import TokenLengths
from quail.execution.metrics import scalar_node_metrics
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.explain import estimated_rows
from quail.logical import cte_schema
from quail.physical import QueryStage, encode_graph
from quail.planner.plan import Refusal, RefusalError
from quail.planner.stages import stage_plan

ASSUMED_ANSWER_TOKENS = 32

SUMMED_REPORT_FIELDS = (
    "wall_s", "model_wall_s", "coordinator_wall_s", "fresh_tokens",
    "cached_tokens", "estimated_seconds", "planning_s", "input_ready_s",
    "physical_prepare_s", "finish_s", "token_wait_s", "gpu_s", "chunks",
    "evaluated_documents", "evaluated_document_pairs", "usd_per_query")


class StageExecution:
    """Own a statement's intermediate relations and token files."""

    def __init__(self, query):
        self.query = query
        self.stages = dict(query.logical.stages())
        self.schemas = {name: cte_schema(logical)
                        for name, logical in self.stages.items() if name}
        self.catalog = Catalog({
            **query.catalog.providers,
            **{name: MemoryTableProvider(schema.empty_table(), id_col="",
                                         identity=f"cte:{name}")
               for name, schema in self.schemas.items()}})
        self.plans = {}
        self.results = {}
        self.assumptions = {}
        self.pending = {name: (lambda _column, name=name: self.lengths(name))
                        for name in self.schemas}
        self.inputs = query.inputs.over(self.catalog, self.pending)

    def close(self):
        self.inputs.close()

    def lengths(self, name: str) -> TokenLengths:
        """Estimate an unexecuted relation from its producer's plan."""
        plan = self.plans[name]
        rows = estimated_rows(plan)
        if rows is None:
            rows = sum(node.num_documents for node in plan.nodes
                       if hasattr(node, "num_documents"))
        return TokenLengths([ASSUMED_ANSWER_TOKENS] * max(0, round(rows)))

    def stage(self, name):
        return type(self.query)(self.query.session, self.stages[name],
                                order=self.query.order,
                                catalog=self.catalog, inputs=self.inputs)

    def plan(self):
        """Plan the dependency graph without executing a stage."""
        for name, logical in self.stages.items():
            query = self.stage(name)
            physical = query.plan()
            if isinstance(physical, Refusal):
                return physical
            self.plans[name] = physical
            self.assumptions[name] = tuple(
                (scan.alias, scan.provider, len(self.lengths(scan.provider)),
                 ASSUMED_ANSWER_TOKENS)
                for scan in logical.operators().scans if scan.provider in self.pending)
        return stage_plan(self.query.logical, self.plans, self.assumptions)


def plan_stages(query):
    """Plan a statement as one graph of dependent physical subgraphs."""
    with closing(StageExecution(query)) as execution:
        return execution.plan()


def execute_stages(query, physical_executor=None, on_backend_ready=None,
                   on_execution_ready=None):
    """Run the physical dependency graph, replanning after materialization."""
    started = time.perf_counter()
    plan = query.plan()
    if isinstance(plan, Refusal):
        raise RefusalError(plan)
    callbacks = [on_backend_ready, on_execution_ready]

    def once(index):
        callback, callbacks[index] = callbacks[index], None
        if callback is not None:
            callback()

    with closing(StageExecution(query)) as execution:
        def execute_stage(node, inputs):
            for name, result in inputs.items():
                if name in execution.pending:
                    execution.catalog.providers[name] = MemoryTableProvider(
                        cte_table(result.collect(), execution.schemas[name]),
                        id_col="", identity=f"cte:{name}")
                    del execution.pending[name]
            stage = execution.stage(node.name)
            result = execute_query(
                stage, physical_executor=physical_executor,
                on_backend_ready=lambda: once(0),
                on_execution_ready=lambda: once(1))
            if node.name:
                result._materialized = result.collect()
                result._num_rows = result._materialized.num_rows
            execution.plans[node.name] = stage.plan()
            execution.results[node.name] = result
            return result

        run = GenericRunner().run(plan.graph, ExecutionContext(
            runtimes=query.session.registry.runtimes,
            functions={QueryStage.runtime_key: execute_stage}))
        query._plan = stage_plan(query.logical, execution.plans)
        leaf_metrics = {name: dict(value.node_metrics)
                        for name, value in execution.results.items()}
        leaf_scalars = {name: dict(value.report.get("node_metrics", {}))
                       for name, value in execution.results.items()}
        result = run.value
        result.report = summed_report(execution.results)
        result.report["worker_total_s"] = round(time.perf_counter() - started, 4)
        result.plan = query._plan.graph
        result.node_metrics = {node_id: value.metrics
                               for node_id, value in run.nodes.items()}
        for node in result.plan.nodes:
            result.node_metrics.update({
                f"{node.node_id}/{node_id}": metrics
                for node_id, metrics in leaf_metrics[node.name].items()
            })
        result.report["node_metrics"] = {
            **scalar_node_metrics(run.nodes),
            **{f"stage:{name}/{node_id}": metrics
               for name, scalars in leaf_scalars.items()
               for node_id, metrics in scalars.items()}
        }
        result.report["executed_plan"] = encode_graph(
            result.plan, query.session.registry.codecs)
        return result


def cte_table(result: pa.Table, schema: pa.Schema) -> pa.Table:
    """Rename a stage result and decode only dictionaries compiled as values."""
    columns = [
        column.cast(expected)
        if pa.types.is_dictionary(column.type) and column.type.value_type == expected
        else column
        for column, expected in zip(result.columns, schema.types)]
    if len(columns) != result.num_columns or [c.type for c in columns] != schema.types:
        raise ValueError(
            f"the CTE result has types {[c.type for c in columns]}, but "
            f"its reader was compiled for {schema.types}")
    return pa.Table.from_arrays(columns, schema=schema)


def summed_report(results: dict) -> dict:
    """Combine stage measurements without losing the producer reports."""
    report = dict(results[""].report)
    for key in SUMMED_REPORT_FIELDS:
        values = [result.report.get(key) for result in results.values()]
        values = [value for value in values if value is not None]
        if values:
            report[key] = sum(values)
    report["ctes"] = {name: result.report for name, result in results.items() if name}
    peaks = [result.report["peak_gib"] for result in results.values()
             if result.report.get("peak_gib") is not None]
    if peaks:
        report["peak_gib"] = max(peaks)
    boot = next((result.report for result in results.values()
                 if result.report.get("boot_s")), None)
    if boot is not None:
        for key in ("boot_s", "boot_kind", "boot"):
            report[key] = boot.get(key)
    for key in ("documents_per_second", "document_pairs_per_second"):
        report.pop(key, None)
    return report
