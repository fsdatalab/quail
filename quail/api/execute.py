"""Run a query in the current process."""

import atexit
import contextlib
import json
import os
import time

import pyarrow as pa

from quail.backends import BackendExecutionContext
from quail.execution.finished import first_rows, reporting
from quail.execution.result import QueryResult, limited_batches
from quail.execution.types import PhysicalRequest, PhysicalResponse
from quail.physical import ExecutionLocation, Scan
from quail.physical.plan import Refusal, RefusalError, graph_from_envelope


class _BackendState:
    """The engine loaded in this process, for one configuration at a time.

    A configuration is the engine, model, device, and GPU count a plan
    names. Binding a different one releases the loaded engine first:
    one model's weights and KV take the GPU.
    """

    def __init__(self):
        self.configuration = None
        self.state = {}
        self.release = None

    def bind(self, envelope, backend) -> dict:
        """Return the runtime state for the plan, releasing another engine."""
        configuration = (getattr(backend, "engine_name", backend.name),
                         envelope["model"], envelope["device"], envelope["gpus"])
        if configuration != self.configuration:
            self.close()
            self.configuration = configuration
            self.release = getattr(backend, "release_request", None)
        return self.state

    def close(self) -> None:
        if self.release is not None:
            self.release(self.state)
        self.state.clear()
        self.configuration = self.release = None


# Loaded models outlive individual sessions.
_BACKEND_STATE = _BackendState()
atexit.register(_BACKEND_STATE.close)

def gpu_problem() -> str | None:
    """Return why this process cannot run a model, or None when it can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    # the default check initializes CUDA, which breaks vLLM's forked engine
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    if not torch.cuda.is_available():
        return "no CUDA GPU is visible to this process"
    return None


def _validate_physical_request(request, registry):
    """Validate and decode a physical execution request."""
    if not isinstance(request, PhysicalRequest):
        raise TypeError("execution requires a PhysicalRequest")
    envelope = request.plan
    graph = graph_from_envelope(envelope)
    backend = registry.backend(envelope["backend"])
    graph.validate(runtime_keys=set(registry.runtimes))
    graph.validate_backend(envelope["backend"])
    needed_inputs = {
        node.input_id for node in graph.nodes
        if isinstance(node, Scan)
    }
    missing = needed_inputs - set(request.inputs)
    extra = set(request.inputs) - needed_inputs
    if missing or extra:
        raise ValueError(
            "execution request has wrong input bindings; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return graph, backend


def _execute_physical(request, registry):
    """Run the backend selected by a registered physical plan."""
    graph, backend = _validate_physical_request(request, registry)
    response = backend.execute_request(BackendExecutionContext(
        request=request,
        graph=graph,
        registry=registry,
        gpus=request.gpus,
        runtime_state=_BACKEND_STATE.bind(request.plan, backend),
    ))
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a model backend must return PhysicalResponse")
    return response


def execute_query(query, physical_executor=None, on_backend_ready=None,
                  on_execution_ready=None, on_rows=None, hint=None):
    """Execute one query in its session's process.

    Args:
        query: The Query to run.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.
        on_backend_ready: Optional callback after model preparation.
        on_execution_ready: Optional callback after backend preparation and
            immediately before query execution.
        on_rows: Optional callable(pyarrow Table) handed a model region's
            rows as they finish, each row once.
        hint: Optional ``InputHint`` the region may skip the rows of the
            tables it reads by.
    """
    from quail.api.query import Query

    if isinstance(query, Query):
        return _execute_in_datafusion(query, physical_executor,
                                      on_backend_ready, on_execution_ready)
    return _execute_region(query, physical_executor,
                           on_backend_ready, on_execution_ready, on_rows, hint)


def _execute_in_datafusion(query, physical_executor, on_backend_ready,
                           on_execution_ready):
    """Run a query's SQL in DataFusion, which calls back for each model region.

    The result holds DataFusion's rows. A query of one model region also
    carries that region's plan, metrics, and answers, and, when the region
    is the whole query, the region's own rows, which keep Quail's added
    columns such as a label's probabilities. A query of several model
    regions retains each executed result in ``region_results``, keyed by its
    query-local logical ID, and reports each report under ``regions``.
    """
    from quail.api.context import SessionContext

    whole = query.bound_whole()
    results, region_tables, region_s = [], [], []
    region_results = {}
    started = time.perf_counter()
    # Rust runs the regions of Quail's own backend on GPU workers
    # itself. A request backend, and a callable that stands in for one,
    # go through the Python engine.
    if streams(query, physical_executor):
        def ready():
            for called in (on_backend_ready, on_execution_ready):
                if called is not None:
                    called()

        table, context, startup_s = _run_in_rust(
            query, physical_executor, ready, results, region_tables,
            region_s, region_results)
        # starting the workers is model startup, not query time
        return _query_result(query, context, table, whole, started + startup_s,
                             results, region_tables, region_s, region_results)

    def run_query(region, on_rows=None, hint=None):
        if region.region_id is None or region.region_id in region_results:
            raise RuntimeError("model region has a missing or duplicate logical ID")
        region_started = time.perf_counter()
        result = _execute_region(region, physical_executor,
                                 on_backend_ready, on_execution_ready, on_rows, hint)
        region_s.append(time.perf_counter() - region_started)
        results.append(result)
        region_results[region.region_id] = result
        table = result.collect()
        region_tables.append(table)
        return table

    context = SessionContext(query.session, order=query.order, run_query=run_query)
    table = context.sql(query.text, query.dialect)
    return _query_result(query, context, table, whole, started, results,
                         region_tables, region_s, region_results)


def _run_in_rust(query, source, ready, results, region_tables, region_s,
                 region_results):
    """Run a query whose model regions Rust runs on GPU workers.

    Each region's QueryResult is built from what Rust kept of its run.

    Args:
        query: The Query.
        source: None for the GPUs' workers, or a test's stand-in:
            ``start_workers(gpus)`` returns workers under "workers",
            ``report`` holds report fields to set, and ``close()`` stops
            the workers.
        ready: Callable() called once the workers have the model loaded
            for a region.
        results: Filled with each region's QueryResult, in run order.
        region_tables: Filled with each region's rows.
        region_s: Filled with each region's seconds.
        region_results: Filled with each region's result by region id.

    Returns:
        The query's rows, the SessionContext that ran it, and the
        seconds it took to start the workers.
    """
    from quail.api.context import SessionContext
    from quail.api.region_inputs import region_query

    session = query.session
    config, registry = session.config, session.registry
    backend = registry.backend(config.backend)
    startup_started = time.perf_counter()
    workers = _start_workers(session, source)
    startup_s = time.perf_counter() - startup_started
    try:
        context = SessionContext(session, order=query.order, workers=workers)
        context.when_ready(ready)
        table = context.sql(query.text, query.dialect)
        regions = context.take_regions()
    finally:
        if source is not None:
            source.close()
    for planned, scans, inputs, relation, ran, boots, seconds, total_s in regions:
        found = json.loads(planned)
        region = region_query(
            session, found["operators"], logical=found.get("logical", ""),
            order=query.order, tables=inputs, keep_order=True,
            region_id=found.get("region_id"))
        region.adopt_plan(found, scans)
        report = backend.report_region(
            region.plan(), ran, [json.loads(boot) for boot in boots], seconds,
            registry.device(config.device))
        if source is not None:
            report.update(source.report)
        response = PhysicalResponse(report.pop("_outputs"), report)
        result = region.finish(response, seconds, relation=relation)
        result.report.update(
            planning_s=round(region.planning_s, 4),
            input_ready_s=round(region.planning_s, 4),
            physical_prepare_s=0.0, model_wall_s=result.report["wall_s"],
            finish_s=0.0, worker_total_s=round(total_s, 4))
        results.append(result)
        region_tables.append(result.collect())
        # the boot is the region's time, not DataFusion's
        region_s.append(total_s)
        region_results[region.region_id] = result
    return table, context, startup_s


def _start_workers(session, source):
    """Start the GPU workers, with the model loaded, and the session's planner.

    Args:
        session: The Session whose configuration the workers run.
        source: None for the GPUs' workers, or a test's stand-in with
            ``start_workers(gpus)``.

    Raises:
        RuntimeError: This machine has no GPU to run the model.
    """
    config = session.config
    if source is None:
        problem = gpu_problem()
        if problem is not None:
            raise RuntimeError(
                f"cannot run the model on this machine: {problem}. "
                "Run your Session on a machine with a CUDA GPU "
                "and the backend installed.")
        workers = session.registry.backend(config.backend).workers(config)
    else:
        workers = source.start_workers(config.gpus)["workers"]
    # loading the tokenizer into the planner is startup as well
    _ = session.planner
    return workers


def streams(query, physical_executor=None) -> bool:
    """Whether Rust runs the query's regions on GPU workers.

    Rust runs them for the ``quail`` backend, or a test's stand-in with
    ``start_workers``, and then streams each region's rows as they
    finish (``stream_query``).
    """
    from quail.api.query import backend_refusal

    session = query.session
    return (query.num_regions > 0 and session.config.backend == "quail"
            and backend_refusal(session) is None
            and (physical_executor is None
                 or hasattr(physical_executor, "start_workers")))


def stream_query(query, physical_executor=None, batch_rows: int = 65_536,
                 limit: int | None = None):
    """Run a query and return its rows as DataFusion produces them.

    A model region sends each row once the model has answered every
    question about it, so the reader receives the rows of an AI filter
    while the region still runs. An operator that reads every row
    first, such as an aggregate, a sort, or a join's build side, holds
    its rows until the regions below it end.

    Args:
        query: The Query; ``streams(query)`` must be true.
        physical_executor: None for the GPUs' workers, or a test's
            stand-in with ``start_workers(gpus)`` and ``close()``.
        batch_rows: The most rows of each batch.
        limit: The most rows to return, or None for every row.

    Returns:
        A pyarrow RecordBatchReader. Closing it early stops reading; a
        running region still runs to its end.
    """
    from quail.api.compile import DATAFUSION_DIALECTS, compile_message
    from quail.api.context import SessionContext

    session = query.session
    workers = _start_workers(session, physical_executor)
    with contextlib.ExitStack() as stack:
        if physical_executor is not None:
            stack.callback(physical_executor.close)
        context = SessionContext(session, order=query.order, workers=workers)
        try:
            rows = context._native.sql_stream(
                query.text, DATAFUSION_DIALECTS[query.dialect])
        except RuntimeError as error:
            raise RuntimeError(compile_message(error, query.text)) from None
        schema = rows.schema
        closing = stack.pop_all()

    def batches():
        # the workers run the query until the stream ends
        with closing:
            yield from limited_batches(
                (batch for table in rows for batch in table.to_batches()),
                batch_rows, limit)

    return pa.RecordBatchReader.from_batches(schema, batches())


def _query_result(query, context, table, whole, started, results, region_tables,
                  region_s, region_results):
    """Return a query's result from its rows and its regions' results."""
    if len(results) != 1:
        result = QueryResult.from_table(table, report={
            "regions": [region_results[key].report for key in sorted(region_results)]
        } if results else None)
        result.region_results = region_results
        return result
    (region,) = results
    # the time outside the region: building the context, planning the
    # SQL, gathering the region's rows, and the operators around it
    region.report["datafusion_s"] = round(
        time.perf_counter() - started - region_s[0], 4)
    result = QueryResult.from_table(region_tables[0] if whole else table,
                                    report=region.report)
    result.plan = region.plan
    result.physical_plan = region.physical_plan
    result.region_id = region.region_id
    result.region_results = region_results
    result.node_metrics = region.node_metrics
    result.answer_tables = region.answer_tables
    result.survivor_indices = region.survivor_indices
    result.true_join_tables = region.true_join_tables
    result.region_rows = region_tables[0].num_rows
    if not whole:
        result.datafusion_plan = context.explain(query.text, query.dialect)
    return result


def _execute_region(query, physical_executor, on_backend_ready,
                    on_execution_ready, on_rows=None, hint=None):
    """Execute one model region's Query in its session's process.

    With ``on_rows``, the region's rows go to it as they finish, each
    row once, and the result holds them all as well.
    """
    total_started = time.perf_counter()
    plan = query.plan()
    if isinstance(plan, Refusal):
        raise RefusalError(plan)
    plan.graph.validate(runtime_keys=set(query.session.registry.runtimes))
    plan.graph.validate_backend(plan.backend)
    if plan.gpus > 8:
        raise NotImplementedError("query execution supports at most 8 GPUs")
    has_compute = any(node.backend is not None or
                      node.location is ExecutionLocation.GPU_EXECUTOR
                      for node in plan.nodes)
    if has_compute and physical_executor is None:
        problem = gpu_problem()
        if problem is not None:
            raise RuntimeError(
                f"cannot run the model in this process: {problem}. "
                "Run your Session in a process with a CUDA GPU "
                "and the backend installed."
            )
    if has_compute and physical_executor is None:
        # Backend startup must not wait for background tokenization.
        _prepare_backend(plan, query.session.registry)
    if has_compute and on_backend_ready is not None:
        on_backend_ready()
    physical_prepare_started = time.perf_counter()
    request = query._prepare_physical() if has_compute else None
    physical_prepare_s = time.perf_counter() - physical_prepare_started
    if on_execution_ready is not None:
        on_execution_ready()
    started = time.perf_counter()
    finished = query.finished_rows(on_rows)
    if has_compute:
        tables = {scan["alias"]: scan["provider"] for scan in query.facts()["scans"]}
        first = {}
        if hint is not None:
            first = first_rows(request.column_tables(),
                               plan.settings.get("deduplicate", {}))
        # with no one taking rows as they finish, they are built at the end
        with reporting(None if on_rows is None else finished, hint, tables,
                       first):
            response = (physical_executor(request) if physical_executor is not None
                        else _execute_physical(request, query.session.registry))
    else:
        response = PhysicalResponse(
            {}, {"wall_s": 0.0, "fresh_tokens": 0, "cached_tokens": 0})
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a physical executor must return PhysicalResponse")
    coordinator_wall_s = time.perf_counter() - started
    finish_started = time.perf_counter()
    result = query.finish(response, coordinator_wall_s, finished)
    finish_s = time.perf_counter() - finish_started
    model_wall_s = result.report["wall_s"]
    result.report.update(
        planning_s=round(query.planning_s, 4),
        # the planner tokenizes the inputs, so they are ready once planned
        input_ready_s=round(query.planning_s, 4),
        physical_prepare_s=round(physical_prepare_s, 4),
        model_wall_s=model_wall_s,
        finish_s=round(finish_s, 4),
    )
    result.report["worker_total_s"] = round(
        time.perf_counter() - total_started, 4)
    return result


def _prepare_backend(plan, registry) -> None:
    """Let the backend boot from the plan alone, before documents arrive."""
    backend = registry.backend(plan.backend)
    prepare = getattr(backend, "prepare_request", None)
    if prepare is None:
        return
    envelope = plan.to_envelope()
    prepare(BackendExecutionContext(
        request=PhysicalRequest(envelope, {}),
        graph=plan.graph,
        registry=registry,
        gpus=plan.gpus,
        runtime_state=_BACKEND_STATE.bind(envelope, backend),
    ))
