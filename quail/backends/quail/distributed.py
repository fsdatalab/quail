"""Typed Quail graph execution across GPU child processes."""

from __future__ import annotations

import time

from quail.backends.quail import coordinator
from quail.backends.quail.graph import executed_join_plan, model_answers
from quail.execution import export_physical_outputs
from quail.physical import (
    AnchoredJoin,
    DocumentInput,
    PackedFilter,
    PhysicalGraph,
)
from quail.planner import balanced_shards
from quail.runtime.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
    StreamedInput,
    compute_subgraph,
    scalar_node_metrics,
)


class DistributedQuailExecution:
    """Dispatch typed Quail model nodes to one child per GPU."""

    def __init__(self, payload, graph, gpu_count, round_fn,
                 model_spec, device, registry):

        self.payload = payload
        self.graph = graph
        self.gpu_count = gpu_count
        self.round_fn = round_fn
        self.model_spec = model_spec
        self.device = device
        self.registry = registry
        self.docs = payload["docs"]
        self.shards = {
            node.alias: node.shards
            for node in graph.nodes
            if isinstance(node, DocumentInput)
            and len(node.shards) == gpu_count
        }
        self.filter_aliases = {
            node.alias
            for node in graph.nodes if isinstance(node, PackedFilter)
        }
        if self.filter_aliases - set(self.shards):
            for alias in self.filter_aliases - set(self.shards):
                shards, _ = balanced_shards(
                    [len(document) for document in self.docs[alias]],
                    gpu_count,
                )
                self.shards[alias] = shards
        self.joins = tuple(stage for node in graph.nodes
                           if isinstance(node, AnchoredJoin) for stage in node.stages)
        self.pre = payload.get("pre_ids") or []
        self.joins_started = False
        self.retained: dict[str, set[int]] = {}
        self.prior_shards: dict[str, list[list[int]]] = {}
        self.started = False
        self.child_totals = [None] * gpu_count
        self.peak_gib = 0.0
        self.kv_stats = {
            "retained_after_filters": 0,
            "retained_pages_after_filters": 0,
            "retained_prefix_tokens_after_filters": 0,
            "join_anchor_hits": 0,
            "join_anchor_misses": 0,
        }

    def _runtime_payload(self):
        return dict(self.payload)

    def execute(self, node, inputs):
        if isinstance(node, PackedFilter):
            return self._execute_filter(node, inputs)
        if isinstance(node, AnchoredJoin):
            return self._execute_join(node, inputs)
        raise TypeError(
            f"distributed Quail cannot execute {node.type_name!r}")

    def begin(self):
        """Start a query that has no filter node."""
        subs = coordinator.begin_query_payloads(
            self.payload, self.gpu_count
        )
        for sub in subs:
            sub["start_query"] = True
        outputs = self.round_fn("filters", subs)
        self.started = True
        self.peak_gib = max(
            self.peak_gib,
            *(output.get("peak_gib", 0.0) for output in outputs),
        )

    def _execute_filter(self, node, inputs):

        document_ids = next(iter(inputs.values()))
        shards = self.shards
        complete = (
            isinstance(document_ids, range)
            and document_ids.start == 0
            and document_ids.stop == len(self.docs[node.alias])
            and document_ids.step == 1
        )
        if node.alias in shards and not complete:
            live = set(document_ids)
            shards = dict(shards)
            shards[node.alias] = tuple(
                tuple(document for document in shard if document in live)
                for shard in shards[node.alias]
            )
        subs = coordinator.filter_node_payloads(
            self.payload,
            node,
            shards,
            self.gpu_count,
            has_joins=bool(self.joins),
        )
        for sub in subs:
            sub["start_query"] = not self.started
        started = time.perf_counter()
        outputs = self.round_fn("filters", subs)
        wall = time.perf_counter() - started
        self.started = True
        self.peak_gib = max(
            self.peak_gib,
            *(output.get("peak_gib", 0.0) for output in outputs),
        )
        merged = coordinator.merge_filter_round(
            outputs,
            limit=(
                None if self.joins
                else self.payload.get("filter_limit")
            ),
        )
        self._retained_placement(outputs)
        survivors = merged["survivors"].get(node.alias, [])
        answers = merged["filters"].get(node.alias, {})
        return NodeResult(
            {
                f"ids:{node.alias}": survivors,
                f"filter_answers:{node.alias}": answers,
            },
            NodeMetrics(
                wall_s=wall,
                input_rows=len(document_ids),
                output_rows=len(survivors),
                evaluated_documents=len(answers),
                fresh_tokens=merged["fresh_tokens"],
            ),
        )

    def _execute_join(self, node, inputs):

        survivors = dict(inputs["survivors"])
        group = inputs["group"]
        stream = inputs.get("anchor_stream")
        filtered_aliases = set(self.retained)
        if stream is not None:
            # the anchor's chain runs inside this round on each GPU,
            # over that GPU's filter shard of every anchor document
            filter_ids = list(next(iter(stream.inputs.values())))
            survivors[node.anchor] = filter_ids
            filtered_aliases.add(node.anchor)
        if not self.joins_started:
            self.snapshot_after_filters()
            self.joins_started = True
        subs = coordinator.join_group_payloads(
            self._runtime_payload(),
            self.gpu_count,
            survivors,
            group,
            prior_shards=self.prior_shards,
            filtered_aliases=filtered_aliases,
            shards=self.shards,
        )
        encoded_node = self.registry.codecs[node.type_name].encode(node)
        encoded_filter = (
            None if stream is None
            else self.registry.codecs[stream.node.type_name].encode(
                stream.node)
        )
        for sub in subs:
            sub.pop("joins", None)
            sub.update(
                retain_anchor=node.keep_anchor_kv,
                final_group=node.node_id == self.graph.nodes_by_type(
                    AnchoredJoin.type_name)[-1].node_id,
                start_query=not self.started,
                physical_node=encoded_node,
                stream_filter_node=encoded_filter,
            )
        started = time.perf_counter()
        outputs = self.round_fn("joins", subs)
        wall = time.perf_counter() - started
        self.started = True
        produced = {}
        if stream is not None:
            merged = coordinator.merge_filter_round([
                {
                    "filters": output["filters"],
                    "survivors": output["survivors"],
                    "fresh_tokens": output["filter_fresh_tokens"],
                }
                for output in outputs
            ])
            filter_answers = merged["filters"].get(node.anchor, {})
            filter_survivors = merged["survivors"].get(node.anchor, [])
            produced[stream.node.node_id] = NodeResult(
                {
                    f"ids:{node.anchor}": filter_survivors,
                    f"filter_answers:{node.anchor}": filter_answers,
                },
                NodeMetrics(
                    input_rows=len(filter_ids),
                    output_rows=len(filter_survivors),
                    evaluated_documents=len(filter_answers),
                    fresh_tokens=merged["fresh_tokens"],
                ),
            )
        stage_outputs = coordinator.merge_join_round(outputs)
        regret = 0
        hits = 0
        misses = 0
        fresh_tokens = 0
        for index, output in enumerate(outputs):
            fresh_tokens += output["fresh_tokens"]
            hits += output["kv_round"]["hits"]
            misses += output["kv_round"]["misses"]
            regret += output["kv_round"]["regret_tokens"]
            self.child_totals[index] = output["kv_totals"]
        self.kv_stats["join_anchor_hits"] += hits
        self.kv_stats["join_anchor_misses"] += misses

        enriched = []
        answer_outputs = {}
        for stage, output, join in zip(node.stages, stage_outputs, group):
            output.update(
                anchor=node.anchor,
                partners=list(stage.partners),
                semantics=stage.semantics,
                selectivity=stage.selectivity,
                written_pos=stage.written_pos,
            )
            enriched.append(output)
            answer_outputs[f"join_answers:{stage.written_pos}"] = output
        anchor_survivors = coordinator.gate_group(
            enriched[-1], node.stages[-1].semantics
        )

        self._retained_placement(outputs)
        return NodeResult(
            {
                f"ids:{node.anchor}": anchor_survivors,
                **answer_outputs,
            },
            NodeMetrics(
                wall_s=wall,
                input_rows=len(enriched[-1]["anchor_index"]) if enriched
                else len(survivors[node.anchor]),
                output_rows=len(anchor_survivors),
                evaluated_document_pairs=sum(
                    sum(len(row) for row in stage["rows"].values())
                    for stage in enriched
                ),
                fresh_tokens=fresh_tokens,
                kv_hits=hits,
                kv_misses=misses,
                regret_tokens=regret,
                extension={"joins": enriched},
            ),
            produced=produced,
        )

    def _retained_placement(self, outputs):
        self.retained = {}
        placement = {}
        for worker, output in enumerate(outputs):
            for alias, documents in output.get("retained", {}).items():
                self.retained.setdefault(alias, set()).update(documents)
                placement.setdefault(
                    alias, [[] for _ in range(self.gpu_count)]
                )
                placement[alias][worker] = list(documents)
        self.prior_shards = placement

    def snapshot_after_filters(self):
        from quail.planner.budgets import PAGE_TOKENS

        lengths = [len(self.pre) + len(self.docs[alias][document])
                   for alias, documents in self.retained.items()
                   for document in documents]
        self.kv_stats.update(
            retained_after_filters=len(lengths),
            retained_pages_after_filters=sum(-(-n // PAGE_TOKENS) for n in lengths),
            retained_prefix_tokens_after_filters=sum(lengths),
            retained_by_alias_after_filters={
                alias: len(documents) for alias, documents in self.retained.items()},
        )

    def report(self):
        for totals in self.child_totals:
            for key, value in (totals or {}).items():
                self.kv_stats[key] = self.kv_stats.get(key, 0) + value
        return {
            "kv_manager": dict(self.kv_stats),
            "peak_gib": self.peak_gib,
        }


def prepare_distributed_inputs(node, inputs, context):
    if isinstance(node, AnchoredJoin):
        survivors = {}
        stream = None
        for port in node.inputs:
            value = inputs[port.name]
            alias = port.source.port.split(":", 1)[1]
            if isinstance(value, StreamedInput):
                if alias != node.anchor:
                    raise TypeError(
                        "only the anchor's filter chain streams into a join")
                stream = value
            else:
                survivors[alias] = list(value)
        return {
            "survivors": survivors,
            "group": [stage.runtime_spec() for stage in node.stages],
            "anchor_stream": stream,
        }
    return inputs


def execute_distributed_graph(payload, graph: PhysicalGraph, gpu_count: int,
                              round_fn, model_spec, device, runtimes,
                              registry) -> dict:
    """Execute one typed Quail graph across GPU child processes."""
    execution = DistributedQuailExecution(
        payload, graph, gpu_count, round_fn, model_spec, device, registry
    )
    sources = {
        alias: range(len(documents))
        for alias, documents in payload["docs"].items()
    }
    context = ExecutionContext(
        runtimes=runtimes,
        model_execution=execution,
        sources=sources,
        model_inputs=prepare_distributed_inputs,
        state={"distributed_execution": execution},
    )
    started = time.perf_counter()
    if not any(isinstance(node, PackedFilter) for node in graph.nodes):
        execution.begin()
    result = GenericRunner().run(compute_subgraph(graph), context)
    elapsed = time.perf_counter() - started
    filters, joins = model_answers(graph, result)
    report = execution.report()

    report.update(
        filters=filters,
        joins=joins,
        _outputs=export_physical_outputs(compute_subgraph(graph), result),
        wall_s=round(elapsed, 2),
        fresh_tokens=result.metrics.fresh_tokens,
        regret_tokens=result.metrics.regret_tokens,
        executed_join_plan=executed_join_plan(graph),
        node_metrics=scalar_node_metrics(result.nodes),
    )
    return report
