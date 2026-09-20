"""Typed Quail graph execution across GPU child processes."""

from __future__ import annotations

import time

from quail.backends.quail import coordinator
from quail.backends.quail.graph import (
    executed_join_plan,
    model_answers,
    throughput,
)
from quail.execution.pairs import columns_key
from quail.execution.reranker import score_in_batches
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
    SurvivorStream,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.tokens import select_documents
from quail.execution.types import export_physical_outputs
from quail.physical import (
    AiFilter,
    AiJoin,
    AiScore,
    PhysicalGraph,
    PhysicalScan,
)
from quail.planner import balanced_shards


class DistributedQuailExecution:
    """Dispatch typed Quail model nodes to one child per GPU."""

    # the children keep the document tokens after the first score round
    score_documents_sent = False

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
            if isinstance(node, PhysicalScan)
            and len(node.shards) == gpu_count
        }
        self.filter_aliases = {
            node.alias
            for node in graph.nodes if isinstance(node, AiFilter)
        }
        if self.filter_aliases - set(self.shards):
            for alias in self.filter_aliases - set(self.shards):
                shards, _ = balanced_shards(
                    [len(document) for document in self.docs[alias]],
                    gpu_count,
                )
                self.shards[alias] = shards
        self.joins = tuple(stage for node in graph.nodes
                           if isinstance(node, AiJoin) for stage in node.stages)
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
        if isinstance(node, AiScore):
            return self._execute_score(node, inputs)
        if isinstance(node, AiFilter):
            return self._execute_filter(node, inputs)
        if isinstance(node, AiJoin):
            return self._execute_join(node, inputs)
        raise TypeError(
            f"distributed Quail cannot execute {node.type_name!r}")

    def _execute_score(self, node, inputs):
        return score_in_batches(
            node, inputs, self._score_round, shards=self.gpu_count
        )

    def _score_round(self, node, batches):
        """Score one batch per GPU child and return their results in order."""
        subs = [{"node": node, "inputs": {"score_rows": batch}}
                for batch in batches]
        if not self.score_documents_sent:
            documents = {
                alias: select_documents(docs, range(len(docs)))
                for alias, docs in self.docs.items()
            }
            for sub in subs:
                sub["inputs"]["documents"] = documents
            self.score_documents_sent = True
        return self.round_fn("scores", subs)

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
        if node.pin_survivors:
            stream = SurvivorStream(node, list(document_ids))

            return NodeResult(
                {f"ids:{node.alias}": stream,
                 f"filter_answers:{node.alias}": {}},
                finalize=stream.finalized_result,
            )
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
            # the anchor's chain runs inside this round, over this GPU's shard
            survivors[node.anchor] = list(stream.document_ids)
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
            pair_tables=inputs.get("pairs"),
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
                    AiJoin.type_name)[-1].node_id,
                start_query=not self.started,
                physical_node=encoded_node,
                stream_filter_node=encoded_filter,
            )
        started = time.perf_counter()
        outputs = self.round_fn("joins", subs)
        wall = time.perf_counter() - started
        self.started = True
        if stream is not None:
            merged = coordinator.merge_filter_round([
                {
                    "filters": output["filters"],
                    "survivors": output["survivors"],
                    "fresh_tokens": output["filter_fresh_tokens"],
                }
                for output in outputs
            ])
            answers = merged["filters"].get(node.anchor, {})
            survivors = merged["survivors"].get(node.anchor, [])
            stream.complete(NodeResult(
                {
                    f"ids:{node.anchor}": survivors,
                    f"filter_answers:{node.anchor}": answers,
                },
                NodeMetrics(
                    input_rows=len(stream.document_ids),
                    output_rows=len(survivors),
                    evaluated_documents=len(answers),
                    fresh_tokens=merged["fresh_tokens"],
                ),
            ))
        stage_outputs = coordinator.merge_join_round(outputs)
        hits = 0
        misses = 0
        fresh_tokens = 0
        for index, output in enumerate(outputs):
            fresh_tokens += output["fresh_tokens"]
            hits += output["kv_round"]["hits"]
            misses += output["kv_round"]["misses"]
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
                input_rows=len(enriched[-1]["anchor_index"]),
                output_rows=len(anchor_survivors),
                evaluated_document_pairs=sum(
                    sum(len(row) for row in stage["rows"].values())
                    for stage in enriched
                ),
                fresh_tokens=fresh_tokens,
                kv_hits=hits,
                kv_misses=misses,
                extension={"joins": enriched},
            ),
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
        from quail.cost.budgets import PAGE_TOKENS

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

_ONE_GPU_APPLY = ("per-batch apply() functions run on one GPU; the "
                  "planner refuses them for several")


def prepare_distributed_inputs(node, inputs, context):
    if isinstance(node, AiJoin):
        survivors = {}
        stream = None
        pairs = {}
        for port in node.inputs:
            value = inputs[port.name]
            if port.source.port.startswith("pairs:"):
                if not hasattr(value, "num_rows"):
                    raise TypeError(_ONE_GPU_APPLY)
                pairs[int(port.source.port.split(":", 1)[1])] = value
                continue
            alias = port.source.port.split(":", 1)[1]
            if isinstance(value, SurvivorStream):
                if alias != node.anchor:
                    raise TypeError(
                        "only the anchor's filter chain streams into a join")
                if value.transforms:
                    raise TypeError(_ONE_GPU_APPLY)
                stream = value
            else:
                survivors[alias] = list(value)
        return {
            "survivors": survivors,
            "group": [stage.runtime_spec() for stage in node.stages],
            "anchor_stream": stream,
            "pairs": pairs,
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
    for alias, table in payload.get("columns", {}).items():
        sources[columns_key(alias)] = table
    context = ExecutionContext(
        runtimes=runtimes,
        model_execution=execution,
        sources=sources,
        model_inputs=prepare_distributed_inputs,
        state={"distributed_execution": execution},
        functions=registry.functions,
    )
    started = time.perf_counter()
    if not any(isinstance(node, AiFilter) for node in graph.nodes):
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
        cached_tokens=result.metrics.cached_tokens,
        evaluated_documents=result.metrics.evaluated_documents,
        evaluated_document_pairs=result.metrics.evaluated_document_pairs,
        usd_per_query=(
            None if device.usd_per_hour is None
            else elapsed / 3600 * gpu_count * device.usd_per_hour
        ),
        backend_metrics={"scores": [
            dict(value.metrics.extension)
            for node_id, value in result.nodes.items()
            if graph.node(node_id).type_name == AiScore.type_name
        ]},
        executed_join_plan=executed_join_plan(graph),
        node_metrics=scalar_node_metrics(result.nodes),
    )
    report.update(throughput(graph, result.metrics, elapsed))
    return report
