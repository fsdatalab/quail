"""Typed Quail graph execution across GPU child processes."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pyarrow as pa

from quail.backends.quail import coordinator
from quail.backends.quail.graph import (
    executed_join_plan,
    model_answers,
    stop_groups,
    throughput,
)
from quail.backends.quail.prompts import join_request
from quail.execution.pairs import columns_key
from quail.execution.reranker import score_in_batches
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.tokens import select_documents
from quail.execution.types import export_physical_outputs
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiInference,
    AiJoin,
    AiScore,
    PhysicalGraph,
    Scan,
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
            if isinstance(node, Scan)
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
        # a per-key stop needs a key's documents on one worker
        for node in graph.nodes:
            if isinstance(node, AiFilter) and node.stop_key and gpu_count > 1:
                documents = self.docs[node.alias]
                self.shards[node.alias] = coordinator.keyed_shards(
                    stop_groups(node, payload.get("columns", {}).get(
                        node.alias), range(len(documents))),
                    [len(document) for document in documents], gpu_count)
        self.joins = tuple(node for node in graph.nodes if isinstance(node, AiJoin))
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
        if isinstance(node, AiExtract):
            return self._execute_extractions((node,), inputs)[node.node_id]
        if isinstance(node, (AiScore, AiClassify)):
            return self._execute_score(node, inputs)
        if isinstance(node, AiFilter):
            return self._execute_filter(node, inputs)
        if isinstance(node, AiJoin):
            return self._execute_join(node, inputs)
        raise TypeError(
            f"distributed Quail cannot execute {node.type_name!r}")

    def execute_pipeline(self, pipeline, inputs, context):
        """Run compatible predicates or extraction questions together on each worker."""
        first = pipeline[0]
        if isinstance(first, AiFilter):
            return self._execute_filters(pipeline, inputs[first.node_id])
        if isinstance(first, AiJoin):
            prepared = prepare_distributed_inputs(
                first, inputs, context, nodes=pipeline)
            return self._execute_joins(pipeline, prepared)
        return self._execute_extractions(pipeline, inputs[first.node_id])

    def _extraction_positions(self, alias, ids):
        """Keep documents on their previous worker, partitioning unplaced rows."""
        owners = np.full(len(self.docs[alias]), -1, dtype=np.int32)
        for worker, shard in enumerate(self.shards.get(alias, ())):
            owners[np.asarray(shard, dtype=np.int64)] = worker
        for worker, shard in enumerate(self.prior_shards.get(alias, ())):
            owners[np.asarray(shard, dtype=np.int64)] = worker
        missing = ids[owners[ids] < 0]
        shards, _ = balanced_shards(
            [len(self.docs[alias][document]) for document in missing],
            self.gpu_count)
        for worker, shard in enumerate(shards):
            owners[missing[np.asarray(shard, dtype=np.int64)]] = worker
        return [np.flatnonzero(owners[ids] == worker)
                for worker in range(self.gpu_count)]

    def _execute_extractions(self, nodes, inputs):
        if not self.started:
            self.begin()
        alias = nodes[0].spec.alias
        (source,) = inputs.values()
        ids = (source.column(alias).to_numpy() if isinstance(source, pa.Table)
               else np.asarray(source, dtype=np.int64))
        positions = self._extraction_positions(alias, ids)
        texts = self.payload["texts"][alias]
        if not isinstance(texts, (pa.Array, pa.ChunkedArray)):
            texts = pa.array(texts, type=pa.large_string())
        subs = []
        for part in positions:
            documents = ids[part]
            subs.append({"nodes": nodes, "inputs": {
                "document_ids": documents,
                "document_positions": range(len(documents)),
                "documents": {alias: select_documents(self.docs[alias], documents)},
                "texts": {alias: texts.take(pa.array(documents))},
                "pre": self.pre,
                "gpu_timing": self.payload.get("gpu_timing", False),
            }})
        started = time.perf_counter()
        outputs = self.round_fn("extract", subs)
        wall = time.perf_counter() - started
        self._retained_placement(outputs)
        for worker, output in enumerate(outputs):
            self.child_totals[worker] = output["kv_totals"]
            self.peak_gib = max(self.peak_gib, output["peak_gib"])
        order = pa.array(np.argsort(np.concatenate(positions)))
        results = {}
        for index, node in enumerate(nodes):
            parts = [output["results"][node.node_id] for output in outputs]
            table = pa.concat_tables([part.outputs["rows"] for part in parts])
            metrics = sum((part.metrics for part in parts), NodeMetrics())
            extension = dict(parts[0].metrics.extension)
            for name in ("input_rows", "none", "steps", "answer_tokens",
                         "suffix_tokens", "borrowed_prefix_tokens", "pack_s"):
                extension[name] = sum(part.metrics.extension.get(name, 0)
                                      for part in parts)
            results[node.node_id] = NodeResult(
                {"rows": table.take(order), f"ids:{alias}": ids},
                replace(metrics, wall_s=wall if index == 0 else 0.0,
                        extension=extension))
        return results

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
        return self._execute_filters((node,), inputs)[node.node_id]

    def _execute_filters(self, nodes, inputs):
        node = nodes[0]

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
            sub["node_ids"] = [member.node_id for member in nodes]
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
        results = {}
        for j, member in enumerate(nodes):
            parts = [output["results"][member.node_id] for output in outputs]
            rows = {document: row for part in parts
                    for document, row in part.outputs[
                        f"filter_answers:{node.alias}"].items()}
            survivors = sorted(document for part in parts
                               for document in part.outputs[f"ids:{node.alias}"])
            if j == len(nodes) - 1:
                survivors = merged["survivors"].get(node.alias, survivors)
            results[member.node_id] = NodeResult(
                {f"ids:{node.alias}": survivors,
                 f"filter_answers:{node.alias}": rows},
                replace(sum((part.metrics for part in parts), NodeMetrics()),
                        wall_s=wall if j == 0 else 0.0,
                        extension={name: sum(part.metrics.extension.get(name, 0)
                                             for part in parts)
                                   for name in ("borrowed_prefix_tokens", "pack_s")}))
        return results

    def _execute_join(self, node, inputs):
        return self._execute_joins((node,), inputs)[node.node_id]

    def _execute_joins(self, nodes, inputs):
        survivors = dict(inputs["survivors"])
        group = inputs["group"]
        filtered_aliases = set(self.retained)
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
        encoded_nodes = [self.registry.codecs[member.type_name].encode(member)
                         for member in nodes]
        for sub in subs:
            sub.pop("joins", None)
            sub.update(
                final_group=nodes[-1].node_id == self.graph.nodes_by_type(
                    AiJoin.type_name)[-1].node_id
                and not self.graph.nodes_by_type(AiExtract.type_name),
                start_query=not self.started,
                physical_nodes=encoded_nodes,
            )
        started = time.perf_counter()
        outputs = self.round_fn("joins", subs)
        wall = time.perf_counter() - started
        self.started = True
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

        results = {}
        for j, (member, output) in enumerate(zip(nodes, stage_outputs)):
            output.update(anchor=member.anchor, partners=list(member.partners),
                          semantics=member.semantics, selectivity=member.selectivity,
                          written_pos=member.written_pos)
            survivors = coordinator.gate_group(output, member.semantics)
            metrics = sum((item["results"][member.node_id].metrics
                           for item in outputs), NodeMetrics())
            results[member.node_id] = NodeResult(
                {f"ids:{member.anchor}": survivors,
                 f"join_answers:{member.written_pos}": output},
                replace(metrics, wall_s=wall if j == 0 else 0.0,
                        extension={"joins": [output]}))
        self._retained_placement(outputs)
        return results

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


def prepare_distributed_inputs(node, inputs, context, nodes=None):
    if isinstance(node, AiJoin):
        grouped = nodes is not None
        nodes = (node,) if nodes is None else nodes
        inside = {member.node_id for member in nodes}
        survivors = {}
        pairs = {}
        for member, port in ((member, port) for member in nodes
                             for port in member.inputs):
            if port.source.node_id in inside:
                continue
            value = (inputs[member.node_id] if grouped else inputs)[port.name]
            if port.source.port.startswith("pairs:"):
                if not hasattr(value, "num_rows"):
                    raise TypeError(_ONE_GPU_APPLY)
                pairs[int(port.source.port.split(":", 1)[1])] = value
                continue
            survivors[port.source.port.split(":", 1)[1]] = list(value)
        return {
            "survivors": survivors,
            "group": [join_request(member) for member in nodes],
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
        workers=execution.gpu_count,
        run_pipeline=execution.execute_pipeline,
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
        backend_metrics={"inferences": [
            dict(value.metrics.extension)
            for node_id, value in result.nodes.items()
            if isinstance(graph.node(node_id), AiInference)
        ]},
        executed_join_plan=executed_join_plan(graph),
        node_metrics=scalar_node_metrics(result.nodes),
    )
    report.update(throughput(graph, result.metrics, elapsed))
    if payload.get("gpu_timing"):
        report.update(gpu_s=result.metrics.gpu_s, chunks=result.metrics.chunks)
    return report
