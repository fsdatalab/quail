"""Typed Quail graph execution across GPU child processes."""

from __future__ import annotations

import time

from quail.physical import (
    AdaptiveJoinPlan,
    AnchoredJoin,
    DocumentInput,
    Exchange,
    PackedFilter,
    PhysicalGraph,
)
from quail.runtime.quail_graph import (
    _child_graph,
    _next_join,
    _possible_anchors,
    scalar_node_metrics,
)
from quail.runtime.runner import (
    compute_subgraph,
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
)


class DistributedAccounting:
    """Combined read only KV accounting for runtime join planning."""

    def __init__(self, execution):
        self.execution = execution
        self.n_pages = execution.total_pages
        self.page_tokens = execution.page_tokens

    @property
    def owned(self):
        return {
            (alias, document)
            for alias, documents in self.execution.retained.items()
            for document in documents
        }

    @property
    def retained(self):
        return self.owned

    @property
    def retained_pages(self):
        return sum(
            -(-(
                len(self.execution.pre)
                + len(self.execution.docs[alias][document])
            ) // self.page_tokens)
            for alias, documents in self.execution.retained.items()
            for document in documents
        )

    @property
    def retained_prefix_tokens(self):
        return sum(
            len(self.execution.pre)
            + len(self.execution.docs[alias][document])
            for alias, documents in self.execution.retained.items()
            for document in documents
        )


class DistributedArenaView:
    """KV capacity and residency visible to the coordinator planner."""

    def __init__(self, execution):
        self.accounting = DistributedAccounting(execution)


class DistributedQuailExecution:
    """Dispatch typed Quail model nodes to one child per GPU."""

    def __init__(self, payload, graph, gpu_count, round_fn,
                 model_spec, device, registry):
        from quail.planner import budgets

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
            from quail.planner.decide import balanced_shards
            for alias in self.filter_aliases - set(self.shards):
                shards, _ = balanced_shards(
                    [len(document) for document in self.docs[alias]],
                    gpu_count,
                )
                self.shards[alias] = shards
        adaptive = next(
            (node for node in graph.nodes
             if isinstance(node, AdaptiveJoinPlan)),
            None,
        )
        self.joins = [] if adaptive is None else list(adaptive.join_specs)
        self.pre = payload.get("pre_ids") or []
        self.page_tokens = budgets.PAGE_TOKENS
        self.total_pages = (
            budgets.arena_tokens(
                model_spec, device, payload["chunk_tokens"]
            ) // self.page_tokens
        ) * gpu_count
        self.retained: dict[str, set[int]] = {}
        self.prior_shards: dict[str, list[list[int]]] = {}
        self.started = False
        self.boot_outputs = []
        self.child_totals = [None] * gpu_count
        self.peak_gib = 0.0
        self.kv_stats = {
            "retained_after_filters": 0,
            "retained_pages_after_filters": 0,
            "retained_prefix_tokens_after_filters": 0,
            "join_anchor_hits": 0,
            "join_anchor_misses": 0,
        }
        self.arena_view = DistributedArenaView(self)

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
        from quail.runtime import coordinator

        subs = coordinator.begin_query_payloads(
            self.payload, self.gpu_count
        )
        for sub in subs:
            sub["start_query"] = True
        outputs = self.round_fn("filters", subs)
        self.started = True
        self.boot_outputs.extend(outputs)
        self.peak_gib = max(
            self.peak_gib,
            *(output.get("peak_gib", 0.0) for output in outputs),
        )

    def _execute_filter(self, node, inputs):
        from quail.runtime import coordinator

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
        self.boot_outputs.extend(outputs)
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
        for output in outputs:
            for alias, documents in output.get("retained", {}).items():
                self.retained.setdefault(alias, set()).update(documents)
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
        from quail.runtime import coordinator

        survivors = inputs["survivors"]
        group = inputs["group"]
        future = set(inputs["future_anchors"])
        drop = [
            alias for alias in self.retained
            if alias not in future and alias != node.anchor
        ]
        for alias in drop:
            self.retained.pop(alias, None)
            self.prior_shards.pop(alias, None)
        subs = coordinator.join_group_payloads(
            self._runtime_payload(),
            self.gpu_count,
            survivors,
            group,
            prior_shards=self.prior_shards,
            filtered_aliases=self.filter_aliases,
            shards=self.shards,
        )
        encoded_node = self.registry.codecs[node.type_name].encode(node)
        for sub in subs:
            sub.pop("joins", None)
            sub.update(
                retain_anchor=node.anchor in future,
                drop_kept=drop,
                final_group=not inputs["remaining"],
                start_query=not self.started,
                physical_node=encoded_node,
            )
        started = time.perf_counter()
        outputs = self.round_fn("joins", subs)
        wall = time.perf_counter() - started
        self.started = True
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
        return NodeResult(
            {
                f"ids:{node.anchor}": anchor_survivors,
                **answer_outputs,
            },
            NodeMetrics(
                wall_s=wall,
                input_rows=len(inputs["survivors"][node.anchor]),
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
        )

    def snapshot_after_filters(self):
        accounting = self.arena_view.accounting
        self.kv_stats.update(
            retained_after_filters=len(accounting.retained),
            retained_pages_after_filters=accounting.retained_pages,
            retained_prefix_tokens_after_filters=(
                accounting.retained_prefix_tokens
            ),
        )

    def reconcile_retained(self, survivors):
        for alias in list(self.retained):
            self.retained[alias].intersection_update(survivors[alias])
            if not self.retained[alias]:
                self.retained.pop(alias)
                self.prior_shards.pop(alias, None)
                continue
            alive = self.retained[alias]
            self.prior_shards[alias] = [
                [document for document in shard if document in alive]
                for shard in self.prior_shards[alias]
            ]

    def report(self):
        for totals in self.child_totals:
            for key, value in (totals or {}).items():
                self.kv_stats[key] = self.kv_stats.get(key, 0) + value
        boot_s = max(
            (output.get("boot_s", 0.0) for output in self.boot_outputs),
            default=0.0,
        )
        slowest = max(
            self.boot_outputs,
            key=lambda output: output.get("boot_s", 0.0),
            default={},
        )
        return {
            "boot_s": round(boot_s, 2),
            "boot_kind": slowest.get("boot_kind"),
            "boot": slowest.get("boot"),
            "kv_manager": dict(self.kv_stats),
            "peak_gib": self.peak_gib,
        }


def prepare_distributed_inputs(node, inputs, context):
    if isinstance(node, PackedFilter):
        return inputs
    if isinstance(node, AnchoredJoin):
        from quail.runtime.coordinator import stage_for_anchor

        state = context.state
        return {
            "survivors": state["survivors"],
            "group": [
                stage_for_anchor(state["joins"][index], node.anchor)
                for index in node.stage_idxs
            ],
            "future_anchors": state["future_anchors"],
            "remaining": state["remaining"],
        }
    return inputs


def run_distributed_adaptive(node, inputs, context):
    """Plan joins and execute typed children across GPU processes."""
    from quail.runtime.coordinator import (
        report_join_plan,
        search_specs,
        thin_survivors,
    )

    state = context.state
    execution = state["distributed_execution"]
    survivors = {}
    for input_port in node.inputs:
        alias = input_port.source.port.split(":", 1)[1]
        survivors[alias] = list(inputs[input_port.name])
    for alias, documents in execution.docs.items():
        survivors.setdefault(alias, list(range(len(documents))))
    state.update(
        survivors=survivors,
        joins=list(node.join_specs),
        search_specs=search_specs(list(node.join_specs)),
        remaining=set(range(len(node.join_specs))),
        already_joined=set(),
        optimizer_runs=[],
        optimizer_sequence=[],
        executed_nodes=[],
        arena=execution.arena_view,
        docs=execution.docs,
        pre=execution.pre,
        chunk_tokens=execution.payload["chunk_tokens"],
        model_spec=execution.model_spec,
        device=execution.device,
        order_rule=execution.payload.get("order_rule", "as_written"),
    )
    execution.snapshot_after_filters()
    finished_full = []
    all_joins = []
    join_answers = {}
    previous_anchor = None
    metrics = NodeMetrics()
    while state["remaining"]:
        selected = _next_join(state)
        state["optimizer_sequence"].extend(
            (
                state["joins"][index].get("written_pos", index),
                selected.anchor,
            )
            for index in selected.stage_idxs
        )
        state["remaining"].difference_update(selected.stage_idxs)
        state["future_anchors"] = _possible_anchors(
            state["remaining"], state["search_specs"]
        )
        aliases = tuple(dict.fromkeys(
            alias
            for stage in selected.stages
            for alias in (stage.anchor, *stage.partners)
        ))
        mutable_sources = context.sources
        if not isinstance(mutable_sources, dict):
            raise TypeError("adaptive join needs mutable runtime sources")
        for alias in aliases:
            mutable_sources[alias] = list(survivors[alias])
        child = _child_graph(selected, previous_anchor, aliases)
        child_result = context.execute_graph(child)
        joined = child_result.nodes[selected.node_id]
        metrics = metrics + child_result.metrics
        stage_outputs = joined.metrics.extension["joins"]
        all_joins.extend(stage_outputs)
        for stage, output in zip(selected.stages, stage_outputs):
            if stage.semantics == "full":
                finished_full.append(output)
            join_answers[stage.written_pos] = output
        survivors[selected.anchor] = list(
            joined.outputs[f"ids:{selected.anchor}"]
        )
        thin_survivors(finished_full, survivors)
        execution.reconcile_retained(survivors)
        for index in selected.stage_idxs:
            state["already_joined"].update(
                state["search_specs"][index]["aliases"]
            )
        state["executed_nodes"].extend(
            child_node
            for child_node in child.nodes
            if isinstance(child_node, (Exchange, AnchoredJoin))
        )
        previous_anchor = selected.anchor

    outputs = {
        f"ids:{alias}": list(survivors[alias]) for alias in node.aliases
    }
    outputs.update({
        f"join_answers:{position}": join_answers[position]
        for position in node.join_positions
    })
    optimizer_runs = state["optimizer_runs"]
    optimizer = None if not optimizer_runs else {
        "states": sum(run["states"] for run in optimizer_runs),
        "generated": sum(run["generated"] for run in optimizer_runs),
        "replans": len(optimizer_runs),
        "sequence": [list(step)
                     for step in state["optimizer_sequence"]],
        "executed_plan": report_join_plan(
            state["optimizer_sequence"], state["joins"]
        ),
    }
    extension = dict(metrics.extension)
    extension.update(
        joins=all_joins,
        join_optimizer=optimizer,
        executed_nodes=tuple(state["executed_nodes"]),
    )
    return NodeResult(
        outputs,
        NodeMetrics(
            wall_s=metrics.wall_s,
            input_rows=metrics.input_rows,
            output_rows=metrics.output_rows,
            evaluated_documents=metrics.evaluated_documents,
            evaluated_document_pairs=metrics.evaluated_document_pairs,
            fresh_tokens=metrics.fresh_tokens,
            cached_tokens=metrics.cached_tokens,
            kv_hits=metrics.kv_hits,
            kv_misses=metrics.kv_misses,
            kv_removals=metrics.kv_removals,
            kv_recomputations=metrics.kv_recomputations,
            regret_tokens=metrics.regret_tokens,
            peak_gpu_bytes=metrics.peak_gpu_bytes,
            extension=extension,
        ),
    )


def execute_distributed_graph(payload, graph: PhysicalGraph, gpu_count: int,
                              round_fn, model_spec, device, runtimes,
                              registry=None) -> dict:
    """Execute one typed Quail graph across GPU child processes."""
    if registry is None:
        from quail.extensions import registry_from_modules
        registry = registry_from_modules(tuple(
            payload["physical_plan"].get("extension_modules", ())
        ))
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
        adaptive_join=run_distributed_adaptive,
        model_inputs=prepare_distributed_inputs,
        state={"distributed_execution": execution},
    )
    started = time.perf_counter()
    if not any(isinstance(node, PackedFilter) for node in graph.nodes):
        execution.begin()
    result = GenericRunner().run(compute_subgraph(graph), context)
    elapsed = time.perf_counter() - started
    filters = {}
    adaptive_result = None
    for physical_node in graph.nodes:
        if isinstance(physical_node, PackedFilter):
            node_result = result.nodes[physical_node.node_id]
            filters[physical_node.alias] = node_result.outputs[
                f"filter_answers:{physical_node.alias}"
            ]
        elif isinstance(physical_node, AdaptiveJoinPlan):
            adaptive_result = result.nodes[physical_node.node_id]
    joins = [] if adaptive_result is None else \
        adaptive_result.metrics.extension["joins"]
    optimizer = None if adaptive_result is None else \
        adaptive_result.metrics.extension["join_optimizer"]
    report = execution.report()
    from quail.execution import export_physical_outputs

    report.update(
        filters=filters,
        joins=joins,
        _outputs=export_physical_outputs(compute_subgraph(graph), result),
        wall_s=round(elapsed - report["boot_s"], 2),
        fresh_tokens=result.metrics.fresh_tokens,
        regret_tokens=result.metrics.regret_tokens,
        join_optimizer=optimizer,
        node_metrics=scalar_node_metrics(result.nodes),
    )
    return report
