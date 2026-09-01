"""Quail runtimes for the typed model graph."""

from __future__ import annotations

import itertools
import time
from dataclasses import replace

from quail.physical import (
    AdaptiveJoinPlan,
    AnchoredJoin,
    DocumentScan,
    Exchange,
    JoinStage,
    PackedFilter,
    PhysicalGraph,
    PortRef,
)
from quail.physical.base import input_ports
from quail.runtime.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
)


def _join_round_kv(anchor_keys, prefix_lens, owned, seen) -> dict:
    hits = 0
    regret = 0
    for key, n_tokens in zip(anchor_keys, prefix_lens):
        if key in owned:
            hits += 1
        elif key in seen:
            regret += n_tokens
    return {
        "hits": hits,
        "misses": len(anchor_keys) - hits,
        "regret_tokens": regret,
    }


def _tuple_suffix(join, docs, member):
    from quail.runtime.tokens import chain_tokens

    parts = []
    for alias, document in zip(join["partners"], member):
        parts.extend((join["labels"][alias], docs[alias][document]))
    parts.append(join["tail"])
    return chain_tokens(*parts)


def _possible_anchors(indices, all_specs) -> set[str]:
    possible = set()
    for index in indices:
        spec = all_specs[index]
        if spec["semantics"] == "full" and spec.get("anchor_free"):
            possible.update(spec["aliases"])
        else:
            possible.add(spec["anchor"])
    return possible


def _runtime_stage(join, index: int, anchor: str,
                   expected_tuples: int, resident: str) -> JoinStage:
    partners = tuple(alias for alias in join["aliases"] if alias != anchor)
    return JoinStage(
        written_pos=int(join.get("written_pos", index)),
        exec_idx=index,
        anchor=anchor,
        partners=partners,
        semantics=join["semantics"],
        selectivity=join.get("selectivity"),
        expected_tuples=float(expected_tuples),
        anchor_frame_tokens=len(join["frames"][anchor]),
        pair_tail_tokens=len(join["tail"]),
        anchor_resident=resident,
        tuple_tokens=0.0,
    )


def _next_join(state) -> AnchoredJoin:
    from quail.planner.joins import search_joins, summarize_alias
    from quail.runtime.coordinator import runtime_join_steps

    remaining = state["remaining"]
    all_specs = state["search_specs"]
    joins = state["joins"]
    survivors = state["survivors"]
    docs = state["docs"]
    arena = state["arena"]
    specs = [all_specs[index] for index in sorted(remaining)]
    involved = sorted({
        alias for spec in specs for alias in spec["aliases"]
    })
    found = search_joins(
        specs,
        {alias: float(len(survivors[alias])) for alias in involved},
        {
            alias: summarize_alias(
                (len(docs[alias][document])
                 for document in survivors[alias]),
                resident_flags=(
                    (alias, document) in arena.accounting.owned
                    for document in survivors[alias]
                ),
            )
            for alias in involved
        },
        {},
        len(state["pre"]),
        state["chunk_tokens"],
        state["model_spec"],
        state["device"],
        fixed_order=state["order_rule"] == "as_written",
        arena_tokens=float(
            arena.accounting.n_pages * arena.accounting.page_tokens
        ),
        page_tokens=arena.accounting.page_tokens,
        already_joined=state["already_joined"],
    )
    if found is not None:
        state["optimizer_runs"].append(found)
        nodes = runtime_join_steps(found["seq"], joins)
        selected = next(
            node for node in nodes if isinstance(node, AnchoredJoin)
        )
    else:
        ordered = sorted(
            remaining,
            key=lambda index: joins[index].get("written_pos", index),
        )
        first = ordered[0]
        anchor = joins[first]["anchor"]
        group = [first]
        if joins[first]["semantics"] == "full":
            for index in ordered[1:]:
                join = joins[index]
                if join["semantics"] != "full" or join["anchor"] != anchor:
                    break
                group.append(index)
        selected = AnchoredJoin(
            node_id=f"runtime-join:{len(state['executed_nodes'])}",
            anchor=anchor,
            stage_idxs=tuple(group),
        )

    anchor = selected.anchor
    stages = []
    for index in selected.stage_idxs:
        join = joins[index]
        partner_count = 1
        for alias in join["aliases"]:
            if alias != anchor:
                partner_count *= len(survivors[alias])
        keys = [(anchor, document) for document in survivors[anchor]]
        resident = "all" if keys and all(
            key in arena.accounting.owned for key in keys
        ) else "some" if any(
            key in arena.accounting.owned for key in keys
        ) else "none"
        stages.append(_runtime_stage(
            join,
            index,
            anchor,
            len(survivors[anchor]) * partner_count,
            resident,
        ))
    return replace(selected, stages=tuple(stages))


def prepare_model_inputs(node, inputs, context: ExecutionContext):
    """Prepare Quail scheduler inputs from typed port values."""
    from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
    from quail.runtime.coordinator import stage_for_anchor
    from quail.runtime.tokens import chain_tokens

    state = context.state
    if isinstance(node, PackedFilter):
        state["pipeline"].attention_mode = FILTER_ATTENTION
        document_ids = list(next(iter(inputs.values())))
        return {
            "documents": [
                chain_tokens(state["pre"], state["docs"][node.alias][index])
                for index in document_ids
            ],
            "document_ids": document_ids,
            "limit": state["filter_limit"],
            "retain_survivors": (
                range(len(document_ids)) if node.keep_kv else ()
            ),
        }
    if not isinstance(node, AnchoredJoin):
        return inputs

    state["pipeline"].attention_mode = JOIN_ATTENTION
    by_alias = {
        input_port.source.port.split(":", 1)[1]: list(
            inputs[input_port.name]
        )
        for input_port in node.inputs
        if input_port.source.port.startswith("ids:")
    }
    anchor_ids = by_alias[node.anchor]
    joins = state["joins"]
    group = [
        stage_for_anchor(joins[index], node.anchor)
        for index in node.stage_idxs
    ]
    stage_suffixes = []
    tuple_indices = {}
    for stage, join in zip(node.stages, group):
        tuples = [list(member) for member in itertools.product(
            *[by_alias[alias] for alias in join["partners"]]
        )]
        tuple_indices[stage.written_pos] = tuples
        stage_suffixes.append([
            _tuple_suffix(join, state["docs"], member)
            for member in tuples
        ])
    prefixes = [
        chain_tokens(state["pre"], state["docs"][node.anchor][document])
        for document in anchor_ids
    ]
    anchor_keys = [(node.anchor, document) for document in anchor_ids]
    round_kv = _join_round_kv(
        anchor_keys,
        [len(prefix) for prefix in prefixes],
        state["arena"].accounting.owned,
        state["seen"],
    )
    state["kv_stats"]["join_anchor_hits"] += round_kv["hits"]
    state["kv_stats"]["join_anchor_misses"] += round_kv["misses"]
    state["regret_tokens"] += round_kv["regret_tokens"]
    future = state["future_anchors"]

    def anchor_done(local_index, row):
        key = anchor_keys[local_index]
        matched = any(row)
        alive = not matched if group[-1]["semantics"] == "anti" \
            else matched
        if alive and node.anchor in future:
            state["arena"].retain(key, len(prefixes[local_index]))
        else:
            state["arena"].free_key(key)

    state["prepared_join"] = {
        "node": node,
        "group": group,
        "anchor_ids": anchor_ids,
        "anchor_keys": anchor_keys,
        "prefixes": prefixes,
        "tuple_indices": tuple_indices,
    }
    return {
        "prefixes": prefixes,
        "stage_suffixes": stage_suffixes,
        "stage_frames": [join.get("frame") or [] for join in group],
        "anchor_keys": anchor_keys,
        "anchor_done": anchor_done,
        "anchor_ids": anchor_ids,
        "partner_indices": tuple_indices,
        "group": group,
    }


def record_model_result(node, result: NodeResult,
                        context: ExecutionContext) -> None:
    """Record keys computed by the current query."""
    state = context.state
    if isinstance(node, PackedFilter):
        answers = result.outputs[f"filter_answers:{node.alias}"]
        state["seen"].update((node.alias, document)
                             for document in answers)
    elif isinstance(node, AnchoredJoin):
        state["seen"].update(state["prepared_join"]["anchor_keys"])


def _child_graph(node: AnchoredJoin, previous_anchor: str | None,
                 aliases: tuple[str, ...]) -> PhysicalGraph:
    scan_nodes = tuple(
        DocumentScan(
            node_id=f"{node.node_id}:scan:{alias}",
            alias=alias,
            provider="runtime",
            column="tokens",
        )
        for alias in aliases
    )
    sources = {
        alias: PortRef(f"{node.node_id}:scan:{alias}", f"ids:{alias}")
        for alias in aliases
    }
    nodes = list(scan_nodes)
    if previous_anchor is not None and previous_anchor != node.anchor:
        exchange = Exchange(
            node_id=f"runtime-exchange:{node.node_id}",
            inputs=input_ports(tuple(
                sources[alias].to_tuple() for alias in aliases
            )),
            next_anchor=node.anchor,
            aliases=aliases,
        )
        nodes.append(exchange)
        sources = {
            alias: PortRef(exchange.node_id, f"ids:{alias}")
            for alias in aliases
        }
    anchored = replace(
        node,
        inputs=input_ports(tuple(
            sources[alias].to_tuple() for alias in aliases
        )),
    )
    nodes.append(anchored)
    return PhysicalGraph(
        tuple(nodes), PortRef(anchored.node_id, f"ids:{anchored.anchor}")
    )


def run_adaptive_join(node, inputs, context: ExecutionContext) -> NodeResult:
    """Plan and execute Quail join child graphs from actual survivors."""
    from quail.runtime.coordinator import (
        gate_group,
        report_join_plan,
        search_specs,
        thin_survivors,
    )

    state = context.state
    survivors = {}
    for input_port in node.inputs:
        alias = input_port.source.port.split(":", 1)[1]
        survivors[alias] = list(inputs[input_port.name])
    for alias, documents in state["docs"].items():
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
        prepared_join=None,
    )
    state["kv_stats"].update(
        retained_after_filters=len(state["arena"].accounting.retained),
        retained_pages_after_filters=(
            state["arena"].accounting.retained_pages
        ),
        retained_prefix_tokens_after_filters=(
            state["arena"].accounting.retained_prefix_tokens
        ),
    )
    finished_full = []
    out_joins = []
    pairs = {}
    remaining = state["remaining"]

    initial_anchors = _possible_anchors(remaining, state["search_specs"])
    for key in [
        key for key in list(state["arena"].accounting.retained)
        if key[0] not in initial_anchors
    ]:
        state["arena"].free_key(key)

    previous_anchor = None
    metrics = NodeMetrics()
    while remaining:
        selected = _next_join(state)
        state["optimizer_sequence"].extend(
            (
                state["joins"][index].get("written_pos", index),
                selected.anchor,
            )
            for index in selected.stage_idxs
        )
        remaining.difference_update(selected.stage_idxs)
        state["future_anchors"] = _possible_anchors(
            remaining, state["search_specs"]
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
        metrics = metrics + child_result.metrics
        joined = child_result.nodes[selected.node_id]
        prepared = state["prepared_join"]
        answers = joined.metrics.extension["answers"]
        group = prepared["group"]
        anchor_ids = prepared["anchor_ids"]
        for stage, stage_answers, join in zip(
                selected.stages, answers, group):
            stage_out = {
                "rows": {
                    int(local): row
                    for local, row in stage_answers.items()
                },
                "anchor_index": anchor_ids,
                "partner_index": prepared["tuple_indices"][
                    stage.written_pos
                ],
                "anchor": selected.anchor,
                "partners": list(stage.partners),
                "semantics": join["semantics"],
                "selectivity": join.get("selectivity"),
                "written_pos": stage.written_pos,
            }
            out_joins.append(stage_out)
            if stage.semantics == "full":
                finished_full.append(stage_out)
                pairs[stage.written_pos] = stage_out

        survivors[selected.anchor] = list(
            joined.outputs[f"ids:{selected.anchor}"]
        )
        alive_after = set(survivors[selected.anchor])
        for document, key, prefix in zip(
                anchor_ids,
                prepared["anchor_keys"],
                prepared["prefixes"]):
            if key not in state["arena"].accounting.owned:
                continue
            if document in alive_after \
                    and selected.anchor in state["future_anchors"]:
                state["arena"].retain(key, len(prefix))
            else:
                state["arena"].free_key(key)
        for key in [
            key for key in list(state["arena"].accounting.retained)
            if key[0] not in state["future_anchors"]
        ]:
            state["arena"].free_key(key)
        before = {alias: set(ids) for alias, ids in survivors.items()}
        thin_survivors(finished_full, survivors)
        for alias, old_ids in before.items():
            for document in old_ids - set(survivors[alias]):
                key = (alias, document)
                if key in state["arena"].accounting.owned:
                    state["arena"].free_key(key)
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
    outputs.update({f"pairs:{position}": pairs[position]
                    for position in node.full_join_positions})
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
        joins=out_joins,
        join_optimizer=optimizer,
        executed_nodes=tuple(state["executed_nodes"]),
    )
    return NodeResult(
        outputs,
        replace(
            metrics,
            regret_tokens=state["regret_tokens"],
            extension=extension,
        ),
    )


def model_subgraph(graph: PhysicalGraph) -> PhysicalGraph:
    """Return the section executed inside the Modal container."""
    nodes = tuple(
        node for node in graph.nodes
        if isinstance(node, (DocumentScan, PackedFilter, AdaptiveJoinPlan))
    )
    adaptive = next(
        (node for node in nodes if isinstance(node, AdaptiveJoinPlan)), None
    )
    if adaptive is not None:
        root = PortRef(adaptive.node_id, adaptive.outputs[0].name)
    else:
        filters = [node for node in nodes if isinstance(node, PackedFilter)]
        if not filters:
            raise ValueError("Quail model graph has no model operation")
        root = PortRef(filters[-1].node_id, filters[-1].outputs[0].name)
    return PhysicalGraph(nodes, root)


def execute_single_graph(state, payload, graph: PhysicalGraph) -> dict:
    """Execute one Quail model graph on one GPU executor."""
    torch = state["torch"]
    arena = state["arena"]
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()
    docs = state["docs"]
    sources = {
        alias: list(range(len(documents)))
        for alias, documents in docs.items()
    }
    runtime_state = {
        **state,
        "pre": payload.get("pre_ids") or [],
        "filter_limit": (
            None if any(isinstance(node, AdaptiveJoinPlan)
                        for node in graph.nodes)
            else payload.get("limit")
        ),
        "order_rule": payload.get("order_rule", "as_written"),
        "seen": set(),
        "regret_tokens": 0,
        "kv_stats": {
            "retained_after_filters": 0,
            "retained_pages_after_filters": 0,
            "retained_prefix_tokens_after_filters": 0,
            "join_anchor_hits": 0,
            "join_anchor_misses": 0,
        },
    }
    context = ExecutionContext(
        runtimes=state["runtimes"],
        model_execution=state["model_execution"],
        sources=sources,
        adaptive_join=run_adaptive_join,
        model_inputs=prepare_model_inputs,
        model_result=record_model_result,
        state=runtime_state,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        result = GenericRunner().run(model_subgraph(graph), context)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

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

    if adaptive_result is not None:
        joins = adaptive_result.metrics.extension["joins"]
        optimizer = adaptive_result.metrics.extension["join_optimizer"]
    else:
        joins = []
        optimizer = None

    for key in list(arena.accounting.owned):
        arena.free_key(key)
    kv_manager = {
        **runtime_state["kv_stats"],
        "evicted_keys": arena.evicted_keys,
        "evicted_pages": arena.evicted_pages,
        "evicted_prefix_tokens": arena.evicted_prefix_tokens,
    }
    return {
        "filters": filters,
        "joins": joins,
        "wall_s": round(wall, 2),
        "fresh_tokens": result.metrics.fresh_tokens,
        "regret_tokens": runtime_state["regret_tokens"],
        "join_optimizer": optimizer,
        "kv_manager": kv_manager,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
