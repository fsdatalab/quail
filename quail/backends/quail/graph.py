"""Quail runtimes for the typed model graph."""

from __future__ import annotations

import itertools
import time

from quail.backends.quail.retention import apply_retention, retain_after_join
from quail.execution import export_physical_outputs
from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION
from quail.physical import (
    AnchoredJoin,
    PackedFilter,
    PhysicalGraph,
)
from quail.runtime.runner import (
    ExecutionContext,
    GenericRunner,
    ModelNodeRuntime,
    NodeResult,
    StreamedInput,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.runtime.tokens import DocumentPrefixes, chain_tokens


def quail_runtimes() -> dict:
    """Return runtimes for the Quail backend's physical nodes."""
    model_runtime = ModelNodeRuntime()
    return {
        PackedFilter.runtime_key: model_runtime,
        AnchoredJoin.runtime_key: model_runtime,
    }


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

    parts = []
    for alias, document in zip(join["partners"], member):
        parts.extend((join["labels"][alias], docs[alias][document]))
    parts.append(join["tail"])
    return chain_tokens(*parts)


def filter_inputs(state, node, document_ids) -> dict:
    """Scheduler inputs for one filter chain over the given documents."""
    return {
        "documents": DocumentPrefixes(
            state["pre"], state["docs"][node.alias], document_ids
        ),
        "document_ids": document_ids,
        "limit": state["filter_limit"],
        "retain_survivors": node.keep_kv,
    }


def prepare_model_inputs(node, inputs, context: ExecutionContext):
    """Prepare Quail scheduler inputs from typed port values."""
    state = context.state
    if isinstance(node, PackedFilter):
        state["pipeline"].attention_mode = FILTER_ATTENTION
        return filter_inputs(state, node, next(iter(inputs.values())))
    if not isinstance(node, AnchoredJoin):
        return inputs

    state["pipeline"].attention_mode = JOIN_ATTENTION
    stream = None
    by_alias = {}
    for input_port in node.inputs:
        if not input_port.source.port.startswith("ids:"):
            continue
        alias = input_port.source.port.split(":", 1)[1]
        value = inputs[input_port.name]
        if isinstance(value, StreamedInput):
            if alias != node.anchor or not isinstance(value.node, PackedFilter):
                raise TypeError(
                    "only the anchor's filter chain streams into a join")
            stream = value
        else:
            by_alias[alias] = list(value)
    if not state["joins_started"]:
        state["joins_started"] = True
        accounting = state["arena"].accounting
        state["kv_stats"].update(
            retained_after_filters=len(accounting.retained),
            retained_pages_after_filters=accounting.retained_pages,
            retained_prefix_tokens_after_filters=accounting.retained_prefix_tokens,
            retained_by_alias_after_filters={
                alias: sum(key[0] == alias for key in accounting.retained)
                for alias in state["docs"]
            },
        )
    config = state["retention"]
    apply_retention(state["arena"], config,
                    config.get("before", {}).get(node.node_id, {}), by_alias)
    group = [stage.runtime_spec() for stage in node.stages]
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
    if stream is None:
        anchor_ids = by_alias[node.anchor]
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
        anchor_stream = None
    else:
        # the join admits anchors as the chain hands them over; the
        # driver appends each one's key and prefix to these lists
        anchor_ids = None
        prefixes = []
        anchor_keys = []
        round_kv = None
        anchor_stream = {
            "node": stream.node,
            **filter_inputs(state, stream.node,
                            next(iter(stream.inputs.values()))),
        }

    def anchor_done(local_index, row):
        key = anchor_keys[local_index]
        matched = any(row)
        alive = not matched if group[-1]["semantics"] == "anti" \
            else matched
        if alive and node.keep_anchor_kv:
            retain_after_join(
                state["arena"], key, len(prefixes[local_index]), config,
                config.get("after", {}).get(node.node_id, {}))
        else:
            state["arena"].free_key(key)

    state["prepared_join"] = {
        "node": node,
        "group": group,
        "anchor_ids": anchor_ids,
        "anchor_keys": anchor_keys,
        "prefixes": prefixes,
        "tuple_indices": tuple_indices,
        "streamed": stream is not None,
    }
    return {
        "prefixes": prefixes,
        "stage_suffixes": stage_suffixes,
        "stage_frames": [join.get("frame") or [] for join in group],
        "anchor_keys": anchor_keys,
        "anchor_done": anchor_done,
        "anchor_stream": anchor_stream,
        "kv_round": round_kv,
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
        prepared = state["prepared_join"]
        keys = prepared["anchor_keys"]
        if prepared["streamed"]:
            # every streamed anchor read its KV from the chain: a hit
            state["kv_stats"]["join_anchor_hits"] += len(keys)
            anchor_ids = [key[1] for key in keys]
        else:
            anchor_ids = prepared["anchor_ids"]
        state["seen"].update(keys)
        live = set(result.outputs[f"ids:{node.anchor}"])
        for document, key, prefix in zip(
                anchor_ids, keys, prepared["prefixes"]):
            if key in state["arena"].accounting.owned:
                if node.keep_anchor_kv and document in live:
                    config = state["retention"]
                    retain_after_join(state["arena"], key, len(prefix), config,
                                      config.get("after", {}).get(node.node_id, {}))
                else:
                    state["arena"].free_key(key)


def execute_single_graph(state, payload, graph: PhysicalGraph) -> dict:
    """Execute one Quail model graph on one GPU executor."""
    torch = state["torch"]
    arena = state["arena"]
    for key in list(arena.accounting.owned):
        arena.free_key(key)
    arena.reset_stats()
    retention = payload.get("retention", {})
    apply_retention(arena, retention, retention.get("initial", {}))
    docs = state["docs"]
    sources = {
        alias: range(len(documents))
        for alias, documents in docs.items()
    }
    runtime_state = {
        **state,
        "pre": payload.get("pre_ids") or [],
        "retention": retention,
        "filter_limit": (
            None if any(isinstance(node, AnchoredJoin)
                        for node in graph.nodes)
            else payload.get("filter_limit")
        ),
        "joins_started": False,
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
        model_inputs=prepare_model_inputs,
        model_result=record_model_result,
        state=runtime_state,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        result = GenericRunner().run(compute_subgraph(graph), context)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    filters, joins = model_answers(graph, result)

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
        "_outputs": export_physical_outputs(compute_subgraph(graph), result),
        "wall_s": round(wall, 2),
        "fresh_tokens": result.metrics.fresh_tokens,
        "regret_tokens": runtime_state["regret_tokens"],
        "node_metrics": scalar_node_metrics(result.nodes),
        "executed_join_plan": executed_join_plan(graph),
        "kv_manager": kv_manager,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def model_answers(graph, result) -> tuple[dict, list]:
    """Collect filter and join answers from executed model nodes."""
    filters, joins = {}, []
    for node in graph.topological_nodes():
        if isinstance(node, PackedFilter):
            filters[node.alias] = result.nodes[node.node_id].outputs[
                f"filter_answers:{node.alias}"
            ]
        elif isinstance(node, AnchoredJoin):
            joins.extend(result.nodes[node.node_id].outputs[
                f"join_answers:{stage.written_pos}"
            ] for stage in node.stages)
    return filters, joins


def executed_join_plan(graph) -> list[dict]:
    """Describe the join nodes executed from the saved graph."""
    from quail.physical import Exchange

    return [
        {"type": node.type_name, "id": node.node_id,
         **node.explain_fields()}
        for node in graph.topological_nodes()
        if isinstance(node, (AnchoredJoin, Exchange))
    ]
