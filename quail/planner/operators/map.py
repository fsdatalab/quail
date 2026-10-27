"""AI.MAP: its spec and cost."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.physical import AiMap, MapSpec, PortRef
from quail.planner.operators.common import (
    bind_inference,
    inference_node,
)


def map_spec(context, call, name, expected_rows, resident=False) -> MapSpec:
    """Build a map's specification and price it.

    The specification carries the token ids of the prompt around the
    document. The cost reuses the extraction estimate: a map runs the
    same rounds, with the map's answer limit and no draft, and its
    answer lengths are assumed geometric with the expected mean. The
    documents' KV is priced as computed afresh unless the map shares a
    pipeline with the step before it, which read the documents.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.MAP call.
        name: Result column name.
        expected_rows: Expected number of documents to map over.
        resident: Whether the documents' KV is already read, by a
            step of the same pipeline before this one.
    """
    prompt = context.lowered(call.prompt)
    (ref,) = call.prompt.args
    lengths = context.document_tokens[ref.alias]
    model, device = context.model, context.device
    spec = bind_inference(
        MapSpec, call.prompt, name=name,
        expected_rows=expected_rows, estimated_seconds=0.0,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        instruction=call.instruction, **_answer_limits(call))
    cost = extract_cost.estimate(
        expected_rows,
        lengths,
        len(prompt.preamble_token_ids),
        len(prompt.tail_token_ids),
        chunk=budgets.chunk_budget(context.model, context.device),
        model=context.model,
        device=context.device,
        resident=resident,
        max_answer_tokens=spec.max_answer_tokens,
        draft_tokens=0,
        draft_policy="fixed",
        round_cost_per_token=budgets.round_cost_per_token(model, device),
        gpus=context.gpus,
        assumptions=extract_cost.ExtractAssumptions(spec.expected_answer_tokens),
        memo=context.memo,
    )
    return replace(spec, estimated_seconds=cost.seconds)


def _answer_limits(call) -> dict:
    """Return a map call's answer limit and expected length, when it sets one."""
    if not call.max_answer_tokens:
        return {}
    expected = min(MapSpec.expected_answer_tokens, call.max_answer_tokens)
    return {"max_answer_tokens": call.max_answer_tokens,
            "expected_answer_tokens": expected}


def emit_map(builder, alias, expected_rows, names, port, resident=False) -> tuple:
    """Write the answers of the alias's documents.

    Args:
        builder: The graph builder.
        alias: The table.
        expected_rows: The documents the maps read.
        names: The maps to emit, in run order.
        port: The port the documents stand at before the maps.
        resident: Whether the step before the run left the documents'
            KV for the first map to read.

    Returns:
        Where the documents stand after them, and the rows ports of
        the answers SELECT returns.
    """
    semantic_maps = {node.name: node for node in builder.operators.maps
                     if node.alias == alias}
    calls = [semantic_maps[name] for name in names]
    row_ports = []
    for index, logical_node in enumerate(calls):
        # the maps of a table run in one pipeline
        # (quail.physical.pipelines): a document is read once for all
        # of them
        spec = map_spec(
            builder.context,
            logical_node.call,
            logical_node.name,
            expected_rows,
            resident=resident or index > 0,
        )
        node = inference_node(
            AiMap, spec, (port,),
            index=sum(isinstance(n, AiMap) for n in builder.nodes),
            backend="quail", model=builder.model.name)
        builder.nodes.append(node)
        port = PortRef(node.node_id, node.rows_port)
        row_ports.append(port)
    return port, row_ports
