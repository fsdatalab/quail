"""AI.MAP: its spec, cost, and plan nodes."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.physical import AiMap, MapSpec
from quail.planner.operators.common import (
    bind_inference,
    emit_calls,
    inference_node,
    node_index,
)

# the planner's guess at an answer's mean length, before any answer is
# written; an answer limit below it caps it
EXPECTED_ANSWER_TOKENS = 64


def map_seconds(context, spec: MapSpec, expected_rows, resident=False,
                shared=None) -> float:
    """Return a map's estimated seconds over its documents.

    A map's answer is an extraction's with no drafts: the prompt with
    the document, then one chosen token per round, each round reading
    the document, the instruction, and the answer so far from KV.

    Args:
        context: The planning context, with the model and device.
        spec: The map's specification.
        expected_rows: Expected number of documents to map.
        resident: Whether the documents' KV is already read, by a step
            of the same pipeline before this one.
        shared: Per-document borrowed prefix lengths, or None.
    """
    head, tail = spec.prompt_token_parts
    return extract_cost.estimate(
        expected_rows, context.document_tokens[spec.aliases[0]], len(head),
        len(tail), chunk=budgets.chunk_budget(context.model, context.device),
        model=context.model, device=context.device, resident=resident,
        shared=shared, max_answer_tokens=spec.max_answer_tokens,
        draft_tokens=0, draft_policy="fixed", gpus=context.gpus,
        assumptions=extract_cost.ExtractAssumptions(spec.expected_answer_tokens),
        memo=context.memo).seconds


def map_spec(context, call, name, expected_rows, resident=False) -> MapSpec:
    """Build a map's specification and price it.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.MAP call.
        name: Result column name.
        expected_rows: Expected number of documents to map.
        resident: Whether the documents' KV is already read, by a step
            of the same pipeline before this one.
    """
    prompt = context.lowered(call.prompt)
    spec = bind_inference(
        MapSpec, call.prompt, name=name, expected_rows=expected_rows,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        max_answer_tokens=call.max_answer_tokens,
        expected_answer_tokens=min(EXPECTED_ANSWER_TOKENS,
                                   call.max_answer_tokens))
    return replace(spec, estimated_seconds=map_seconds(
        context, spec, expected_rows, resident))


def emit_map(builder, alias, expected_rows, names, port, resident=False) -> tuple:
    """Write the answers of the alias's documents, one node per map.

    The maps of a table run in one pipeline after the table's other
    steps: a document is read once for all of them.

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
    calls = {node.name: node.call for node in builder.operators.maps
             if node.alias == alias}

    def node_of(name, port, expected_rows, index):
        spec = map_spec(builder.context, calls[name], name, expected_rows,
                        resident=resident or index > 0)
        return inference_node(
            AiMap, spec, (port,), index=node_index(builder, AiMap),
            backend="quail", model=builder.model.name)

    return emit_calls(builder, alias, expected_rows, port, names, node_of)
