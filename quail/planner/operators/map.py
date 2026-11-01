"""AI.MAP: its spec, cost, and plan nodes."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.physical import AiMap, MapSpec
from quail.planner.operators.common import ColumnPlanner, bind_inference

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


class MapPlanner(ColumnPlanner):
    """Build and price AI.MAP nodes."""

    node_type = AiMap

    def spec(self, context, node, expected_rows, resident=False) -> MapSpec:
        call = node.call
        prompt = context.lowered(call.prompt)
        spec = bind_inference(
            MapSpec, call.prompt, name=node.name, expected_rows=expected_rows,
            prompt_token_parts=(tuple(prompt.preamble_token_ids),
                                tuple(prompt.tail_token_ids)),
            max_answer_tokens=call.max_answer_tokens,
            expected_answer_tokens=min(EXPECTED_ANSWER_TOKENS,
                                       call.max_answer_tokens))
        return replace(spec, estimated_seconds=map_seconds(
            context, spec, expected_rows, resident))
