"""AI.EXTRACT: its spec and cost."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.logical import effective_selectivity
from quail.logical.prompts import (
    shared_preamble,
)
from quail.physical import (
    AiExtract,
    Condition,
    ExtractSpec,
    Filter,
    PortRef,
)
from quail.physical.base import input_ports
from quail.planner.operators.common import bind_inference, inference_node


def extract_spec(context, call, name, live, shared=False) -> ExtractSpec:
    """Build an extraction's specification and price it.

    The specification carries the token ids of the prompt around the
    document. The documents' KV is priced as computed afresh unless
    the extraction shares a pipeline with the one before it, which
    read the documents.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.EXTRACT call.
        name: Result column name.
        live: Expected number of documents to extract from.
        shared: Whether an extraction of the same table runs right
            before this one.
    """
    prompt = context.lowered(call.prompt)
    # under the query's shared preamble, the prompt's document KV is
    # the one a filter writes
    shares_head = tuple(prompt.preamble_token_ids) == tuple(
        context.tokenizer(shared_preamble(
            context.model.turn_prefix, context.model.prompt_layout)))
    (ref,) = call.prompt.args
    lengths = context.document_tokens[ref.alias]
    spec = bind_inference(
        ExtractSpec, call.prompt, name=name,
        expected_inputs=live, estimated_seconds=0.0,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        question=call.question, trim=call.trim, shares_head=shares_head)
    cost = extract_cost.estimate(
        live, lengths, len(prompt.preamble_token_ids), len(prompt.tail_token_ids),
        chunk=budgets.chunk_budget(context.model, context.device),
        model=context.model, device=context.device, resident=shared,
        max_answer_tokens=spec.max_answer_tokens, draft_tokens=spec.draft_tokens,
        workers=context.gpu_count)
    return replace(spec, estimated_seconds=cost.seconds)


def emit_extract(builder, alias, live=None):
    """Copy the answers of the alias's documents, after its labels.

    A Filter on an extraction's answer or span follows that extraction
    and keeps the documents the next extraction reads, as a label
    filter does after its classification.
    """
    stats = builder.operator_order.statistics.stats
    calls = [node for node in builder.operators.extracts if node.alias == alias]
    if not calls or (alias in builder.extract_after and live is None):
        return
    if live is None:
        live = builder.live_asked.get(alias, float(stats[alias].n_docs))
        for test in builder.operators.label_filters.get(alias, ()):
            live *= effective_selectivity(test.selectivity)
    tests = builder.operators.extract_filters.get(alias, ())
    shared = False
    for logical_node in calls:
        # consecutive extractions of a table run in one pipeline
        # (quail.physical.pipelines): a document is read once for
        # all of them; a Filter between two ends the pipeline
        spec = extract_spec(builder.context, logical_node.call,
                            logical_node.name, live, shared=shared)
        node = inference_node(
            AiExtract, spec, (builder.rows_src.get(alias, builder.ids_src[alias]),),
            index=sum(isinstance(n, AiExtract) for n in builder.nodes),
            backend="quail", model=builder.model.name)
        builder.nodes.append(node)
        scores = PortRef(node.node_id, "rows")
        builder.ids_src[alias] = PortRef(node.node_id, f"ids:{alias}")
        shared = True
        for test in tests:
            if test.name != logical_node.name:
                continue
            lid = f"filter:{alias}:{test.position}"
            builder.nodes.append(Filter(
                node_id=lid, inputs=input_ports((scores,)),
                predicate=Condition.from_predicate(
                    test.condition.column.column, test.condition),
                aliases=(alias,), selectivity=test.selectivity,
                written_pos=test.position))
            scores = PortRef(lid, "rows")
            builder.ids_src[alias] = PortRef(lid, f"ids:{alias}")
            live *= effective_selectivity(test.selectivity)
            shared = False
        builder.rows_src[alias] = scores
        builder.label_ports.append(scores)

