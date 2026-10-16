"""AI.EXTRACT: its spec and cost."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.logical import effective_selectivity
from quail.physical import (
    AiExtract,
    ExtractSpec,
    Filter,
    PortRef,
)
from quail.physical.base import input_ports
from quail.planner.operators.common import (
    bind_inference,
    inference_node,
    shares_query_head,
)


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
    shares_head = shares_query_head(context, prompt)
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


def emit_extract(builder, alias, live, names):
    """Copy the answers of the alias's documents.

    A Filter on an extraction's answer or span follows that extraction
    and drops the documents the next extraction would read, as a label
    filter does after its classification.

    Args:
        builder: The graph builder.
        alias: The table.
        live: The documents the extractions read.
        names: The extractions to emit, in run order.
    """
    extracts = {node.name: node for node in builder.operators.extracts
                if node.alias == alias}
    calls = [extracts[name] for name in names]
    tests = builder.operators.extract_filters.get(alias, ())
    for index, logical_node in enumerate(calls):
        # the extractions of a table and the Filters on their answers
        # run in one pipeline (quail.physical.pipelines): a document
        # is read once for all of them, and a Filter drops it before
        # the next question
        spec = extract_spec(builder.context, logical_node.call,
                            logical_node.name, live, shared=index > 0)
        node = inference_node(
            AiExtract, spec, (builder.source[alias],),
            index=sum(isinstance(n, AiExtract) for n in builder.nodes),
            backend="quail", model=builder.model.name)
        builder.nodes.append(node)
        scores = PortRef(node.node_id, "rows")
        builder.source[alias] = scores
        for test in tests:
            if test.name != logical_node.name:
                continue
            lid = f"filter:{alias}:{test.position}"
            builder.nodes.append(Filter(
                node_id=lid, inputs=input_ports((scores,)),
                predicate=test.condition, aliases=(alias,),
                selectivity=test.selectivity, written_pos=test.position))
            scores = PortRef(lid, "rows")
            builder.source[alias] = scores
            live *= effective_selectivity(test.selectivity)
        builder.label_ports.append(scores)

