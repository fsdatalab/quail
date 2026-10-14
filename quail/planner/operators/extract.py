"""AI.EXTRACT: its spec and cost."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.logical.prompts import (
    shared_preamble,
)
from quail.physical import (
    ExtractSpec,
)
from quail.planner.operators.common import bind_inference


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
