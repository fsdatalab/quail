"""AI.EXTRACT: its spec and cost."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost import extract as extract_cost
from quail.cost.retention import coefficients
from quail.physical import AiExtract, ExtractSpec
from quail.planner.operators.common import ColumnPlanner, bind_inference


class ExtractPlanner(ColumnPlanner):
    """Build and price AI.EXTRACT nodes.

    The specification carries the token ids of the prompt around the
    document. The documents' KV is priced as computed afresh unless
    the extraction shares a pipeline with the step before it, which
    read the documents.
    """

    node_type = AiExtract

    def spec(self, context, node, expected_rows, resident=False) -> ExtractSpec:
        call, name = node.call, node.name
        prompt = context.lowered(call.prompt)
        (ref,) = call.prompt.args
        lengths = context.document_tokens[ref.alias]
        model, device = context.model, context.device
        # one document token's KV read per round, in tokens of compute: the
        # pass is compute-bound, so a fed token costs its dense FLOPs
        token_seconds = coefficients(model, device)["linear_seconds"]
        round_cost_per_token = model.kappa / device.hbm_bw / token_seconds
        spec = bind_inference(
            ExtractSpec, call.prompt, name=name,
            expected_rows=expected_rows, estimated_seconds=0.0,
            prompt_token_parts=(tuple(prompt.preamble_token_ids),
                                tuple(prompt.tail_token_ids)),
            question=call.question,
            expected_answer_tokens=extract_cost.ExtractAssumptions().answer_tokens,
            round_cost_per_token=round_cost_per_token)
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
            draft_tokens=spec.draft_tokens,
            draft_policy=spec.draft_policy,
            round_cost_per_token=spec.round_cost_per_token,
            gpus=context.gpus,
            memo=context.memo,
        )
        return replace(spec, estimated_seconds=cost.seconds)
