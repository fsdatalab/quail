"""AI.SAE_FEATURES: its spec, cost, and plan nodes."""

from quail.cost import budgets
from quail.cost.sae_features import SaeFeaturesCost
from quail.physical import AiSaeFeatures, SaeFeaturesSpec
from quail.planner.operators.common import ColumnPlanner, bind_inference
from quail.planner.plan import Refusal
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import SaeFeaturesRefusedError


class SaeFeaturesPlanner(ColumnPlanner):
    """Build and price AI.SAE_FEATURES nodes.

    The feature calls of a table run in one pipeline: a document is
    read once for all of them, and when any of them pools by 'max',
    each document's maxima are kept, so a later call encodes only its
    question tokens.
    """

    node_type = AiSaeFeatures

    def spec(self, context, node, expected_rows, resident=False) -> SaeFeaturesSpec:
        model = context.model
        if model.sae is None:
            raise SaeFeaturesRefusedError(Refusal(
                reasons=(f"AI.SAE_FEATURES needs a model loaded with an SAE; "
                         f"{model.name!r} has none",),
                constraint="unsupported_sae_features_query", needed=1,
                available=0, unit="queries"))
        call = node.call
        max_pooling = context.logical_plan is not None and any(
            other.alias == node.alias and other.pooling == "max"
            for other in context.logical_plan.operators().sae_features)
        prompt = context.lowered(call.prompt)
        (ref,) = call.prompt.args
        summary = summarize_alias(
            context.document_tokens[ref.alias], model.sliding_window,
            memo=context.memo)
        cost = SaeFeaturesCost(
            token_lengths=(len(prompt.preamble_token_ids),
                           len(prompt.tail_token_ids)),
            summary=summary, pooling=call.pooling, model=model,
            device=context.device,
            chunk_tokens=budgets.chunk_budget(model, context.device),
            gpus=context.gpus, max_pooling=max_pooling)
        _work, seconds = cost.estimate(expected_rows, resident=resident)
        return bind_inference(
            SaeFeaturesSpec, call.prompt, name=node.name,
            expected_rows=expected_rows, estimated_seconds=seconds,
            prompt_token_parts=(tuple(prompt.preamble_token_ids),
                                tuple(prompt.tail_token_ids)),
            pooling=call.pooling)
