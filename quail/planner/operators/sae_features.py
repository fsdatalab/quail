"""AI.SAE_FEATURES: its spec, cost, and plan nodes."""

from quail.cost import budgets
from quail.cost.sae_features import SaeFeaturesCost
from quail.physical import AiSaeFeatures
from quail.planner.operators.common import (
    bind_inference,
    emit_calls,
    inference_node,
    node_index,
)
from quail.planner.plan import Refusal
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import SaeFeaturesRefusedError


def sae_features_spec(context, call, name, expected_rows, resident=False,
                      max_pooling=False):
    """Build a feature call's specification and price it.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.SAE_FEATURES call.
        name: Result column name.
        expected_rows: Expected number of documents to read.
        resident: Whether the documents' KV is already read, by a
            step of the same pipeline before this one.
        max_pooling: Whether another call requires all token rows.
    """
    model = context.model
    if model.sae is None:
        raise SaeFeaturesRefusedError(Refusal(
            reasons=(f"AI.SAE_FEATURES needs a model loaded with an SAE; "
                     f"{model.name!r} has none",),
            constraint="unsupported_sae_features_query", needed=1,
            available=0, unit="queries"))
    prompt = context.lowered(call.prompt)
    (ref,) = call.prompt.args
    summary = summarize_alias(
        context.document_tokens[ref.alias], model.sliding_window,
        memo=context.memo)
    chunk = budgets.chunk_budget(model, context.device)
    cost = SaeFeaturesCost(
        token_lengths=(len(prompt.preamble_token_ids),
                       len(prompt.tail_token_ids)),
        summary=summary, pooling=call.pooling, model=model,
        device=context.device, chunk_tokens=chunk, gpus=context.gpus,
        max_pooling=max_pooling)
    _work, seconds = cost.estimate(expected_rows, resident=resident)
    from quail.physical import SaeFeaturesSpec

    return bind_inference(
        SaeFeaturesSpec, call.prompt, name=name,
        expected_rows=expected_rows, estimated_seconds=seconds,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        pooling=call.pooling)


def emit_sae_features(builder, alias, expected_rows, names, port,
                      resident=False) -> tuple:
    """Read the features of the alias's documents, one node per call.

    The feature calls of a table run in one pipeline: a document is
    read once for all of them, and with pooling 'max' each document's
    maxima are kept, so a later call encodes only its question tokens.

    Args:
        builder: The graph builder.
        alias: The table.
        expected_rows: The documents the calls read.
        names: The calls to emit, in run order.
        port: The port the documents stand at before the calls.
        resident: Whether the step before the run left the documents'
            KV for the first call to read.

    Returns:
        Where the documents stand after them, and the rows ports of
        the feature lists SELECT returns.
    """
    feature_calls = {node.name: node
                     for node in builder.operators.sae_features
                     if node.alias == alias}
    max_pooling = any(feature_calls[name].pooling == "max" for name in names)

    def node_of(name, port, expected_rows, index):
        logical_node = feature_calls[name]
        spec = sae_features_spec(
            builder.context, logical_node.call, name, expected_rows,
            resident=resident or index > 0, max_pooling=max_pooling)
        return inference_node(
            AiSaeFeatures, spec, (port,),
            index=node_index(builder, AiSaeFeatures),
            backend="quail", model=builder.model.name)

    return emit_calls(builder, alias, expected_rows, port, names, node_of)
