"""AI.SAE_FEATURES: its spec, cost, and plan nodes."""

from quail.cost import budgets
from quail.cost.sae_features import SaeFeaturesCost
from quail.physical import AiSaeFeatures, PortRef
from quail.planner.operators.common import bind_inference, inference_node
from quail.planner.plan import Refusal
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import SaeFeaturesRefusedError


def sae_features_spec(context, call, name, expected_rows, resident=False):
    """Build a feature call's specification and price it.

    Args:
        context: The planning context, with the model and tokenizer.
        call: Logical AI.SAE_FEATURES call.
        name: Result column name.
        expected_rows: Expected number of documents to read.
        resident: Whether the documents' KV is already read, by a
            step of the same pipeline before this one.
    """
    model = context.model
    if not model.sae_hf_name or not model.residual_readout:
        raise SaeFeaturesRefusedError(Refusal(
            reasons=(f"AI.SAE_FEATURES needs a model with an SAE; "
                     f"{model.name!r} has none",),
            constraint="model_without_sae", needed=1, available=0,
            unit="sae"))
    prompt = context.lowered(call.prompt)
    (ref,) = call.prompt.args
    summary = summarize_alias(
        context.document_tokens[ref.alias], model.sliding_window,
        memo=context.memo)
    chunk = budgets.chunk_budget(model, context.device)
    cost = SaeFeaturesCost(
        token_lengths=(len(prompt.preamble_token_ids),
                       len(prompt.tail_token_ids)),
        summary=summary, reading=call.reading, model=model,
        device=context.device, chunk_tokens=chunk, gpus=context.gpus)
    _work, seconds = cost.estimate(expected_rows, resident=resident)
    from quail.physical import SaeFeaturesSpec

    return bind_inference(
        SaeFeaturesSpec, call.prompt, name=name,
        expected_rows=expected_rows, estimated_seconds=seconds,
        prompt_token_parts=(tuple(prompt.preamble_token_ids),
                            tuple(prompt.tail_token_ids)),
        reading=call.reading)


def emit_sae_features(builder, alias, expected_rows, names, port,
                      resident=False) -> tuple:
    """Read the features of the alias's documents, one node per call.

    The feature calls of a table run in one pipeline: a document is
    read once for all of them, and with reading 'max' each document's
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
    calls = [feature_calls[name] for name in names]
    row_ports = []
    for index, logical_node in enumerate(calls):
        spec = sae_features_spec(
            builder.context, logical_node.call, logical_node.name,
            expected_rows, resident=resident or index > 0)
        node = inference_node(
            AiSaeFeatures, spec, (port,),
            index=sum(isinstance(n, AiSaeFeatures) for n in builder.nodes),
            backend="quail", model=builder.model.name)
        builder.nodes.append(node)
        port = PortRef(node.node_id, node.rows_port)
        row_ports.append(port)
    return port, row_ports
