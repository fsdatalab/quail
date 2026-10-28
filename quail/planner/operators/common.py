"""What every AI operator shares: prompt token counts, specs, and plan nodes."""

from quail.logical import effective_selectivity
from quail.logical.expressions import column_name
from quail.logical.prompts import prompt_aliases
from quail.physical import Filter, PortRef
from quail.physical.base import input_ports
from quail.planner.prompt_lowering import LoweredPrompt


def question_tokens(lowered: LoweredPrompt, canvas: int = 0) -> int:
    """Token count of the prompt's per-evaluation tail.

    canvas is the rows a diffusion model appends to every evaluation
    to answer on; a decoder answers on the tail's last row.
    """
    if lowered.tail_tokens is None:
        raise ValueError("a score prompt has no question tail")
    return lowered.tail_tokens + canvas


def preamble_tokens(lowered_prompts) -> int:
    """Return the engine preamble's token count from any lowered prompt."""
    for lowered in lowered_prompts:
        if lowered.preamble_tokens is not None:
            return lowered.preamble_tokens
    return 0


def bind_inference(spec_type, prompt, *, name, expected_rows,
                   prompt_token_parts, aliases=None, query_template=None,
                   estimated_seconds=0.0, **settings):
    """Build an operator's spec from its bound prompt and execution settings."""
    return spec_type(
        name=name,
        aliases=prompt_aliases(prompt) if aliases is None else aliases,
        query_template=(prompt.template if query_template is None
                        else query_template),
        arguments=tuple((ref.alias, column_name(ref, qualified=False))
                        for ref in prompt.args),
        expected_rows=expected_rows, estimated_seconds=estimated_seconds,
        prompt_token_parts=prompt_token_parts, **settings)


def inference_node(node_type, spec, sources, *, index, backend, model):
    """Connect a model call to its input rows or document IDs."""
    name = node_type.type_name.removeprefix("quail.").replace("_", "-")
    return node_type(
        node_id=f"{name}:{index}", inputs=input_ports(tuple(sources)),
        backend_name=backend, model=model, spec=spec)


def emit_calls(builder, alias, expected_rows, port, calls, node_of, *,
               projected=None) -> tuple:
    """Emit one table's calls in run order, each followed by its Filters.

    The calls run in one pipeline (quail.physical.pipelines): a document
    is read once for all of them, and a Filter on a call's answer drops
    the document before the next call reads it.

    Args:
        builder: The graph builder.
        alias: The table.
        expected_rows: The documents the first call reads.
        port: The port the documents stand at before the calls.
        calls: The calls, in run order.
        node_of: Builds a call's node from (call, port, expected_rows,
            index); index counts the calls emitted before it.
        projected: Whether SELECT returns a call's answer; None returns
            every call's.

    Returns:
        Where the documents stand after the calls, and the rows ports
        of the answers SELECT returns.
    """
    tests = {}
    for test in builder.operators.column_filters.get(alias, ()):
        tests.setdefault(test.name, []).append(test)
    row_ports = []
    for index, call in enumerate(calls):
        node = node_of(call, port, expected_rows, index)
        builder.nodes.append(node)
        port = PortRef(node.node_id, node.rows_port)
        if projected is None or projected(call):
            row_ports.append(port)
        for test in tests.get(node.spec.name, ()):
            filter_id = f"filter:{alias}:{test.position}"
            builder.nodes.append(Filter(
                node_id=filter_id, inputs=input_ports((port,)),
                predicate=test.condition, aliases=(alias,),
                selectivity=test.selectivity, written_position=test.position))
            port = PortRef(filter_id, f"rows:{alias}")
            expected_rows *= effective_selectivity(test.selectivity)
    return port, row_ports


def node_index(builder, node_type) -> int:
    """Return how many nodes of a type the builder holds."""
    return sum(isinstance(node, node_type) for node in builder.nodes)
