"""Bind model call inputs and connect their physical operators."""

from quail.logical.prompts import prompt_aliases
from quail.physical.base import input_ports


def bind_inference(spec_type, prompt, *, name, expected_inputs,
                   prompt_token_parts, aliases=None, query_template=None,
                   estimated_seconds=0.0, **settings):
    """Build an operator's spec from its bound prompt and execution settings."""
    return spec_type(
        name=name,
        aliases=prompt_aliases(prompt) if aliases is None else aliases,
        query_template=(prompt.template if query_template is None
                        else query_template),
        arguments=tuple((ref.alias, ref.column) for ref in prompt.args),
        expected_inputs=expected_inputs, estimated_seconds=estimated_seconds,
        prompt_token_parts=prompt_token_parts, **settings)


def inference_node(node_type, spec, sources, *, index, backend, model):
    """Connect a model call to its input rows or document IDs."""
    name = node_type.type_name.removeprefix("quail.").replace("_", "-")
    return node_type(
        node_id=f"{name}:{index}", inputs=input_ports(tuple(sources)),
        backend_name=backend, model=model, spec=spec)
