"""What every AI operator shares: prompt token counts, specs, and plan nodes."""

from quail.logical.prompts import prompt_aliases
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
        arguments=tuple((ref.alias, ref.column) for ref in prompt.args),
        expected_rows=expected_rows, estimated_seconds=estimated_seconds,
        prompt_token_parts=prompt_token_parts, **settings)


def inference_node(node_type, spec, sources, *, index, backend, model):
    """Connect a model call to its input rows or document IDs."""
    name = node_type.type_name.removeprefix("quail.").replace("_", "-")
    return node_type(
        node_id=f"{name}:{index}", inputs=input_ports(tuple(sources)),
        backend_name=backend, model=model, spec=spec)
