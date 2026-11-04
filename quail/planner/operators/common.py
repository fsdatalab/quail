"""What every AI operator shares: prompt token counts, specs, and plan nodes."""

from typing import ClassVar

from quail.common.expressions import column_name
from quail.common.prompts import prompt_aliases
from quail.physical.base import input_ports
from quail.physical.lowering import LoweredPrompt


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


def node_index(builder, node_type) -> int:
    """Return how many nodes of a type the builder holds."""
    return sum(isinstance(node, node_type) for node in builder.nodes)


class ColumnPlanner:
    """How the planner builds and prices one kind of column operator.

    A column operator is one model call that adds columns to a table's
    rows (quail.logical.nodes.column.SemanticColumn). Each kind
    implements this interface in its own module, and the graph builder
    and the table step pricing call it without branching on the kind,
    as DataFusion's physical operators implement ExecutionPlan.

    Attributes:
        node_type: The physical node class the kind builds.
    """

    node_type: ClassVar[type]

    def spec(self, context, node, expected_rows, resident=False):
        """Return the operator's priced specification.

        Args:
            context: The planning context, with the model and tokenizer.
            node: The logical column operator.
            expected_rows: Expected number of documents it reads.
            resident: Whether the documents' KV is already read, by a
                step of the same pipeline before this one.
        """
        raise NotImplementedError

    def seconds(self, context, node, resident, expected_rows) -> float:
        """Return the operator's estimated seconds over expected_rows documents."""
        return self.spec(context, node, expected_rows, resident).estimated_seconds

    def node(self, builder, node, port, expected_rows, resident):
        """Return the operator's physical node over the rows at port."""
        spec = self.spec(builder.context, node, expected_rows, resident)
        return inference_node(
            self.node_type, spec, (port,), index=node_index(builder, self.node_type),
            backend="quail", model=builder.model.name)

    def projected(self, operators, node) -> bool:
        """Return whether SELECT returns the operator's columns."""
        return True
