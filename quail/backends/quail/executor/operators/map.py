"""Run a pipeline's maps and build their answer tables."""

import pyarrow as pa

from quail.backends.quail.executor.decoding import AnswerPart
from quail.backends.quail.executor.map import MapStages
from quail.execution.rows import map_rows
from quail.physical import AiMap


class MapPart(AnswerPart):
    """Map stages and per-instruction results for the shared executor."""

    node_type = AiMap
    stages_type = MapStages
    operator = "AI.MAP"
    branches = True

    @property
    def attention(self) -> str:
        """The attention path for the pipeline's rounds.

        Branches run on the path the plan chose for the maps (the
        choose_attention_algorithm rule); on the tree path each
        document's pages are read once per pass for all its branches
        (see Stage.document_reads). A pipeline without branches runs
        unified, as does a model without tree attention.
        """
        if not getattr(self.answer_stages, "branches", False):
            return "unified"
        return self.node.attention or "unified"

    def _answer_rows(self, node, q: int, kept, document_ids):
        """Return one map's written answers."""
        return map_rows(node.spec.alias, node.spec.name, document_ids,
                        self.answer_stages.answers(q, kept))

    @property
    def tested_columns(self) -> tuple[str, ...]:
        """The columns a Filter in the pipeline may test."""
        return tuple(node.spec.name for node in self.nodes)

    def tested_values(self, column: str, positions) -> pa.Array:
        """Return the answers settled so far at the positions.

        Args:
            column: A map's answer column.
            positions: Positions among the part's documents.
        """
        q = next(q for q, node in enumerate(self.nodes)
                 if column == node.spec.name)
        return pa.array(
            [self.answer_stages.documents[position].answers[q].answer
             for position in positions], pa.string())
