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
    # a round's rows join their pages and one causal paged call reads
    # them with the document, rather than the tree path's call over
    # the fresh rows, its read of the document, and their merge; maps
    # put to a document at once also read its shared pages once for
    # all of their branches (see attention_shared)
    attention = "unified"

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
