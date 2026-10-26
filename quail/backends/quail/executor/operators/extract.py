"""Run a pipeline's extractions and build their answer tables."""

import pyarrow as pa

from quail.backends.quail.executor.decoding import AnswerPart
from quail.backends.quail.executor.extract import ExtractionStages
from quail.execution.rows import extraction_rows
from quail.physical import AiExtract


class ExtractPart(AnswerPart):
    """Extraction stages and per-question results for the shared executor."""

    node_type = AiExtract
    stages_type = ExtractionStages
    operator = "AI.EXTRACT"

    def _answer_rows(self, node, q: int, kept, document_ids):
        """Return one extraction's answers and nullable character spans."""
        answers, answer_spans = self.answer_stages.answers(q, kept)
        return extraction_rows(node.spec.alias, node.spec.name,
                               node.spec.span_name, document_ids, answers,
                               answer_spans)

    @property
    def tested_columns(self) -> tuple[str, ...]:
        """The columns a Filter in the pipeline may test."""
        return tuple(name for node in self.nodes
                     for name in (node.spec.name, node.spec.span_name))

    def tested_values(self, column: str, positions) -> pa.Array:
        """Return the answers or spans settled so far at the positions.

        Args:
            column: An extraction's answer or span column.
            positions: Positions among the part's documents.
        """
        q, span = next(
            (q, column == node.spec.span_name) for q, node in enumerate(self.nodes)
            if column in (node.spec.name, node.spec.span_name))
        answers = [
            self.answer_stages.documents[position].answers[q]
            for position in positions
        ]
        if not span:
            return pa.array([answer.answer for answer in answers], pa.string())
        return pa.array(
            [None if answer.span is None
             else {"start": answer.span[0], "end": answer.span[1]}
             for answer in answers],
            pa.struct([("start", pa.int32()), ("end", pa.int32())]))
