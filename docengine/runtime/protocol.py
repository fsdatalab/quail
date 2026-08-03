"""Runtime inputs and benchmark-only ground truth."""

from dataclasses import dataclass
from typing import Mapping, Sequence


TokenRows = tuple[tuple[int, ...], ...]


def _freeze_rows(rows: Sequence[Sequence[int]]) -> TokenRows:
    return tuple(tuple(int(token) for token in row) for row in rows)


@dataclass(frozen=True)
class FilterQuery:
    body_token_ids: TokenRows
    question_token_ids: TokenRows
    yes_token_ids: frozenset[int]

    @classmethod
    def from_sequences(
        cls,
        body_token_ids: Sequence[Sequence[int]],
        question_token_ids: Sequence[Sequence[int]],
        yes_token_ids: Sequence[int],
    ) -> "FilterQuery":
        query = cls(
            body_token_ids=_freeze_rows(body_token_ids),
            question_token_ids=_freeze_rows(question_token_ids),
            yes_token_ids=frozenset(int(token) for token in yes_token_ids),
        )
        query.validate()
        return query

    @property
    def n_documents(self) -> int:
        return len(self.body_token_ids)

    @property
    def n_filters(self) -> int:
        return len(self.question_token_ids)

    def validate(self) -> None:
        if not self.body_token_ids:
            raise ValueError("query needs at least one document")
        if not self.question_token_ids:
            raise ValueError("query needs at least one filter")
        if any(not row for row in self.body_token_ids):
            raise ValueError("document token rows must be nonempty")
        if any(not row for row in self.question_token_ids):
            raise ValueError("filter token rows must be nonempty")
        if not self.yes_token_ids:
            raise ValueError("query needs at least one true-answer token id")


@dataclass(frozen=True)
class GroundTruthLabels:
    outcomes: tuple[tuple[int, ...], ...]

    @classmethod
    def from_sequences(
        cls,
        outcomes: Sequence[Sequence[int]],
    ) -> "GroundTruthLabels":
        labels = cls(
            outcomes=tuple(
                tuple(int(value) for value in row)
                for row in outcomes
            )
        )
        for row in labels.outcomes:
            if any(value not in (0, 1) for value in row):
                raise ValueError("ground truth values must be zero or one")
        return labels

    def validate_for(self, query: FilterQuery) -> None:
        if len(self.outcomes) != query.n_documents:
            raise ValueError("ground truth document count does not match query")
        if any(len(row) != query.n_filters for row in self.outcomes):
            raise ValueError("ground truth filter count does not match query")


@dataclass(frozen=True)
class AnswerEvaluation:
    attempted: int
    correct: int
    wrong: tuple[tuple[int, int], ...]
    expected_survivors: tuple[int, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0


def evaluate_answers(
    answers: Mapping[tuple[int, int], int],
    labels: GroundTruthLabels,
) -> AnswerEvaluation:
    wrong = []
    correct = 0
    for (document, stage), answer in answers.items():
        expected = labels.outcomes[document][stage - 1]
        if int(answer) == expected:
            correct += 1
        else:
            wrong.append((int(document), int(stage)))
    survivors = tuple(
        document
        for document, row in enumerate(labels.outcomes)
        if all(row)
    )
    return AnswerEvaluation(
        attempted=len(answers),
        correct=correct,
        wrong=tuple(sorted(wrong)),
        expected_survivors=survivors,
    )
