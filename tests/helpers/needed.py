"""Which model answers a query's rows depend on, found by changing them.

An answer is needed when some other value of it changes the rows the
query returns. ``NeededAnswers`` runs a generated query with every
pushdown off, so it asks every answer the query could read, and then
again with each answer changed to each other value it may take. It
also runs the query with Quail's rules on and records the answers that
run asks.

Changing one answer at a time misses answers that matter only
together: when two rows each make an ``EXISTS`` true, neither change
alone changes the rows, though one of the two is needed.
``sufficient`` finds a set of answers that together decide the rows,
so the fewest answers a query needs lies between the two counts.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import quail
from helpers.generated import NAMES, PAIR_TEXT, tables, truth
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER
from quail.builtins import LOGICAL_RULES, PHYSICAL_RULES, built_in_registry

# Quail's logical rewrites, and the executor's ways to ask fewer rows
QUAIL_RULES = ("derive", "place_ai_nodes", "merge_repeated_ai_calls")
EARLY_STOPS = ("should_stop_answers_early", "should_deduplicate",
               "should_push_down_limit")
# the values an answer of each kind may take, besides its true one
VALUES = {
    "filter": (True, False, None),
    "join": (True, False, None),
    "label": ("a", "b", "c", None),
    "score": (0.0, 0.25, 0.5, 0.75, 1.0, None),
    "map": (*NAMES, "", "zz", None),
    "extract": (None, "", "Acme", "acme", "Globex", "zz"),
}

Answer = tuple[str, str, tuple[str, ...]]


def answer_key(kind: str, question: str, texts) -> Answer:
    """Return one key per answer, whatever prompt layout asked it.

    A pair question may be asked as a join of two documents, or as a
    filter of one text that holds both; either is the same answer.
    """
    pair = PAIR_TEXT.fullmatch(texts[0]) if kind == "filter" and texts[0] else None
    if pair is not None:
        kind, texts = "join", pair.groups()
    number = re.search(r"q(\d):", question).group(1)
    # a join's answer is the same with its documents in either order
    texts = tuple(sorted(texts)) if kind == "join" else tuple(texts)
    return kind, f"q{number}:", texts


def true_answer(key: Answer):
    """Return the answer the generated tables' plain SQL computes."""
    return truth(*key)


def _rows(table) -> list:
    return sorted(zip(*(column.to_pylist() for column in table.columns)), key=repr)


@dataclass
class Asked:
    """What one query's runs asked, and which answers its rows need.

    Attributes:
        sql: The query.
        every: The answers the run with no pushdown asks.
        needed: The answers some other value of which changes the rows.
        asked: The answers the run with Quail's rules asks.
    """

    sql: str
    every: set[Answer]
    needed: set[Answer]
    asked: set[Answer]

    @property
    def missed(self) -> set[Answer]:
        """Needed answers Quail did not ask: its rows were right by chance."""
        return self.needed - self.asked

    @property
    def unneeded(self) -> set[Answer]:
        """Answers Quail asked that no change of one answer shows a need for."""
        return self.asked - self.needed


class NeededAnswers:
    """Runs generated queries over ``tables(seed)`` with an oracle model.

    Args:
        seed: The seed of the tables.
    """

    def __init__(self, seed: int):
        self._changed: dict[Answer, object] = {}
        self._asked: set[Answer] = set()
        self._sessions = []
        self.plain = self._context(seed, optimized=False)
        self.optimized = self._context(seed, optimized=True)

    def close(self) -> None:
        """Close both sessions."""
        for session in self._sessions:
            session.close()

    def _answer(self, kind, question, texts):
        key = answer_key(kind, question, texts)
        self._asked.add(key)
        if key in self._changed:
            return self._changed[key]
        return truth(kind, question, texts)

    def _context(self, seed: int, optimized: bool):
        registry = built_in_registry() if optimized else built_in_registry(
            logical_rules=tuple(rule for rule in LOGICAL_RULES
                                if rule not in QUAIL_RULES),
            physical_rules=tuple(rule for rule in PHYSICAL_RULES
                                 if rule not in EARLY_STOPS))
        session = quail.Session(CONFIG, tokenizer=TOKENIZER, registry=registry)
        self._sessions.append(session)
        oracle = Oracle(truth=self._answer)
        oracle.catalog = session.catalog
        context = quail.SessionContext(
            session, physical_executor=oracle_executor(oracle))
        for name, table in tables(seed).items():
            context.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        return context

    def run(self, context, sql: str, changed: dict | None = None):
        """Return a query's rows and the answers it asked.

        Args:
            context: ``plain`` or ``optimized``.
            sql: The query.
            changed: Answers to give in place of the true ones.

        Returns:
            The rows, sorted, or None when the query fails with these
            answers, and the answers asked.
        """
        self._changed, self._asked = dict(changed or {}), set()
        try:
            found = _rows(context.sql(sql))
        except Exception:  # noqa: BLE001 - a changed answer may not fit the query
            found = None
        finally:
            self._changed = {}
        return found, set(self._asked)

    def asked(self, sql: str) -> Asked:
        """Return what each run asks, and the answers that one change shows needed."""
        rows, every = self.run(self.plain, sql)
        optimized, asked = self.run(self.optimized, sql)
        assert optimized == rows, sql
        needed = set()
        for key in sorted(every, key=repr):
            truth_value = true_answer(key)
            for value in VALUES[key[0]]:
                if value != truth_value and self.run(
                        self.plain, sql, {key: value})[0] not in (rows, None):
                    needed.add(key)
                    break
        return Asked(sql, every, needed, asked)

    def sufficient(self, found: Asked, samples: int = 8,
                   seed: int = 0) -> set[Answer]:
        """Return answers that together decide the query's rows.

        Greedy: each answer no single change shows needed becomes unknown
        in turn, when the rows stay the same for ``samples`` random values
        of every unknown answer, this one changed in each. The answers
        left known are a sufficient set if the samples missed nothing.

        Args:
            found: The query's ``asked()``.
            samples: Random values tried for each answer.
            seed: The seed of the random values.
        """
        rng = random.Random(seed)
        rows, _ = self.run(self.plain, found.sql)
        unknown: list[Answer] = []
        for key in sorted(found.every - found.needed, key=repr):
            trial = [*unknown, key]
            truth_value = true_answer(key)
            changes = [value for value in VALUES[key[0]] if value != truth_value]
            same = True
            for _ in range(samples):
                values = {item: rng.choice(VALUES[item[0]]) for item in trial}
                values[key] = rng.choice(changes)
                if self.run(self.plain, found.sql, values)[0] not in (rows, None):
                    same = False
                    break
            if same:
                unknown.append(key)
        return found.every - set(unknown)
