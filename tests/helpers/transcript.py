"""Transcripts of what Quail asks the model, for a person to read.

A transcript test runs a query on an oracle model and writes, as text,
what Quail did: each AI operator of the plan with its guard, filter, and
fetch; each question the model answered, by prompt and row, in the order
the operators asked; and the rows the query returned. The test passes
when the text matches the expected transcript saved beside the test, in
``transcripts/<test name>.txt``. A change in what Quail asks shows up as
a diff of that file, which a reviewer reads in place of counts.

Each query also runs with every pushdown off. The transcript fails if the
two return different rows, and says how many of that run's questions
Quail asked.

Run ``QUAIL_UPDATE_TRANSCRIPTS=1 uv run pytest ...`` to write the
expected transcripts from the current behavior, then read the diff.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

import quail
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER
from quail.builtins import LOGICAL_RULES, PHYSICAL_RULES, built_in_registry

# Quail's logical rewrites, and the executor's ways to ask fewer rows
QUAIL_RULES = ("merge_repeated_ai_calls", "derive", "place_ai_nodes")
EARLY_STOPS = ("should_stop_answers_early", "should_deduplicate",
               "should_push_down_limit")
UPDATE = "QUAIL_UPDATE_TRANSCRIPTS"
# how a document of several is shown, numbered from 0
PAIR_LAYOUT = "\n\nDOCUMENT {}:\n"


class _Stopped(NamedTuple):
    """A map or extraction answer the executor stopped before its end."""

    answer: str
    written: str


def _shown(answer) -> str:
    if isinstance(answer, _Stopped):
        return f"{answer.answer!r} stopped at {answer.written!r}"
    if answer is True:
        return "TRUE"
    if answer is False:
        return "FALSE"
    if answer is None:
        return "NULL"
    return repr(answer)


def _plan_line(line: str) -> str:
    line = line.strip().removeprefix("AiNode: ")
    while (start := line.find('prompt(Utf8("')) >= 0:
        # drop prompt( and its closing parenthesis, outside quoted text
        depth, quoted = 0, False
        for end in range(start + len("prompt"), len(line)):
            if line[end] == '"':
                quoted = not quoted
            elif not quoted and line[end] in "()":
                depth += 1 if line[end] == "(" else -1
                if depth == 0:
                    break
        line = line[:start] + line[start + len("prompt("):end] + line[end + 1:]
    return re.sub(r'Utf8\("(.*?)"\)', r"'\1'", line)


class Transcripts:
    """Runs queries over ``tables`` on an oracle model and writes transcripts.

    Args:
        tables: Each table by name, or (table, id column) when its id
            column is not ``id``; its other text columns hold its
            documents. A text several rows hold names them all, as
            ``r0/r1``.
        answer: Callable(kind, prompt, rows) -> the model's answer. ``kind``
            is "filter", "join", "label", "score", "map", "extract", or
            "features"; ``prompt`` is the SQL string literal the question was written
            with, such as ``'kind: {0}'``; ``rows`` holds the id of each
            document asked about, or the quoted text of a document no
            table holds, such as a computed one.
        rules: The logical rules of the optimized run; every rule when None.
        config: The sessions' engine configuration; ``CONFIG`` when None.
        written: Whether to show where the executor stopped each map or
            extraction answer, and how many answer tokens the model wrote,
            one a character, of those the run with every pushdown off
            wrote.
        pipelines: Whether the oracle runs chains of operators, as the
            GPU executor does.
    """

    def __init__(self, tables: dict[str, pa.Table], answer, rules=None, *,
                 config=None, written=False, pipelines=False):
        self._answer = answer
        self._pipelines = pipelines
        self._written = written
        self._oracles = {}
        self._unwritten: list[int] = []
        self._tokens_written = 0
        self._config = config or CONFIG
        tables = {name: found if isinstance(found, tuple) else (found, "id")
                  for name, found in tables.items()}
        holders: dict[str, list] = {}
        self._rows = set()
        for table, id_column in tables.values():
            ids = table.column(id_column).to_pylist()
            self._rows.update(ids)
            for name in table.column_names:
                if name != id_column and pa.types.is_string(
                        table.schema.field(name).type):
                    for row, text in zip(ids, table.column(name).to_pylist(),
                                         strict=True):
                        if text is not None and row not in holders.get(text, []):
                            holders.setdefault(text, []).append(row)
        self._ids = {text: "/".join(map(str, rows))
                     for text, rows in holders.items()}
        self._prompts: list[str] = []
        self._asked: list[tuple[str, tuple, object]] = []
        self._sessions = []
        everything = built_in_registry(
            logical_rules=tuple(rules) if rules is not None else LOGICAL_RULES)
        nothing = built_in_registry(
            logical_rules=tuple(rule for rule in LOGICAL_RULES
                                if rule not in QUAIL_RULES),
            physical_rules=tuple(rule for rule in PHYSICAL_RULES
                                 if rule not in EARLY_STOPS))
        self.session, self._optimized = self._context(tables, everything)
        _, self._plain = self._context(tables, nothing)

    def close(self) -> None:
        """Close the sessions."""
        for session in self._sessions:
            session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _context(self, tables, registry):
        session = quail.Session(self._config, tokenizer=TOKENIZER,
                                registry=registry)
        self._sessions.append(session)
        oracle = Oracle(truth=self._truth, pipelines=self._pipelines)
        oracle.features = lambda spec, document: self._truth(
            "features", spec.query_template,
            oracle._texts(spec.aliases[:1], [document]))
        oracle.catalog = session.catalog
        if self._written:
            oracle._written = self._writer(oracle._written)
        context = quail.SessionContext(
            session, physical_executor=oracle_executor(oracle))
        self._oracles[id(context)] = oracle
        for name, (table, id_column) in tables.items():
            context.register(
                name, quail.DocumentProvider.from_table(table, id_col=id_column))
        return session, context

    def _prompt(self, question: str) -> str:
        """The SQL string literal a question was written with."""
        found = [literal for literal in self._prompts
                 if (pieces := [piece.strip() for piece in
                                re.split(r"\{\d\}", literal) if piece.strip()])
                 and all(piece in question for piece in pieces)]
        return max(found, key=len) if found else "?"

    def _truth(self, kind, question, texts):
        if len(texts) == 1 and texts[0].startswith(PAIR_LAYOUT.format(0)):
            # a pair asked as one document, by the run with every pushdown off
            texts = tuple(re.split(PAIR_LAYOUT.format(r"\d+"), texts[0])[1:])
        rows = tuple(self._ids.get(text, repr(text)) for text in texts)
        prompt = self._prompt(question)
        answer = self._answer(kind, prompt, rows)
        if kind in ("map", "extract"):
            self._unwritten.append(len(self._asked))
        self._asked.append((prompt, rows, answer))
        return answer

    def _writer(self, write):
        """Wrap the oracle's writer to mark each answer it stops early.

        The oracle writes map and extraction answers in the order it
        asked for them.
        """
        def written(node, text, source):
            found = write(node, text, source)
            index = self._unwritten.pop(0)
            if found != text:
                prompt, rows, _ = self._asked[index]
                self._asked[index] = (prompt, rows, _Stopped(text, found))
            return found
        return written

    def _run(self, context, sql: str, dialect: str, ordered: bool):
        self._asked = []
        self._unwritten = []
        oracle = self._oracles[id(context)]
        before = oracle.answer_tokens
        table = context.sql(sql, dialect)
        self._tokens_written = oracle.answer_tokens - before
        self._schema = table.schema
        rows = list(zip(*(column.to_pylist() for column in table.columns)))
        return (rows if ordered else sorted(rows, key=repr)), self._asked

    def query(self, sql: str, dialect: str = "snowflake",
              ordered: bool = False, columns: bool = False) -> str:
        """Run a query and return its transcript.

        Args:
            sql: The query.
            dialect: The SQL dialect it is written in.
            ordered: Whether to show the rows in the order the query
                returns them; they are sorted otherwise.
            columns: Whether to show each returned column's name and type.

        Raises:
            AssertionError: The run with every pushdown off returns
                other rows.
        """
        self._prompts = re.findall(r"'((?:[^']|'')*)'", sql)
        plain_rows, plain_asked = self._run(self._plain, sql, dialect, ordered)
        plain_tokens = self._tokens_written
        rows, asked = self._run(self._optimized, sql, dialect, ordered)
        assert rows == plain_rows, (sql, rows, plain_rows)
        query = self.session.sql(sql, dialect=dialect)
        regions = query.regions() if hasattr(query, "regions") else [query]
        lines = [f"--- {sql} ---", "plan"]
        for region in regions:
            lines += [f"  {_plan_line(line)}" for line in region.logical.splitlines()
                      if "AiNode:" in line]
        asked_by: dict[str, list] = {}
        for prompt, ids, answer in asked:
            asked_by.setdefault(prompt, []).append((ids, answer))
        lines.append(f"asked {len(asked)} of {len(plain_asked)}")
        for prompt, answers in asked_by.items():
            shown = ", ".join(f"{'+'.join(ids)} {_shown(answer)}"
                              for ids, answer in sorted(answers, key=repr))
            lines.append(f"  '{prompt}': {shown}")
        if self._written:
            lines.append(f"wrote {self._tokens_written} of {plain_tokens} "
                         "answer tokens")
        if columns:
            lines.append("columns " + ", ".join(
                f"{field.name} {field.type}" for field in self._schema))
        lines.append(f"rows {len(rows)}{', in order' if ordered else ''}")
        lines += ["  " + ", ".join(
            value if isinstance(value, str) and value in self._rows
            else _shown(value) for value in row) for row in rows]
        return "\n".join(lines) + "\n"


def assert_transcript(text: str, name: str | None = None) -> None:
    """Compare a transcript with the one saved for the calling test.

    Args:
        text: The transcript.
        name: The saved transcript's name; the calling test's when None.
    """
    caller = inspect.stack()[1]
    name = name or caller.function
    path = Path(caller.filename).parent / "transcripts" / f"{name}.txt"
    if os.environ.get(UPDATE):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no transcript {path}; run with {UPDATE}=1"
    assert text == path.read_text(), f"{path} differs; run with {UPDATE}=1 and diff"
