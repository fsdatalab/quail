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

import pyarrow as pa

import quail
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER
from quail.builtins import LOGICAL_RULES, PHYSICAL_RULES, built_in_registry

# Quail's logical rewrites, and the executor's ways to ask fewer rows
QUAIL_RULES = ("derive", "place_ai_nodes", "merge_repeated_ai_calls")
EARLY_STOPS = ("should_stop_answers_early", "should_deduplicate",
               "should_push_down_limit")
UPDATE = "QUAIL_UPDATE_TRANSCRIPTS"


def _shown(answer) -> str:
    if answer is True:
        return "TRUE"
    if answer is False:
        return "FALSE"
    if answer is None:
        return "NULL"
    return repr(answer)


def _plan_line(line: str) -> str:
    line = line.strip().removeprefix("AiNode: ")
    line = re.sub(r'prompt\(Utf8\("(.*?)"\), ([^)]*)\)', r"'\1', \2", line)
    return re.sub(r'Utf8\("(.*?)"\)', r"'\1'", line)


class Transcripts:
    """Runs queries over ``tables`` on an oracle model and writes transcripts.

    Args:
        tables: Each table by name, or (table, id column) when its id
            column is not ``id``; its other text columns hold its
            documents. A text several rows hold names them all, as
            ``r0/r1``.
        answer: Callable(kind, prompt, rows) -> the model's answer. ``kind``
            is "filter", "join", "label", "score", "map", or "extract";
            ``prompt`` is the SQL string literal the question was written
            with, such as ``'kind: {0}'``; ``rows`` holds the id of each
            document asked about.
        rules: The logical rules of the optimized run; every rule when None.
    """

    def __init__(self, tables: dict[str, pa.Table], answer, rules=None):
        self._answer = answer
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
        self._ids = {text: "/".join(rows) for text, rows in holders.items()}
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
        session = quail.Session(CONFIG, tokenizer=TOKENIZER, registry=registry)
        self._sessions.append(session)
        oracle = Oracle(truth=self._truth)
        oracle.catalog = session.catalog
        context = quail.SessionContext(
            session, physical_executor=oracle_executor(oracle))
        for name, (table, id_column) in tables.items():
            context.register(
                name, quail.DocumentProvider.from_table(table, id_col=id_column))
        return session, context

    def _prompt(self, question: str) -> str:
        """The SQL string literal a question was written with."""
        found = [literal for literal in self._prompts
                 if (core := re.sub(r"\{\d\}", "", literal).strip())
                 and core in question]
        return max(found, key=len) if found else "?"

    def _truth(self, kind, question, texts):
        rows = tuple(self._ids.get(text, "?") for text in texts)
        prompt = self._prompt(question)
        answer = self._answer(kind, prompt, rows)
        self._asked.append((prompt, rows, answer))
        return answer

    def _run(self, context, sql: str):
        self._asked = []
        table = context.sql(sql)
        rows = sorted(zip(*(column.to_pylist() for column in table.columns)),
                      key=repr)
        return rows, self._asked

    def query(self, sql: str) -> str:
        """Run a query and return its transcript.

        Raises:
            AssertionError: The run with every pushdown off returns
                other rows.
        """
        self._prompts = re.findall(r"'((?:[^']|'')*)'", sql)
        plain_rows, plain_asked = self._run(self._plain, sql)
        rows, asked = self._run(self._optimized, sql)
        assert rows == plain_rows, (sql, rows, plain_rows)
        query = self.session.sql(sql)
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
        lines.append(f"rows {len(rows)}")
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
