"""Generated AI queries, each with the plain SQL that computes its rows.

A query writes each AI call as a macro, such as ``IF2(r.body)``.
``ai_sql`` expands the macros to AI calls. ``plain_sql`` expands them to
SQL that computes the same answer from the document's text, so plain
DataFusion returns the rows the AI query must return. ``truth`` answers
the test oracle from the same text. An AI call of a NULL document is
NULL, in both.
"""

import random
import re

import pyarrow as pa

NAMES = ["Acme", "acme", "Globex", "Hooli", "ACME corp", "Initech"]
MACRO = re.compile(r"\b(IF|CL|SC|MP|EX|JN)(\d)\(")
PAIR_TEXT = re.compile(r"\s*DOCUMENT 0:\n(.*)\n\nDOCUMENT 1:\n(.*)", re.S)


def _weight(text: str) -> int:
    return ord(text[0]) + len(text) + ord(text[-1]) if text else 0


def _weight_sql(expr: str) -> str:
    expr = f"coalesce({expr}, '')"
    return f"(ascii({expr}) + character_length({expr}) + ascii(right({expr}, 1)))"


def _extracted(key: int, text: str):
    value = _weight(text) + key
    if not text or (key != 1 and value % 4 == 3):
        return None
    words = text.split(" ")
    return words[value % 3] if value % 3 < len(words) and words[value % 3] else None


def _extracted_sql(key: int, expr: str) -> str:
    weight = _weight_sql(expr)
    word = f"nullif(split_part({expr}, ' ', 1 + ({weight} + {key}) % 3), '')"
    none = f"coalesce({expr}, '') = ''"
    if key != 1:
        none += f" OR ({weight} + {key}) % 4 = 3"
    return f"(CASE WHEN {none} THEN NULL ELSE {word} END)"


def _mapped_sql(key: int, expr: str) -> str:
    whens = " ".join(f"WHEN {index} THEN '{name}'"
                     for index, name in enumerate(NAMES))
    last = "''" if key == 3 else "'Acme'"
    return f"(CASE ({_weight_sql(expr)} + {key}) % 7 {whens} ELSE {last} END)"


def _label_sql(key: int, expr: str) -> str:
    last = "'b'" if key == 3 else "'a'"
    return (f"(CASE ({_weight_sql(expr)} + {key}) % 4 WHEN 0 THEN 'a' "
            f"WHEN 1 THEN 'b' WHEN 2 THEN 'c' ELSE {last} END)")


PLAIN = {
    "IF": lambda key, expr: f"(({_weight_sql(expr)} + {key}) % {key + 1} = 0)",
    "CL": _label_sql,
    "SC": lambda key, expr: f"((({_weight_sql(expr)} + {key}) % 5) / 4.0)",
    "MP": _mapped_sql,
    "EX": _extracted_sql,
    "JN": lambda key, left, right:
        f"(({_weight_sql(left)} + {_weight_sql(right)} + {key}) % 3 = 0)",
}
AI = {
    "IF": lambda key, expr: f"AI.IF(PROMPT('q{key}: {{0}}', {expr}))",
    "CL": lambda key, expr:
        f"AI.CLASSIFY(PROMPT('q{key}: {{0}}', {expr}), ['a', 'b', 'c'])",
    "SC": lambda key, expr: f"AI.SCORE(PROMPT('q{key}: {{0}}', {expr}))",
    "MP": lambda key, expr: f"AI.MAP({expr}, 'q{key}: name it', 16)",
    "EX": lambda key, expr:
        f"AI.EXTRACT({expr}, {{'f{key}': 'q{key}: which'}})['f{key}']",
    "JN": lambda key, left, right:
        f"AI.IF(PROMPT('q{key}: {{0}} {{1}}', {left}, {right}))",
}


def truth(kind: str, question: str, texts: tuple):
    """Return the answer the plain SQL computes, for ``Oracle(truth=...)``."""
    key = int(re.search(r"q(\d):", question).group(1))
    # a filter of two columns reads both as one document
    pair = PAIR_TEXT.fullmatch(texts[0]) if kind == "filter" else None
    if pair is not None:
        kind, texts = "join", pair.groups()
    weight = sum(_weight(text) for text in texts) + key
    if kind == "filter":
        return weight % (key + 1) == 0
    if kind == "join":
        return weight % 3 == 0
    if kind == "label":
        last = "b" if key == 3 else "a"
        return None if len(texts) > 1 else ["a", "b", "c", last][weight % 4]
    if kind == "score":
        return (weight % 5) / 4.0
    if kind == "map":
        return "" if weight % 7 == 6 and key == 3 else NAMES[weight % 7 % 6]
    return _extracted(key, texts[0])


def _arguments(text: str) -> list[str]:
    arguments, depth, quoted, current = [], 0, False, ""
    for char in text:
        quoted ^= char == "'"
        if not quoted:
            depth += {"(": 1, ")": -1}.get(char, 0)
            if char == "," and depth == 0:
                arguments.append(current.strip())
                current = ""
                continue
        current += char
    return [*arguments, current.strip()]


def _expand(sql: str, calls: dict) -> str:
    found = MACRO.search(sql)
    if found is None:
        return sql
    depth, end = 1, found.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(sql[end], 0)
        end += 1
    arguments = [_expand(argument, calls)
                 for argument in _arguments(sql[found.end():end - 1])]
    call = calls[found.group(1)](int(found.group(2)), *arguments)
    return sql[:found.start()] + call + _expand(sql[end:], calls)


def ai_sql(sql: str) -> str:
    """Return the query with each macro as its AI call."""
    return _expand(sql, AI)


def _strict(call):
    """Return the macro's SQL, NULL when a document is NULL."""
    def expanded(key, *documents):
        absent = " OR ".join(f"{document} IS NULL" for document in documents)
        return f"(CASE WHEN {absent} THEN NULL ELSE {call(key, *documents)} END)"
    return expanded


def plain_sql(sql: str) -> str:
    """Return the query with each macro as SQL over the document's text."""
    return _expand(sql, {name: _strict(call) for name, call in PLAIN.items()})


def tables(seed: int) -> dict[str, pa.Table]:
    """Return the tables the queries read: documents r and t, and names v."""
    rng = random.Random(seed)
    words = ["Acme", "acme", "Globex", "Hooli", "Initech", "Zed", "Bolt"]

    def documents(prefix, num_documents):
        # the last document is NULL
        return pa.table({
            "id": [f"{prefix}{index}" for index in range(num_documents)],
            "body": [f"{rng.choice(words)} {rng.choice(words)} {prefix}{index}"
                     f"{'x' * rng.randrange(3)}"
                     for index in range(num_documents - 1)] + [None],
            "stars": [rng.randrange(1, 5) for _ in range(num_documents)]})

    names = ["Acme", "acme", "Globex", "Globex", None, "Hooli", "Zed", None, "r3"]
    return {"r": documents("r", 9), "t": documents("t", 7), "v": pa.table({
        "id": [f"v{index}" for index in range(len(names))], "name": names,
        "rid": [f"r{rng.randrange(11)}" for _ in names]})}


class Queries:
    """Random queries over ``tables``, each returned as SQL with macros."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def _pick(self, *choices):
        return self.rng.choice(choices)

    def _key(self) -> int:
        return self.rng.choice((1, 2, 3))

    def _text(self, column: str) -> str:
        return f"{self._pick('MP', 'EX', 'CL')}{self._key()}({column})"

    def _value(self, column: str) -> str:
        kind = self._pick("text", "text", "SC", "IF", "lower", "case")
        if kind in ("SC", "IF"):
            return f"{kind}{self._key()}({column})"
        if kind == "lower":
            return f"lower({self._text(column)})"
        if kind == "case":
            return (f"CASE WHEN {self._test(column)} THEN {self._text(column)} "
                    f"ELSE 'none' END")
        return self._text(column)

    def _test(self, column: str) -> str:
        key, alias = self._key(), column.split(".")[0]
        name = self._pick("Acme", "acme", "Globex", "Hooli")
        return self._pick(
            f"IF{key}({column})", f"NOT IF{key}({column})",
            f"IF{key}({column}) = FALSE",
            f"CL{key}({column}) {self._pick('=', '<>')} '{self._pick(*'abc')}'",
            f"CL{key}({column}) {self._pick('IN', 'NOT IN')} ('a', 'c')",
            f"SC{key}({column}) {self._pick('>', '<=')} 0.5",
            f"MP{key}({column}) {self._pick('ILIKE', 'NOT LIKE')} "
            f"'{self._pick('%acme%', 'A%', '%o%')}'",
            f"lower(MP{key}({column})) = '{name.lower()}'",
            f"length(MP{key}({column})) > {self._pick(4, 5)}",
            f"EX{key}({column}) IS {self._pick('', 'NOT ')}NULL",
            f"left(EX{key}({column}), 1) = '{name[0]}'",
            f"{alias}.stars {self._pick('>', '<=')} {self._pick(1, 2, 3)}")

    def _where(self, column: str, depth: int = 0) -> str:
        shape = self._pick("test", "test", "and", "or", "not")
        if depth == 2 or shape == "test":
            return self._test(column)
        left, right = (self._where(column, depth + 1) for _ in range(2))
        if shape == "not":
            return f"NOT ({left})"
        return f"({left} {shape.upper()} {right})"

    def _one_table(self) -> str:
        table = self._pick("r", "t")
        column = f"{table}.body"
        values = "".join(f", {self._value(column)} AS c{index}"
                         for index in range(self.rng.randrange(3)))
        return (f"SELECT {table}.id{values} FROM {table} "
                f"WHERE {self._where(column)}")

    def _key_join(self) -> str:
        table = self._pick("r", "t")
        column, answer = f"{table}.body", self._text(f"{table}.body")
        join = self._pick("JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
                          "LEFT SEMI JOIN", "LEFT ANTI JOIN")
        term = self._pick("", "", f" AND v.rid = {table}.id",
                          f" AND {self._test(column)}", " AND IF1(v.name)")
        selected = f"{table}.id" if "SEMI" in join or "ANTI" in join else (
            f"{table}.id, v.id" + self._pick("", f", {answer} AS c0"))
        where = self._pick("", f" WHERE {self._where(column)}")
        return (f"SELECT {selected} FROM {table} {join} v "
                f"ON v.name = {answer}{term}{where}")

    def _pair_join(self) -> str:
        key = self._key()
        join = self._pick("JOIN", "JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN")
        term = self._pick("", " AND r.stars = t.stars",
                          f" AND {self._test('r.body')}")
        where = self._pick("", f" WHERE {self._where(self._pick('r', 't') + '.body')}")
        return (f"SELECT r.id, t.id FROM r {join} t "
                f"ON JN{key}(r.body, t.body){term}{where}")

    def _subquery(self) -> str:
        table = self._pick("r", "t")
        column, answer = f"{table}.body", self._text(f"{table}.body")
        test = self._where(column)
        return self._pick(
            f"SELECT {table}.id FROM {table} WHERE {answer} "
            f"{self._pick('IN', 'NOT IN')} "
            f"(SELECT v.name FROM v WHERE v.name IS NOT NULL)",
            f"SELECT {table}.id FROM {table} WHERE {self._pick('', 'NOT ')}EXISTS "
            f"(SELECT 1 FROM v WHERE v.name = {answer})",
            f"SELECT v.id FROM v WHERE {self._pick('', 'NOT ')}EXISTS "
            f"(SELECT 1 FROM {table} WHERE {table}.id = v.rid AND {test})",
            f"SELECT {table}.id, (SELECT COUNT(*) FROM v WHERE v.name = {answer}) "
            f"AS n FROM {table}",
            f"SELECT r.id FROM r WHERE EXISTS "
            f"(SELECT 1 FROM v WHERE JN{self._key()}(r.body, v.name))")

    def _grouped(self) -> str:
        table = self._pick("r", "t")
        column = f"{table}.body"
        where = self._pick("", f" WHERE {self._where(column)}")
        return self._pick(
            f"SELECT {self._text(column)} AS g, COUNT(*) AS n FROM {table}{where} "
            f"GROUP BY 1",
            f"SELECT {table}.stars, SUM(CASE WHEN {self._test(column)} THEN 1 "
            f"ELSE 0 END) AS n FROM {table}{where} GROUP BY {table}.stars",
            f"SELECT DISTINCT {self._value(column)} AS c0 FROM {table}{where}",
            f"SELECT {table}.id, {self._text(column)} AS c0 FROM {table}{where} "
            f"{self._pick('UNION ALL', 'INTERSECT', 'EXCEPT')} "
            f"SELECT v.rid, v.name FROM v",
            f"SELECT {table}.id, SC{self._key()}({column}) AS c0 FROM {table}"
            f"{where} ORDER BY c0 DESC, {table}.id LIMIT {self._pick(2, 4)}")

    def query(self) -> str:
        """Return one query, with its AI calls as macros."""
        shape = self._pick(self._one_table, self._one_table, self._key_join,
                           self._pair_join, self._subquery, self._grouped)
        return shape()
