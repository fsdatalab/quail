"""Write a QUAIL-B query as SQL and compile it on a Quail session.

quail-b reads each query's Substrait plan into relations, operators,
and relational steps (`QuerySpec.info`). `query_sql` writes that
reading as one SQL statement in the Snowflake dialect: each AI
operator is an AI function in SELECT, WHERE, or a JOIN's ON, and the
aggregate, having, sort, and fetch steps are the clauses that follow.
"""

from __future__ import annotations

from quail_b.substrait import (
    Aggregate,
    Classify,
    ColumnTest,
    Fetch,
    Filter,
    Having,
    InList,
    PlanInfo,
    Sort,
)

AGGREGATE_SQL = {"count": "COUNT", "sum": "SUM", "avg": "AVG", "min": "MIN",
                 "max": "MAX"}

# an AI.MAP prompt is the document, a blank line, then the instruction
MAP_DOCUMENT = "{0}\n\n"


def build_query(session, info: PlanInfo, selectivity=None,
                order: str | None = None):
    """Compile a QUAIL-B query from the benchmark's reading of a plan.

    Args:
        session: A session with every relation's table registered.
        info: The plan as `QuerySpec.info` reads it.
        selectivity: Prompt -> the fraction of documents or pairs
            expected to pass, given to the planner for ordering; a label
            filter's key is (classification prompt, frozenset of
            accepted labels).
        order: The filter order rule `Session.sql` takes.
    """
    return session.sql(query_sql(info, selectivity), order=order)


# Snowflake string literal escapes, backslash first
_ESCAPES = (("\\", "\\\\"), ("'", "\\'"), ("\n", "\\n"), ("\t", "\\t"),
            ("\f", "\\f"), ("\b", "\\b"),
            ("\r", "\\r"))


def sql_string(text: str) -> str:
    """Quote text as a Snowflake string literal."""
    for character, escaped in _ESCAPES:
        text = text.replace(character, escaped)
    return f"'{text}'"


def _literal(value) -> str:
    return sql_string(value) if isinstance(value, str) else repr(value)


def _options(fraction) -> str:
    return "" if fraction is None else f", {{'selectivity': {fraction!r}}}"


def query_sql(info: PlanInfo, selectivity=None) -> str:
    """Return one QUAIL-B query as a SQL statement in the Snowflake dialect.

    Args:
        info: The plan as `QuerySpec.info` reads it.
        selectivity: As `build_query` takes it.
    """
    selectivity = selectivity or {}

    def document(alias: str) -> str:
        return f"{alias}.{info.relation(alias).text_column}"

    def prompt(template: str, *aliases: str) -> str:
        return (f"PROMPT({sql_string(template)}, "
                f"{', '.join(document(alias) for alias in aliases)})")

    def classify(item: Classify) -> str:
        aliases = (item.relation, *(() if item.partner is None else (item.partner,)))
        if any(item.descriptions):
            labels = ", ".join(f"({sql_string(label)}, {sql_string(description)})"
                               for label, description
                               in zip(item.labels, item.descriptions))
        else:
            labels = ", ".join(sql_string(label) for label in item.labels)
        return f"AI.CLASSIFY({prompt(item.prompt, *aliases)}, [{labels}])"

    # the AI functions that compute a named output, by that name
    outputs = {}
    for item in info.classifies:
        outputs[item.output] = classify(item)
    for item in info.scores:
        outputs[item.output] = f"AI.SCORE({prompt(item.prompt, item.relation)})"
    for item in info.sae_features:
        outputs[item.output] = (
            f"AI.SAE_FEATURES({prompt(item.prompt, item.relation)}, "
            f"pooling => {sql_string(item.pooling)})")
    for item in info.maps:
        outputs[item.output] = (
            f"AI.MAP({document(item.relation)}, "
            f"{sql_string(item.prompt.removeprefix(MAP_DOCUMENT))}, {item.max_tokens})")
    for struct in info.structs:
        extracts = [extract for extract in info.extracts
                    if info.struct_of(extract) == struct]
        questions = ", ".join(
            f"{sql_string(extract.output)}: {sql_string(extract.prompt)}"
            for extract in extracts)
        outputs[struct.name] = (
            f"AI.EXTRACT({document(extracts[0].relation)}, {{{questions}}})")
    names = {f"{item.relation}.{item.output}": item.output
             for item in (*info.classifies, *info.scores, *info.sae_features,
                          *info.maps)}
    names.update({name: struct.name for struct in info.structs
                  for name in struct.fields})

    def field(name: str) -> str:
        return names.get(name, name)

    selected = list(dict.fromkeys(field(name) for name in info.select))

    def tested(name: str) -> str:
        """Name an output SELECT computes; otherwise write its call."""
        name = field(name)
        return outputs[name] if name in outputs and name not in selected else name

    measures = {}
    if info.aggregate is not None:
        for name, function, argument in info.aggregate.measures:
            if function == "count_distinct":
                measures[name] = f"COUNT(DISTINCT {tested(argument)})"
            else:
                measures[name] = (f"{AGGREGATE_SQL[function]}"
                                  f"({'*' if argument is None else tested(argument)})")
    items = []
    for name in selected:
        if name in outputs:
            items.append(f"{outputs[name]} AS {name}")
        elif name in measures:
            items.append(f"{measures[name]} AS {name}")
        else:
            items.append(name)

    where = []
    prompts = {(item.relation, item.output): item.prompt
               for item in info.classifies}
    for item in info.operators:
        if isinstance(item, ColumnTest):
            where.append(f"{item.relation}.{item.column} {item.comparison} "
                         f"{_literal(item.value)}")
        elif isinstance(item, Filter):
            where.append(f"AI_FILTER({prompt(item.prompt, item.relation)}"
                         f"{_options(selectivity.get(item.prompt))})")
        elif isinstance(item, InList):
            labels = ", ".join(sql_string(label) for label in item.accepted)
            test = f"{tested(item.output)} IN ({labels})"
            fraction = selectivity.get(
                (prompts[(item.relation, item.output)], frozenset(item.accepted)))
            where.append(test if fraction is None
                         else f"HINT({test}{_options(fraction)})")
    joins, joined = [], {info.base_alias}
    for join in info.joins:
        left, right = join.relations
        added = [alias for alias in join.relations if alias not in joined]
        if len(added) != 1:
            raise ValueError(f"join {join.id!r} must add one relation")
        (new,) = added
        joined.add(new)
        conditions = [f"{left}.{left_column} = {right}.{right_column}"
                      for left_column, right_column in join.on]
        conditions.append(f"AI_FILTER({prompt(join.prompt, left, right)}"
                          f"{_options(selectivity.get(join.prompt))})")
        joins.append(f"JOIN {info.relation(new).table} {new} ON "
                     + " AND ".join(conditions))
    distinct = ""
    clauses = []
    for step in info.tail:
        if isinstance(step, Aggregate):
            if not step.measures and set(step.keys) == set(info.select):
                distinct = "DISTINCT "     # an aggregate with no measures
                continue
            clauses.append("GROUP BY " + ", ".join(field(key) for key in step.keys))
        elif isinstance(step, Having):
            clauses.append("HAVING " + " AND ".join(
                f"{measures[name]} {comparison} {_literal(value)}"
                for name, comparison, value in step.tests))
        elif isinstance(step, Sort):
            clauses.append("ORDER BY " + ", ".join(
                f"{field(name)} {'DESC' if descending else 'ASC'} NULLS LAST"
                for name, descending in step.keys))
        elif isinstance(step, Fetch):
            clauses.append(f"LIMIT {step.count}")
            if step.offset:
                clauses.append(f"OFFSET {step.offset}")
    base = info.base_alias
    lines = [f"SELECT {distinct}{', '.join(items)}",
             f"FROM {info.relation(base).table} {base}", *joins]
    if where:
        lines.append("WHERE " + " AND ".join(where))
    return "\n".join([*lines, *clauses])

