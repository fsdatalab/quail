"""Read the benchmark information in a QUAIL-B Substrait plan."""

from __future__ import annotations

from dataclasses import dataclass

from substrait import algebra_pb2, plan_pb2

SUBSTRAIT_VERSION = (0, 103, 0)
AI_EXTENSION_URN = "extension:org.fsdatalab.quail_b:functions_ai"
COMPARISON_EXTENSION_URN = "extension:io.substrait:functions_comparison"
BOOLEAN_EXTENSION_URN = "extension:io.substrait:functions_boolean"

AI_FILTER_NAME = "ai_filter:str_str"
AI_JOIN_NAME = "ai_join:str_str_str"
EQUAL_NAME = "equal:any_any"
LTE_NAME = "lte:any_any"
AND_NAME = "and:bool"

_FUNCTION_URNS = {
    AI_FILTER_NAME: AI_EXTENSION_URN,
    AI_JOIN_NAME: AI_EXTENSION_URN,
    EQUAL_NAME: COMPARISON_EXTENSION_URN,
    LTE_NAME: COMPARISON_EXTENSION_URN,
    AND_NAME: BOOLEAN_EXTENSION_URN,
}


@dataclass(frozen=True)
class _Bound:
    """An ordinary upper bound on one integer column of a relation.

    Rows whose column is at most `value` reach the AI operators; the
    others are not part of the query's input.
    """

    column: str
    value: int


@dataclass(frozen=True)
class _Relation:
    alias: str
    table: str
    text_column: str
    bounds: tuple[_Bound, ...] = ()


@dataclass(frozen=True)
class _Filter:
    id: str
    relation: str
    prompt: str


@dataclass(frozen=True)
class _Join:
    id: str
    relations: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


type _Operator = _Filter | _Join


@dataclass(frozen=True)
class _PlanInfo:
    relations: tuple[_Relation, ...]
    operators: tuple[_Operator, ...]
    select: tuple[str, ...]

    @property
    def filters(self) -> tuple[_Filter, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, _Filter)
        )

    @property
    def joins(self) -> tuple[_Join, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, _Join)
        )

    @property
    def base_alias(self) -> str:
        return self.relations[0].alias

    def relation(self, alias: str) -> _Relation:
        return next(
            relation for relation in self.relations if relation.alias == alias
        )


@dataclass(frozen=True)
class _Decoded:
    fields: tuple[tuple[str, str], ...]
    tables: tuple[tuple[str, str], ...]
    text_columns: tuple[tuple[str, str], ...]
    operators: tuple[_Operator, ...]
    bounds: tuple[tuple[str, _Bound], ...] = ()


def _with_text_column(
    text_columns: tuple[tuple[str, str], ...],
    field: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    """Record the column an AI function reads from one relation."""
    alias, column = field
    known = dict(text_columns)
    if known.get(alias, column) != column:
        raise ValueError(f"relation {alias!r} is used with two text columns")
    if alias in known:
        return text_columns
    return (*text_columns, (alias, column))


def _function_names(plan: plan_pb2.Plan) -> dict[int, str]:
    urns = {}
    for declaration in plan.extension_urns:
        anchor = declaration.extension_urn_anchor
        if anchor in urns:
            raise ValueError(f"duplicate Substrait extension URN anchor {anchor}")
        urns[anchor] = declaration.urn
    names = {}
    for declaration in plan.extensions:
        if not declaration.HasField("extension_function"):
            raise ValueError("QUAIL-B plans support scalar functions only")
        function = declaration.extension_function
        if function.function_anchor in names:
            raise ValueError(
                f"duplicate Substrait function anchor {function.function_anchor}"
            )
        expected_urn = _FUNCTION_URNS.get(function.name)
        if expected_urn is None:
            raise ValueError(
                f"unsupported Substrait function {function.name!r}"
            )
        if urns.get(function.extension_urn_reference) != expected_urn:
            raise ValueError(
                f"Substrait function {function.name!r} has the wrong URN"
            )
        names[function.function_anchor] = function.name
    return names


def _selection_index(expression: algebra_pb2.Expression) -> int:
    if not expression.HasField("selection"):
        raise ValueError("Substrait argument must be a field selection")
    reference = expression.selection
    if not reference.HasField("direct_reference"):
        raise ValueError("Substrait field must use a direct reference")
    segment = reference.direct_reference
    if not segment.HasField("struct_field") or segment.struct_field.HasField(
        "child"
    ):
        raise ValueError("Substrait field must select one top-level column")
    return segment.struct_field.field


def _selected(
    fields: tuple[tuple[str, str], ...],
    expression: algebra_pb2.Expression,
) -> tuple[str, str]:
    index = _selection_index(expression)
    if index >= len(fields):
        raise ValueError(f"Substrait field index {index} is out of range")
    return fields[index]


def _arguments(
    expression: algebra_pb2.Expression,
) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        raise ValueError("Substrait predicate must be a scalar function")
    return [
        argument.value for argument in expression.scalar_function.arguments
    ]


def _literal_string(expression: algebra_pb2.Expression) -> str:
    if not expression.HasField("literal"):
        raise ValueError("Substrait AI prompt must be a string literal")
    literal = expression.literal
    if literal.WhichOneof("literal_type") != "string":
        raise ValueError("Substrait AI prompt must be a string literal")
    return literal.string


def _literal_int(expression: algebra_pb2.Expression) -> int:
    if not expression.HasField("literal"):
        raise ValueError("Substrait column bound must be an integer literal")
    literal = expression.literal
    kind = literal.WhichOneof("literal_type")
    if kind not in ("i8", "i16", "i32", "i64"):
        raise ValueError("Substrait column bound must be an integer literal")
    return int(getattr(literal, kind))


def _flatten(
    expression: algebra_pb2.Expression,
    functions: dict[int, str],
) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        return [expression]
    function = expression.scalar_function
    if functions.get(function.function_reference) != AND_NAME:
        return [expression]
    conditions = []
    for argument in function.arguments:
        conditions.extend(_flatten(argument.value, functions))
    return conditions


def _decode(
    rel: algebra_pb2.Rel,
    functions: dict[int, str],
) -> _Decoded:
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        read = rel.read
        alias = read.common.hint.alias
        if not alias:
            raise ValueError("QUAIL-B reads need a relation alias in hint.alias")
        if not read.HasField("named_table") or not read.named_table.names:
            raise ValueError("QUAIL-B reads need a named table")
        if len(read.base_schema.names) != len(read.base_schema.struct.types):
            raise ValueError("Substrait read schema names and types do not match")
        fields = tuple((alias, name) for name in read.base_schema.names)
        if (alias, "id") not in fields:
            raise ValueError("QUAIL-B reads need an id column")
        return _Decoded(fields, ((alias, read.named_table.names[-1]),), (), ())

    if kind == "filter":
        child = _decode(rel.filter.input, functions)
        condition = rel.filter.condition
        function = condition.scalar_function
        name = functions.get(function.function_reference)
        if name == LTE_NAME:
            if child.operators:
                raise ValueError(
                    "a QUAIL-B column bound must precede the AI operators")
            arguments = _arguments(condition)
            if len(arguments) != 2:
                raise ValueError("lte needs a field and an integer literal")
            alias, column = _selected(child.fields, arguments[0])
            bound = _Bound(column, _literal_int(arguments[1]))
            return _Decoded(
                child.fields,
                child.tables,
                child.text_columns,
                child.operators,
                (*child.bounds, (alias, bound)),
            )
        if name != AI_FILTER_NAME:
            raise ValueError("QUAIL-B FilterRel must call ai_filter or lte")
        arguments = _arguments(condition)
        if len(arguments) != 2:
            raise ValueError("ai_filter needs a prompt and document")
        prompt = _literal_string(arguments[0])
        field = _selected(child.fields, arguments[1])
        operator = _Filter(rel.filter.common.hint.alias, field[0], prompt)
        return _Decoded(
            child.fields,
            child.tables,
            _with_text_column(child.text_columns, field),
            (*child.operators, operator),
            child.bounds,
        )

    if kind == "join":
        if rel.join.type != algebra_pb2.JoinRel.JOIN_TYPE_INNER:
            raise ValueError("QUAIL-B JoinRel must be an inner join")
        left = _decode(rel.join.left, functions)
        right = _decode(rel.join.right, functions)
        fields = left.fields + right.fields
        conditions = _flatten(rel.join.expression, functions)
        ai_conditions = [
            condition
            for condition in conditions
            if condition.HasField("scalar_function")
            and functions.get(
                condition.scalar_function.function_reference
            ) == AI_JOIN_NAME
        ]
        if len(ai_conditions) != 1:
            raise ValueError("QUAIL-B JoinRel needs one ai_join condition")
        arguments = _arguments(ai_conditions[0])
        if len(arguments) != 3:
            raise ValueError("ai_join needs a prompt and two documents")
        prompt = _literal_string(arguments[0])
        document_fields = (
            _selected(fields, arguments[1]),
            _selected(fields, arguments[2]),
        )
        relation_aliases = tuple(field[0] for field in document_fields)
        text_columns = (*left.text_columns, *right.text_columns)
        for field in document_fields:
            text_columns = _with_text_column(text_columns, field)
        on = []
        for condition in conditions:
            function = condition.scalar_function
            name = functions.get(function.function_reference)
            if name == AI_JOIN_NAME:
                continue
            if name != EQUAL_NAME:
                raise ValueError("unsupported QUAIL-B join condition")
            equality = _arguments(condition)
            if len(equality) != 2:
                raise ValueError("equal needs two fields")
            first, second = (
                _selected(fields, expression) for expression in equality
            )
            if (first[0], second[0]) == relation_aliases:
                on.append((first[1], second[1]))
            elif (second[0], first[0]) == relation_aliases:
                on.append((second[1], first[1]))
            else:
                raise ValueError("join equality uses unrelated relations")
        operator = _Join(
            rel.join.common.hint.alias,
            relation_aliases,
            prompt,
            tuple(on),
        )
        return _Decoded(
            fields,
            (*left.tables, *right.tables),
            text_columns,
            (*left.operators, *right.operators, operator),
            (*left.bounds, *right.bounds),
        )

    raise ValueError(f"unsupported QUAIL-B Substrait relation {kind!r}")


def _field_alias(name: str) -> str:
    alias, separator, column = name.partition(".")
    if not separator or not alias or not column or "." in column:
        raise ValueError(f"invalid field name {name!r}")
    return alias


def _validate_info(info: _PlanInfo) -> None:
    aliases = [relation.alias for relation in info.relations]
    if not aliases or len(set(aliases)) != len(aliases):
        raise ValueError("QUAIL-B relation aliases must be nonempty and unique")
    operator_ids = [operator.id for operator in info.operators]
    if any(not operator_id for operator_id in operator_ids):
        raise ValueError("QUAIL-B operator IDs must be nonempty")
    if len(set(operator_ids)) != len(operator_ids):
        raise ValueError("QUAIL-B operator IDs must be unique")
    if len(info.relations) != len(info.joins) + 1:
        raise ValueError("QUAIL-B plan must join every relation")
    joined = {info.base_alias}
    for join in info.joins:
        referenced = set(join.relations)
        if len(referenced & joined) != 1 or len(referenced - joined) != 1:
            raise ValueError(f"join {join.id!r} must add one relation")
        joined.update(referenced)
    first_join = {alias: len(info.operators) for alias in aliases}
    for index, operator in enumerate(info.operators):
        if isinstance(operator, _Join):
            for alias in operator.relations:
                first_join[alias] = min(first_join[alias], index)
    for index, operator in enumerate(info.operators):
        if (
            isinstance(operator, _Filter)
            and index > first_join[operator.relation]
        ):
            raise ValueError(
                f"filter {operator.id!r} must precede joins on its relation"
            )
    selected = {_field_alias(name) for name in info.select}
    if not selected or not selected <= set(aliases):
        raise ValueError("QUAIL-B projection uses an unknown relation")


def _inspect_plan(plan: plan_pb2.Plan) -> _PlanInfo:
    """Return the benchmark information in a supported Substrait plan."""
    version = plan.version
    if (
        version.major_number,
        version.minor_number,
        version.patch_number,
    ) != SUBSTRAIT_VERSION:
        raise ValueError(
            f"QUAIL-B requires Substrait {'.'.join(map(str, SUBSTRAIT_VERSION))}"
        )
    if len(plan.relations) != 1 or not plan.relations[0].HasField("root"):
        raise ValueError("QUAIL-B needs one Substrait root relation")
    root = plan.relations[0].root
    if not root.input.HasField("project"):
        raise ValueError("QUAIL-B root must contain a ProjectRel")
    project = root.input.project
    decoded = _decode(project.input, _function_names(plan))
    select = tuple(
        ".".join(_selected(decoded.fields, expression))
        for expression in project.expressions
    )
    expected_mapping = tuple(
        len(decoded.fields) + index
        for index in range(len(project.expressions))
    )
    if tuple(project.common.emit.output_mapping) != expected_mapping:
        raise ValueError("QUAIL-B ProjectRel has an invalid output mapping")
    expected_names = tuple(name.split(".", 1)[0] for name in select)
    if tuple(root.names) != expected_names:
        raise ValueError("QUAIL-B root names do not match selected relations")
    text_columns = dict(decoded.text_columns)
    relations = []
    for alias, table in decoded.tables:
        if alias not in text_columns:
            raise ValueError(f"relation {alias!r} is not read by an AI function")
        bounds = tuple(
            bound for bound_alias, bound in decoded.bounds if bound_alias == alias)
        relations.append(_Relation(alias, table, text_columns[alias], bounds))
    info = _PlanInfo(tuple(relations), decoded.operators, select)
    _validate_info(info)
    return info
