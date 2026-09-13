"""Build and inspect the Substrait plans used by QUAIL-B."""

from __future__ import annotations

from dataclasses import dataclass

from google.protobuf import json_format, struct_pb2
from substrait import algebra_pb2, plan_pb2, type_pb2
from substrait.extensions import extensions_pb2

SUBSTRAIT_VERSION = (0, 103, 0)
AI_EXTENSION_URN = "extension:org.fsdatalab.quail_b:functions_ai"
COMPARISON_EXTENSION_URN = "extension:io.substrait:functions_comparison"
BOOLEAN_EXTENSION_URN = "extension:io.substrait:functions_boolean"

AI_FILTER_ANCHOR = 1
AI_JOIN_ANCHOR = 2
EQUAL_ANCHOR = 3
AND_ANCHOR = 4

AI_FILTER_NAME = "ai_filter:str_str"
AI_JOIN_NAME = "ai_join:str_str_str"
EQUAL_NAME = "equal:any_any"
AND_NAME = "and:bool"

_RELATION_METADATA = "quail_b.relation"
_OPERATOR_METADATA = "quail_b.operator"
_STRUCT_TYPE_URL = "type.googleapis.com/google.protobuf.Struct"
_FUNCTION_URNS = {
    AI_FILTER_NAME: AI_EXTENSION_URN,
    AI_JOIN_NAME: AI_EXTENSION_URN,
    EQUAL_NAME: COMPARISON_EXTENSION_URN,
    AND_NAME: BOOLEAN_EXTENSION_URN,
}


@dataclass(frozen=True)
class RelationSpec:
    """One document relation decoded from a Substrait ReadRel."""

    alias: str
    table: str
    text_column: str


@dataclass(frozen=True)
class FilterSpec:
    """One AI filter decoded from a Substrait FilterRel."""

    id: str
    relation: str
    prompt: str


@dataclass(frozen=True)
class JoinSpec:
    """One AI join decoded from a Substrait JoinRel."""

    id: str
    relations: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


type OperatorSpec = FilterSpec | JoinSpec


@dataclass(frozen=True)
class PlanDetails:
    """The benchmark fields decoded from one Substrait plan."""

    relations: tuple[RelationSpec, ...]
    operators: tuple[OperatorSpec, ...]
    select: tuple[str, ...]

    @property
    def filters(self) -> tuple[FilterSpec, ...]:
        """Return AI filters in logical plan order."""
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, FilterSpec)
        )

    @property
    def joins(self) -> tuple[JoinSpec, ...]:
        """Return AI joins in logical plan order."""
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, JoinSpec)
        )


@dataclass(frozen=True)
class _Node:
    rel: algebra_pb2.Rel
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _DecodedNode:
    fields: tuple[tuple[str, str], ...]
    relations: tuple[RelationSpec, ...]
    operators: tuple[OperatorSpec, ...]


def _string_type() -> type_pb2.Type:
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _boolean_type() -> type_pb2.Type:
    return type_pb2.Type(
        bool=type_pb2.Type.Boolean(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _metadata(kind: str, **values) -> extensions_pb2.AdvancedExtension:
    metadata = struct_pb2.Struct()
    metadata.update({"kind": kind, **values})
    extension = extensions_pb2.AdvancedExtension()
    extension.optimization.add().Pack(metadata, deterministic=True)
    return extension


def _read_metadata(
    extension: extensions_pb2.AdvancedExtension,
    kind: str,
) -> dict:
    for packed in extension.optimization:
        if not packed.Is(struct_pb2.Struct.DESCRIPTOR):
            continue
        metadata = struct_pb2.Struct()
        packed.Unpack(metadata)
        values = json_format.MessageToDict(metadata)
        if values.get("kind") == kind:
            return values
    raise ValueError(f"Substrait relation lacks {kind} metadata")


def _selection(index: int) -> algebra_pb2.Expression:
    return algebra_pb2.Expression(
        selection=algebra_pb2.Expression.FieldReference(
            direct_reference=algebra_pb2.Expression.ReferenceSegment(
                struct_field=(
                    algebra_pb2.Expression.ReferenceSegment.StructField(
                        field=index
                    )
                )
            ),
            root_reference=algebra_pb2.Expression.FieldReference.RootReference(),
        )
    )


def _string_literal(value: str) -> algebra_pb2.Expression:
    return algebra_pb2.Expression(
        literal=algebra_pb2.Expression.Literal(string=value)
    )


def _scalar(
    anchor: int,
    arguments: list[algebra_pb2.Expression],
) -> algebra_pb2.Expression:
    return algebra_pb2.Expression(
        scalar_function=algebra_pb2.Expression.ScalarFunction(
            function_reference=anchor,
            arguments=[
                algebra_pb2.FunctionArgument(value=argument)
                for argument in arguments
            ],
            output_type=_boolean_type(),
        )
    )


def _extension_function(
    urn_reference: int,
    function_anchor: int,
    name: str,
) -> extensions_pb2.SimpleExtensionDeclaration:
    return extensions_pb2.SimpleExtensionDeclaration(
        extension_function=(
            extensions_pb2.SimpleExtensionDeclaration.ExtensionFunction(
                extension_urn_reference=urn_reference,
                function_anchor=function_anchor,
                name=name,
            )
        )
    )


def _selected_column(name: str) -> tuple[str, str]:
    alias, separator, column = name.partition(".")
    if not separator or not alias or not column:
        raise ValueError(f"invalid selected column {name!r}")
    return alias, column


def _validate_components(
    query_id: str,
    relations: tuple[RelationSpec, ...],
    operators: tuple[OperatorSpec, ...],
    select: tuple[str, ...],
) -> None:
    if not relations:
        raise ValueError(f"{query_id}: a query needs at least one relation")
    for relation in relations:
        values = (relation.alias, relation.table, relation.text_column)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(
                f"{query_id}: relation names must be nonempty strings"
            )
    aliases = [relation.alias for relation in relations]
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"{query_id}: relation aliases must be unique")
    operator_ids = [operator.id for operator in operators]
    if any(
        not isinstance(operator_id, str) or not operator_id
        for operator_id in operator_ids
    ):
        raise ValueError(f"{query_id}: operator ids must be nonempty strings")
    if len(set(operator_ids)) != len(operator_ids):
        raise ValueError(f"{query_id}: operator ids must be unique")

    names = set(aliases)
    for operator in operators:
        if not isinstance(operator, (FilterSpec, JoinSpec)):
            raise TypeError(f"{query_id}: unsupported operator {operator!r}")
        if not isinstance(operator.prompt, str) or not operator.prompt:
            raise ValueError(
                f"{query_id}: operator {operator.id!r} needs a prompt"
            )
        if isinstance(operator, FilterSpec):
            referenced = (operator.relation,)
        else:
            if len(operator.relations) != 2:
                raise ValueError(
                    f"{query_id}: join {operator.id!r} needs two relations"
                )
            referenced = operator.relations
            if any(
                not isinstance(column, str) or not column
                for columns in operator.on
                for column in columns
            ):
                raise ValueError(
                    f"{query_id}: join {operator.id!r} has an invalid column"
                )
        if not set(referenced) <= names:
            raise ValueError(
                f"{query_id}: operator {operator.id!r} references an "
                "unknown relation"
            )
        if isinstance(operator, JoinSpec) and len(set(referenced)) != 2:
            raise ValueError(
                f"{query_id}: join {operator.id!r} needs two relations"
            )

    joins = tuple(
        operator for operator in operators if isinstance(operator, JoinSpec)
    )
    if len(relations) != len(joins) + 1:
        raise ValueError(
            f"{query_id}: {len(relations)} relations need "
            f"{len(relations) - 1} joins, not {len(joins)}"
        )
    joined = {relations[0].alias}
    for join in joins:
        referenced = set(join.relations)
        if len(referenced & joined) != 1 or len(referenced - joined) != 1:
            raise ValueError(
                f"{query_id}: join {join.id!r} must add one relation"
            )
        joined.update(referenced)

    first_join = {alias: len(operators) for alias in aliases}
    for index, operator in enumerate(operators):
        if isinstance(operator, JoinSpec):
            for alias in operator.relations:
                first_join[alias] = min(first_join[alias], index)
    for index, operator in enumerate(operators):
        if (
            isinstance(operator, FilterSpec)
            and index > first_join[operator.relation]
        ):
            raise ValueError(
                f"{query_id}: filter {operator.id!r} must precede joins "
                f"on relation {operator.relation!r}"
            )

    if not select:
        raise ValueError(f"{query_id}: a query must select at least one column")
    for name in select:
        alias, _column = _selected_column(name)
        if alias not in names:
            raise ValueError(
                f"{query_id}: selected column {name!r} has an unknown relation"
            )


def _relation_columns(
    relation: RelationSpec,
    operators: tuple[OperatorSpec, ...],
    select: tuple[str, ...],
) -> tuple[str, ...]:
    columns = ["id", relation.text_column]
    for name in select:
        alias, column = _selected_column(name)
        if alias == relation.alias and column not in columns:
            columns.append(column)
    for operator in operators:
        if not isinstance(operator, JoinSpec):
            continue
        left_alias, right_alias = operator.relations
        for left_column, right_column in operator.on:
            for alias, column in (
                (left_alias, left_column),
                (right_alias, right_column),
            ):
                if alias == relation.alias and column not in columns:
                    columns.append(column)
    return tuple(columns)


def _read_node(
    relation: RelationSpec,
    columns: tuple[str, ...],
    rel_anchor: int,
) -> _Node:
    schema = type_pb2.NamedStruct(
        names=columns,
        struct=type_pb2.Type.Struct(
            types=[_string_type() for _column in columns],
            nullability=type_pb2.Type.NULLABILITY_REQUIRED,
        ),
    )
    read = algebra_pb2.ReadRel(
        common=algebra_pb2.RelCommon(
            direct=algebra_pb2.RelCommon.Direct(),
            rel_anchor=rel_anchor,
        ),
        base_schema=schema,
        named_table=algebra_pb2.ReadRel.NamedTable(names=[relation.table]),
        advanced_extension=_metadata(
            _RELATION_METADATA,
            alias=relation.alias,
            text_column=relation.text_column,
        ),
    )
    return _Node(
        algebra_pb2.Rel(read=read),
        tuple((relation.alias, column) for column in columns),
    )


def build_plan(
    query_id: str,
    relations: tuple[RelationSpec, ...],
    operators: tuple[OperatorSpec, ...],
    select: tuple[str, ...],
) -> plan_pb2.Plan:
    """Build the canonical Substrait plan for a benchmark query."""
    _validate_components(query_id, relations, operators, select)
    next_anchor = 1
    nodes = {}
    relation_by_alias = {relation.alias: relation for relation in relations}
    for relation in relations:
        columns = _relation_columns(relation, operators, select)
        nodes[relation.alias] = _read_node(relation, columns, next_anchor)
        next_anchor += 1

    current = nodes[relations[0].alias]
    joined = {relations[0].alias}
    uses_equal = False
    for operator in operators:
        if isinstance(operator, FilterSpec):
            child = nodes[operator.relation]
            relation = relation_by_alias[operator.relation]
            field = child.fields.index(
                (operator.relation, relation.text_column)
            )
            condition = _scalar(
                AI_FILTER_ANCHOR,
                [_string_literal(operator.prompt), _selection(field)],
            )
            filtered = _Node(
                algebra_pb2.Rel(
                    filter=algebra_pb2.FilterRel(
                        common=algebra_pb2.RelCommon(
                            direct=algebra_pb2.RelCommon.Direct(),
                            rel_anchor=next_anchor,
                        ),
                        input=child.rel,
                        condition=condition,
                        advanced_extension=_metadata(
                            _OPERATOR_METADATA,
                            id=operator.id,
                        ),
                    )
                ),
                child.fields,
            )
            next_anchor += 1
            nodes[operator.relation] = filtered
            if operator.relation == relations[0].alias:
                current = filtered
            continue

        (added,) = set(operator.relations) - joined
        right = nodes[added]
        fields = current.fields + right.fields
        left_relation, right_relation = operator.relations
        left_column = relation_by_alias[left_relation].text_column
        right_column = relation_by_alias[right_relation].text_column
        condition = _scalar(
            AI_JOIN_ANCHOR,
            [
                _string_literal(operator.prompt),
                _selection(fields.index((left_relation, left_column))),
                _selection(fields.index((right_relation, right_column))),
            ],
        )
        for left_on, right_on in operator.on:
            uses_equal = True
            equality = _scalar(
                EQUAL_ANCHOR,
                [
                    _selection(fields.index((left_relation, left_on))),
                    _selection(fields.index((right_relation, right_on))),
                ],
            )
            condition = _scalar(AND_ANCHOR, [condition, equality])
        current = _Node(
            algebra_pb2.Rel(
                join=algebra_pb2.JoinRel(
                    common=algebra_pb2.RelCommon(
                        direct=algebra_pb2.RelCommon.Direct(),
                        rel_anchor=next_anchor,
                    ),
                    left=current.rel,
                    right=right.rel,
                    expression=condition,
                    type=algebra_pb2.JoinRel.JOIN_TYPE_INNER,
                    advanced_extension=_metadata(
                        _OPERATOR_METADATA,
                        id=operator.id,
                    ),
                )
            ),
            fields,
        )
        next_anchor += 1
        joined.add(added)

    selections = [_selected_column(name) for name in select]
    expressions = [
        _selection(current.fields.index(selection))
        for selection in selections
    ]
    project = algebra_pb2.Rel(
        project=algebra_pb2.ProjectRel(
            common=algebra_pb2.RelCommon(
                emit=algebra_pb2.RelCommon.Emit(
                    output_mapping=[
                        len(current.fields) + index
                        for index in range(len(expressions))
                    ]
                ),
                rel_anchor=next_anchor,
            ),
            input=current.rel,
            expressions=expressions,
        )
    )

    extension_urns = [
        extensions_pb2.SimpleExtensionURN(
            extension_urn_anchor=1,
            urn=AI_EXTENSION_URN,
        )
    ]
    extensions = []
    if any(isinstance(operator, FilterSpec) for operator in operators):
        extensions.append(
            _extension_function(1, AI_FILTER_ANCHOR, AI_FILTER_NAME)
        )
    if any(isinstance(operator, JoinSpec) for operator in operators):
        extensions.append(
            _extension_function(1, AI_JOIN_ANCHOR, AI_JOIN_NAME)
        )
    if uses_equal:
        extension_urns.extend([
            extensions_pb2.SimpleExtensionURN(
                extension_urn_anchor=2,
                urn=COMPARISON_EXTENSION_URN,
            ),
            extensions_pb2.SimpleExtensionURN(
                extension_urn_anchor=3,
                urn=BOOLEAN_EXTENSION_URN,
            ),
        ])
        extensions.extend([
            _extension_function(2, EQUAL_ANCHOR, EQUAL_NAME),
            _extension_function(3, AND_ANCHOR, AND_NAME),
        ])

    return plan_pb2.Plan(
        version=plan_pb2.Version(
            major_number=SUBSTRAIT_VERSION[0],
            minor_number=SUBSTRAIT_VERSION[1],
            patch_number=SUBSTRAIT_VERSION[2],
            producer="quail-b",
        ),
        extension_urns=extension_urns,
        extensions=extensions,
        relations=[
            plan_pb2.PlanRel(
                root=algebra_pb2.RelRoot(
                    input=project,
                    names=[alias for alias, _column in selections],
                )
            )
        ],
        expected_type_urls=[_STRUCT_TYPE_URL],
        execution_behavior=plan_pb2.ExecutionBehavior(
            variable_eval_mode=(
                plan_pb2.ExecutionBehavior.VARIABLE_EVALUATION_MODE_PER_PLAN
            )
        ),
    )


def _function_names(plan: plan_pb2.Plan) -> dict[int, str]:
    urns = {}
    for declaration in plan.extension_urns:
        anchor = declaration.extension_urn_anchor
        if anchor in urns:
            raise ValueError(f"duplicate Substrait extension URN anchor {anchor}")
        urns[anchor] = declaration.urn

    names = {}
    for declaration in plan.extensions:
        if declaration.HasField("extension_function"):
            function = declaration.extension_function
            if function.function_anchor in names:
                raise ValueError(
                    "duplicate Substrait function anchor "
                    f"{function.function_anchor}"
                )
            expected_urn = _FUNCTION_URNS.get(function.name)
            if expected_urn is None:
                raise ValueError(
                    f"unsupported Substrait function {function.name!r}"
                )
            actual_urn = urns.get(function.extension_urn_reference)
            if actual_urn != expected_urn:
                raise ValueError(
                    f"Substrait function {function.name!r} must use "
                    f"{expected_urn!r}"
                )
            names[function.function_anchor] = function.name
    return names


def _selection_index(expression: algebra_pb2.Expression) -> int:
    if not expression.HasField("selection"):
        raise ValueError("Substrait AI argument must be a field selection")
    reference = expression.selection
    if not reference.HasField("direct_reference"):
        raise ValueError("Substrait AI field must use a direct reference")
    segment = reference.direct_reference
    if not segment.HasField("struct_field") or segment.struct_field.HasField(
        "child"
    ):
        raise ValueError("Substrait AI field must select one top-level column")
    return segment.struct_field.field


def _literal_string(expression: algebra_pb2.Expression) -> str:
    if not expression.HasField("literal"):
        raise ValueError("Substrait AI prompt must be a string literal")
    literal = expression.literal
    if literal.WhichOneof("literal_type") != "string":
        raise ValueError("Substrait AI prompt must be a string literal")
    return literal.string


def _selected_field(
    fields: tuple[tuple[str, str], ...],
    expression: algebra_pb2.Expression,
) -> tuple[str, str]:
    index = _selection_index(expression)
    if index >= len(fields):
        raise ValueError(f"Substrait field index {index} is out of range")
    return fields[index]


def _arguments(expression: algebra_pb2.Expression) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        raise ValueError("Substrait predicate must be a scalar function")
    return [argument.value for argument in expression.scalar_function.arguments]


def _flatten_conditions(
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
        conditions.extend(_flatten_conditions(argument.value, functions))
    return conditions


def _decode_rel(
    rel: algebra_pb2.Rel,
    functions: dict[int, str],
) -> _DecodedNode:
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        read = rel.read
        metadata = _read_metadata(
            read.advanced_extension,
            _RELATION_METADATA,
        )
        if not read.HasField("named_table") or not read.named_table.names:
            raise ValueError("QUAIL-B Substrait reads need a named table")
        alias = metadata["alias"]
        relation = RelationSpec(
            alias,
            read.named_table.names[-1],
            metadata["text_column"],
        )
        if len(read.base_schema.names) != len(read.base_schema.struct.types):
            raise ValueError("Substrait read schema names and types do not match")
        fields = tuple((alias, name) for name in read.base_schema.names)
        if (alias, "id") not in fields:
            raise ValueError("QUAIL-B Substrait reads need an id column")
        if (alias, relation.text_column) not in fields:
            raise ValueError("QUAIL-B Substrait reads need their text column")
        return _DecodedNode(fields, (relation,), ())

    if kind == "filter":
        child = _decode_rel(rel.filter.input, functions)
        metadata = _read_metadata(
            rel.filter.advanced_extension,
            _OPERATOR_METADATA,
        )
        condition = rel.filter.condition
        function = condition.scalar_function
        if functions.get(function.function_reference) != AI_FILTER_NAME:
            raise ValueError("QUAIL-B FilterRel must call ai_filter")
        arguments = _arguments(condition)
        if len(arguments) != 2:
            raise ValueError("ai_filter needs a prompt and document")
        prompt = _literal_string(arguments[0])
        field = _selected_field(child.fields, arguments[1])
        relation = next(
            relation
            for relation in child.relations
            if relation.alias == field[0]
        )
        if field[1] != relation.text_column:
            raise ValueError("ai_filter must receive the relation text column")
        operator = FilterSpec(metadata["id"], field[0], prompt)
        return _DecodedNode(
            child.fields,
            child.relations,
            (*child.operators, operator),
        )

    if kind == "join":
        if rel.join.type != algebra_pb2.JoinRel.JOIN_TYPE_INNER:
            raise ValueError("QUAIL-B JoinRel must be an inner join")
        left = _decode_rel(rel.join.left, functions)
        right = _decode_rel(rel.join.right, functions)
        fields = left.fields + right.fields
        conditions = _flatten_conditions(rel.join.expression, functions)
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
        left_field = _selected_field(fields, arguments[1])
        right_field = _selected_field(fields, arguments[2])
        relation_aliases = (left_field[0], right_field[0])
        relations = {
            relation.alias: relation
            for relation in (*left.relations, *right.relations)
        }
        if (
            left_field[1] != relations[left_field[0]].text_column
            or right_field[1] != relations[right_field[0]].text_column
        ):
            raise ValueError("ai_join must receive relation text columns")

        on = []
        for condition in conditions:
            function = condition.scalar_function
            function_name = functions.get(function.function_reference)
            if function_name == AI_JOIN_NAME:
                continue
            if function_name != EQUAL_NAME:
                raise ValueError("unsupported QUAIL-B join condition")
            equality = _arguments(condition)
            if len(equality) != 2:
                raise ValueError("equal needs two fields")
            first = _selected_field(fields, equality[0])
            second = _selected_field(fields, equality[1])
            if (first[0], second[0]) == relation_aliases:
                on.append((first[1], second[1]))
            elif (second[0], first[0]) == relation_aliases:
                on.append((second[1], first[1]))
            else:
                raise ValueError("join equality uses unrelated relations")

        metadata = _read_metadata(
            rel.join.advanced_extension,
            _OPERATOR_METADATA,
        )
        operator = JoinSpec(
            metadata["id"],
            relation_aliases,
            prompt,
            tuple(on),
        )
        return _DecodedNode(
            fields,
            (*left.relations, *right.relations),
            (*left.operators, *right.operators, operator),
        )

    raise ValueError(f"unsupported QUAIL-B Substrait relation {kind!r}")


def plan_details(plan: plan_pb2.Plan) -> PlanDetails:
    """Validate a QUAIL-B Substrait plan and return its benchmark fields."""
    version = plan.version
    if (
        version.major_number,
        version.minor_number,
        version.patch_number,
    ) != SUBSTRAIT_VERSION:
        raise ValueError(
            f"QUAIL-B requires Substrait {'.'.join(map(str, SUBSTRAIT_VERSION))}"
        )
    if _STRUCT_TYPE_URL not in plan.expected_type_urls:
        raise ValueError("Substrait plan must declare its Struct metadata")
    if (
        plan.execution_behavior.variable_eval_mode
        != plan_pb2.ExecutionBehavior.VARIABLE_EVALUATION_MODE_PER_PLAN
    ):
        raise ValueError("QUAIL-B requires per-plan variable evaluation")
    if len(plan.relations) != 1 or not plan.relations[0].HasField("root"):
        raise ValueError("QUAIL-B needs one Substrait root relation")
    root = plan.relations[0].root
    if not root.input.HasField("project"):
        raise ValueError("QUAIL-B Substrait root must contain a ProjectRel")
    project = root.input.project
    decoded = _decode_rel(project.input, _function_names(plan))
    select = tuple(
        ".".join(_selected_field(decoded.fields, expression))
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
    details = PlanDetails(decoded.relations, decoded.operators, select)
    _validate_components("plan", details.relations, details.operators, select)
    return details
