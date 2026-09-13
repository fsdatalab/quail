"""Small Substrait constructors for tests."""

from dataclasses import dataclass

from substrait import algebra_pb2 as algebra
from substrait import plan_pb2, type_pb2
from substrait.extensions import extensions_pb2

from quail_b.substrait import (
    AI_EXTENSION_URN,
    AI_FILTER_NAME,
    AI_JOIN_NAME,
    SUBSTRAIT_VERSION,
)
from quail_b.substrait_metadata_pb2 import (
    OperatorMetadata,
    RelationMetadata,
)


@dataclass(frozen=True)
class Node:
    rel: algebra.Rel
    fields: tuple[str, ...]
    functions: frozenset[str] = frozenset()


def _string_type():
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _bool_type():
    return type_pb2.Type(
        bool=type_pb2.Type.Boolean(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _metadata(message):
    extension = extensions_pb2.AdvancedExtension()
    extension.optimization.add().Pack(message, deterministic=True)
    return extension


def _selection(node, field):
    index = node.fields.index(field)
    return algebra.Expression(
        selection=algebra.Expression.FieldReference(
            direct_reference=algebra.Expression.ReferenceSegment(
                struct_field=algebra.Expression.ReferenceSegment.StructField(
                    field=index
                )
            ),
            root_reference=algebra.Expression.FieldReference.RootReference(),
        )
    )


def _literal(value):
    return algebra.Expression(
        literal=algebra.Expression.Literal(string=value)
    )


def _scalar(anchor, arguments):
    return algebra.Expression(
        scalar_function=algebra.Expression.ScalarFunction(
            function_reference=anchor,
            arguments=[
                algebra.FunctionArgument(value=argument)
                for argument in arguments
            ],
            output_type=_bool_type(),
        )
    )


def read_rel(table, alias, text_column):
    names = ("id", text_column)
    read = algebra.ReadRel(
        common=algebra.RelCommon(direct=algebra.RelCommon.Direct()),
        base_schema=type_pb2.NamedStruct(
            names=names,
            struct=type_pb2.Type.Struct(
                types=[_string_type() for _name in names],
                nullability=type_pb2.Type.NULLABILITY_REQUIRED,
            ),
        ),
        named_table=algebra.ReadRel.NamedTable(names=[table]),
        advanced_extension=_metadata(
            RelationMetadata(
                schema_version=1,
                alias=alias,
                text_column=text_column,
            )
        ),
    )
    return Node(
        algebra.Rel(read=read),
        tuple(f"{alias}.{name}" for name in names),
    )


def filter_rel(node, operator_id, prompt, field):
    relation = algebra.FilterRel(
        common=algebra.RelCommon(direct=algebra.RelCommon.Direct()),
        input=node.rel,
        condition=_scalar(
            1,
            [_literal(prompt), _selection(node, field)],
        ),
        advanced_extension=_metadata(
            OperatorMetadata(schema_version=1, operator_id=operator_id)
        ),
    )
    return Node(
        algebra.Rel(filter=relation),
        node.fields,
        node.functions | {AI_FILTER_NAME},
    )


def join_rel(left, right, operator_id, prompt, fields):
    node = Node(
        algebra.Rel(),
        left.fields + right.fields,
        left.functions | right.functions | {AI_JOIN_NAME},
    )
    relation = algebra.JoinRel(
        common=algebra.RelCommon(direct=algebra.RelCommon.Direct()),
        left=left.rel,
        right=right.rel,
        expression=_scalar(
            2,
            [
                _literal(prompt),
                _selection(node, fields[0]),
                _selection(node, fields[1]),
            ],
        ),
        type=algebra.JoinRel.JOIN_TYPE_INNER,
        advanced_extension=_metadata(
            OperatorMetadata(schema_version=1, operator_id=operator_id)
        ),
    )
    return Node(algebra.Rel(join=relation), node.fields, node.functions)


def project_plan(node, select):
    expressions = [_selection(node, name) for name in select]
    project = algebra.Rel(
        project=algebra.ProjectRel(
            common=algebra.RelCommon(
                emit=algebra.RelCommon.Emit(
                    output_mapping=[
                        len(node.fields) + index
                        for index in range(len(expressions))
                    ]
                )
            ),
            input=node.rel,
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
    for name, anchor in (
        (AI_FILTER_NAME, 1),
        (AI_JOIN_NAME, 2),
    ):
        if name not in node.functions:
            continue
        extensions.append(
            extensions_pb2.SimpleExtensionDeclaration(
                extension_function=(
                    extensions_pb2.SimpleExtensionDeclaration.ExtensionFunction(
                        extension_urn_reference=1,
                        function_anchor=anchor,
                        name=name,
                    )
                )
            )
        )
    type_urls = [
        f"type.googleapis.com/{RelationMetadata.DESCRIPTOR.full_name}",
        f"type.googleapis.com/{OperatorMetadata.DESCRIPTOR.full_name}",
    ]
    return plan_pb2.Plan(
        version=plan_pb2.Version(
            major_number=SUBSTRAIT_VERSION[0],
            minor_number=SUBSTRAIT_VERSION[1],
            patch_number=SUBSTRAIT_VERSION[2],
            producer="quail-b-test",
        ),
        extension_urns=extension_urns,
        extensions=extensions,
        relations=[
            plan_pb2.PlanRel(
                root=algebra.RelRoot(
                    input=project,
                    names=[name.split(".", 1)[0] for name in select],
                )
            )
        ],
        expected_type_urls=type_urls,
        execution_behavior=plan_pb2.ExecutionBehavior(
            variable_eval_mode=(
                plan_pb2.ExecutionBehavior.VARIABLE_EVALUATION_MODE_PER_PLAN
            )
        ),
    )
