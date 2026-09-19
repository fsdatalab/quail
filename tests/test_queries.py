"""CPU checks for the QUAIL-B query catalog."""

import json
from importlib.resources import files

import pytest
from substrait import plan_pb2

from quail_b.prompts import SERIOUS_ADVERSE_EVENT
from quail_b.queries import (
    FILTER_SELECTIVITY_ESTIMATES,
    JOIN_SELECTIVITY_ESTIMATES,
    PRIVACY_QUERIES,
    QUERIES,
    QUERY_FAMILY_WORKLOADS,
    QUERY_ORDER,
    QuerySpec,
    queries,
    query_family_name,
    split_query_families,
    split_query_ids,
)
from quail_b.run import _query_hash
from quail_b.substrait import (
    AI_JOIN_NAME,
    AND_NAME,
    EQUAL_NAME,
    _inspect_plan,
)
from tools.make_substrait_plans import write_plans


def test_catalog_has_the_30_default_queries_and_two_privacy_queries():
    assert len(QUERIES) == 30
    assert QUERY_ORDER == (
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 4)),
        *(f"FEV-{i}" for i in range(1, 11)),
        *(f"LEP-{i}" for i in range(1, 6)),
        "AGENT-1", "AGENT-2",
    )
    assert [spec.id for spec in PRIVACY_QUERIES] == ["PRIV-1", "PRIV-2"]
    assert list(queries(include_privacy=True)) == [*QUERY_ORDER, "PRIV-1", "PRIV-2"]
    for spec in QUERIES:
        info = _inspect_plan(spec.plan)
        assert all(
            filter_spec.prompt in FILTER_SELECTIVITY_ESTIMATES
            for filter_spec in info.filters
        ), spec.id
        assert all(
            join.prompt in JOIN_SELECTIVITY_ESTIMATES
            for join in info.joins
        ), spec.id
        assert all(name.endswith(".id") for name in info.select), spec.id
    for spec in PRIVACY_QUERIES:
        info = _inspect_plan(spec.plan)
        assert any(
            filter_spec.prompt not in FILTER_SELECTIVITY_ESTIMATES
            for filter_spec in info.filters
        ), spec.id


def test_query_spec_contains_only_identity_and_plan_bytes():
    assert tuple(QuerySpec.__dataclass_fields__) == (
        "id",
        "description",
        "plan_bytes",
    )


@pytest.mark.parametrize("query_id", ["BIO-1", "BIO-3"])
def test_biodex_filters_match_the_replacement_reference(query_id):
    info = _inspect_plan(queries()[query_id].plan)
    assert [(op.id, op.relation, op.prompt) for op in info.filters] == [
        ("filter-1", "r", SERIOUS_ADVERSE_EVENT),
    ]
    assert len(info.joins) == (1 if query_id == "BIO-3" else 0)


def test_filters_are_substrait_relations_over_their_input():
    spec = queries()["IMDB-4"]

    plan = spec.plan
    assert isinstance(plan, plan_pb2.Plan)
    info = _inspect_plan(plan)
    assert [
        (relation.alias, relation.table, relation.text_column)
        for relation in info.relations
    ] == [
        ("r", "reviews", "body"),
        ("a", "aspects", "aspect"),
    ]
    assert [
        (operator.id, operator.relation) for operator in info.filters
    ] == [
        ("filter-1", "r"),
        ("filter-2", "r"),
    ]
    assert [operator.id for operator in info.joins] == ["join-1"]

    project = plan.relations[0].root.input.project
    join = project.input.join
    assert join.common.hint.alias == "join-1"
    assert join.left.filter.common.hint.alias == "filter-2"
    assert join.left.filter.input.filter.common.hint.alias == "filter-1"
    assert join.left.filter.input.filter.input.read.common.hint.alias == "r"
    assert join.right.read.common.hint.alias == "a"
    assert not plan.expected_type_urls


def test_fev_10_combines_ai_and_ordinary_join_conditions():
    spec = queries()["FEV-10"]
    plan = spec.plan
    names = {
        declaration.extension_function.function_anchor:
        declaration.extension_function.name
        for declaration in plan.extensions
    }
    assert set(names.values()) == {
        AI_JOIN_NAME,
        AND_NAME,
        EQUAL_NAME,
        "ai_filter:str_str",
    }
    condition = plan.relations[0].root.input.project.input.join.expression
    assert names[condition.scalar_function.function_reference] == AND_NAME
    assert {
        names[argument.value.scalar_function.function_reference]
        for argument in condition.scalar_function.arguments
    } == {AI_JOIN_NAME, EQUAL_NAME}
    join = plan.relations[0].root.input.project.input.join
    assert tuple(join.left.filter.input.read.base_schema.names) == (
        "id",
        "claim",
        "evidence_wiki_url",
    )
    assert tuple(join.right.filter.input.read.base_schema.names) == (
        "id",
        "text",
    )


def test_repeated_tables_and_join_order_round_trip_through_substrait():
    spec = queries()["IMDB-9"]
    details = _inspect_plan(spec.plan)

    assert [(relation.alias, relation.table) for relation in details.relations] == [
        ("r1", "reviews"),
        ("a1", "aspects"),
        ("r2", "reviews"),
        ("a2", "aspects"),
    ]
    assert [join.relations for join in details.joins] == [
        ("r1", "a1"),
        ("r2", "a1"),
        ("r2", "a2"),
    ]


def test_definition_hash_uses_query_semantics_not_protobuf_bytes():
    original = queries()["IMDB-4"]
    copy = QuerySpec.from_plan(original.id, original.description, original.plan)

    assert copy.plan_bytes == original.plan_bytes
    assert _query_hash(copy) == _query_hash(original)
    different_wire = original.plan
    different_wire.version.producer = "another-producer"
    same_query = QuerySpec.from_plan(
        original.id,
        original.description,
        different_wire,
    )
    assert same_query.plan_bytes != original.plan_bytes
    assert _query_hash(same_query) == _query_hash(original)
    changed_description = QuerySpec.from_plan(
        original.id,
        "changed",
        original.plan,
    )
    assert _query_hash(changed_description) == _query_hash(original)
    changed_prompt = original.plan
    first_filter = (
        changed_prompt.relations[0]
        .root.input.project.input.join.left.filter.input.filter
    )
    first_filter.condition.scalar_function.arguments[0].value.literal.string = (
        "changed"
    )
    changed_query = QuerySpec.from_plan(
        original.id,
        original.description,
        changed_prompt,
    )
    assert _query_hash(changed_query) != _query_hash(original)


def test_ai_extension_definition_is_packaged():
    extension = files("quail_b").joinpath("substrait_extensions.yaml").read_text()

    assert "urn: extension:org.fsdatalab.quail_b:functions_ai" in extension
    assert "name: ai_filter" in extension
    assert "name: ai_join" in extension


def test_substrait_plans_are_packaged():
    package = files("quail_b")
    catalog = package.joinpath("plans", "catalog.json")
    entries = json.loads(catalog.read_text())

    assert len(entries) == 32
    assert all(
        package.joinpath("plans", f"{entry['id']}.json").is_file()
        for entry in entries
    )


def test_checked_in_plans_equal_the_generator_output(tmp_path):
    write_plans(tmp_path)
    package = files("quail_b").joinpath("plans")

    generated = sorted(path.name for path in tmp_path.iterdir())
    assert generated == sorted(path.name for path in package.iterdir())
    for name in generated:
        assert (tmp_path / name).read_text() == package.joinpath(name).read_text(), (
            f"{name} differs; run tools/make_substrait_plans.py"
        )


def test_parallel_query_split_matches_stock_vllm():
    assert split_query_ids(QUERY_ORDER, 4) == (
        QUERY_ORDER[0:8],
        QUERY_ORDER[8:16],
        QUERY_ORDER[16:23],
        QUERY_ORDER[23:30],
    )


def test_query_family_split_matches_benchmark_catalog():
    assert QUERY_FAMILY_WORKLOADS == {
        "IMDB": "imdb",
        "BIO": "biodex",
        "FEV": "fever",
        "LEP": "lepard",
        "AGENT": "agent",
    }
    assert split_query_families(QUERY_ORDER) == (
        QUERY_ORDER[0:10],
        QUERY_ORDER[10:13],
        QUERY_ORDER[13:23],
        QUERY_ORDER[23:28],
        QUERY_ORDER[28:30],
    )
    assert query_family_name(QUERY_ORDER[0:10]) == "imdb"


def test_query_family_rejects_mixed_or_unknown_queries():
    with pytest.raises(ValueError, match="expected one query family"):
        query_family_name(("IMDB-1", "BIO-1"))
    with pytest.raises(ValueError, match="unknown query family"):
        split_query_families(("OTHER-1",))
