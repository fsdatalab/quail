"""Model calls as expressions, and the plan's operator walk."""

import pytest

from quail.logical import (
    Alias,
    ColumnRef,
    Compare,
    CompileError,
    FilterPredicate,
    Join,
    LogicalPlan,
    ModelCall,
    Project,
    Scan,
    SemanticFilter,
    SemanticJoin,
    bind_join_prompt,
    bind_prompt,
    is_score,
    model_call,
)

R = ColumnRef("r", "reviews", "review")
P = ColumnRef("p", "products", "description")


def _filter_call(kind="boolean"):
    return ModelCall(bind_prompt("q {0}", (R,)), kind)


def _pair_call(kind="boolean"):
    return ModelCall(bind_join_prompt("same {0} {1}", (R, P)), kind)


def test_model_call_reports_its_aliases_and_kind():
    call = _pair_call("score")
    assert call.aliases() == ("r", "p")
    assert is_score(call)
    assert not is_score(_filter_call())
    compare = Compare(call, ">=", 0.5)
    assert model_call(compare) is call
    assert model_call(Alias(call, "s")) is call


@pytest.mark.parametrize("expression,message", [
    (Compare(_filter_call(), ">=", 0.5), "only an AI.SCORE call compares"),
    (Compare(_filter_call("score"), "=", 0.5), "unsupported AI.SCORE comparison"),
    (Compare(_filter_call("score"), ">=", 1.5), "between 0 and 1"),
    (_filter_call("score"), "only when compared"),
    (ModelCall(bind_prompt("q {0}", (R,)), "label"), "kind must be one of"),
])
def test_predicates_must_answer_yes_or_no(expression, message):
    node = SemanticFilter(
        Scan("reviews", "r", "review"), (FilterPredicate(expression),)
    )
    with pytest.raises(CompileError, match=message):
        node.validate()


def test_semantic_join_checks_its_predicate_against_its_input():
    r = Scan("reviews", "r", "review")
    p = Scan("products", "p", "description")
    SemanticJoin(Join(r, p), _pair_call()).validate()
    with pytest.raises(CompileError, match="does not produce"):
        SemanticJoin(r, _pair_call()).validate()
    with pytest.raises(CompileError, match="one input"):
        SemanticJoin(Join(r, p), _pair_call()).with_children((r, p))


def test_operators_walk_lists_each_operator_in_written_order():
    r = Scan("reviews", "r", "review")
    p = Scan("products", "p", "description")
    first = FilterPredicate(_filter_call(), selectivity=0.5)
    second = FilterPredicate(Compare(_filter_call("score"), ">", 0.2))
    joined = SemanticJoin(
        Join(SemanticFilter(r, (first, second)), p), _pair_call()
    )
    plan = LogicalPlan(Project(joined, (
        ColumnRef("r", "reviews", "id"), Alias(_filter_call("score"), "s"),
    )))
    plan.validate()
    operators = plan.operators()
    assert [scan.alias for scan in operators.scans] == ["r", "p"]
    assert operators.filters == {"r": (first, second)}
    assert operators.joins == (joined,)
    assert operators.applies == ()
    assert operators.prompts == (
        first.prompt, second.prompt, joined.prompt,
    )
    assert plan.root.explain_fields()["columns"] == [
        "r.id", "AI.SCORE('DOCUMENT:\\n{0}\\n\\nq') AS s",
    ]
