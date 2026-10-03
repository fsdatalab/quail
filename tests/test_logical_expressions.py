"""Model calls as expressions, and the plan's operator walk."""

from dataclasses import replace

import pytest

from quail.logical import (
    Alias,
    ColumnRef,
    Compare,
    CompileError,
    Filter,
    FilterPredicate,
    InList,
    Join,
    LogicalPlan,
    ModelCall,
    Project,
    Scan,
    SemanticClassify,
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


INVALID_PREDICATES = [
    (Compare(_filter_call(), ">=", 0.5), "only an AI.SCORE call compares"),
    (Compare(_filter_call("score"), "=", 0.5), "unsupported AI.SCORE comparison"),
    (Compare(_filter_call("score"), ">=", 1.5), "between 0 and 1"),
    (_filter_call("score"), "only when compared"),
    (ModelCall(bind_prompt("q {0}", (R,)), "text"), "kind must be one of"),
    (ModelCall(bind_prompt("q {0}", (R,)), "label"), "at least two labels"),
    (ModelCall(bind_prompt("q {0}", (R,)), "label", ("a", "A")),
     "differ ignoring case"),
    (ModelCall(bind_prompt("q {0}", (R,)), "label", ("a", "b")),
     "tested by a Filter on its"),
    (ModelCall(bind_prompt("q {0}", (R,)), "boolean", ("a", "b")),
     "only an AI.CLASSIFY call has labels"),
    (InList(ColumnRef("r", "reviews", "tone"), ("a",)),
     "a model call or a comparison"),
]


def test_model_calls_and_predicates_validate_kind_comparison_and_input():
    call = _pair_call("score")
    assert call.aliases() == ("r", "p")
    assert is_score(call)
    assert not is_score(_filter_call())
    compare = Compare(call, ">=", 0.5)
    assert model_call(compare) is call
    assert model_call(Alias(call, "s")) is call

    for expression, message in INVALID_PREDICATES:
        node = SemanticFilter(
            Scan("reviews", "r", "review"), (FilterPredicate(expression),)
        )
        try:
            node.validate()
        except CompileError as error:
            assert message in str(error), expression
        else:
            raise AssertionError(f"validated: {expression}")
    with pytest.raises(CompileError, match="lists a value twice"):
        InList(ColumnRef("r", "reviews", "tone"), ("a", "a")).validate()

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

    # Each classification sits below the first filter testing its
    # label; a classification of joined rows sits above its join.
    # Repeated tests share a call.
    named = ModelCall(bind_prompt("category {0}", (R,)), "label", ("a", "b"))
    hidden = ModelCall(bind_prompt("hidden {0}", (R,)), "label", ("a", "b"))
    pair = ModelCall(bind_join_prompt("pair {0} {1}", (R, P)),
                     "label", ("a", "b"))
    classified = SemanticClassify(SemanticFilter(r, (first,)), named,
                                  "projected_name")
    named_column = ColumnRef("r", "reviews", "projected_name")
    tested = Filter(Filter(classified, InList(named_column, ("a",))),
                    InList(named_column, ("b",)))
    hidden_node = SemanticClassify(tested, hidden, "__label_r_3")
    chain = Filter(hidden_node,
                   InList(ColumnRef("r", "reviews", "__label_r_3"), ("a",)))
    joined = SemanticJoin(Join(chain, p), _pair_call())
    pair_node = SemanticClassify(joined, pair, "pair_label")
    plan = LogicalPlan(Project(pair_node, (
        Alias(named, "projected_name"), Alias(pair, "pair_label"),
        Alias(_filter_call("score"), "score"),
    )))
    plan.validate()
    operators = plan.operators()
    assert operators.classifies == (classified, hidden_node, pair_node)
    assert operators.labels.calls == ((named, "r"), (hidden, "r"), (pair, "r"))
    assert operators.labels.names == {
        named: "projected_name", hidden: "__label_r_3", pair: "pair_label"}
    assert operators.labels.tests == {named: [1, 2], hidden: [3]}
    assert operators.labels.projected == {
        named: "projected_name", pair: "pair_label"}
    assert operators.prompts == (
        first.prompt, named.prompt, named.prompt, hidden.prompt,
        joined.prompt, pair.prompt, _filter_call("score").prompt,
    )
    assert plan.root.explain_fields()["columns"][:2] == [
        "projected_name", "pair_label"]
    assert pair_node.explain_fields() == {
        "name": "pair_label", "probabilities": False,
        "expression": "AI.CLASSIFY('pair {0} {1}', ['a', 'b'])"}

    # the label column follows the input columns; probabilities add one
    assert classified.output_schema() == (
        R, ColumnRef("r", "reviews", "projected_name"))
    probable = SemanticClassify(r, replace(named, probabilities=True), "tone")
    assert probable.probabilities
    assert probable.output_schema()[1:] == (
        ColumnRef("r", "reviews", "tone"),
        ColumnRef("r", "reviews", "tone_probabilities"))
    assert pair_node.output_schema() == (
        R, ColumnRef("r", "reviews", "projected_name"),
        ColumnRef("r", "reviews", "__label_r_3"), P,
        ColumnRef("r", "reviews", "pair_label"))

    # a label is tested or returned only above the node computing it
    with pytest.raises(CompileError, match="nothing below computes"):
        Filter(r, InList(named_column, ("a",))).validate()
    assert chain.explain_fields() == {
        "condition": "r.__label_r_3 IN ['a']", "selectivity": None}
    with pytest.raises(CompileError, match="needs a SemanticClassify"):
        Project(classified, (Alias(hidden, "x"),)).validate()
    with pytest.raises(CompileError, match="already used"):
        SemanticClassify(classified, hidden, "projected_name").validate()
    with pytest.raises(CompileError, match="does not produce"):
        SemanticClassify(r, pair, "pair_label").validate()
    with pytest.raises(CompileError, match="needs a join of the two"):
        SemanticClassify(Join(r, p), pair, "pair_label").validate()
    with pytest.raises(CompileError, match="needs an AI.CLASSIFY"):
        SemanticClassify(r, _filter_call(), "flag").validate()
