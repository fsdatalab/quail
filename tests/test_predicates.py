"""CPU checks for predicate identities and exact prompt rendering."""

from dataclasses import replace

from quail_b.predicates import (
    PREDICATES,
    example_identity,
    judgment_identity,
    label_set_identity,
    predicate_payload,
    predicate_version,
    render_filter_prompt,
    render_join_prompt,
)


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def test_stable_ids_cover_predicate_semantics_and_inputs():
    assert len(PREDICATES) == 21
    assert len({spec.key for spec in PREDICATES}) == len(PREDICATES)
    original = PREDICATES[0]

    join_spec = _spec("quailb.biodex.report.experienced_reaction")
    assert (predicate_payload(join_spec)["render"]
            == "join_arg0_anchor_then_arg1_v1")
    changed_prompt = replace(join_spec, template=join_spec.template + "\n")
    changed_roles = replace(join_spec, left_role="medical_report")
    assert predicate_version(join_spec) != predicate_version(changed_prompt)
    assert predicate_version(join_spec) != predicate_version(changed_roles)

    left = {"role": "report", "table": "reports", "row_id": "rp0"}
    right = {"role": "reaction", "table": "terms", "row_id": "tm0"}
    example, full = example_identity("c_test", [left, right])
    reversed_example, _ = example_identity("c_test", [right, left])
    assert example != reversed_example
    assert (judgment_identity("ls_one", full)
            != judgment_identity("ls_two", full))

    first = label_set_identity(original, "c_one", "1" * 64)
    second = label_set_identity(original, "c_two", "2" * 64)
    assert first["label_set_id"] != second["label_set_id"]


def test_filter_and_join_prompts_use_the_engine_layout():
    filter_prompt = render_filter_prompt(PREDICATES[0], "review text")
    assert filter_prompt.startswith("DOCUMENT:\nreview text")
    assert "Evaluate TRUE or FALSE" in filter_prompt
    assert filter_prompt.endswith("\nANSWER:")

    join_prompt = render_join_prompt(PREDICATES[3], "review", "aspect")
    assert join_prompt.startswith("DOCUMENT:\nreview")
    assert "(The document above is DOCUMENT {0}.)" in join_prompt
    assert "DOCUMENT {1}:\naspect" in join_prompt
    assert join_prompt.endswith("\nANSWER:")


def test_biodex_replacement_matches_saved_reference_identity():
    spec = _spec("quailb.biodex.report.describes_serious_adverse_event")
    identity = label_set_identity(
        spec, "c_d7a294f1a0d83293b31ed8519df4262e",
        "d7a294f1a0d83293b31ed8519df4262e5cf3f347a535cdb9abeb109f20ae75c8",
    )
    assert identity["label_set_id"] == "ls_5627a6d5416349ee3e418c41594bc3a3"
