"""CPU checks for predicate identities and exact prompt rendering."""

from dataclasses import replace

import pytest

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
    assert len(PREDICATES) == 35
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
    from quail_b.rendering import ANSWER_CUE, SHARED_PRE

    filter_prompt = render_filter_prompt(PREDICATES[0], "review text")
    assert filter_prompt.startswith(SHARED_PRE + "review text")
    assert "Evaluate TRUE or FALSE" in filter_prompt
    assert filter_prompt.endswith(ANSWER_CUE)

    join_prompt = render_join_prompt(PREDICATES[3], "review", "aspect")
    assert join_prompt.startswith(SHARED_PRE + "review")
    assert "(The document above is DOCUMENT {0}.)" in join_prompt
    assert "DOCUMENT {1}:\naspect" in join_prompt
    assert join_prompt.endswith(ANSWER_CUE)


def test_biodex_replacement_matches_saved_reference_identity():
    spec = _spec("quailb.biodex.report.describes_serious_adverse_event")
    identity = label_set_identity(
        spec, "c_d7a294f1a0d83293b31ed8519df4262e",
        "d7a294f1a0d83293b31ed8519df4262e5cf3f347a535cdb9abeb109f20ae75c8",
    )
    assert identity["label_set_id"] == "ls_5627a6d5416349ee3e418c41594bc3a3"


def test_task_instruction_changes_label_identity(monkeypatch):
    from quail_b import rendering

    spec = PREDICATES[0]
    current = label_set_identity(spec, "c_one", "1" * 64)
    monkeypatch.setattr(
        rendering, "TASK_INSTRUCTION",
        "You are performing a data processing task. "
        "Evaluate TRUE or FALSE for the following question: ",
    )
    previous = label_set_identity(spec, "c_one", "1" * 64)
    assert current["label_set_id"] != previous["label_set_id"]


def test_answer_cue_changes_label_identity(monkeypatch):
    from quail_b import rendering

    spec = PREDICATES[0]
    current = label_set_identity(spec, "c_one", "1" * 64)
    monkeypatch.setattr(rendering, "ANSWER_CUE", "\nANSWER: modified")
    previous = label_set_identity(spec, "c_one", "1" * 64)
    assert current["label_set_id"] != previous["label_set_id"]


def test_cuad_predicates_are_labeled_by_the_annotation():
    from quail_b.predicates import annotation_answer, label_sources

    spec = _spec("quailb.cuad.contract.non_compete")
    (source,) = label_sources(spec)
    assert source["id"].startswith("s_")
    assert source["spec"]["dataset"] == "zenodo/CUAD_v1"
    assert annotation_answer(spec, {"clauses": ["Exclusivity", "Non-Compete"]})
    assert not annotation_answer(spec, {"clauses": ["Exclusivity"]})
    with pytest.raises(ValueError, match="not a filter the annotation"):
        annotation_answer(PREDICATES[0], {"clauses": []})
    # the category names the label, not the prompt: the version hash
    # depends on the words the model reads and nothing else
    renamed = replace(spec, source_category="Other")
    assert predicate_version(renamed) == predicate_version(spec)
    prompt = render_filter_prompt(spec, "<pages>")
    assert prompt.startswith("DOCUMENT:\n<pages>\n\nEvaluate TRUE or FALSE for "
                             "the following question: Judge strictly from "
                             "the contract above")


def test_raw_join_restores_the_published_predicate_hash():
    from quail_b.predicates import predicate_version

    spec = _spec("quailb.imdb.review.discusses_aspect")
    assert predicate_version(spec)[0] == "pv_7fd88f0450b6e15acbae8810ef0405ae"


def test_financebench_join_is_labeled_by_the_evidence_pages():
    from quail_b.predicates import (
        annotation_pair_answer,
        annotation_sourced,
        label_sources,
    )

    spec = _spec("quailb.financebench.page.answers_question")
    assert annotation_sourced(spec)
    assert not annotation_sourced(
        _spec("quailb.financebench.question.needs_calculation"))
    (source,) = label_sources(spec)
    assert source["spec"]["dataset"] == "github/patronus-ai/financebench"
    question = {"id": "fq0", "filing": "fl0", "evidence_pages": [60, 61]}
    assert annotation_pair_answer(
        spec, question, {"id": "fl0p60", "filing": "fl0", "page_number": 60})
    assert not annotation_pair_answer(
        spec, question, {"id": "fl0p59", "filing": "fl0", "page_number": 59})
    assert not annotation_pair_answer(
        spec, question, {"id": "fl1p60", "filing": "fl1", "page_number": 60})
    with pytest.raises(ValueError, match="not a join the annotation"):
        annotation_pair_answer(_spec("quailb.imdb.review.discusses_aspect"),
                               {}, {})


def test_officeqa_join_is_labeled_by_the_source_page():
    from quail_b.predicates import (
        annotation_pair_answer,
        annotation_sourced,
        label_sources,
    )

    spec = _spec("quailb.officeqa.page.answers_question")
    assert annotation_sourced(spec)
    assert not annotation_sourced(
        _spec("quailb.officeqa.question.combines_figures"))
    (source,) = label_sources(spec)
    assert source["spec"]["dataset"] == "databricks/officeqa-pro-v2"
    question = {"id": "tq0", "statement": "ts0", "evidence_pages": [12]}
    assert annotation_pair_answer(
        spec, question, {"id": "ts0p12", "statement": "ts0", "page_number": 12})
    assert not annotation_pair_answer(
        spec, question, {"id": "ts0p13", "statement": "ts0", "page_number": 13})
    assert not annotation_pair_answer(
        spec, question, {"id": "ts1p12", "statement": "ts1", "page_number": 12})
