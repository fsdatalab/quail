"""CPU checks for label loading and scoring, with no engine."""

import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from quail_b.data import GROUND_TRUTH_ROOT
from quail_b.labels import (
    GroundTruthCollection,
    PredicateLabels,
    _validate_label_set_corpora,
    load_ground_truth,
)
from quail_b.predicates import PREDICATES, predicate_payload
from quail_b.queries import QuerySpec, queries
from quail_b.scoring import (
    RunOutput,
    evaluate,
    expected_rows,
    rows_from_answers,
)
from tools.make_substrait_plans import Filter, Join, Scan, build_plan


def test_bio_4_scores_both_term_aliases_with_shared_reaction_labels():
    spec = queries()["BIO-4"]
    corpus = {
        "reports": pa.table({"id": ["r0", "r1", "r2"]}),
        "terms": pa.table({"id": ["t0", "t1", "t2", "t3"]}),
    }
    positive_filters = {
        "report_describes_serious_adverse_event": {"r0", "r1"},
        "reaction_is_neurological": {"t0", "t2"},
        "reaction_is_cardiovascular": {"t1", "t2"},
    }
    predicates = {}
    for predicate in PREDICATES:
        if predicate.workload != "biodex":
            continue
        if predicate.kind == "filter":
            answers = {
                (row_id, None): row_id in positive_filters[predicate.slug]
                for row_id in corpus[predicate.left_table]["id"].to_pylist()
            }
        else:
            answers = {
                (report, term): term != "t3" and (report != "r1" or term == "t0")
                for report in corpus["reports"]["id"].to_pylist()
                for term in corpus["terms"]["id"].to_pylist()
            }
        predicates[predicate.key] = PredicateLabels(
            predicate.key, f"ls_{predicate.slug}", predicate_payload(predicate),
            answers, {},
        )
    truth = GroundTruthCollection(
        "gt_bio4", "c_bio4", 0.1, "qwen3-32b-fp8", predicates)
    expected = expected_rows(spec, truth, corpus)
    assert _tuples(expected, ["r", "n", "c"]) == {
        ("r0", "t0", "t1"), ("r0", "t0", "t2"),
        ("r0", "t2", "t1"), ("r0", "t2", "t2"),
    }
    scored = evaluate(spec, RunOutput(None, None, expected), truth, corpus)
    assert scored["output_accuracy"]["precision"] == 1.0
    assert scored["output_accuracy"]["recall"] == 1.0
    assert scored["input_document_rows"] == 11
    assert scored["unique_input_documents"] == 7


def _spec(query_id, description, tree):
    return QuerySpec.from_plan(query_id, description, build_plan(tree))


FILTER = "Judge the review.\n\n{0}\nAnswer TRUE or FALSE."
JOIN = "Judge the pair.\n\n{0}\nAspect: {1}\nAnswer TRUE or FALSE."
SPEC = _spec(
    "TEST-1",
    "one filter then one join",
    Join(
        Filter(Scan("reviews", "r", "body"), FILTER),
        Scan("aspects", "a", "aspect"),
        ("r", "a"),
        JOIN,
    ),
)
CORPUS = {
    "reviews": [{"id": "r0", "body": "good film"},
                {"id": "r1", "body": "bad film"}],
    "aspects": [{"id": "a0", "aspect": "acting"},
                {"id": "a1", "aspect": "ending"}],
}


def _predicate(key, template, kind, left_table, right_table=None):
    columns = {"reviews": "body", "aspects": "aspect"}
    return {
        "key": key,
        "template": template,
        "kind": kind,
        "left_table": left_table,
        "left_column": columns[left_table],
        "right_table": right_table,
        "right_column": columns[right_table] if right_table else None,
    }


def _truth():
    filter_key = "test.review.filter"
    join_key = "test.review.aspect"
    return GroundTruthCollection(
        collection_id="gt_test",
        corpus_id="c_test",
        scale_factor=0.1,
        reference_model="qwen3-32b-fp8",
        predicates={
            filter_key: PredicateLabels(
                key=filter_key,
                label_set_id="ls_filter",
                predicate=_predicate(
                    filter_key, FILTER, "filter", "reviews"),
                answers={("r0", None): True, ("r1", None): False},
                source_rows={"qwen3-32b-fp8": 2},
            ),
            join_key: PredicateLabels(
                key=join_key,
                label_set_id="ls_join",
                predicate=_predicate(
                    join_key, JOIN, "join", "reviews", "aspects"),
                answers={
                    ("r0", "a0"): True,
                    ("r0", "a1"): False,
                    ("r1", "a0"): False,
                    ("r1", "a1"): True,
                },
                source_rows={"qwen3-32b-fp8": 4},
            ),
        },
    )


def _output(join_answers, rows):
    return RunOutput(
        filter_answers={"filter-1": pa.table({
            "r": ["r0", "r1"], "answer": [True, False]})},
        join_answers={"join-1": pa.table({
            "r": ["r0", "r0"], "a": ["a0", "a1"],
            "answer": [bool(answer) for answer in join_answers]})},
        rows=pa.table({"r": [row[0] for row in rows],
                       "a": [row[1] for row in rows]},
                      schema=pa.schema([("r", pa.string()),
                                        ("a", pa.string())])),
    )


def test_scores_answers_and_final_rows():
    evaluation = evaluate(SPEC, _output([0, 0], []), _truth(), CORPUS)

    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert evaluation["answer_accuracy"]["accuracy"] == 0.75
    assert [item["op"] for item in evaluation["per_predicate"]] == [
        "filter", "join"
    ]
    # r0 passes the filter and pairs with a0; r1 pairs with a1 but is
    # filtered out, so one row is expected and none was returned
    assert evaluation["output_accuracy"] == {
        "predicted_rows": 0,
        "expected_rows": 1,
        "matching_rows": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_match": False,
        "false_positive_rows": 0,
        "false_negative_rows": 1,
    }
    assert evaluation["input_document_rows"] == 4
    assert evaluation["unique_input_documents"] == 4


def test_an_answer_without_a_label_is_an_error():
    output = _output([0, 0], [])
    output.join_answers["join-1"] = pa.table({
        "r": ["r0", "r9"], "a": ["a0", "a1"], "answer": [True, False]})
    with pytest.raises(KeyError, match="no ground truth"):
        evaluate(SPEC, output, _truth(), CORPUS)


def test_rows_from_answers_joins_true_pairs_of_surviving_documents():
    output = _output([1, 1], [])
    rows = rows_from_answers(spec=SPEC, filter_answers=output.filter_answers,
                             join_answers=output.join_answers)
    # r0 passes its filter and pairs with both aspects; r1 was never asked
    assert rows.sort_by("a").to_pydict() == {"a": ["a0", "a1"], "r": ["r0", "r0"]}



def fever_truth():
    """Labels for FEV-9 over a three claim, three evidence corpus."""
    from quail_b.predicates import PREDICATES, predicate_payload
    from quail_b.prompts import F11, F13, REFUTE, SUPPORT

    # a claim's evidence_wiki_url names its page, which is an evidence
    # row's id; c1 names e0, so its label pair with e1 is off-page
    corpus = {
        "claims": pa.table({"id": ["c0", "c1", "c2"],
                            "claim": ["person one", "person two", "a place"],
                            "evidence_wiki_url": ["e0", "e0", "e2"]}),
        "evidence": pa.table({"id": ["e0", "e1", "e2"],
                              "text": ["person one", "person two", "a place"]}),
    }
    true_pairs = {
        SUPPORT: {("c0", "e0"), ("c1", "e1"), ("c2", "e0"), ("c1", "e2")},
        REFUTE: {("c1", "e0"), ("c2", "e0"), ("c0", "e2")},
    }
    predicates = {}
    for spec in PREDICATES:
        if spec.template not in (F11, F13, SUPPORT, REFUTE):
            continue
        ids = corpus[spec.left_table]["id"].to_pylist()
        answers = (
            {(row_id, None): index < 2 for index, row_id in enumerate(ids)}
            if spec.kind == "filter" else {
                (left, right): (left, right) in true_pairs[spec.template]
                for left in ids
                for right in corpus[spec.right_table]["id"].to_pylist()
            }
        )
        predicates[spec.key] = PredicateLabels(
            spec.key, f"ls_{spec.key}", predicate_payload(spec), answers, {},
        )
    truth = GroundTruthCollection("gt_fev9", "c_fev9", 0.1, None, predicates)
    return corpus, truth


def test_fev9_expected_rows_follow_the_join_chain_and_every_filter():
    corpus, truth = fever_truth()
    from quail_b.prompts import F11, F13

    spec = queries()["FEV-9"]
    assert [(filter_spec.relation, filter_spec.prompt)
            for filter_spec in spec._info.filters] == [
        ("c1", F11), ("e1", F13), ("c2", F11), ("e2", F13)]
    assert [join.relations for join in spec._info.joins] == [
        ("c1", "e1"), ("c2", "e1"), ("c2", "e2")]

    rows = expected_rows(spec, truth, corpus)

    # c2 is filtered out; e1 supports c0, refutes c1, and c1 has e1 as
    # different supporting evidence
    assert rows.to_pydict() == {
        "c1": ["c0"], "c2": ["c1"], "e1": ["e0"], "e2": ["e1"]}


def test_fev10_rows_keep_only_pairs_on_the_claims_own_page():
    corpus, truth = fever_truth()
    spec = queries()["FEV-10"]
    cross = queries()["FEV-5"]
    assert spec._info.joins[0].on == (("evidence_wiki_url", "id"),)
    assert (
        spec._info.relations == cross._info.relations
        and not cross._info.joins[0].on
    )

    # FEV-5 keeps every supported pair of surviving documents; FEV-10
    # drops (c1, e1) because c1's page is e0
    assert expected_rows(cross, truth, corpus).to_pydict() == {
        "c": ["c0", "c1"], "e": ["e0", "e1"]}
    assert expected_rows(spec, truth, corpus).to_pydict() == {
        "c": ["c0"], "e": ["e0"]}

    # an engine's answers over every pair are held to the same equality
    filter_answers = {
        "filter-1": pa.table({"c": ["c0", "c1", "c2"],
                              "answer": [True, True, False]}),
        "filter-2": pa.table({"e": ["e0", "e1", "e2"],
                              "answer": [True, True, False]}),
    }
    join_answers = {"join-1": pa.table({
        "c": ["c0", "c1", "c1"], "e": ["e0", "e1", "e0"],
        "answer": [True, True, False]})}
    assert rows_from_answers(spec, filter_answers, join_answers,
                             corpus).to_pydict() == {"c": ["c0"], "e": ["e0"]}
    assert rows_from_answers(cross, filter_answers, join_answers
                             ).to_pydict() == {"c": ["c0", "c1"],
                                               "e": ["e0", "e1"]}
    with pytest.raises(ValueError, match="needs the corpus rows"):
        rows_from_answers(spec, filter_answers, join_answers)


def test_load_benchmark_with_local_reference_labels(tmp_path):
    collection_id = "gt_test"
    label_set_id = "ls_filter"
    import quail_b as benchmark
    from quail_b.data import (
        DATA_SEED,
        PUBLISHED_CORPORA,
        SOURCE_REVISIONS,
        corpus_identity,
    )

    corpus_id = PUBLISHED_CORPORA[0.1]
    spec = benchmark.get_query("IMDB-1")
    template = spec._info.filters[0].prompt
    predicate_key = "test.review.filter"
    collection_dir = (tmp_path / GROUND_TRUTH_ROOT / "collections"
                      / collection_id)
    label_dir = (tmp_path / GROUND_TRUTH_ROOT / "label_sets" / "test"
                 / "review_filter" / label_set_id)
    collection_dir.mkdir(parents=True)
    (label_dir / "parts").mkdir(parents=True)
    collection = {
        "status": "complete",
        "collection_id": collection_id,
        "corpus_id": corpus_id,
        "scale_factor": 0.1,
        "label_sets": {predicate_key: label_set_id},
        "summary": {"model": "qwen3-32b-fp8"},
    }
    (collection_dir / "manifest.json").write_text(json.dumps(collection))
    old_dir = (tmp_path / GROUND_TRUTH_ROOT / "collections" / "gt_old")
    old_dir.mkdir(parents=True)
    (old_dir / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "collection_id": "gt_old",
        "corpus_id": corpus_id,
        "scale_factor": 0.1,
    }))
    corpus_dir = (tmp_path / GROUND_TRUTH_ROOT / "corpora" / corpus_id)
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "active_collection.json").write_text(json.dumps({
        "collection_id": collection_id,
    }))
    manifest = {
        "status": "complete",
        "rows": 2,
        "source_rows": {"qwen3-32b-fp8": 2},
        "predicate": _predicate(
            predicate_key, template, "filter", "reviews"),
    }
    (label_dir / "manifest.json").write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist([
        {"predicate_key": predicate_key, "label_set_id": label_set_id,
         "answer": True, "left_id": "r0", "right_id": None},
        {"predicate_key": predicate_key, "label_set_id": label_set_id,
         "answer": False, "left_id": "r1", "right_id": None},
    ]), label_dir / "parts" / "part_000.parquet")

    loaded = load_ground_truth(
        tmp_path, scale_factor=0.1,
        corpus_id=corpus_id)

    assert loaded.collection_id == collection_id
    assert loaded.answer(predicate_key, "r0") is True
    assert loaded.answer(predicate_key, "r1") is False
    assert loaded.key_for_template(template) == predicate_key
    assert loaded.predicates[predicate_key].table.to_pydict() == {
        "left_id": ["r0", "r1"], "right_id": [None, None],
        "answer": [True, False]}
    # only the label sets of the given templates are read
    narrowed = load_ground_truth(
        tmp_path, scale_factor=0.1, corpus_id=corpus_id, templates={"other"})
    assert not narrowed.predicates
    pq.write_table(pa.Table.from_pylist([
        {"predicate_key": predicate_key, "label_set_id": label_set_id,
         "answer": True, "left_id": "r1", "right_id": None},
    ]), label_dir / "parts" / "part_001.parquet")
    with pytest.raises(ValueError, match="duplicate ground truth"):
        load_ground_truth(tmp_path, scale_factor=0.1, corpus_id=corpus_id)
    (label_dir / "parts" / "part_001.parquet").unlink()

    reviews = pa.Table.from_pylist(CORPUS["reviews"])
    corpus = corpus_identity(
        {"reviews": reviews}, 0.1, DATA_SEED, SOURCE_REVISIONS)
    corpus["corpus_id"] = corpus_id
    (corpus_dir / "manifest.json").write_text(json.dumps(corpus))
    pq.write_table(reviews, corpus_dir / "reviews.parquet")
    calls = []

    def run_query(query, tables):
        calls.append(query.id)
        assert set(tables) == {"reviews"}
        return RunOutput(
            {"filter-1": pa.table({
                "r": ["r0", "r1"], "answer": [True, False]})},
            {}, pa.table({"r": ["r0"]}), runtime_s=2.0,
            measurements={"fresh_tokens": 100})

    output_dir = tmp_path / "run"
    record = benchmark.run(
        run_query, queries=["IMDB-1"], output_dir=output_dir, root=tmp_path,
        gpu_hourly_rate_usd=3.6, metadata={"engine": "test"})
    row = record["queries"][0]["metrics"]
    assert row["accuracy"]["answer_accuracy"]["accuracy"] == 1.0
    assert row["documents_per_second"] == 1.0
    assert row["cost_usd"] == 0.002
    assert row["fresh_tokens"] == 100
    assert row["minimum_tokens"] is None
    assert row["regret_tokens"] is None
    assert row["evaluated_document_pairs"] is None
    assert "document_pairs_per_second" not in row
    assert record["collection_id"] == collection_id
    assert pq.read_table(output_dir / "IMDB-1/rows.parquet").to_pydict() == {
        "r": ["r0"]}
    before = (output_dir / "report.md").read_text()
    (corpus_dir / "active_collection.json").write_text(
        json.dumps({"collection_id": "gt_missing"}))
    assert benchmark.report(output_dir, root=tmp_path) == output_dir / "report.md"
    assert (output_dir / "report.md").read_text() == before
    assert calls == ["IMDB-1"]
    with pytest.raises(FileExistsError):
        benchmark.run(run_query, queries=["IMDB-1"],
                      output_dir=output_dir, root=tmp_path)

    def output_only(query, tables):
        return RunOutput(None, None, pa.table({"r": ["r0"]}), runtime_s=2.0)

    untraced = benchmark.run(
        output_only, queries=["IMDB-1"], output_dir=tmp_path / "untraced",
        root=tmp_path, collection_id=collection_id)
    accuracy = untraced["queries"][0]["metrics"]["accuracy"]
    assert accuracy["answer_accuracy"] is None
    assert accuracy["output_accuracy"]["precision"] == 1.0
    assert "unavailable" in (tmp_path / "untraced/report.md").read_text()

    def wrong_id(query, tables):
        return RunOutput(None, None, pa.table({"r": ["unknown"]}), runtime_s=2.0)

    with pytest.raises(ValueError, match="unknown document ID"):
        benchmark.run(
            wrong_id, queries=["IMDB-1"], output_dir=tmp_path / "bad",
            root=tmp_path, collection_id=collection_id)

    def repeated_row(query, tables):
        return RunOutput(None, None, pa.table({"r": ["r0", "r0"]}), runtime_s=2.0)

    with pytest.raises(ValueError, match="duplicate document IDs"):
        benchmark.run(
            repeated_row, queries=["IMDB-1"], output_dir=tmp_path / "repeated",
            root=tmp_path, collection_id=collection_id)
    failed = json.loads((tmp_path / "bad/run.json").read_text())
    assert failed["queries"][0]["status"] == "scoring_failed"
    assert (tmp_path / "bad/IMDB-1/rows.parquet").exists()
    pq.write_table(pa.table({"r": ["r0"]}), tmp_path / "bad/IMDB-1/rows.parquet")
    benchmark.report(tmp_path / "bad", root=tmp_path)
    recovered = json.loads((tmp_path / "bad/run.json").read_text())
    assert recovered["status"] == "complete"
    assert "error" not in recovered["queries"][0]

    def broken(query, tables):
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        benchmark.run(
            broken, queries=["IMDB-1"], output_dir=tmp_path / "broken",
            root=tmp_path, collection_id=collection_id)
    assert json.loads((tmp_path / "broken/run.json").read_text())["status"] == "failed"

    def invalid_duration(query, tables):
        return RunOutput(None, None, pa.table({"r": ["r0"]}), runtime_s="invalid")

    with pytest.raises(ValueError, match="runtime_s"):
        benchmark.run(
            invalid_duration, queries=["IMDB-1"],
            output_dir=tmp_path / "bad-duration",
            root=tmp_path, collection_id=collection_id)
    assert (tmp_path / "bad-duration/report.md").exists()

    record["queries"][0]["definition_hash"] = "changed"
    (output_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="query definition changed"):
        benchmark.report(output_dir, root=tmp_path)


def _reused_label_layout(tmp_path):
    predicate_key = "test.review.filter"
    table_manifest = {
        "rows": 2,
        "full_hash": "same-review-table-hash",
    }
    for corpus_id in ("c_source", "c_target"):
        path = (tmp_path / GROUND_TRUTH_ROOT / "corpora" / corpus_id)
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({
            "corpus_id": corpus_id,
            "corpus_full_hash": f"hash-{corpus_id}",
            "tables": {"reviews": table_manifest},
        }))
    source_path = (tmp_path / GROUND_TRUTH_ROOT / "collections"
                   / "gt_source")
    source_path.mkdir(parents=True)
    (source_path / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "collection_id": "gt_source",
        "corpus_id": "c_source",
        "label_sets": {predicate_key: "ls_source"},
    }))
    collection = {
        "corpus_id": "c_target",
        "reused_label_sets": {
            predicate_key: {
                "source_collection_id": "gt_source",
                "source_corpus_id": "c_source",
                "required_tables": ["reviews"],
                "verified_table_manifests": {
                    "reviews": table_manifest,
                },
            },
        },
    }
    manifests = {
        predicate_key: {
            "label_set_id": "ls_source",
            "corpus_id": "c_source",
            "predicate": _predicate(
                predicate_key, FILTER, "filter", "reviews"),
        },
    }
    return tmp_path, collection, manifests


def test_reused_label_set_accepts_identical_table_manifest(tmp_path):
    root, collection, manifests = _reused_label_layout(tmp_path)

    _validate_label_set_corpora(root, collection, manifests)


def test_reused_label_set_accepts_transitive_collection_reuse(tmp_path):
    root, collection, manifests = _reused_label_layout(tmp_path)
    source_path = (tmp_path / GROUND_TRUTH_ROOT / "collections"
                   / "gt_source" / "manifest.json")
    source = json.loads(source_path.read_text())
    source["corpus_id"] = "c_intermediate"
    source_path.write_text(json.dumps(source))

    _validate_label_set_corpora(root, collection, manifests)


def test_reused_label_set_rejects_changed_table_manifest(tmp_path):
    root, collection, manifests = _reused_label_layout(tmp_path)
    target_path = (tmp_path / GROUND_TRUTH_ROOT / "corpora" / "c_target"
                   / "manifest.json")
    target = json.loads(target_path.read_text())
    target["tables"]["reviews"]["full_hash"] = "changed"
    target_path.write_text(json.dumps(target))

    try:
        _validate_label_set_corpora(root, collection, manifests)
    except ValueError as exc:
        assert "table reviews changed" in str(exc)
    else:
        raise AssertionError("changed table manifest was accepted")


def _chain_spec():
    # FEV-8's shape: filters on the claims, three joins in a chain, every
    # alias selected
    c1 = Filter(Scan("claims", "c1", "claim"), FILTER)
    first = Join(c1, Scan("evidence", "e1", "text"), ("c1", "e1"), JOIN)
    c2 = Filter(Scan("claims", "c2", "claim"), FILTER)
    second = Join(first, c2, ("e1", "c2"), JOIN)
    return _spec(
        "CHAIN",
        "three joins",
        Join(second, Scan("evidence", "e2", "text"), ("c2", "e2"), JOIN),
    )


def _random_output(spec, rng, claims=12, evidence=8):
    filters = {}
    for filter_spec in spec._info.filters:
        alias = filter_spec.relation
        filters[filter_spec.id] = pa.table({
            alias: [f"c{i}" for i in range(claims)],
            "answer": [rng.random() < 0.7 for _ in range(claims)]})
    joins = {}
    for join in spec._info.joins:
        left, right = join.relations
        sizes = {"c": claims, "e": evidence}
        pairs = [(f"{left[0]}{i}", f"{right[0]}{j}")
                 for i in range(sizes[left[0]]) for j in range(sizes[right[0]])]
        joins[join.id] = pa.table({
            left: [a for a, _ in pairs], right: [b for _, b in pairs],
            "answer": [rng.random() < 0.4 for _ in pairs]})
    return RunOutput(filters, joins, None)


def test_answers_count_and_check_rows_without_building_them():
    import random

    from quail_b.scoring import (
        implied_row_count,
        implied_rows_mask,
        scores_from_answers,
    )

    spec = _chain_spec()
    for seed in range(6):
        rng = random.Random(seed)
        output = _random_output(spec, rng)
        built = rows_from_answers(spec, output.filter_answers, output.join_answers)
        survivors, relations = scores_from_answers(spec, output, None)
        assert implied_row_count(spec, survivors, relations) == built.num_rows
        # every built row is implied; a row with one id changed is not
        columns = built.column_names
        mask = implied_rows_mask(built, survivors, relations, spec)
        assert built.num_rows == 0 or pc.all(mask).as_py()
        if built.num_rows:
            broken = built.set_column(
                columns.index("e1"), "e1",
                pa.array(["e99"] * built.num_rows, pa.string()))
            assert not pc.any(implied_rows_mask(
                broken, survivors, relations, spec)).as_py()


def _fever_chain_spec(select=None):
    # FEV-9's shape with every join written claim first, as the labels are
    c1 = Filter(Scan("claims", "c1", "claim"), FILTER)
    first = Join(c1, Scan("evidence", "e1", "text"), ("c1", "e1"), JOIN)
    c2 = Filter(Scan("claims", "c2", "claim"), FILTER)
    second = Join(first, c2, ("c2", "e1"), JOIN)
    tree = Join(second, Scan("evidence", "e2", "text"), ("c2", "e2"), JOIN)
    return QuerySpec.from_plan("CHAIN-2", "three joins", build_plan(tree, select))


def _chain_truth(rng, claims=12, evidence=8):
    """Random labels for the chain's two prompts over a small corpus."""
    corpus = {
        "claims": pa.table({"id": [f"c{i}" for i in range(claims)],
                            "claim": ["a claim"] * claims}),
        "evidence": pa.table({"id": [f"e{j}" for j in range(evidence)],
                              "text": ["a passage"] * evidence}),
    }
    columns = {"claims": "claim", "evidence": "text"}

    def predicate(key, template, kind, left, right=None):
        return {"key": key, "template": template, "kind": kind,
                "left_table": left, "left_column": columns[left],
                "right_table": right,
                "right_column": columns[right] if right else None}

    filter_key, join_key = "test.claim.filter", "test.claim.evidence"
    truth = GroundTruthCollection("gt_chain", "c_chain", 0.1, "qwen3-32b-fp8", {
        filter_key: PredicateLabels(
            filter_key, "ls_filter",
            predicate(filter_key, FILTER, "filter", "claims"),
            {(f"c{i}", None): rng.random() < 0.7 for i in range(claims)}, {}),
        join_key: PredicateLabels(
            join_key, "ls_join",
            predicate(join_key, JOIN, "join", "claims", "evidence"),
            {(f"c{i}", f"e{j}"): rng.random() < 0.4
             for i in range(claims) for j in range(evidence)}, {}),
    })
    return corpus, truth


def _tuples(table, selected):
    table = table.select(selected)
    return set(zip(*(table.column(name).to_pylist() for name in selected)))


def test_row_counts_agree_with_built_rows():
    import random

    from quail_b.scoring import row_counts

    # counted in DuckDB against the rows built in Arrow: with every alias
    # selected (a traced run scores from its answers) and with a
    # projection (which scores the saved rows)
    for select in (None, ["c1", "e2"]):
        spec = _fever_chain_spec(select)
        selected = [name.split(".")[0] for name in spec._info.select]
        for seed in range(4):
            rng = random.Random(seed)
            corpus, truth = _chain_truth(rng)
            output = _random_output(spec, rng)
            expected = _tuples(expected_rows(spec, truth, corpus), selected)
            built = rows_from_answers(
                spec, output.filter_answers, output.join_answers)
            predicted = _tuples(built, selected)
            counts = (len(predicted), len(expected), len(predicted & expected))
            assert counts[1] > 0
            output.rows = built.select(selected)
            assert row_counts(spec, output, truth, corpus) == counts
            # an untraced run is scored from its rows alone
            untraced = RunOutput(None, None, built.select(selected))
            assert row_counts(spec, untraced, truth, corpus) == counts
    # a traced run with every alias selected needs no rows at all
    spec = _fever_chain_spec()
    rng = random.Random(7)
    corpus, truth = _chain_truth(rng)
    output = _random_output(spec, rng)
    assert output.rows is None
    assert row_counts(spec, output, truth, corpus)[2] >= 0


def test_traced_rows_must_agree_with_the_answers():
    import random

    from quail_b.run import _validate_output

    spec = _chain_spec()
    output = _random_output(spec, random.Random(1))
    rows = rows_from_answers(spec, output.filter_answers, output.join_answers)
    assert rows.num_rows > 1
    tables = {"claims": pa.table({"id": [f"c{i}" for i in range(12)]}),
              "evidence": pa.table({"id": [f"e{i}" for i in range(8)]})}
    output.rows = rows
    output.runtime_s = 1.0
    _validate_output(spec, output, tables)
    output.rows = rows.slice(1)
    with pytest.raises(ValueError, match="answers imply"):
        _validate_output(spec, output, tables)
    swapped = rows.set_column(
        rows.column_names.index("e2"), "e2",
        pa.array(["e0"] + rows.column("e2").to_pylist()[1:], pa.string()))
    if swapped.to_pylist() != rows.to_pylist():
        output.rows = swapped
        with pytest.raises(ValueError, match="not implied|duplicate"):
            _validate_output(spec, output, tables)


def test_row_sample_is_taken_chunk_by_chunk():
    from quail_b.run import _sample_rows

    # 24 chunks of uneven size; the sample must be the evenly spaced
    # rows of the whole table, never a take over a concatenated column
    bounds = [0, *range(7, 230, 10), 233]
    chunks = [pa.record_batch({"r": [f"r{i}" for i in range(lo, hi)]})
              for lo, hi in zip(bounds, bounds[1:])]
    table = pa.Table.from_batches(chunks)
    assert len(chunks) == 24 and table.num_rows == 233
    sample = _sample_rows(table, 50)
    step = table.num_rows // 50
    assert sample.column("r").to_pylist() == [
        f"r{i}" for i in range(0, step * 50, step)]
    assert _sample_rows(table, 1000) is table
