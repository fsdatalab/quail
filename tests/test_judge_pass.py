"""CPU checks for judge-pass identities and exact prompt rendering."""

from dataclasses import replace

from quail.bench.judge_pass import (
    MODEL_NAME,
    PREDICATES,
    WORKLOAD_PLAN,
    WORKLOADS,
    _compact_label_parts,
    _corpus_identity,
    _saved_verification_sample,
    example_identity,
    judgment_identity,
    label_set_identity,
    predicate_version,
    render_filter_prompt,
    render_join_prompt,
    workload_specs,
)


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def test_stable_ids_cover_predicate_semantics_and_inputs():
    assert len(PREDICATES) == 19
    assert len({spec.key for spec in PREDICATES}) == len(PREDICATES)
    original = PREDICATES[0]
    renamed = replace(original, legacy_code="ANOTHER_F1")
    assert renamed.key == original.key
    assert predicate_version(renamed) == predicate_version(original)

    join_spec = _spec("quailb.biodex.report.experienced_reaction")
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


def test_corpus_identity_uses_source_rows_and_order():
    rows = {
        "reviews": [{"id": "r0", "body": "a"},
                    {"id": "r1", "body": "b"}],
    }
    same = {"reviews": list(rows["reviews"])}
    reversed_rows = {"reviews": list(reversed(rows["reviews"]))}
    changed = {"reviews": [{"id": "r0", "body": "a"},
                           {"id": "r1", "body": "changed"}]}
    assert _corpus_identity(rows, 0.1) == _corpus_identity(same, 0.1)
    assert (_corpus_identity(rows, 0.1)["corpus_id"]
            != _corpus_identity(reversed_rows, 0.1)["corpus_id"])
    assert (_corpus_identity(rows, 0.1)["corpus_id"]
            != _corpus_identity(changed, 0.1)["corpus_id"])


def test_filter_and_join_prompts_use_the_engine_layout():
    filter_prompt = render_filter_prompt(PREDICATES[0], "review text")
    assert filter_prompt.startswith("DOCUMENT:\nreview text")
    assert "Evaluate TRUE or FALSE" in filter_prompt
    assert filter_prompt.endswith("\nANSWER:")

    join_prompt = render_join_prompt(PREDICATES[3], "review", "aspect")
    assert join_prompt.startswith("DOCUMENT:\nreview")
    assert "(The document above is {0}.)" in join_prompt
    assert "DOCUMENT {1}:\naspect" in join_prompt
    assert join_prompt.endswith("\nANSWER:")


def test_saved_verification_sample_covers_completed_parts_after_resume(
        monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import quail.bench.judge_pass as judge_pass

    spec = _spec("quailb.imdb.review.discusses_ending")
    identity = {"label_set_id": "ls_test"}
    monkeypatch.setattr(judge_pass, "PREDICATES", (spec,))
    monkeypatch.setattr(judge_pass, "VOLUME_ROOT", tmp_path)
    parts = (tmp_path / "label_sets" / spec.workload / spec.slug
             / identity["label_set_id"] / "parts")
    parts.mkdir(parents=True)
    saved = [{
        "answer": i % 2 == 0,
        "label_source": MODEL_NAME,
        "left_id": f"rv{i}",
        "right_id": None,
    } for i in range(20)]
    pq.write_table(pa.Table.from_pylist(saved), parts / "part_000.parquet")
    corpus = {
        "reviews": [{"id": f"rv{i}", "body": f"review {i}"}
                    for i in range(20)]
    }

    sample = _saved_verification_sample(corpus, {spec.key: identity})

    assert len(sample.rows[spec.key]) == 16
    assert sample.rows[spec.key][0] == (
        render_filter_prompt(spec, "review 0"), True)


def test_compact_label_parts_keeps_every_saved_row(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = tmp_path / "parts"
    parts.mkdir()
    pq.write_table(pa.table({"id": ["a", "b"], "answer": [True, False]}),
                   parts / "part_000000_000002.parquet")
    pq.write_table(pa.table({"id": ["c"], "answer": [True]}),
                   parts / "part_000002_000003.parquet")

    path, rows = _compact_label_parts(tmp_path)
    second_path, second_rows = _compact_label_parts(tmp_path)

    assert rows == second_rows == 3
    assert path == second_path == tmp_path / "labels.parquet"
    assert pq.read_table(path).to_pylist() == [
        {"id": "a", "answer": True},
        {"id": "b", "answer": False},
        {"id": "c", "answer": True},
    ]


# ---- the parallel split

def test_every_predicate_belongs_to_exactly_one_workload():
    """The four containers between them must cover the collection: a
    predicate in no workload is silently never labelled, and one in
    two workloads is judged twice."""
    seen = [spec.key for w in WORKLOADS for spec in workload_specs(w)]
    assert sorted(seen) == sorted(spec.key for spec in PREDICATES)
    assert len(seen) == len(set(seen)) == 19


def test_the_plan_names_every_predicate_its_workload_owns():
    """What a container actually judges is the plan, not the workload
    field, so the two have to agree."""
    for workload, plan in WORKLOAD_PLAN.items():
        named = {k for _, keys in plan["filters"] for k in keys}
        for slot in ("join", "source_join"):
            if slot in plan:
                named.add(plan[slot][0])
        assert named == {spec.key for spec in workload_specs(workload)}, (
            workload)


def test_the_plan_reads_the_tables_its_predicates_declare():
    for workload, plan in WORKLOAD_PLAN.items():
        for (table, batch), keys in plan["filters"]:
            assert batch > 0
            for key in keys:
                assert _spec(key).left_table == table
        for slot in ("join", "source_join"):
            if slot not in plan:
                continue
            key, left, right = plan[slot][0], plan[slot][1], plan[slot][2]
            spec = _spec(key)
            assert spec.kind == "join"
            assert (spec.left_table, spec.right_table) == (left, right)


def test_verification_sample_can_be_scoped_to_one_workload():
    """Each container reruns only its own predicates, so the sample
    has to take a subset rather than always walking PREDICATES."""
    spec = _spec("quailb.imdb.review.discusses_ending")
    sample = _saved_verification_sample({}, {}, specs=())
    assert sample.rows == {}
    assert spec in workload_specs("imdb")
