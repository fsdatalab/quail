"""CPU checks for judge-pass identities and exact prompt rendering."""

from dataclasses import replace

from quail.bench.judge_pass import (
    JUDGE_SPEC,
    MODEL_NAME,
    PREDICATES,
    _compact_label_parts,
    _corpus_identity,
    _label_dir_by_id,
    _part_bounds,
    _parts_stats,
    _saved_verification_sample,
    example_identity,
    judgment_identity,
    label_set_identity,
    predicate_version,
    render_filter_prompt,
    render_join_prompt,
)


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def test_stable_ids_cover_predicate_semantics_and_inputs():
    assert len(PREDICATES) == 23
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
    assert "(The document above is DOCUMENT {0}.)" in join_prompt
    assert "DOCUMENT {1}:\naspect" in join_prompt
    assert join_prompt.endswith("\nANSWER:")


def test_saved_verification_sample_covers_completed_parts_after_resume(
        monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import quail.bench.judge_pass as judge_pass

    spec = _spec("quailb.imdb.review.discusses_ending")
    identity = {"label_set_id": "ls_test"}
    monkeypatch.setattr(judge_pass, "VOLUME_ROOT", tmp_path)
    parts = (tmp_path / "label_sets" / spec.workload / spec.slug
             / identity["label_set_id"] / "parts")
    parts.mkdir(parents=True)
    saved = [{
        "answer": i % 2 == 0,
        "label_source": MODEL_NAME,
        "left_id": f"rv{i}",
        "right_id": None,
    } for i in range(85)]
    # only the first of two parts is on disk, the resume case
    pq.write_table(pa.Table.from_pylist(saved),
                   parts / "part_000000_000085.parquet")
    corpus = {
        "reviews": [{"id": f"rv{i}", "body": f"review {i}"}
                    for i in range(170)]
    }

    sample = _saved_verification_sample(
        corpus, {spec.key: identity}, specs=(spec,))

    assert len(sample.rows[spec.key]) == 16
    assert sample.rows[spec.key][0] == (
        render_filter_prompt(spec, "review 0"), True)


def test_compact_label_parts_keeps_every_saved_row(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    pq.write_table(pa.table({"id": ["a", "b"], "answer": [True, False]}),
                   parts_dir / "part_000000_000002.parquet")
    pq.write_table(pa.table({"id": ["c"], "answer": [True]}),
                   parts_dir / "part_000002_000003.parquet")
    # a leftover from an earlier generation of part boundaries, which
    # the caller does not name and compaction must therefore ignore
    pq.write_table(pa.table({"id": ["a", "b", "c"],
                             "answer": [True, False, True]}),
                   parts_dir / "part_000000_000003.parquet")
    parts = [parts_dir / "part_000000_000002.parquet",
             parts_dir / "part_000002_000003.parquet"]

    path, rows = _compact_label_parts(tmp_path, parts)
    second_path, second_rows = _compact_label_parts(tmp_path, parts)

    assert rows == second_rows == 3
    assert path == second_path == tmp_path / "labels.parquet"
    assert pq.read_table(path).to_pylist() == [
        {"id": "a", "answer": True},
        {"id": "b", "answer": False},
        {"id": "c", "answer": True},
    ]


def test_judge_spec_holds_no_scheduler_capacity_knobs():
    """These change throughput and memory, never the token a greedy
    one-token decode picks, so they must not move label_set_id."""
    for field in ("max_num_batched_tokens", "max_num_seqs",
                  "gpu_memory_utilization"):
        assert field not in JUDGE_SPEC


def test_part_bounds_match_the_writers():
    reviews = [{"id": f"rv{i}"} for i in range(5000)]
    reports = [{"id": f"rp{i}"} for i in range(200)]
    terms = [{"id": f"tm{i}"} for i in range(614)]
    citations = [{"id": f"lp{i}"} for i in range(200)]
    rows = {"reviews": reviews, "reports": reports, "terms": terms,
            "citations": citations}

    # three imdb filters share reviews.body, so 256 // 3 rows per part
    bounds = _part_bounds(_spec("quailb.imdb.review.discusses_ending"), rows)
    assert bounds[0] == (0, 85)
    assert bounds[-1] == (4930, 5000)
    assert sum(end - start for start, end in bounds) == 5000

    # a join over 614 right rows cannot fit even one left row in 256
    # prompts, so it falls back to one report per part
    bounds = _part_bounds(
        _spec("quailb.biodex.report.experienced_reaction"), rows)
    assert bounds[0] == (0, 1)
    assert len(bounds) == 200

    # the LePaRD source join has its own fixed anchor batch
    bounds = _part_bounds(_spec("quailb.lepard.excerpt.cites_passage"), rows)
    assert bounds == [(0, 50), (50, 100), (100, 150), (150, 200)]


def test_parts_stats_ignores_files_it_was_not_given(tmp_path):
    """A label directory can hold more than one generation of part
    files; globbing it counts the same answers twice."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    current = parts_dir / "part_000000_000002.parquet"
    pq.write_table(pa.table({"answer": [True, False],
                            "label_source": [MODEL_NAME] * 2}), current)
    pq.write_table(pa.table({"answer": [True, False],
                            "label_source": [MODEL_NAME] * 2}),
                   parts_dir / "part_000000_000001.parquet")

    stats = _parts_stats([current])
    assert stats == {"rows": 2, "true_rows": 1, "false_rows": 1,
                     "source_rows": {MODEL_NAME: 2}}
