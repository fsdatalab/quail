"""CPU checks for the QUAIL-B labeling pass."""

from pathlib import Path

import pytest

import quail.bench.labeling as labeling
from quail.bench.labeling import (
    _check_reused_label_set,
    _compact_label_parts,
    _corpus_identity,
    _saved_verification_sample,
)
from quail_b.data import DATA_SEED, SOURCE_REVISIONS, corpus_identity
from quail_b.predicates import MODEL_NAME, PREDICATES


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def test_corpus_identity_and_label_reuse(monkeypatch, tmp_path):
    rows = {"reviews": [{"id": "r0", "body": "text"}]}
    assert (corpus_identity(rows, 0.1, DATA_SEED, SOURCE_REVISIONS)
            == _corpus_identity(rows, 0.1))

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

    with monkeypatch.context() as patch:
        import json

        spec = _spec("quailb.imdb.review.mentions_positive_aspect")
        label_set_id = "ls_old"
        table_manifest = {"rows": 2, "ordered_rows_full_hash": "abc"}
        corpus = {
            "corpus_id": "c_original",
            "corpus_full_hash": "full-original",
            "tables": {"reviews": table_manifest},
        }
        corpus_dir = tmp_path / "corpora" / corpus["corpus_id"]
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "manifest.json").write_text(json.dumps(corpus))
        label_dir = (
            tmp_path / "label_sets" / spec.workload / spec.slug / label_set_id
        )
        label_dir.mkdir(parents=True)
        (label_dir / "manifest.json").write_text(json.dumps({
            "status": "complete",
            "label_set_id": label_set_id,
            "corpus_id": corpus["corpus_id"],
            "corpus_full_hash": corpus["corpus_full_hash"],
        }))
        source_collection = {
            "collection_id": "gt_middle",
            "corpus_id": "c_middle",
            "label_sets": {spec.key: label_set_id},
        }
        target = {"tables": {"reviews": dict(table_manifest)}}
        patch.setattr(labeling, "ROOT", tmp_path)

        reused = _check_reused_label_set(
            spec, label_set_id, source_collection, target)

        assert reused["source_collection_id"] == "gt_middle"
        assert reused["source_corpus_id"] == "c_original"

        target["tables"]["reviews"]["ordered_rows_full_hash"] = "changed"
        with pytest.raises(ValueError, match="table reviews changed"):
            _check_reused_label_set(spec, label_set_id, source_collection, target)


def test_saved_label_parts_and_resume(monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        import pyarrow as pa
        import pyarrow.parquet as pq

        spec = _spec("quailb.imdb.review.discusses_ending")
        identity = {"label_set_id": "ls_test"}
        patch.setattr(labeling, "ROOT", tmp_path)
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
            {"id": "rv0", "body": "review 0"}, None, True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    pq.write_table(pa.table({"id": ["a", "b"], "answer": [True, False]}),
                   parts_dir / "part_000000_000002.parquet")
    pq.write_table(pa.table({"id": ["c"], "answer": [True]}),
                   parts_dir / "part_000002_000003.parquet")
    # a leftover generation of boundaries, which the caller does not
    # name and compaction must therefore ignore
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


class _FakeResult:
    def __init__(self, answer_tables):
        self.answer_tables = answer_tables


class _FakeQuery:
    """Stands in for a bound builder; answers TRUE when the text has 'yes'."""

    def __init__(self, session, name):
        self.session = session
        self.names = [name]

    def alias(self, _alias):
        return self

    def ai_filter(self, _prompt, **_kwargs):
        return self

    def ai_join(self, other, _prompt, **kwargs):
        self.session.join_anchors.append(kwargs.get("anchor"))
        self.names.append(other.names[0])
        return self

    def select(self, *_columns):
        return self

    def run(self):
        import pyarrow as pa

        tables = [self.session.tables[name] for name in self.names]
        if len(tables) == 1:
            text = tables[0].column(1).to_pylist()
            table = pa.table({"l": list(range(len(text))),
                              "answer": ["yes" in value for value in text]})
            return _FakeResult({"filters": {("l", 0): table}, "joins": {}})
        left, right = (table.column(1).to_pylist() for table in tables)
        rows = [(i, j, "yes" in a and "yes" in b)
                for i, a in enumerate(left) for j, b in enumerate(right)]
        table = pa.table({"l": [r[0] for r in rows], "r": [r[1] for r in rows],
                          "answer": [r[2] for r in rows]})
        return _FakeResult({"filters": {}, "joins": {0: table}})


class _FakeSession:
    def __init__(self, _config):
        self.tables = {}
        self.join_anchors = []

    def register(self, name, provider):
        self.tables[name] = provider.table

    def docs(self, name):
        return _FakeQuery(self, name)

    def close(self):
        pass


def test_judge_answers_and_source_labels(monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        import quail

        patch.setattr(quail, "Session", _FakeSession)
        judge = labeling.QuailJudge()
        spec = _spec("quailb.imdb.review.discusses_aspect")
        reviews = [{"id": "rv0", "body": "no"}, {"id": "rv1", "body": "yes"}]
        aspects = [{"id": "as0", "aspect": "yes the plot"},
                   {"id": "as1", "aspect": "the acting"}]

        assert judge.filter(_spec("quailb.imdb.review.discusses_ending"),
                            reviews) == [False, True]
        assert judge.join(spec, reviews, aspects) == {
            (0, 0): False, (0, 1): False, (1, 0): True, (1, 1): False}
        assert judge.queries == 2
        assert judge.rows_answered == 6
        assert judge.session.join_anchors == ["l"]

    with monkeypatch.context() as patch:
        import pyarrow.parquet as pq

        import quail

        patch.setattr(quail, "Session", _FakeSession)
        patch.setattr(labeling, "ROOT", tmp_path)
        judge = labeling.QuailJudge()
        spec = _spec("quailb.fever.passage.supports_claim")
        claims = [{"id": "cl0", "claim": "yes one", "label": "REFUTES",
                   "evidence_wiki_url": "page0"},
                  {"id": "cl1", "claim": "two", "label": "SUPPORTS",
                   "evidence_wiki_url": "page1"}]
        evidence = [{"id": "page0", "text": "yes"}, {"id": "page1", "text": "no"}]
        identity = labeling.label_set_identity(spec, "c_test", "0" * 64)

        labeling._write_qwen_join_parts(
            judge, labeling.VerificationSample(), spec, claims, evidence,
            identity, "c_test", source_label=labeling._fever_source_label)

        parts = labeling._expected_parts(
            spec, identity, {"claims": claims, "evidence": evidence})
        assert len(parts) == 1 and parts[0].exists()
        rows = {(row["left_id"], row["right_id"]): (row["answer"],
                                                     row["label_source"])
                for row in pq.read_table(parts[0]).to_pylist()}
        assert rows == {
            ("cl0", "page0"): (False, "fever_annotation"),   # source wins
            ("cl0", "page1"): (False, MODEL_NAME),
            ("cl1", "page0"): (False, MODEL_NAME),
            ("cl1", "page1"): (True, "fever_annotation"),
        }


def _write_corpus(root, sf, rows):
    """Write one corpus under root the way _materialize_corpus does."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = labeling._corpus_identity(rows, sf)
    target = root / "corpora" / identity["corpus_id"]
    target.mkdir(parents=True, exist_ok=True)
    for table, table_rows in rows.items():
        pq.write_table(pa.Table.from_pylist(table_rows),
                       target / f"{table}.parquet")
    manifest = {**identity, "columns": labeling.CORPUS_COLUMNS}
    (target / "manifest.json").write_text(json.dumps(manifest))
    return target, manifest, rows


def _corpus_rows(reviews, reports, terms, claims, evidence, contexts,
                 passages, contracts=(), filings=()):
    """Corpus rows.

    `contracts` lists each contract's clause categories. `filings` lists
    (question, evidence pages) per FinanceBench question; question i asks
    about filing f{i}, which has two pages.
    """
    return {
        "filing_questions": [
            {"id": f"fq{i}", "financebench_id": f"financebench_id_{i:05d}",
             "filing": f"f{i}", "doc_name": f"FILING{i}", "company": "Co",
             "question_type": "metrics-generated", "question": question,
             "answer": "42", "evidence_pages": list(pages)}
            for i, (question, pages) in enumerate(filings)],
        "filing_pages": [
            {"id": f"f{i}p{page}", "filing": f"f{i}", "doc_name": f"FILING{i}",
             "page_number": page, "page_count": 2, "pdf_sha256": "0" * 64,
             "document": f"files/f{i}.pdf#page={page}"}
            for i in range(len(filings)) for page in (1, 2)],
        "contracts": [
            {"id": f"ct{i}", "title": f"contract {i}", "page_count": 2,
             "pdf_sha256": "0" * 64, "document": f"files/ct{i}.pdf",
             "clauses": list(clauses)}
            for i, clauses in enumerate(contracts)],
        "contract_pages": [
            {"id": f"ct{i}p1", "contract_id": f"ct{i}", "page_number": 1,
             "pdf_sha256": "0" * 64, "document": f"files/ct{i}.pdf#page=1",
             "clauses": list(clauses)}
            for i, clauses in enumerate(contracts)],
        "reviews": [{"id": f"rv{i}", "body": body}
                    for i, body in enumerate(reviews)],
        "aspects": [{"id": "as0", "aspect": "the plot"}],
        "reports": [{"id": f"rp{i}", "report": text, "reactions": ["x"]}
                    for i, text in enumerate(reports)],
        "terms": [{"id": f"tm{i}", "term": term}
                  for i, term in enumerate(terms)],
        "claims": [{"id": f"cl{i}", "claim": claim, "label": label,
                    "evidence_wiki_url": page}
                   for i, (claim, label, page) in enumerate(claims)],
        "evidence": [{"id": page, "text": text}
                     for page, text in evidence],
        "citation_contexts": [
            {"id": f"lc{i}", "destination_context": text,
             "cited_passage_ids": cited}
            for i, (text, cited) in enumerate(contexts)],
        "citation_passages": [
            {"id": f"lp{i}", "passage_text": text, "passage_ids": ids}
            for i, (text, ids) in enumerate(passages)],
        "agent_traces": [{"id": "at0000-t005", "trace": "yes trace",
                          "trajectory_id": "at0000", "turn_index": 5,
                          "token_count": 2}],
    }


def test_derive_collection_copies_labels_by_content(monkeypatch, tmp_path):
    import json

    import pyarrow.parquet as pq

    import quail

    monkeypatch.setattr(quail, "Session", _FakeSession)
    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    specs = (
        _spec("quailb.imdb.review.mentions_positive_aspect"),
        _spec("quailb.biodex.report.experienced_reaction"),
        _spec("quailb.fever.passage.supports_claim"),
        _spec("quailb.lepard.excerpt.cites_passage"),
        _spec("quailb.cuad.contract.non_compete"),
        _spec("quailb.financebench.page.answers_question"),
    )
    monkeypatch.setattr(labeling, "PREDICATES", specs)

    # the large corpus: two of everything, the LePaRD context cites p2
    source_rows = _corpus_rows(
        reviews=["yes good", "bad", "yes again"],
        reports=["yes report", "other report"],
        terms=["yes fever", "cough"],
        claims=[("yes claim", "REFUTES", "page0"),
                ("second claim", "SUPPORTS", "page1")],
        evidence=[("page0", "yes page"), ("page1", "no")],
        contexts=[("context one", ["p1", "p2"]), ("context two", ["p3"])],
        passages=[("passage a", ["p2"]), ("passage b", ["p1"])],
        contracts=[("Exclusivity",), ("Non-Compete", "Exclusivity")],
        filings=[("ratio?", [2]), ("revenue?", [1])])
    _, source_corpus, _ = _write_corpus(tmp_path, 1.0, source_rows)
    source_id = source_corpus["corpus_id"]
    identities = {
        spec.key: labeling.label_set_identity(
            spec, source_id, source_corpus["corpus_full_hash"])
        for spec in specs}
    judge = labeling.QuailJudge()
    sample = labeling.VerificationSample()
    labeling._write_filter_parts(
        judge, sample, source_rows["reviews"], [specs[0]], identities,
        source_id, 4096 // 3)
    labeling._write_qwen_join_parts(
        judge, sample, specs[1], source_rows["reports"],
        source_rows["terms"], identities[specs[1].key], source_id)
    labeling._write_qwen_join_parts(
        judge, sample, specs[2], source_rows["claims"],
        source_rows["evidence"], identities[specs[2].key], source_id,
        source_label=labeling._fever_source_label)
    labeling._write_lepard_source(
        specs[3], source_rows["citation_contexts"],
        source_rows["citation_passages"], identities[specs[3].key],
        source_id)
    labeling._write_filter_parts(
        labeling.AnnotationLabeler(), None, source_rows["contracts"],
        [specs[4]], identities, source_id, labeling.filter_batch_rows(specs[4]))
    labeling._write_annotation_join(
        specs[5], source_rows["filing_questions"],
        source_rows["filing_pages"], identities[specs[5].key], source_id)
    for spec in specs:
        labeling._complete_manifest(spec, identities[spec.key], source_rows)
    collection_dir = tmp_path / "collections" / "gt_source"
    collection_dir.mkdir(parents=True)
    (collection_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "collection_id": "gt_source",
        "scale_factor": 1.0, "corpus_id": source_id,
        "label_sets": {key: identity["label_set_id"]
                       for key, identity in identities.items()}}))

    # the small corpus: a prefix of the documents, terms renumbered,
    # the context no longer cites p2 because that pair was not sampled,
    # and the one sampled contract is the source's second, renumbered ct0
    target_rows = _corpus_rows(
        reviews=["yes good", "bad"],
        reports=["yes report"],
        terms=["cough", "yes fever"],
        claims=[("yes claim", "REFUTES", "page0")],
        evidence=[("page0", "yes page")],
        contexts=[("context one", ["p1"])],
        passages=[("passage a", ["p2"]), ("passage b", ["p1"])],
        contracts=[("Non-Compete", "Exclusivity")],
        filings=[("revenue?", [1])])
    monkeypatch.setattr(
        labeling, "_materialize_corpus",
        lambda sf: _write_corpus(tmp_path, sf, target_rows))

    summary = labeling.derive_collection(0.1, "gt_source")

    assert summary["cell"] == "quailb_ground_truth_collection_derived"
    assert summary["source_collection_id"] == "gt_source"
    assert summary["total_labels"] == 2 + 2 + 1 + 2 + 1 + 2
    assert summary["qwen_judgments"] == 2 + 2
    sets = summary["label_sets"]
    assert sets[specs[1].key]["copied_from"] == identities[
        specs[1].key]["label_set_id"]
    assert "recomputed_from" in sets[specs[3].key]
    assert "recomputed_from" in sets[specs[4].key]
    assert "recomputed_from" in sets[specs[5].key]

    def answers(spec):
        label_dir = labeling._label_dir_by_id(
            spec, sets[spec.key]["label_set_id"])
        table = pq.read_table(label_dir / "labels.parquet")
        return {(row["left_id"], row["right_id"]):
                (row["answer"], row["label_source"])
                for row in table.to_pylist()}

    assert answers(specs[0]) == {("rv0", None): (True, MODEL_NAME),
                                 ("rv1", None): (False, MODEL_NAME)}
    # tm1 is "yes fever" here but was tm0 in the source
    assert answers(specs[1]) == {("rp0", "tm0"): (False, MODEL_NAME),
                                 ("rp0", "tm1"): (True, MODEL_NAME)}
    assert answers(specs[2]) == {
        ("cl0", "page0"): (False, "fever_annotation")}
    assert answers(specs[3]) == {
        ("lc0", "lp0"): (False, "lepard_citation_edge"),
        ("lc0", "lp1"): (True, "lepard_citation_edge")}
    assert answers(specs[4]) == {("ct0", None): (True, "cuad_annotation")}
    assert answers(specs[5]) == {
        ("fq0", "f0p1"): (True, "financebench_evidence"),
        ("fq0", "f0p2"): (False, "financebench_evidence")}
    active = json.loads((tmp_path / "corpora"
                         / summary["corpus_id"]
                         / f"active_collection.{labeling.PROMPT_FORMAT}.json"
                         ).read_text())
    assert active["collection_id"] == summary["collection_id"]


def test_cuad_workload_is_labeled_from_the_annotation_without_a_gpu(
        monkeypatch, tmp_path):
    import pyarrow.parquet as pq

    from quail_b.predicates import workload_specs

    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    assert labeling.annotation_workloads() == ("cuad",)
    rows = _corpus_rows(
        reviews=["a"], reports=["r"], terms=["t"],
        claims=[("c", "SUPPORTS", "page0")], evidence=[("page0", "e")],
        contexts=[("x", ["p1"])], passages=[("y", ["p1"])],
        contracts=[("Exclusivity",), ("Non-Compete", "Exclusivity"), ()],
        filings=[("ratio?", [1])])
    _, corpus, _ = _write_corpus(tmp_path, 0.1, rows)

    partial = labeling.label_annotated_workload(corpus["corpus_id"], "cuad")

    specs = workload_specs("cuad")
    assert set(partial["manifests"]) == {spec.key for spec in specs}
    assert partial["model_wall_s"] == 0.0
    assert partial["deterministic_rerun"]["compared"] == 0
    non_compete = partial["manifests"]["quailb.cuad.contract.non_compete"]
    assert non_compete["status"] == "complete"
    assert (non_compete["rows"], non_compete["true_rows"]) == (3, 1)
    assert non_compete["source_rows"] == {"cuad_annotation": 3}
    labels = pq.read_table(non_compete["compact_path"]).to_pylist()
    assert [(row["left_id"], row["answer"]) for row in labels] == [
        ("ct0", False), ("ct1", True), ("ct2", False)]
    caps = partial["manifests"]["quailb.cuad.page.caps_liability"]
    assert (caps["rows"], caps["true_rows"]) == (3, 0)
    with pytest.raises(ValueError, match="labeled from its annotation"):
        labeling.judge_workload(corpus["corpus_id"], "cuad")
    with pytest.raises(ValueError, match="needs a model judge"):
        labeling.label_annotated_workload(corpus["corpus_id"], "imdb")


def test_financebench_workload_judges_the_filter_and_reads_the_join(
        monkeypatch, tmp_path):
    """The question filter runs on the model; the page join is annotated."""
    import pyarrow.parquet as pq

    import quail

    monkeypatch.setattr(quail, "Session", _FakeSession)
    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    # one anchor per join part, so the annotation join writes two parts
    monkeypatch.setattr(labeling, "JOIN_PAIRS_PER_CALL", 4)
    assert "financebench" not in labeling.annotation_workloads()
    rows = _corpus_rows(
        reviews=["a"], reports=["r"], terms=["t"],
        claims=[("c", "SUPPORTS", "page0")], evidence=[("page0", "e")],
        contexts=[("x", ["p1"])], passages=[("y", ["p1"])],
        contracts=[("Exclusivity",)],
        filings=[("yes, what is the ratio?", [2]),
                 ("what is the revenue?", [1, 2])])
    _, corpus, _ = _write_corpus(tmp_path, 0.1, rows)

    partial = labeling.judge_workload(corpus["corpus_id"], "financebench")

    needs = partial["manifests"][
        "quailb.financebench.question.needs_calculation"]
    assert (needs["rows"], needs["true_rows"]) == (2, 1)
    assert needs["source_rows"] == {MODEL_NAME: 2}
    answers = partial["manifests"]["quailb.financebench.page.answers_question"]
    assert answers["status"] == "complete"
    assert (answers["rows"], answers["true_rows"]) == (2 * 4, 3)
    assert answers["source_rows"] == {"financebench_evidence": 8}
    parts = sorted((Path(answers["compact_path"]).parent / "parts").iterdir())
    assert [part.name for part in parts] == [
        "part_000000_000001.parquet", "part_000001_000002.parquet"]
    labels = pq.read_table(answers["compact_path"]).to_pylist()
    assert {(row["left_id"], row["right_id"]) for row in labels
            if row["answer"]} == {("fq0", "f0p2"), ("fq1", "f1p1"),
                                  ("fq1", "f1p2")}


def test_activate_collection_preserves_raw_prompt_pointer(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    directory = tmp_path / "corpora" / "c_test"
    directory.mkdir(parents=True)
    raw = directory / "active_collection.json"
    raw.write_text('{"collection_id": "gt_raw"}')
    labeling._activate_collection("c_test", "gt_chat")
    assert raw.read_text() == '{"collection_id": "gt_raw"}'
    active = directory / f"active_collection.{labeling.PROMPT_FORMAT}.json"
    assert json.loads(active.read_text()) == {"collection_id": "gt_chat"}


class _FakeS3:
    """Records uploads; the bucket already holds one label file."""

    def __init__(self, existing):
        self.existing = existing
        self.uploads = []

    def get_paginator(self, _name):
        existing = self.existing

        class _Paginator:
            def paginate(self, **_kwargs):
                return [{"Contents": [{"Key": key, "Size": size}
                                      for key, size in existing.items()]}]

        return _Paginator()

    def upload_file(self, path, _bucket, key):
        self.uploads.append(key)


def test_publish_uploads_only_what_a_reader_needs(monkeypatch, tmp_path):
    import json

    from quail_b.data import GROUND_TRUTH_ROOT

    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    spec = _spec("quailb.imdb.review.discusses_ending")
    monkeypatch.setattr(labeling, "PREDICATES", (spec,))
    monkeypatch.setattr(labeling, "PREDICATE_BY_KEY", {})
    corpus_dir = tmp_path / "corpora" / "c_x"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text("{}")
    (corpus_dir / "reviews.parquet").write_bytes(b"rows")
    (corpus_dir / "active_collection.json").write_text("{}")
    (corpus_dir / "files").mkdir()
    (corpus_dir / "files" / "ct0.pdf").write_bytes(b"%PDF")
    label_dir = tmp_path / "label_sets" / spec.workload / spec.slug / "ls_x"
    (label_dir / "parts").mkdir(parents=True)
    (label_dir / "manifest.json").write_text("{}")
    (label_dir / "labels.parquet").write_bytes(b"labels")
    (label_dir / "parts" / "part_000000_000002.parquet").write_bytes(b"p")
    collection_dir = tmp_path / "collections" / "gt_x"
    collection_dir.mkdir(parents=True)
    (collection_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "corpus_id": "c_x",
        "label_sets": {spec.key: "ls_x"}}))
    (collection_dir / "summary.json").write_text("{}")
    prefix = f"{GROUND_TRUTH_ROOT}/label_sets/{spec.workload}/{spec.slug}"
    client = _FakeS3({f"{prefix}/ls_x/labels.parquet": 6})

    result = labeling.publish(tmp_path, ["gt_x"], client=client)

    assert result["uploaded"] == 7 and result["skipped"] == 1
    assert not any("/parts/" in key for key in client.uploads)
    assert f"{GROUND_TRUTH_ROOT}/corpora/c_x/reviews.parquet" in client.uploads
    assert f"{GROUND_TRUTH_ROOT}/corpora/c_x/files/ct0.pdf" in client.uploads
    assert (f"{GROUND_TRUTH_ROOT}/corpora/c_x/active_collection.json"
            in client.uploads)


def test_fixed_join_order_does_not_reuse_automatic_anchor_labels():
    for key, changed in (
        ("quailb.fever.passage.supports_claim", True),
        ("quailb.imdb.review.discusses_ending", False),
        ("quailb.lepard.excerpt.cites_passage", False),
    ):
        spec = _spec(key)
        old = labeling._label_set_identity(
            spec, "c_test", "0" * 64, judge=labeling.QUAIL_JUDGE_SPEC)
        new = labeling.label_set_identity(spec, "c_test", "0" * 64)
        assert (new["label_set_id"] != old["label_set_id"]) == changed


def test_activate_new_predicate_reuses_existing_workload_labels(monkeypatch, tmp_path):
    import json

    old = _spec("quailb.biodex.report.describes_serious_adverse_event")
    new = _spec("quailb.biodex.reaction.is_neurological")
    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    monkeypatch.setattr(labeling, "PREDICATES", (old, new))
    report_table = {"rows": 1, "ordered_rows_full_hash": "same-reports"}
    for corpus_id in ("c_source", "c_target"):
        manifest = {
            "corpus_id": corpus_id, "corpus_full_hash": corpus_id + "-full",
            "scale_factor": 0.1,
            "tables": {"reports": report_table,
                       "terms": {"rows": 1, "ordered_rows_full_hash": corpus_id}},
        }
        labeling._atomic_json(
            tmp_path / "corpora" / corpus_id / "manifest.json", manifest)
    source = {
        "collection_id": "gt_source", "status": "complete",
        "corpus_id": "c_source", "scale_factor": 0.1,
        "label_sets": {old.key: "ls_original"},
    }
    labeling._atomic_json(
        tmp_path / "collections/gt_source/manifest.json", source)
    identity = labeling.label_set_identity(new, "c_target", "c_target-full")
    for spec, label_id, corpus in (
            (old, "ls_original", "c_source"),
            (new, identity["label_set_id"], "c_target")):
        labeling._atomic_json(
            labeling._label_dir(spec, {"label_set_id": label_id}) / "manifest.json",
            {"label_set_id": label_id, "status": "complete",
             "corpus_id": corpus, "corpus_full_hash": corpus + "-full",
             "rows": 1, "true_rows": 1, "source_rows": {MODEL_NAME: 1}})
    summary = labeling.activate_reused_collection(
        0.1, "c_target", "gt_source", "", relabeled_predicates=(new.key,))
    assert summary["reused_predicates"] == 1
    assert summary["new_predicates"] == 1
    assert summary["label_sets"][old.key]["label_set_id"] == "ls_original"
    assert summary["label_sets"][new.key]["label_set_id"] == identity["label_set_id"]
    active = tmp_path / "corpora/c_target/active_collection.raw-v1.json"
    assert json.loads(active.read_text())["collection_id"] == summary["collection_id"]
    with pytest.raises(ValueError, match="unknown relabeled predicates"):
        labeling.activate_reused_collection(
            0.1, "c_target", "gt_source", "", relabeled_predicates=("unknown",))
