"""Build the QUAIL-B reference labels with Quail, one H100 per workload.

Each predicate runs as a Quail query over the corpus with Qwen3 32B
fp8: a filter over every document, a full join over every pair. FEVER
and LePaRD supply source truth where the dataset gives the exact
answer; CUAD's clause predicates are answered by its lawyer annotation
alone, on the CPU. Every part file is written under a content-addressed path on
the `quail-results` volume and skipped when it already exists, so an
interrupted pass resumes where it stopped. Scale factors 0.1, 0.5 and
1.0 are supported; a smaller one samples a prefix of a larger one's
documents, so its labels are derived from the larger pass on the CPU
by matching document content.

    uv run modal run --detach -m quail.bench.labeling --sf 1.0
    uv run modal run --detach -m quail.bench.labeling --sf 0.1 \
      --derive-from-collection <collection>

Publish finished collections to the public bucket, with the AWS
credentials of the machine that runs the command (a profile, the
environment, or SSO, resolved the way the aws CLI resolves them):

    uv run modal run -m quail.bench.labeling \
      --publish-collections <collection>,<collection>

Label-set and collection ids come from `quail_b.predicates` and
depend only on the corpus, the prompts, and the judge.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import modal
import pyarrow as pa

from quail.bench.images import cpu_image, gpu_image
from quail_b import data
from quail_b.cuad import FILES_DIR
from quail_b.data import CORPUS_COLUMNS, GROUND_TRUTH_ROOT, PUBLIC_BUCKET
from quail_b.predicates import (
    MODEL_NAME,
    MODEL_REVISION,
    PREDICATE_BY_KEY,
    PREDICATES,
    QUAIL_JUDGE_SPEC,
    SCHEMA_VERSION,
    WORKLOADS,
    PredicateSpec,
    _full_hash,
    _named_id,
    _text_hash,
    annotation_answer,
    example_identity,
    judgment_identity,
    predicate_payload,
    workload_specs,
)
from quail_b.predicates import label_set_identity as _label_set_identity
from quail_b.rendering import PROMPT_FORMAT

SCALE_FACTOR = 0.1
SUPPORTED_SCALE_FACTORS = (0.1, 0.5, 1.0)
VERIFY_PER_PREDICATE = 16

ROOT = Path.home() / ".cache" / "quail-b" / GROUND_TRUTH_ROOT


def _no_commit() -> None:
    return None


# called after every part file lands; the Modal wrapper commits the volume
after_write = _no_commit

# Label counts follow from the table sizes each scale factor samples.
# The hours use the 14,000 fresh tokens per second measured for the
# BIO-2 join at 32B. These estimates exclude BIO-4's new term filters.
_PREDICTIONS = {
    0.1: ("21 predicates need 1,210,264 labels: 993,450 model judgments "
          "through Quail and 216,814 source labels, about 1.5 H100 hours "
          "in total."),
    0.5: ("21 predicates need 17,618,467 labels: 13,233,865 model "
          "judgments through Quail and 4,384,602 source labels, about 8 "
          "H100 hours in total with agent traces the slowest workload."),
    1.0: ("21 predicates need 51,801,003 labels: 36,926,465 model "
          "judgments through Quail and 14,874,538 source labels, about 23 "
          "H100 hours in total with no workload over 8 hours."),
}
REUSE_PREDICTION_TEXT = (
    "The unchanged table manifests will match exactly. Their label sets "
    "can be reused with the relabeled workloads."
)
DERIVE_PREDICTION_TEXT = (
    "Every document of the smaller corpus appears in the larger one, so "
    "every model judgment and FEVER source label copies over by content "
    "hash. The LePaRD citation join is recomputed from the smaller "
    "corpus; at sf=0.1 from sf=1.0, 2 of its 216,500 labels differ."
)
BIO4_PREDICTION_TEXT = (
    "BIO-4 adds two filter labels per reaction term; "
    "their time is not included in that estimate. "
    "The rerun of 16 saved answers per predicate shows no differences, "
    "and every workload finishes on one H100 without an out-of-memory failure."
)


def prediction_text(sf: float) -> str:
    """The stated prediction for one scale factor's labeling pass."""
    try:
        counts = _PREDICTIONS[float(sf)]
    except KeyError as error:
        raise ValueError(
            f"scale factor {sf} is not one of {SUPPORTED_SCALE_FACTORS}"
        ) from error
    return f"At sf={sf}, {counts} {BIO4_PREDICTION_TEXT}"


# One Quail query is one Parquet part, setting the resume granularity.
# The vLLM judge used 256; a label set's manifest records its value so
# older sets still compact.
PROMPTS_PER_CALL = 4096
LEGACY_PROMPTS_PER_CALL = 256
# A join part holds this many pairs. The planner may anchor on the
# right table, and every part then computes every right-table anchor
# again: at sf=1.0 FEVER's 1,478 passages take about 45 seconds per
# part, so a part must carry many left rows. 563,500 pairs ran as one
# query at sf=0.1. Label sets without this field split joins by
# PROMPTS_PER_CALL.
JOIN_PAIRS_PER_CALL = 500_000


def rows_per_call(prompts_per_row: int,
                  prompts_per_call: int = PROMPTS_PER_CALL) -> int:
    """Rows per call, never fewer than one.

    A part cannot be smaller than one left row, even when a single row
    already exceeds the target.
    """
    return max(1, prompts_per_call // prompts_per_row)


def filter_groups(specs, prompts_per_call: int = PROMPTS_PER_CALL) -> list:
    """Filter predicates grouped by the column they read, in spec order."""
    groups: dict = {}
    for spec in specs:
        if spec.kind == "filter":
            groups.setdefault((spec.left_table, spec.left_column),
                              []).append(spec)
    return [(table, tuple(members),
             rows_per_call(len(members), prompts_per_call))
            for (table, _column), members in groups.items()]


def label_set_identity(spec: PredicateSpec, corpus_id: str,
                       corpus_full_hash: str) -> dict:
    """This pass's label-set identity: the Quail judge, this part size."""
    judge = QUAIL_JUDGE_SPEC
    if spec.kind == "join":
        judge = {**judge, "join_anchor": "arg0"}
    identity = _label_set_identity(
        spec, corpus_id, corpus_full_hash, judge=judge)
    return {**identity, "prompts_per_call": PROMPTS_PER_CALL,
            "join_pairs_per_call": JOIN_PAIRS_PER_CALL}


def join_anchor_batch(identity: dict, right_rows: int) -> int:
    """Left rows per join part for one label set, never fewer than one."""
    pairs = identity.get("join_pairs_per_call")
    if pairs is None:
        return rows_per_call(
            right_rows,
            identity.get("prompts_per_call", LEGACY_PROMPTS_PER_CALL))
    return max(1, pairs // max(1, right_rows))


def part_size_fields(manifest: dict) -> dict:
    """The part-size settings a saved manifest records."""
    fields = {"prompts_per_call": manifest.get(
        "prompts_per_call", LEGACY_PROMPTS_PER_CALL)}
    if "join_pairs_per_call" in manifest:
        fields["join_pairs_per_call"] = manifest["join_pairs_per_call"]
    return fields


def filter_batch_rows(spec: PredicateSpec,
                      prompts_per_call: int = PROMPTS_PER_CALL) -> int:
    """Left rows per part of one filter: its group's share of a call.

    The group is every filter of the workload over the same column,
    so a part of any member covers the same rows.
    """
    return next(n for _table, members, n
                in filter_groups(workload_specs(spec.workload), prompts_per_call)
                if spec in members)


def join_specs(specs) -> tuple:
    return tuple(p for p in specs if p.kind == "join")


# Filters answered from the dataset's own annotation, without a model.
ANNOTATION_SOURCE = "cuad_annotation"


def annotation_sourced(spec: PredicateSpec) -> bool:
    """Whether the dataset's annotation answers this filter outright."""
    return spec.kind == "filter" and spec.source_policy == ANNOTATION_SOURCE


def annotation_workloads() -> tuple[str, ...]:
    """The workloads whose every predicate the annotation answers."""
    return tuple(workload for workload in WORKLOADS
                 if all(annotation_sourced(spec)
                        for spec in workload_specs(workload)))


def _load_corpus(corpus_id: str) -> tuple[Path, dict, dict]:
    """Read a corpus already materialized on the volume."""
    target = ROOT / "corpora" / corpus_id
    with open(target / "manifest.json") as f:
        manifest = json.load(f)
    return target, manifest, _read_rows(target)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(temp, path)


def _atomic_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    schema = pa.schema([
        ("judgment_id", pa.string()),
        ("example_id", pa.string()),
        ("example_full_hash", pa.string()),
        ("label_set_id", pa.string()),
        ("predicate_key", pa.string()),
        ("predicate_version", pa.string()),
        ("answer", pa.bool_()),
        ("label_source", pa.string()),
        ("left_role", pa.string()),
        ("left_table", pa.string()),
        ("left_id", pa.string()),
        ("left_content_sha256", pa.string()),
        ("right_role", pa.string()),
        ("right_table", pa.string()),
        ("right_id", pa.string()),
        ("right_content_sha256", pa.string()),
        ("selected_token_id", pa.int64()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temp, compression="zstd",
                   use_dictionary=True)
    os.replace(temp, path)


def _read_rows(data_dir: Path) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    rows = {}
    for table, columns in CORPUS_COLUMNS.items():
        rows[table] = pq.read_table(
            data_dir / f"{table}.parquet", columns=list(columns)).to_pylist()
    return rows


def _corpus_identity(rows: dict[str, list[dict]], sf: float) -> dict:
    """The corpus id quail-b computes for these tables, with this code's pins."""
    return data.corpus_identity(rows, sf, data.DATA_SEED, data.SOURCE_REVISIONS)


def _materialize_corpus(sf: float) -> tuple[Path, dict, dict]:
    """Build the corpus and copy its tables and referenced files to ROOT.

    A file table's PDFs land under `files/` beside the tables, where
    the table's relative references point.
    """
    with tempfile.TemporaryDirectory(prefix="quailb_judge_") as temp:
        data_dir = data.build_sets(temp, sf=sf)
        rows = _read_rows(data_dir)
        identity = _corpus_identity(rows, sf)
        target = ROOT / "corpora" / identity["corpus_id"]
        target.mkdir(parents=True, exist_ok=True)
        for source in data_dir.glob("*.parquet"):
            destination = target / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        files = data_dir / FILES_DIR
        if files.is_dir():
            shutil.copytree(files, target / FILES_DIR, dirs_exist_ok=True)
        manifest = {**identity, "columns": CORPUS_COLUMNS}
        _atomic_json(target / "manifest.json", manifest)
    return target, manifest, _read_rows(target)


def _content_hash(row: dict, column: str) -> str:
    value = row[column]
    if not isinstance(value, str):
        raise TypeError(f"{column} is not text")
    return _text_hash(value)


def _operand(role: str, table: str, row: dict, column: str) -> dict:
    return {
        "role": role,
        "table": table,
        "row_id": str(row["id"]),
        "column": column,
        "content_sha256": _content_hash(row, column),
    }


def _label_dir(spec: PredicateSpec, identity: dict) -> Path:
    return (ROOT / "label_sets" / spec.workload / spec.slug
            / identity["label_set_id"])


def _part_path(spec: PredicateSpec, identity: dict,
               start: int, end: int) -> Path:
    return (_label_dir(spec, identity) / "parts"
            / f"part_{start:06d}_{end:06d}.parquet")


def _answer_row(spec: PredicateSpec, identity: dict, corpus_id: str,
                left: dict, right: dict | None, label: bool,
                source: str, selected_token_id: int | None) -> dict:
    operands = [_operand(spec.left_role, spec.left_table, left,
                         spec.left_column)]
    if right is not None:
        operands.append(_operand(spec.right_role, spec.right_table, right,
                                 spec.right_column))
    example_id, example_full = example_identity(corpus_id, operands)
    return {
        "judgment_id": judgment_identity(identity["label_set_id"],
                                           example_full),
        "example_id": example_id,
        "example_full_hash": example_full,
        "label_set_id": identity["label_set_id"],
        "predicate_key": spec.key,
        "predicate_version": identity["predicate_version"],
        "answer": bool(label),
        "label_source": source,
        "left_role": spec.left_role,
        "left_table": spec.left_table,
        "left_id": str(left["id"]),
        "left_content_sha256": _content_hash(left, spec.left_column),
        "right_role": spec.right_role,
        "right_table": spec.right_table,
        "right_id": (str(right["id"]) if right is not None else None),
        "right_content_sha256": (
            _content_hash(right, spec.right_column)
            if right is not None else None),
        "selected_token_id": selected_token_id,
    }


def _part_bounds(spec: PredicateSpec, identity: dict,
                 corpus_rows: dict[str, list[dict]]) -> list[tuple[int, int]]:
    """The left-row ranges the writers split this predicate into.

    Must stay in step with _write_filter_parts, _write_qwen_join_parts
    and _write_lepard_source, which is why the batch sizes are derived
    the same way here rather than restated.
    """
    left = len(corpus_rows[spec.left_table])
    per_call = identity.get("prompts_per_call", LEGACY_PROMPTS_PER_CALL)
    if spec.kind == "filter":
        step = filter_batch_rows(spec, per_call)
    elif spec.source_policy == "lepard_citation_edge":
        step = 50
    else:
        step = join_anchor_batch(identity,
                                 len(corpus_rows[spec.right_table]))
    return [(start, min(start + step, left))
            for start in range(0, left, step)]


def _expected_parts(spec: PredicateSpec, identity: dict,
                    corpus_rows: dict[str, list[dict]]) -> list[Path]:
    return [_part_path(spec, identity, start, end)
            for start, end in _part_bounds(spec, identity, corpus_rows)]


def _parts_stats(parts: list[Path]) -> dict:
    """Stats over exactly the parts named, never a glob of the directory.

    A label directory can hold more than one generation of part files:
    a change in filter-group membership or in a join's right-hand table
    moves the boundaries and leaves the older files in place under
    their own names. Globbing counts those twice.
    """
    import pyarrow.parquet as pq

    rows = true_rows = 0
    sources = {}
    for part in parts:
        table = pq.read_table(part, columns=["answer", "label_source"])
        answers = table["answer"].to_pylist()
        labels = table["label_source"].to_pylist()
        rows += len(answers)
        true_rows += sum(bool(v) for v in answers)
        for source in labels:
            sources[source] = sources.get(source, 0) + 1
    return {"rows": rows, "true_rows": true_rows,
            "false_rows": rows - true_rows,
            "source_rows": sources}


def _compact_label_parts(label_dir: Path,
                         parts: list[Path]) -> tuple[Path, int]:
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{label_dir}: missing {len(missing)} part files, first "
            f"{missing[0]}")
    import pyarrow.parquet as pq

    destination = label_dir / "labels.parquet"
    expected = sum(pq.read_metadata(part).num_rows for part in parts)
    if destination.exists():
        rows = pq.read_metadata(destination).num_rows
        if rows != expected:
            raise ValueError(
                f"{destination} has {rows} rows, expected {expected}")
        return destination, rows
    temp = destination.with_name(destination.name + ".tmp")
    writer = None
    try:
        for part in parts:
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(
                    temp, table.schema, compression="zstd",
                    use_dictionary=True)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temp, destination)
    return destination, expected


class QuailJudge:
    """Answer predicates by running them as Quail queries on this GPU.

    One session keeps the model loaded across queries. Each call
    registers its rows as an in-memory table under a fresh name, so
    the catalog grows by one entry per part.
    """

    source = MODEL_NAME

    def __init__(self, gpus: int = 1):
        import quail

        t0 = time.perf_counter()
        self.session = quail.Session(
            quail.EngineConfig(
                gpus=gpus,
                model=MODEL_NAME,
                backend="quail",
                device="h100-sxm",
            )
        )
        self.boot_s = round(time.perf_counter() - t0, 2)
        self.queries = 0
        self.rows_answered = 0
        self.model_wall_s = 0.0
        self._tables = 0

    def close(self) -> None:
        self.session.close()

    def _register(self, rows: list[dict], column: str) -> str:
        import quail

        self._tables += 1
        name = f"labeling_{self._tables}"
        table = pa.table({
            "id": pa.array([str(row["id"]) for row in rows], pa.string()),
            column: pa.array([row[column] for row in rows], pa.string()),
        })
        self.session.register(
            name, quail.DocumentProvider.from_table(table, id_col="id"))
        return name

    def _run(self, query, expected: int):
        t0 = time.perf_counter()
        result = query.run()
        self.model_wall_s += time.perf_counter() - t0
        self.queries += 1
        self.rows_answered += expected
        return result

    def filter(self, spec: PredicateSpec, rows: list[dict]) -> list[bool]:
        """One answer per row, in row order."""
        import quail

        if not rows:
            return []
        name = self._register(rows, spec.left_column)
        query = (self.session.docs(name).alias("l")
                 .ai_filter(quail.prompt(
                     spec.template, quail.col(f"l.{spec.left_column}")))
                 .select("l.id"))
        result = self._run(query, len(rows))
        table = result.answer_tables["filters"][("l", 0)]
        answers = [None] * len(rows)
        for index, answer in zip(table.column("l").to_pylist(),
                                 table.column("answer").to_pylist()):
            answers[int(index)] = bool(answer)
        if any(answer is None for answer in answers):
            raise ValueError(f"{spec.key}: Quail answered "
                             f"{sum(a is not None for a in answers)} of "
                             f"{len(rows)} rows")
        return answers

    def join(self, spec: PredicateSpec, left_rows: list[dict],
             right_rows: list[dict]) -> dict[tuple[int, int], bool]:
        """Answers for every (left index, right index) pair."""
        import quail

        if not left_rows or not right_rows:
            return {}
        left = self._register(left_rows, spec.left_column)
        right = self._register(right_rows, spec.right_column)
        query = (self.session.docs(left).alias("l")
                 .ai_join(self.session.docs(right).alias("r"),
                          quail.prompt(spec.template,
                                       quail.col(f"l.{spec.left_column}"),
                                       quail.col(f"r.{spec.right_column}")),
                          anchor="l", semantics="full")
                 .select("l.id", "r.id"))
        expected = len(left_rows) * len(right_rows)
        result = self._run(query, expected)
        table = result.answer_tables["joins"][0]
        answers = {
            (int(left_index), int(right_index)): bool(answer)
            for left_index, right_index, answer in zip(
                table.column("l").to_pylist(),
                table.column("r").to_pylist(),
                table.column("answer").to_pylist())
        }
        if len(answers) != expected:
            raise ValueError(f"{spec.key}: Quail answered {len(answers)} of "
                             f"{expected} pairs")
        return answers


class AnnotationLabeler:
    """Answer filters from the dataset's annotation; no model, no GPU.

    Answers the same filter interface as `QuailJudge`, so the part
    writers do not care which one they hold.
    """

    source = ANNOTATION_SOURCE

    def filter(self, spec: PredicateSpec, rows: list[dict]) -> list[bool]:
        """One answer per row, in row order."""
        if not annotation_sourced(spec):
            raise ValueError(f"{spec.key} is not answered by an annotation")
        return [annotation_answer(spec, row) for row in rows]


class VerificationSample:
    """A few saved answers per predicate, asked again at the end."""

    def __init__(self):
        self.rows: dict[str, list[tuple[dict, dict | None, bool]]] = {}

    def add(self, spec: PredicateSpec, left: dict, right: dict | None,
            answer: bool) -> None:
        sample = self.rows.setdefault(spec.key, [])
        if len(sample) < VERIFY_PER_PREDICATE:
            sample.append((left, right, answer))

    def run(self, judge: QuailJudge) -> dict:
        compared = flips = 0
        for key, values in sorted(self.rows.items()):
            spec = PREDICATE_BY_KEY[key]
            values = list(reversed(values))
            if spec.kind == "filter":
                got = judge.filter(spec, [left for left, _, _ in values])
            else:
                got = [judge.join(spec, [left], [right])[(0, 0)]
                       for left, right, _ in values]
            compared += len(values)
            flips += sum(answer != expected
                         for answer, (_, _, expected) in zip(got, values))
        return {"compared": compared, "answer_differences": flips,
                "submission_order": "reverse of first pass"}


def _saved_verification_sample(
        corpus_rows: dict[str, list[dict]],
        identities: dict[str, dict],
        specs: tuple | None = None) -> VerificationSample:
    # read PREDICATES at call time, not as a default: the module
    # attribute is patchable and a default would freeze it
    import pyarrow.parquet as pq

    specs = PREDICATES if specs is None else specs

    rows_by_table = {
        table: {str(row["id"]): row for row in rows}
        for table, rows in corpus_rows.items()
    }
    verification = VerificationSample()
    for spec in specs:
        identity = identities[spec.key]
        for part in _expected_parts(spec, identity, corpus_rows):
            if not part.exists():
                break
            table = pq.read_table(
                part,
                columns=["answer", "label_source", "left_id", "right_id"])
            for saved in table.to_pylist():
                if saved["label_source"] != MODEL_NAME:
                    continue
                left = rows_by_table[spec.left_table][saved["left_id"]]
                right = (None if spec.kind == "filter" else
                         rows_by_table[spec.right_table][saved["right_id"]])
                verification.add(spec, left, right, bool(saved["answer"]))
                if (len(verification.rows[spec.key])
                        == VERIFY_PER_PREDICATE):
                    break
            if (len(verification.rows.get(spec.key, []))
                    == VERIFY_PER_PREDICATE):
                break
    return verification


def _write_filter_parts(labeler, verification: VerificationSample | None,
                        rows: list[dict], specs: list[PredicateSpec],
                        identities: dict[str, dict], corpus_id: str,
                        batch_rows: int) -> None:
    """Write the missing parts of these filters, `batch_rows` rows at a time.

    Args:
        labeler: A `QuailJudge` or `AnnotationLabeler`; its `source`
            names the label source of every row it answers.
        verification: Where a sample of the answers is kept to ask
            again, or None when the answers are not a model's.
        rows: The left table's rows.
        specs: The filters, all over the same column of that table.
        identities: Predicate key -> label-set identity.
        corpus_id: The corpus the rows come from.
        batch_rows: Rows per part; `filter_batch_rows` of every spec.
    """
    for start in range(0, len(rows), batch_rows):
        end = min(start + batch_rows, len(rows))
        missing = [spec for spec in specs
                   if not _part_path(spec, identities[spec.key],
                                     start, end).exists()]
        if not missing:
            continue
        for spec in missing:
            answers = labeler.filter(spec, rows[start:end])
            output = []
            for row, answer in zip(rows[start:end], answers):
                output.append(_answer_row(
                    spec, identities[spec.key], corpus_id, row, None,
                    answer, labeler.source, None))
                if verification is not None:
                    verification.add(spec, row, None, answer)
            _atomic_parquet(
                _part_path(spec, identities[spec.key], start, end), output)
        after_write()
        print(f"[{labeler.source}] filters {specs[0].workload} rows "
              f"{start}:{end}", flush=True)


def _write_qwen_join_parts(judge: QuailJudge,
                           verification: VerificationSample,
                           spec: PredicateSpec, left_rows: list[dict],
                           right_rows: list[dict], identity: dict,
                           corpus_id: str, source_label=None) -> None:
    anchor_batch = join_anchor_batch(identity, len(right_rows))
    for start in range(0, len(left_rows), anchor_batch):
        end = min(start + anchor_batch, len(left_rows))
        part = _part_path(spec, identity, start, end)
        if part.exists():
            continue
        # every pair goes through the model; a source label, where the
        # dataset gives one, replaces the model's answer for that pair
        answers = judge.join(spec, left_rows[start:end], right_rows)
        output_rows = []
        model_rows = 0
        for i, left in enumerate(left_rows[start:end]):
            for j, right in enumerate(right_rows):
                known = source_label(left, right) if source_label else None
                if known is not None:
                    label, source = known
                    output_rows.append(_answer_row(
                        spec, identity, corpus_id, left, right,
                        label, source, None))
                    continue
                answer = answers[(i, j)]
                output_rows.append(_answer_row(
                    spec, identity, corpus_id, left, right,
                    answer, MODEL_NAME, None))
                verification.add(spec, left, right, answer)
                model_rows += 1
        output_rows.sort(key=lambda row: (row["left_id"], row["right_id"]))
        _atomic_parquet(part, output_rows)
        after_write()
        print(f"[judge] join {spec.workload} anchors {start}:{end}, "
              f"model={model_rows}, source={len(output_rows) - model_rows}",
              flush=True)


def _lepard_source_answer(cited_passage_ids, passage_ids) -> bool:
    if not isinstance(cited_passage_ids, set):
        cited_passage_ids = set(cited_passage_ids)
    return not cited_passage_ids.isdisjoint(passage_ids)


def _write_lepard_source(spec: PredicateSpec, left_rows: list[dict],
                          right_rows: list[dict], identity: dict,
                          corpus_id: str, anchor_batch: int = 50) -> None:
    right_passage_ids = [set(right["passage_ids"]) for right in right_rows]
    for start in range(0, len(left_rows), anchor_batch):
        end = min(start + anchor_batch, len(left_rows))
        part = _part_path(spec, identity, start, end)
        if part.exists():
            continue
        output = []
        for left in left_rows[start:end]:
            cited_passage_ids = set(left["cited_passage_ids"])
            for right, passage_ids in zip(right_rows, right_passage_ids):
                output.append(_answer_row(
                    spec, identity, corpus_id, left, right,
                    _lepard_source_answer(cited_passage_ids, passage_ids),
                    "lepard_citation_edge", None))
        _atomic_parquet(part, output)
        after_write()
        print(f"[judge] LePaRD source anchors {start}:{end}", flush=True)


def _expected_rows(spec: PredicateSpec,
                   corpus_rows: dict[str, list[dict]]) -> int:
    count = len(corpus_rows[spec.left_table])
    if spec.kind == "join":
        count *= len(corpus_rows[spec.right_table])
    return count


def _collection_identity(corpus_manifest: dict,
                         identities: dict[str, dict]) -> dict:
    mapping = {key: identities[key]["label_set_id"]
               for key in sorted(identities)}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "quailb",
        "scale_factor": corpus_manifest["scale_factor"],
        "corpus_id": corpus_manifest["corpus_id"],
        "label_sets": mapping,
    }
    full = _full_hash(payload)
    return {**payload, "collection_id": _named_id("gt", full),
            "collection_full_hash": full}


def _activate_collection(corpus_id: str, collection_id: str) -> None:
    _atomic_json(
        ROOT / "corpora" / corpus_id / f"active_collection.{PROMPT_FORMAT}.json",
        {"collection_id": collection_id})


def _required_tables(spec: PredicateSpec) -> tuple[str, ...]:
    tables = {spec.left_table}
    if spec.kind == "join":
        tables.add(spec.right_table)
    return tuple(sorted(tables))


def _label_manifest(label_set_id: str) -> tuple[Path, dict]:
    matches = list((ROOT / "label_sets").glob(
        f"*/*/{label_set_id}/manifest.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one manifest for {label_set_id}, found "
            f"{len(matches)}")
    with open(matches[0]) as f:
        return matches[0], json.load(f)


def _check_reused_label_set(
        spec: PredicateSpec, label_set_id: str, source_collection: dict,
        target_corpus: dict) -> dict:
    if source_collection["label_sets"].get(spec.key) != label_set_id:
        raise ValueError(
            f"source collection does not contain {spec.key} as "
            f"{label_set_id}")
    _, manifest = _label_manifest(label_set_id)
    if manifest.get("status") != "complete":
        raise ValueError(f"label set {label_set_id} is not complete")
    label_corpus_id = manifest.get("corpus_id")
    label_corpus_path = (
        ROOT / "corpora" / str(label_corpus_id) / "manifest.json")
    if not label_corpus_path.exists():
        raise FileNotFoundError(
            f"label set {label_set_id} refers to missing corpus "
            f"{label_corpus_id}")
    with open(label_corpus_path) as f:
        label_corpus = json.load(f)
    if (label_corpus.get("corpus_id") != label_corpus_id
            or manifest.get("corpus_full_hash")
            != label_corpus.get("corpus_full_hash")):
        raise ValueError(f"label set {label_set_id} has the wrong corpus")

    tables = _required_tables(spec)
    for table in tables:
        if label_corpus["tables"].get(table) != target_corpus["tables"].get(
                table):
            raise ValueError(
                f"cannot reuse {spec.key}: table {table} changed")
    return {
        "source_collection_id": source_collection["collection_id"],
        "source_corpus_id": label_corpus_id,
        "required_tables": list(tables),
        "verified_table_manifests": {
            table: label_corpus["tables"][table] for table in tables
        },
    }


def _complete_manifest(spec: PredicateSpec, identity: dict,
                       corpus_rows: dict[str, list[dict]]) -> dict:
    label_dir = _label_dir(spec, identity)
    expected_rows = _expected_rows(spec, corpus_rows)
    parts = _expected_parts(spec, identity, corpus_rows)
    stats = _parts_stats(parts)
    if stats["rows"] != expected_rows:
        raise ValueError(
            f"{spec.key}: saved {stats['rows']} rows, expected {expected_rows}")
    compact_path, compact_rows = _compact_label_parts(label_dir, parts)
    manifest = {
        **identity,
        "status": "complete",
        "predicate": asdict(spec),
        "predicate_payload": predicate_payload(spec),
        "expected_rows": expected_rows,
        "compact_path": str(compact_path),
        "compact_rows": compact_rows,
        **stats,
    }
    _atomic_json(label_dir / "manifest.json", manifest)
    return manifest


def _label_dir_by_id(spec: PredicateSpec, label_set_id: str) -> Path:
    return (ROOT / "label_sets" / spec.workload / spec.slug
            / label_set_id)


def compact_ground_truth(collection_id: str) -> dict:
    collection_path = (ROOT / "collections" / collection_id
                       / "manifest.json")
    if not collection_path.exists():
        raise FileNotFoundError(f"unknown collection {collection_id}")
    with open(collection_path) as f:
        collection = json.load(f)
    if collection.get("status") != "complete":
        raise ValueError(f"collection {collection_id} is not complete")
    _, _corpus_manifest, corpus_rows = _load_corpus(collection["corpus_id"])
    compacted = {}
    for key, label_set_id in sorted(collection["label_sets"].items()):
        spec = PREDICATE_BY_KEY[key]
        label_dir = _label_dir_by_id(spec, label_set_id)
        if not label_dir.is_dir():
            raise FileNotFoundError(f"no directory for {label_set_id}")
        manifest_path = label_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        identity = {"label_set_id": label_set_id,
                    **part_size_fields(manifest)}
        compact_path, rows = _compact_label_parts(
            label_dir, _expected_parts(spec, identity, corpus_rows))
        manifest["compact_path"] = str(compact_path)
        manifest["compact_rows"] = rows
        _atomic_json(manifest_path, manifest)
        compacted[key] = {"path": str(compact_path), "rows": rows}
    result = {
        "collection_id": collection_id,
        "label_sets": len(compacted),
        "rows": sum(item["rows"] for item in compacted.values()),
        "compacted": compacted,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _fever_source_label(claim: dict, passage: dict):
    if claim["evidence_wiki_url"] != passage["id"]:
        return None
    return claim["label"] == "SUPPORTS", "fever_annotation"


def prepare_corpus(sf: float = SCALE_FACTOR) -> dict:
    """Build the corpus and check its collection on the volume.

    Reports whether the collection the corpus implies is already
    complete.
    """
    if float(sf) not in SUPPORTED_SCALE_FACTORS:
        raise ValueError(
            f"scale factor {sf} is not one of {SUPPORTED_SCALE_FACTORS}")
    _, corpus_manifest, _ = _materialize_corpus(sf)
    identities = _identities(PREDICATES, corpus_manifest)
    collection = _collection_identity(corpus_manifest, identities)
    path = (ROOT / "collections" / collection["collection_id"]
            / "manifest.json")
    complete = False
    if path.exists():
        with open(path) as f:
            complete = json.load(f).get("status") == "complete"
    print(f"[judge] corpus {corpus_manifest['corpus_id']}, collection "
          f"{collection['collection_id']}, complete={complete}", flush=True)
    return {"corpus": corpus_manifest,
            "collection_id": collection["collection_id"],
            "complete": complete}


def _identities(specs, corpus_manifest: dict) -> dict[str, dict]:
    """Predicate key -> this pass's label-set identity on that corpus."""
    return {
        spec.key: label_set_identity(
            spec, corpus_manifest["corpus_id"],
            corpus_manifest["corpus_full_hash"])
        for spec in specs
    }


def _start_label_sets(specs, identities: dict[str, dict],
                      rows: dict[str, list[dict]]) -> None:
    """Write a running manifest for each label set that has none yet."""
    for spec in specs:
        label_dir = _label_dir(spec, identities[spec.key])
        label_dir.mkdir(parents=True, exist_ok=True)
        if not (label_dir / "manifest.json").exists():
            _atomic_json(label_dir / "manifest.json", {
                **identities[spec.key],
                "status": "running",
                "predicate": asdict(spec),
                "expected_rows": _expected_rows(spec, rows),
            })


def _workload_specs(workload: str) -> tuple:
    specs = workload_specs(workload)
    if not specs:
        raise ValueError(f"no predicates for workload {workload!r}")
    return specs


def judge_workload(corpus_id: str, workload: str) -> dict:
    """Label one workload's predicates on one GPU.

    Every part file is written under a content-addressed path and
    skipped when it already exists, so a container that dies part way
    resumes where it stopped.
    """
    specs = _workload_specs(workload)
    if workload in annotation_workloads():
        raise ValueError(
            f"{workload} is labeled from its annotation; run "
            "label_annotated_workload instead of a GPU judge")
    t_total = time.perf_counter()
    _, corpus_manifest, rows = _load_corpus(corpus_id)
    identities = _identities(specs, corpus_manifest)
    _start_label_sets(specs, identities, rows)

    judge = QuailJudge()
    boot_s = judge.boot_s
    print(f"[judge] {workload}: model ready in {boot_s:.1f} seconds",
          flush=True)
    verification = VerificationSample()

    corpus_id_ = corpus_manifest["corpus_id"]
    for table, group, rows_per_call in filter_groups(specs):
        _write_filter_parts(judge, verification, rows[table], list(group),
                            identities, corpus_id_, rows_per_call)

    for spec in join_specs(specs):
        left, right = rows[spec.left_table], rows[spec.right_table]
        if spec.source_policy == "lepard_citation_edge":
            _write_lepard_source(spec, left, right, identities[spec.key],
                                 corpus_id_)
            continue
        _write_qwen_join_parts(
            judge, verification, spec, left, right, identities[spec.key],
            corpus_id_,
            source_label=(_fever_source_label
                          if spec.source_policy.startswith("fever")
                          else None))

    manifests = {spec.key: _complete_manifest(
        spec, identities[spec.key], rows)
        for spec in specs}

    saved = _saved_verification_sample(rows, identities, specs)
    deterministic = saved.run(judge)
    judge.close()
    partial = {
        "workload": workload,
        "manifests": manifests,
        "boot_s": round(boot_s, 2),
        "model_wall_s": round(judge.model_wall_s, 2),
        "total_wall_s": round(time.perf_counter() - t_total, 2),
        "queries_this_call": judge.queries,
        "rows_answered_this_call_including_verification": judge.rows_answered,
        "deterministic_rerun": deterministic,
    }
    print(f"[judge] {workload} done in {partial['total_wall_s']:.1f}s",
          flush=True)
    return partial


def label_annotated_workload(corpus_id: str, workload: str) -> dict:
    """Label one workload from its dataset's annotation, on the CPU.

    Returns the same partial record as `judge_workload`, with no model
    time and nothing to ask again, so `finalize_collection` takes it
    unchanged.
    """
    specs = _workload_specs(workload)
    if workload not in annotation_workloads():
        raise ValueError(f"{workload} needs a model judge for some predicate")
    t_total = time.perf_counter()
    _, corpus_manifest, rows = _load_corpus(corpus_id)
    identities = _identities(specs, corpus_manifest)
    _start_label_sets(specs, identities, rows)
    labeler = AnnotationLabeler()
    for table, group, rows_per_call in filter_groups(specs):
        _write_filter_parts(labeler, None, rows[table], list(group),
                            identities, corpus_manifest["corpus_id"],
                            rows_per_call)
    manifests = {spec.key: _complete_manifest(spec, identities[spec.key], rows)
                 for spec in specs}
    partial = {
        "workload": workload,
        "manifests": manifests,
        "boot_s": 0.0,
        "model_wall_s": 0.0,
        "total_wall_s": round(time.perf_counter() - t_total, 2),
        "queries_this_call": 0,
        "rows_answered_this_call_including_verification": 0,
        "deterministic_rerun": {
            "compared": 0, "answer_differences": 0,
            "submission_order": "none: the annotation answers every row"},
    }
    print(f"[{labeler.source}] {workload} done in "
          f"{partial['total_wall_s']:.1f}s", flush=True)
    return partial


def finalize_collection(sf: float, corpus_id: str,
                        partials: dict[str, dict]) -> dict:
    """Assemble five workloads and activate their ground truth."""
    _, corpus_manifest, _ = _load_corpus(corpus_id)
    identities = _identities(PREDICATES, corpus_manifest)
    collection = _collection_identity(corpus_manifest, identities)
    collection_dir = (ROOT / "collections"
                      / collection["collection_id"])

    by_workload = partials
    manifests = {}
    for partial in by_workload.values():
        manifests.update(partial["manifests"])
    missing = [spec.key for spec in PREDICATES if spec.key not in manifests]
    if missing:
        raise ValueError(f"no label set reported for: {missing}")

    qwen_rows = sum(m["source_rows"].get(MODEL_NAME, 0)
                    for m in manifests.values())
    total_rows = sum(m["rows"] for m in manifests.values())
    summary = {
        "cell": "quailb_judge_pass",
        "judge": QUAIL_JUDGE_SPEC,
        "prediction": prediction_text(sf),
        "collection_id": collection["collection_id"],
        "corpus_id": corpus_manifest["corpus_id"],
        "scale_factor": sf,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "qwen_judgments": qwen_rows,
        "source_labels": total_rows - qwen_rows,
        "total_labels": total_rows,
        "predicate_count": len(PREDICATES),
        "workloads": {
            name: {k: partial[k] for k in (
                "boot_s", "model_wall_s", "total_wall_s",
                "queries_this_call",
                "rows_answered_this_call_including_verification",
                "deterministic_rerun")}
            for name, partial in sorted(by_workload.items())},
        "wall_s_sum_over_workloads": round(
            sum(p["total_wall_s"] for p in by_workload.values()), 2),
        "wall_s_slowest_workload": round(
            max(p["total_wall_s"] for p in by_workload.values()), 2),
        "deterministic_rerun": {
            "compared": sum(p["deterministic_rerun"]["compared"]
                            for p in by_workload.values()),
            "answer_differences": sum(
                p["deterministic_rerun"]["answer_differences"]
                for p in by_workload.values()),
            "submission_order": "reverse of first pass"},
        "label_sets": {
            key: {"label_set_id": m["label_set_id"], "rows": m["rows"],
                  "true_rows": m["true_rows"],
                  "source_rows": m["source_rows"]}
            for key, m in sorted(manifests.items())},
        "volume_path": str(collection_dir),
    }
    _atomic_json(collection_dir / "manifest.json", {
        **collection, "status": "complete",
        "corpus_manifest": str(
            ROOT / "corpora" / corpus_id / "manifest.json"),
        "summary": summary})
    _atomic_json(collection_dir / "summary.json", summary)
    _activate_collection(corpus_manifest["corpus_id"],
                         collection["collection_id"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def activate_reused_collection(
        sf: float, target_corpus_id: str, source_collection_id: str,
        relabeled_workloads: str, *,
        relabeled_predicates: tuple[str, ...] = ()) -> dict:
    """Build one collection from new labels and verified unchanged tables."""
    with open(ROOT / "corpora" / target_corpus_id / "manifest.json") as f:
        target_corpus = json.load(f)
    source_collection_path = (
        ROOT / "collections" / source_collection_id / "manifest.json")
    with open(source_collection_path) as f:
        source_collection = json.load(f)
    if (source_collection.get("status") != "complete"
            or source_collection.get("collection_id")
            != source_collection_id):
        raise ValueError(
            f"source collection {source_collection_id} is invalid")
    if float(source_collection["scale_factor"]) != float(sf):
        raise ValueError(
            f"source collection has scale factor "
            f"{source_collection['scale_factor']}, expected {sf}")

    source_corpus_id = source_collection["corpus_id"]
    with open(ROOT / "corpora" / source_corpus_id / "manifest.json") as f:
        source_corpus = json.load(f)
    if source_corpus.get("corpus_id") != source_corpus_id:
        raise ValueError(f"source corpus {source_corpus_id} is invalid")
    if (target_corpus.get("corpus_id") != target_corpus_id
            or float(target_corpus["scale_factor"]) != float(sf)):
        raise ValueError(f"target corpus {target_corpus_id} is invalid")
    names = {name.strip() for name in relabeled_workloads.split(",")
             if name.strip()}
    unknown = names - set(WORKLOADS)
    if unknown:
        raise ValueError(f"unknown relabeled workloads: {sorted(unknown)}")
    unknown_predicates = set(relabeled_predicates) - set(PREDICATE_BY_KEY)
    if unknown_predicates:
        raise ValueError(f"unknown relabeled predicates: {sorted(unknown_predicates)}")

    identities = {}
    manifests = {}
    reused = {}
    for spec in PREDICATES:
        if spec.workload in names or spec.key in relabeled_predicates:
            identity = label_set_identity(
                spec, target_corpus["corpus_id"],
                target_corpus["corpus_full_hash"])
            _, manifest = _label_manifest(identity["label_set_id"])
            if (manifest.get("status") != "complete"
                    or manifest.get("corpus_id")
                    != target_corpus["corpus_id"]
                    or manifest.get("corpus_full_hash")
                    != target_corpus["corpus_full_hash"]):
                raise ValueError(
                    f"new label set {identity['label_set_id']} is not a "
                    "complete label set for the target corpus")
        else:
            label_set_id = source_collection["label_sets"].get(spec.key)
            if not label_set_id:
                raise ValueError(
                    f"source collection has no label set for {spec.key}")
            identity = {"label_set_id": label_set_id}
            reused[spec.key] = _check_reused_label_set(
                spec, label_set_id, source_collection, target_corpus)
            _, manifest = _label_manifest(label_set_id)
        identities[spec.key] = identity
        manifests[spec.key] = manifest

    collection = _collection_identity(target_corpus, identities)
    collection_dir = (
        ROOT / "collections" / collection["collection_id"])
    qwen_rows = sum(m["source_rows"].get(MODEL_NAME, 0)
                    for m in manifests.values())
    total_rows = sum(m["rows"] for m in manifests.values())
    summary = {
        "cell": "quailb_ground_truth_collection_reuse",
        "prediction": REUSE_PREDICTION_TEXT,
        "collection_id": collection["collection_id"],
        "corpus_id": target_corpus["corpus_id"],
        "scale_factor": sf,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_collection_id": source_collection_id,
        "source_corpus_id": source_corpus_id,
        "relabeled_workloads": sorted(names),
        "relabeled_predicates": sorted(relabeled_predicates),
        "reused_predicates": len(reused),
        "new_predicates": len(PREDICATES) - len(reused),
        "predicate_count": len(PREDICATES),
        "qwen_judgments": qwen_rows,
        "source_labels": total_rows - qwen_rows,
        "total_labels": total_rows,
        "label_sets": {
            key: {
                "label_set_id": manifest["label_set_id"],
                "rows": manifest["rows"],
                "true_rows": manifest["true_rows"],
                "source_rows": manifest["source_rows"],
                "reused": key in reused,
            }
            for key, manifest in sorted(manifests.items())
        },
        "volume_path": str(collection_dir),
    }
    collection_manifest = {
        **collection,
        "status": "complete",
        "corpus_manifest": str(
            ROOT / "corpora" / target_corpus_id / "manifest.json"),
        "reused_label_sets": reused,
        "summary": summary,
    }
    _atomic_json(collection_dir / "manifest.json", collection_manifest)
    _atomic_json(collection_dir / "summary.json", summary)
    _activate_collection(target_corpus["corpus_id"],
                         collection["collection_id"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _source_labels_by_content(
        spec: PredicateSpec, source_label_set_id: str,
        target_rows: dict[str, list[dict]]) -> dict:
    """Read one finished label set, keyed by document content hashes.

    Only rows whose documents also appear in the target corpus are
    kept, so a join over a large corpus does not have to fit in memory
    as Python objects.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    _, manifest = _label_manifest(source_label_set_id)
    if manifest.get("status") != "complete":
        raise ValueError(f"label set {source_label_set_id} is not complete")
    _, _source_manifest, source_rows = _load_corpus(manifest["corpus_id"])
    source_identity = {"label_set_id": source_label_set_id,
                       **part_size_fields(manifest)}
    label_dir = _label_dir_by_id(spec, source_label_set_id)
    compact_path, _rows = _compact_label_parts(
        label_dir, _expected_parts(spec, source_identity, source_rows))
    table = pq.read_table(compact_path, columns=[
        "left_content_sha256", "right_content_sha256", "answer",
        "label_source"])
    left_hashes = pa.array(
        sorted({_content_hash(row, spec.left_column)
                for row in target_rows[spec.left_table]}))
    table = table.filter(
        pc.is_in(table["left_content_sha256"], value_set=left_hashes))
    if spec.kind == "join":
        right_hashes = pa.array(
            sorted({_content_hash(row, spec.right_column)
                    for row in target_rows[spec.right_table]}))
        table = table.filter(
            pc.is_in(table["right_content_sha256"], value_set=right_hashes))
    return {
        (left, right): (bool(answer), source)
        for left, right, answer, source in zip(
            table["left_content_sha256"].to_pylist(),
            table["right_content_sha256"].to_pylist(),
            table["answer"].to_pylist(),
            table["label_source"].to_pylist())
    }


def _copy_label_set(spec: PredicateSpec, source_label_set_id: str,
                    identity: dict, corpus_id: str,
                    rows: dict[str, list[dict]]) -> int:
    """Write one predicate's target parts from a larger corpus's labels.

    Returns the number of labels written. Raises when a target document
    or pair has no label in the source set.
    """
    parts = _expected_parts(spec, identity, rows)
    if all(part.exists() for part in parts):
        return 0
    labels = _source_labels_by_content(spec, source_label_set_id, rows)
    left_rows = rows[spec.left_table]
    right_rows = rows[spec.right_table] if spec.kind == "join" else [None]
    right_hashes = [
        None if right is None else _content_hash(right, spec.right_column)
        for right in right_rows]
    written = 0
    for part, (start, end) in zip(parts,
                                  _part_bounds(spec, identity, rows)):
        if part.exists():
            continue
        output = []
        missing = 0
        for left in left_rows[start:end]:
            left_hash = _content_hash(left, spec.left_column)
            for right, right_hash in zip(right_rows, right_hashes):
                found = labels.get((left_hash, right_hash))
                if found is None:
                    missing += 1
                    continue
                answer, source = found
                output.append(_answer_row(
                    spec, identity, corpus_id, left, right, answer, source,
                    None))
        if missing:
            raise ValueError(
                f"{spec.key}: {missing} target labels are not in "
                f"{source_label_set_id}")
        if spec.kind == "join":
            output.sort(key=lambda row: (row["left_id"], row["right_id"]))
        _atomic_parquet(part, output)
        after_write()
        written += len(output)
        print(f"[derive] {spec.key} rows {start}:{end}", flush=True)
    return written


def derive_collection(sf: float, source_collection_id: str) -> dict:
    """Build one scale factor's collection from a larger one, on the CPU.

    Every model judgment and FEVER source label is copied by document
    content hash. The LePaRD citation join depends on which citation
    pairs the corpus sampled, and CUAD's annotation labels on the ids
    the corpus gave its contracts, so those are recomputed from the
    target corpus instead.
    """
    if float(sf) not in SUPPORTED_SCALE_FACTORS:
        raise ValueError(
            f"scale factor {sf} is not one of {SUPPORTED_SCALE_FACTORS}")
    source_path = (ROOT / "collections" / source_collection_id
                   / "manifest.json")
    with open(source_path) as f:
        source = json.load(f)
    if (source.get("status") != "complete"
            or source.get("collection_id") != source_collection_id):
        raise ValueError(
            f"source collection {source_collection_id} is invalid")
    if float(source["scale_factor"]) <= float(sf):
        raise ValueError(
            f"source collection has scale factor {source['scale_factor']}, "
            f"which is not larger than {sf}")
    missing = [spec.key for spec in PREDICATES
               if spec.key not in source["label_sets"]]
    if missing:
        raise ValueError(f"source collection has no label set for: {missing}")

    t_total = time.perf_counter()
    _, corpus_manifest, rows = _materialize_corpus(sf)
    corpus_id = corpus_manifest["corpus_id"]
    identities = _identities(PREDICATES, corpus_manifest)
    _start_label_sets(PREDICATES, identities, rows)
    origin = {}
    for spec in PREDICATES:
        identity = identities[spec.key]
        if spec.source_policy == "lepard_citation_edge":
            _write_lepard_source(spec, rows[spec.left_table],
                                 rows[spec.right_table], identity, corpus_id)
            origin[spec.key] = {"recomputed_from": corpus_id}
            continue
        if annotation_sourced(spec):
            # the ids a smaller corpus gives its documents differ, so the
            # annotation is asked again rather than copied by content
            _write_filter_parts(
                AnnotationLabeler(), None, rows[spec.left_table], [spec],
                identities, corpus_id, filter_batch_rows(spec))
            origin[spec.key] = {"recomputed_from": corpus_id}
            continue
        source_label_set_id = source["label_sets"][spec.key]
        _copy_label_set(spec, source_label_set_id, identity, corpus_id, rows)
        origin[spec.key] = {"copied_from": source_label_set_id}

    manifests = {spec.key: _complete_manifest(spec, identities[spec.key], rows)
                 for spec in PREDICATES}
    collection = _collection_identity(corpus_manifest, identities)
    collection_dir = ROOT / "collections" / collection["collection_id"]
    qwen_rows = sum(m["source_rows"].get(MODEL_NAME, 0)
                    for m in manifests.values())
    total_rows = sum(m["rows"] for m in manifests.values())
    summary = {
        "cell": "quailb_ground_truth_collection_derived",
        "judge": QUAIL_JUDGE_SPEC,
        "prediction": DERIVE_PREDICTION_TEXT,
        "collection_id": collection["collection_id"],
        "corpus_id": corpus_id,
        "scale_factor": sf,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_collection_id": source_collection_id,
        "source_corpus_id": source["corpus_id"],
        "source_scale_factor": source["scale_factor"],
        "qwen_judgments": qwen_rows,
        "source_labels": total_rows - qwen_rows,
        "total_labels": total_rows,
        "predicate_count": len(PREDICATES),
        "total_wall_s": round(time.perf_counter() - t_total, 2),
        "label_sets": {
            key: {"label_set_id": m["label_set_id"], "rows": m["rows"],
                  "true_rows": m["true_rows"],
                  "source_rows": m["source_rows"], **origin[key]}
            for key, m in sorted(manifests.items())},
        "volume_path": str(collection_dir),
    }
    _atomic_json(collection_dir / "manifest.json", {
        **collection, "status": "complete",
        "corpus_manifest": str(
            ROOT / "corpora" / corpus_id / "manifest.json"),
        "derived_from_collection": source_collection_id,
        "summary": summary})
    _atomic_json(collection_dir / "summary.json", summary)
    _activate_collection(corpus_id, collection["collection_id"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def collection_files(root: Path, collection_id: str) -> list[Path]:
    """Every file a reader of one collection needs, under root.

    The collection and corpus manifests, the corpus tables and the
    files they refer to, each label set's manifest and compact
    `labels.parquet`, and the manifests of any collection and corpus
    it reuses labels from. Part files are not needed once a label set
    is compacted.
    """
    collection_dir = root / "collections" / collection_id
    with open(collection_dir / "manifest.json") as f:
        manifest = json.load(f)
    if manifest.get("status") != "complete":
        raise ValueError(f"collection {collection_id} is not complete")
    files = [collection_dir / "manifest.json", collection_dir / "summary.json"]
    corpus_dir = root / "corpora" / manifest["corpus_id"]
    files += sorted(path for path in corpus_dir.iterdir()
                    if path.is_file() and path.suffix in (".json", ".parquet"))
    files += sorted((corpus_dir / FILES_DIR).rglob("*.pdf"))
    for label_set_id in sorted(manifest["label_sets"].values()):
        matches = list((root / "label_sets").glob(
            f"*/*/{label_set_id}/manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one manifest for {label_set_id}, found "
                f"{len(matches)}")
        label_dir = matches[0].parent
        compact = label_dir / "labels.parquet"
        if not compact.exists():
            raise FileNotFoundError(
                f"{label_set_id} has no labels.parquet; compact the "
                "collection first")
        files += [label_dir / "manifest.json", compact]
    for record in manifest.get("reused_label_sets", {}).values():
        files.append(root / "collections" / record["source_collection_id"]
                     / "manifest.json")
        files.append(root / "corpora" / record["source_corpus_id"]
                     / "manifest.json")
    return list(dict.fromkeys(files))


def publish(root: Path, collection_ids: list[str],
            bucket: str = PUBLIC_BUCKET, client=None) -> dict:
    """Upload the files of these collections that the bucket lacks.

    A parquet file whose key already exists with the same size is
    skipped; every JSON file is uploaded, because a corpus's
    `active_collection.json` changes without changing size.
    """
    if client is None:
        import boto3

        client = boto3.client("s3")
    prefix = GROUND_TRUTH_ROOT
    existing = {}
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix + "/"):
        for item in page.get("Contents", ()):
            existing[item["Key"]] = item["Size"]
    files = []
    for collection_id in collection_ids:
        files += collection_files(root, collection_id)
    uploaded = skipped = 0
    uploaded_bytes = 0
    for path in dict.fromkeys(files):
        key = f"{prefix}/{path.relative_to(root).as_posix()}"
        size = path.stat().st_size
        if path.suffix == ".parquet" and existing.get(key) == size:
            skipped += 1
            continue
        client.upload_file(str(path), bucket, key)
        uploaded += 1
        uploaded_bytes += size
        print(f"[publish] {key} ({size / 2**20:.1f} MiB)", flush=True)
    result = {"bucket": bucket, "collections": list(collection_ids),
              "files": len(set(files)), "uploaded": uploaded,
              "skipped": skipped, "uploaded_bytes": uploaded_bytes}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


# ---------------------------------------------------------------- Modal

# The judge runs through Quail's executor, so the GPU and volume
# functions use the worker image the benchmark runner uses. Publishing
# needs boto3 and no engine, so it gets a small image; the AWS keys
# come from the machine that runs the command.
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
image = gpu_image()
# the dev group carries boto3 for the upload; no GPU stack here
publish_image = cpu_image()



def _aws_credentials() -> dict[str, str]:
    """The launching machine's AWS credentials, resolved like the aws CLI.

    Empty where boto3 is missing or finds none, so the module imports
    inside containers and the GPU functions never need AWS.
    """
    try:
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()
    except Exception:  # no credentials is a normal state, not an error
        return {}
    if credentials is None:
        return {}
    frozen = credentials.get_frozen_credentials()
    found = {"AWS_ACCESS_KEY_ID": frozen.access_key,
             "AWS_SECRET_ACCESS_KEY": frozen.secret_key}
    if frozen.token:
        found["AWS_SESSION_TOKEN"] = frozen.token
    if session.region_name:
        found["AWS_DEFAULT_REGION"] = session.region_name
    return found


aws_from_launcher = modal.Secret.from_dict(_aws_credentials())
# Experiment cells attach to this existing app so its caches remain useful.
app = modal.App("quail-milestone1")

# Compaction and finalize read every part of a label set; at sf=1.0
# the BioDEX join alone has 20.7 million rows.
CPU_TIMEOUT_S = 3600
CPU_MEMORY_MB = 16384


def _mount() -> None:
    """Point the pass at the volume: labels live there, parts commit as written."""
    global ROOT, after_write
    results_vol.reload()
    ROOT = Path("/results") / GROUND_TRUTH_ROOT
    after_write = results_vol.commit


def parse_function_calls(value: str) -> dict[str, str]:
    calls = {}
    for item in value.split(","):
        try:
            workload, function_call_id = item.split("=", 1)
        except ValueError as exc:
            raise ValueError(
                "function calls must use workload=fc-id") from exc
        workload = workload.strip()
        function_call_id = function_call_id.strip()
        if workload in calls:
            raise ValueError(f"duplicate workload {workload!r}")
        calls[workload] = function_call_id
    missing = set(WORKLOADS) - set(calls)
    unknown = set(calls) - set(WORKLOADS)
    if missing or unknown:
        raise ValueError(
            f"function calls have missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")
    if any(not value.startswith("fc-") for value in calls.values()):
        raise ValueError("every function call id must start with fc-")
    return calls


@app.function(
    image=image, memory=CPU_MEMORY_MB, timeout=CPU_TIMEOUT_S,
    volumes={"/results": results_vol})
def run_compact(collection_id: str) -> str:
    _mount()
    result = compact_ground_truth(collection_id)
    results_vol.commit()
    return json.dumps(result, sort_keys=True)


# Building the sf=1.0 corpus downloads SWE-Next and tokenizes 17,711
# trace snapshots, which is far slower than the other tables.
@app.function(
    image=image, memory=CPU_MEMORY_MB, timeout=2 * CPU_TIMEOUT_S,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/results": results_vol})
def run_prepare_corpus(sf: float = SCALE_FACTOR) -> str:
    _mount()
    result = prepare_corpus(sf)
    results_vol.commit()
    return json.dumps(result, sort_keys=True)


# The slowest sf=1.0 workload took 7.7 hours; 24 hours is Modal's
# limit. Parts commit as written, so a container that dies loses only
# the part it was on.
@app.function(
    image=image, gpu="H100!", memory=98304, timeout=86400,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/root/.cache/kernels": kernel_cache,
             "/results": results_vol})
def run_judge_workload(corpus_id: str, workload: str) -> str:
    """Label one workload's predicates on one GPU."""
    _mount()
    partial = judge_workload(corpus_id, workload)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(partial, sort_keys=True)


@app.function(
    image=image, memory=CPU_MEMORY_MB, timeout=CPU_TIMEOUT_S,
    volumes={"/results": results_vol})
def run_label_annotated_workload(corpus_id: str, workload: str) -> str:
    """Label one workload from its dataset's annotation, on the CPU."""
    _mount()
    partial = label_annotated_workload(corpus_id, workload)
    results_vol.commit()
    return json.dumps(partial, sort_keys=True)


def spawn_workload(corpus_id: str, workload: str):
    """Start the labeling call a workload needs: a GPU judge or the CPU."""
    if workload in annotation_workloads():
        return run_label_annotated_workload.spawn(corpus_id, workload)
    return run_judge_workload.spawn(corpus_id, workload)


@app.function(
    image=image, memory=CPU_MEMORY_MB, timeout=CPU_TIMEOUT_S,
    volumes={"/results": results_vol})
def run_finalize(sf: float, corpus_id: str, partials: str) -> str:
    """Assemble five workloads and activate their ground truth."""
    _mount()
    summary = finalize_collection(sf, corpus_id, json.loads(partials))
    results_vol.commit()
    return json.dumps(summary, sort_keys=True)


@app.function(
    image=image, memory=CPU_MEMORY_MB, timeout=CPU_TIMEOUT_S,
    volumes={"/results": results_vol})
def run_activate_reused(
        sf: float, target_corpus_id: str, source_collection_id: str,
        relabeled_workloads: str) -> str:
    """Build one collection from new labels and verified unchanged tables."""
    _mount()
    summary = activate_reused_collection(
        sf, target_corpus_id, source_collection_id, relabeled_workloads)
    results_vol.commit()
    return json.dumps(summary, sort_keys=True)


# The derive step holds the target's share of a source join in memory:
# at sf=0.5 from sf=1.0, 7.3 million BioDEX pairs.
@app.function(
    image=image, memory=2 * CPU_MEMORY_MB, timeout=2 * CPU_TIMEOUT_S,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/results": results_vol})
def run_derive(sf: float, source_collection_id: str) -> str:
    """Build one scale factor's collection from a larger one, on the CPU."""
    _mount()
    summary = derive_collection(sf, source_collection_id)
    results_vol.commit()
    return json.dumps(summary, sort_keys=True)


@app.function(
    image=publish_image, memory=CPU_MEMORY_MB, timeout=2 * CPU_TIMEOUT_S,
    volumes={"/results": results_vol}, secrets=[aws_from_launcher])
def run_publish(collection_ids: str) -> str:
    """Upload the named collections from the volume to the public bucket."""
    _mount()
    ids = [value.strip() for value in collection_ids.split(",")
           if value.strip()]
    return json.dumps(publish(ROOT, ids), sort_keys=True)


@app.local_entrypoint()
def main(sf: float = SCALE_FACTOR, compact_collection: str | None = None,
         only: str | None = None, finalize_from: str | None = None,
         reuse_from_collection: str | None = None,
         relabeled_workloads: str = "lepard",
         target_corpus: str | None = None,
         derive_from_collection: str | None = None,
         publish_collections: str | None = None):
    """Run the five workloads side by side, then activate the result.

    ``--only imdb,fever`` restricts the pass to those workloads; the
    finalize step is skipped because a collection needs all label sets.
    ``--derive-from-collection`` copies a finished larger collection's
    labels to this scale factor instead of running the model.
    ``--publish-collections`` uploads finished collections to the
    public bucket.
    """
    if publish_collections:
        call = run_publish.spawn(publish_collections)
        print(f"function call id (publish): {call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    if compact_collection:
        call = run_compact.spawn(compact_collection)
        print(f"function call id: {call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    if derive_from_collection:
        print(f"PREDICTION: {DERIVE_PREDICTION_TEXT}", flush=True)
        call = run_derive.spawn(sf, derive_from_collection)
        print(f"function call id (derive): {call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    prediction = (REUSE_PREDICTION_TEXT if reuse_from_collection
                  else prediction_text(sf))
    print(f"PREDICTION: {prediction}", flush=True)

    if reuse_from_collection and target_corpus:
        corpus_id = target_corpus
        prepared = None
    else:
        call = run_prepare_corpus.spawn(sf)
        print(f"function call id (prepare_corpus): {call.object_id}",
              flush=True)
        prepared = json.loads(call.get())
        corpus_id = prepared["corpus"]["corpus_id"]
    if reuse_from_collection:
        call = run_activate_reused.spawn(
            sf, corpus_id, reuse_from_collection, relabeled_workloads)
        print("function call id (activate_reused_collection): "
              f"{call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    if finalize_from:
        partials = {}
        for workload, function_call_id in parse_function_calls(
                finalize_from).items():
            result = modal.FunctionCall.from_id(function_call_id).get()
            partial = json.loads(result)
            if partial["workload"] != workload:
                raise ValueError(
                    f"{function_call_id} returned workload "
                    f"{partial['workload']!r}, expected {workload!r}")
            partials[workload] = partial
        call = run_finalize.spawn(sf, corpus_id, json.dumps(partials))
        print(f"function call id (finalize_collection): {call.object_id}",
              flush=True)
        print(call.get(), flush=True)
        return
    if prepared["complete"] and not only:
        print(f"collection {prepared['collection_id']} is already complete",
              flush=True)
        return

    names = ([w.strip() for w in only.split(",")] if only
             else list(WORKLOADS))
    unknown = [w for w in names if w not in WORKLOADS]
    if unknown:
        raise ValueError(f"unknown workloads: {unknown}")

    calls = {w: spawn_workload(corpus_id, w) for w in names}
    for w, c in calls.items():
        print(f"function call id (workload {w}): {c.object_id}", flush=True)
    partials = {}
    for w, c in calls.items():
        partials[w] = json.loads(c.get())
        print(f"[main] {w} finished in "
              f"{partials[w]['total_wall_s']:.1f}s", flush=True)

    if only:
        print("--only was given, so the collection is not finalized",
              flush=True)
        return
    call = run_finalize.spawn(sf, corpus_id, json.dumps(partials))
    print(f"function call id (finalize_collection): {call.object_id}",
          flush=True)
    print(call.get(), flush=True)
