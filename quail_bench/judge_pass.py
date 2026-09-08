"""Build the sf=0.1 QUAIL-B ground-truth collection on Modal.

Qwen3 32B labels the predicates without exact source labels. FEVER and
LePaRD supply source truth where available. The run uses stable label-set
IDs and skips completed Parquet parts.

    uv run modal run -m quail_bench.judge_pass

Reuse labels after an unrelated table changes in a new corpus:

    uv run modal run --detach -m quail_bench.judge_pass \
      --reuse-from-collection <collection> \
      --target-corpus <corpus> \
      --relabeled-workloads lepard
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

from quail_bench import data, prompts, rendering
from quail_bench.rendering import SHARED_PRE

SCHEMA_VERSION = 1
SCALE_FACTOR = 0.1
MODEL_REPO = "Qwen/Qwen3-32B-FP8"
MODEL_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
MODEL_NAME = "qwen3-32b-fp8"
MAX_MODEL_LEN = 32_768
MAX_BATCH_TOKENS = 25_305
MAX_SEQS = 4_096
# The 32B judge needs more free memory than the 4B engine runs do: at
# 0.92 the KV reservation left 5.38 GiB free on an 80 GiB H100 and a
# 5.41 GiB prefill activation then failed to allocate.
GPU_MEMORY_UTILIZATION = 0.85
VERIFY_PER_PREDICATE = 16

VOLUME_ROOT = Path("/results/ground_truth/quailb/schema_v1")

PREDICTION_TEXT = (
    "At sf=0.1, 21 predicates require 1,210,264 labels. Qwen3 32B "
    "produces 993,450 judgments, and dataset annotations produce 216,814 "
    "source labels. The agent workload adds 3,544 Qwen judgments over "
    "1,772 cumulative SWE-Next trace snapshots. Five H100 GPUs can run "
    "one workload each. Based on the prior agent labeling run, the agent "
    "workload should take 11 to 14 minutes including model load and cost "
    "$0.72 to $0.92. The deterministic "
    "rerun sample should have no answer differences, and each run should "
    "finish without an out-of-memory failure at "
    "gpu_memory_utilization=0.85."
)
REUSE_PREDICTION_TEXT = (
    "The unchanged table manifests will match exactly. Their label sets "
    "can be reused with the relabeled workloads."
)


@dataclass(frozen=True)
class PredicateSpec:
    key: str
    workload: str
    slug: str
    kind: str
    template: str
    left_role: str
    left_table: str
    left_column: str
    right_role: str | None = None
    right_table: str | None = None
    right_column: str | None = None
    source_policy: str = "qwen3_32b"


PREDICATES = (
    PredicateSpec(
        "quailb.imdb.review.mentions_positive_aspect", "imdb",
        "review_mentions_positive_aspect", "filter", prompts.F1,
        "review", "reviews", "body"),
    PredicateSpec(
        "quailb.imdb.review.discusses_ending", "imdb",
        "review_discusses_ending", "filter", prompts.F4,
        "review", "reviews", "body"),
    PredicateSpec(
        "quailb.imdb.review.mentions_named_actor", "imdb",
        "review_mentions_named_actor", "filter", prompts.F5,
        "review", "reviews", "body"),
    PredicateSpec(
        "quailb.imdb.review.discusses_aspect", "imdb",
        "review_discusses_aspect", "join", prompts.DISCUSS_ASPECT,
        "review", "reviews", "body", "aspect", "aspects", "aspect"),
    PredicateSpec(
        "quailb.imdb.review.positive_sentiment_about_aspect",
        "imdb",
        "review_positive_sentiment_about_aspect", "join",
        prompts.ASPECT_SENTIMENT,
        "review", "reviews", "body", "aspect", "aspects", "aspect"),
    PredicateSpec(
        "quailb.biodex.report.involves_female_patient", "biodex",
        "report_involves_female_patient", "filter", prompts.F7,
        "report", "reports", "report"),
    PredicateSpec(
        "quailb.biodex.report.experienced_reaction", "biodex",
        "report_experienced_reaction", "join", prompts.REACTION,
        "report", "reports", "report", "reaction", "terms", "term"),
    PredicateSpec(
        "quailb.fever.claim.about_person", "fever",
        "claim_about_person", "filter", prompts.F11,
        "claim", "claims", "claim"),
    PredicateSpec(
        "quailb.fever.claim.contains_date", "fever",
        "claim_contains_date", "filter", prompts.F12,
        "claim", "claims", "claim"),
    PredicateSpec(
        "quailb.fever.passage.about_person", "fever",
        "passage_about_person", "filter", prompts.F13,
        "passage", "evidence", "text"),
    PredicateSpec(
        "quailb.fever.passage.supports_claim", "fever",
        "passage_supports_claim", "join", prompts.SUPPORT,
        "claim", "claims", "claim", "passage", "evidence", "text",
        "fever_annotation_then_qwen3_32b"),
    PredicateSpec(
        "quailb.fever.passage.refutes_claim", "fever",
        "passage_refutes_claim", "join", prompts.REFUTE,
        "claim", "claims", "claim", "passage", "evidence", "text"),
    PredicateSpec(
        "quailb.lepard.excerpt.reasoning_does_not_apply", "lepard",
        "excerpt_reasoning_does_not_apply", "filter", prompts.LEP1,
        "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.procedural_or_jurisdictional", "lepard",
        "excerpt_procedural_or_jurisdictional", "filter",
        prompts.LEP2, "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.treats_passage_as_binding", "lepard",
        "excerpt_treats_passage_as_binding", "filter", prompts.LEP3,
        "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.supports_liability_or_guilt", "lepard",
        "excerpt_supports_liability_or_guilt", "filter",
        prompts.LEP4, "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.acknowledges_court_disagreement", "lepard",
        "excerpt_acknowledges_court_disagreement", "filter",
        prompts.LEP5, "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.passage.states_general_rule", "lepard",
        "passage_states_general_rule", "filter", prompts.LEPS1,
        "passage", "citation_passages", "passage_text"),
    PredicateSpec(
        "quailb.lepard.excerpt.cites_passage", "lepard",
        "excerpt_cites_passage", "join", prompts.LEPJOIN,
        "excerpt", "citation_contexts", "destination_context",
        "passage", "citation_passages", "passage_text",
        "lepard_citation_edge"),
    PredicateSpec(
        "quailb.agent.trace.recovered_after_unsuccessful_approach",
        "agent", "recovered_after_unsuccessful_approach",
        "filter", prompts.AGENT_RECOVERED,
        "agent_trace", "agent_traces", "trace"),
    PredicateSpec(
        "quailb.agent.trace.implemented_plausible_fix",
        "agent", "implemented_plausible_fix",
        "filter", prompts.AGENT_IMPLEMENTED_FIX,
        "agent_trace", "agent_traces", "trace"),
)

PREDICATE_BY_KEY = {p.key: p for p in PREDICATES}


# One model call is one Parquet part, setting the resume granularity.
PROMPTS_PER_CALL = 256


def rows_per_call(prompts_per_row: int) -> int:
    """Rows per call, never fewer than one.

    A part cannot be smaller than one left row, even when a single row
    already exceeds the target.
    """
    return max(1, PROMPTS_PER_CALL // prompts_per_row)


# One GPU container per workload, so the five run side by side. The
# split is by workload rather than by predicate because a container
# boots the 32B model once (~107 s) and then amortizes it over
# everything it judges.
WORKLOADS = tuple(dict.fromkeys(spec.workload for spec in PREDICATES))


def workload_specs(workload: str) -> tuple:
    return tuple(p for p in PREDICATES if p.workload == workload)


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


def filter_groups(specs) -> list:
    """Filter predicates grouped by the column they read, in spec order."""
    groups: dict = {}
    for spec in specs:
        if spec.kind == "filter":
            groups.setdefault((spec.left_table, spec.left_column),
                              []).append(spec)
    return [(table, tuple(members), rows_per_call(len(members)))
            for (table, _column), members in groups.items()]


def join_specs(specs) -> tuple:
    return tuple(p for p in specs if p.kind == "join")


def _load_corpus(corpus_id: str) -> tuple[Path, dict, dict]:
    """Read a corpus already materialized on the volume."""
    target = VOLUME_ROOT / "corpora" / corpus_id
    with open(target / "manifest.json") as f:
        manifest = json.load(f)
    return target, manifest, _read_rows(target)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _full_hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _named_id(prefix: str, full_hash: str) -> str:
    return f"{prefix}_{full_hash[:32]}"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def predicate_payload(spec: PredicateSpec) -> dict:
    render = ("filter_document_then_question_v1" if spec.kind == "filter"
              else "join_arg0_anchor_then_arg1_v1")
    return {
        "schema_version": SCHEMA_VERSION,
        "predicate_key": spec.key,
        "kind": spec.kind,
        "template": spec.template,
        "left_role": spec.left_role,
        "left_table": spec.left_table,
        "left_column": spec.left_column,
        "right_role": spec.right_role,
        "right_table": spec.right_table,
        "right_column": spec.right_column,
        "render": render,
        "shared_preamble": SHARED_PRE,
    }


def predicate_version(spec: PredicateSpec) -> tuple[str, str]:
    full = _full_hash(predicate_payload(spec))
    return _named_id("pv", full), full


# Only fields that can change the model's answer belong in this hash:
# it flows into JUDGE_ID, then label_set_id, then every label path, so
# any field added here invalidates all existing labels. Scheduler
# capacity knobs (max_num_batched_tokens, max_num_seqs,
# gpu_memory_utilization) change throughput and memory, not the token
# a greedy 1-token decode picks, so they stay out. They used to be in
# here, which made an out-of-memory fix cost a full relabel.
JUDGE_SPEC = {
    "model_repo": MODEL_REPO,
    "model_revision": MODEL_REVISION,
    "tokenizer_revision": MODEL_REVISION,
    "temperature": 0.0,
    "max_tokens": 1,
    "min_tokens": 1,
    "seed": data.DATA_SEED,
    "allowed_answers": ["TRUE", "FALSE"],
    "prefix_caching": True,
    "max_model_len": MAX_MODEL_LEN,
}
JUDGE_FULL_HASH = _full_hash(JUDGE_SPEC)
JUDGE_ID = _named_id("j", JUDGE_FULL_HASH)

SOURCE_SPECS = {
    "fever_annotation": {
        "dataset": "fever/fever",
        "revision": data.SOURCE_REVISIONS["fever/fever"],
        "rule": "matching evidence_wiki_url; SUPPORTS is true; REFUTES false",
    },
    "lepard_citation_edge": {
        "dataset": "rmahari/LePaRD",
        "revision": data.SOURCE_REVISIONS["rmahari/LePaRD"],
        "rule": ("anchor cited_passage_ids intersects candidate "
                 "passage_ids"),
    },
}


def label_sources(spec: PredicateSpec) -> list[dict]:
    sources = []
    if "qwen3_32b" in spec.source_policy:
        sources.append({"id": JUDGE_ID, "full_hash": JUDGE_FULL_HASH,
                        "spec": JUDGE_SPEC})
    if spec.source_policy.startswith("fever_annotation"):
        payload = SOURCE_SPECS["fever_annotation"]
        full = _full_hash(payload)
        sources.append({"id": _named_id("s", full), "full_hash": full,
                        "spec": payload})
    if spec.source_policy == "lepard_citation_edge":
        payload = SOURCE_SPECS["lepard_citation_edge"]
        full = _full_hash(payload)
        sources.append({"id": _named_id("s", full), "full_hash": full,
                        "spec": payload})
    return sources


def label_set_identity(spec: PredicateSpec, corpus_id: str,
                       corpus_full_hash: str) -> dict:
    pversion, pfull = predicate_version(spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "corpus_full_hash": corpus_full_hash,
        "predicate_key": spec.key,
        "predicate_version": pversion,
        "predicate_full_hash": pfull,
        "sources": label_sources(spec),
    }
    full = _full_hash(payload)
    return {**payload, "label_set_id": _named_id("ls", full),
            "label_set_full_hash": full}


def example_identity(corpus_id: str, operands: list[dict]) -> tuple[str, str]:
    full = _full_hash({"corpus_id": corpus_id, "operands": operands})
    return _named_id("ex", full), full


def judgment_identity(label_set_id: str, example_full_hash: str) -> str:
    full = _full_hash({"label_set_id": label_set_id,
                       "example_full_hash": example_full_hash})
    return _named_id("jd", full)


def render_filter_prompt(spec: PredicateSpec, document: str) -> str:
    return rendering.render_filter_prompt(spec.template, document)


def render_join_prompt(spec: PredicateSpec, left: str, right: str) -> str:
    return rendering.render_join_prompt(spec.template, (left, right), anchor=0)


IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
          "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail_bench")
)

data_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow")
    .add_local_python_source("quail_bench")
)

# Building the corpus reads the source datasets off HuggingFace, so it
# needs more than parquet - but not vllm. Its own image keeps
# prepare_corpus light, where data_image is only enough to read parquet
# back.
corpus_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow", "pandas", "huggingface_hub",
                 "datasets", "transformers>=5.2.0")
    .add_local_python_source("quail_bench")
)

# Experiment cells attach to this existing app so its caches remain useful.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)


# Every table here feeds corpus_id, so adding or removing one
# invalidates every label-set identity and forces a full relabel.
CORPUS_COLUMNS = {
    "reviews": ("id", "body"),
    "aspects": ("id", "aspect"),
    "reports": ("id", "report", "reactions"),
    "terms": ("id", "term"),
    "claims": ("id", "claim", "label", "evidence_wiki_url"),
    "evidence": ("id", "text"),
    "citation_contexts": ("id", "destination_context",
                          "cited_passage_ids"),
    "citation_passages": ("id", "passage_text", "passage_ids"),
    "agent_traces": ("id", "trace", "trajectory_id", "turn_index",
                     "token_count"),
}


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
    tables = {}
    for table in sorted(rows):
        row_hashes = [_full_hash(row) for row in rows[table]]
        tables[table] = {
            "rows": len(row_hashes),
            "ordered_rows_full_hash": _full_hash(row_hashes),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "quailb",
        "scale_factor": sf,
        "data_seed": data.DATA_SEED,
        "source_revisions": data.SOURCE_REVISIONS,
        "tables": tables,
    }
    full = _full_hash(payload)
    return {**payload, "corpus_id": _named_id("c", full),
            "corpus_full_hash": full}


def _materialize_corpus(sf: float) -> tuple[Path, dict, dict]:
    with tempfile.TemporaryDirectory(prefix="quailb_judge_") as temp:
        data_dir = data.build_sets(temp, sf=sf)
        rows = _read_rows(data_dir)
        identity = _corpus_identity(rows, sf)
        target = VOLUME_ROOT / "corpora" / identity["corpus_id"]
        target.mkdir(parents=True, exist_ok=True)
        for source in data_dir.glob("*.parquet"):
            destination = target / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
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
    return (VOLUME_ROOT / "label_sets" / spec.workload / spec.slug
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


def _part_bounds(spec: PredicateSpec,
                 corpus_rows: dict[str, list[dict]]) -> list[tuple[int, int]]:
    """The left-row ranges the writers split this predicate into.

    Must stay in step with _write_filter_parts, _write_qwen_join_parts
    and _write_lepard_source, which is why the batch sizes are derived
    the same way here rather than restated.
    """
    left = len(corpus_rows[spec.left_table])
    if spec.kind == "filter":
        step = next(n for _table, members, n
                    in filter_groups(workload_specs(spec.workload))
                    if spec in members)
    elif spec.source_policy == "lepard_citation_edge":
        step = 50
    else:
        step = rows_per_call(len(corpus_rows[spec.right_table]))
    return [(start, min(start + step, left))
            for start in range(0, left, step)]


def _expected_parts(spec: PredicateSpec, identity: dict,
                    corpus_rows: dict[str, list[dict]]) -> list[Path]:
    return [_part_path(spec, identity, start, end)
            for start, end in _part_bounds(spec, corpus_rows)]


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


class ModelJudge:
    def __init__(self):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from quail_bench.rendering import true_false_ids

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION)
        true_ids, false_ids = true_false_ids(self.tokenizer)
        self.true_ids = set(true_ids)
        self.false_ids = set(false_ids)
        allowed = sorted(self.true_ids | self.false_ids)
        self.llm = LLM(
            model=MODEL_REPO,
            revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
            seed=data.DATA_SEED,
            kv_cache_dtype="auto",
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_BATCH_TOKENS,
            max_num_seqs=MAX_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=True,
            disable_log_stats=True)
        self.sampling = SamplingParams(
            temperature=0.0, max_tokens=1, min_tokens=1,
            allowed_token_ids=allowed, logprobs=len(allowed),
            seed=data.DATA_SEED)
        self.boot_s = round(time.perf_counter() - t0, 2)
        self.requests = 0
        self.prompt_tokens = 0
        self.model_wall_s = 0.0

    def answer(self, prompts: list[str]) -> list[tuple[bool, int]]:
        if not prompts:
            return []
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        self.model_wall_s += time.perf_counter() - t0
        self.requests += len(outputs)
        answers = []
        for output in outputs:
            token_id = int(output.outputs[0].token_ids[0])
            if token_id in self.true_ids:
                answer = True
            elif token_id in self.false_ids:
                answer = False
            else:
                raise ValueError(f"unexpected answer token {token_id}")
            prompt_ids = getattr(output, "prompt_token_ids", None)
            if prompt_ids is not None:
                self.prompt_tokens += len(prompt_ids)
            answers.append((answer, token_id))
        return answers


class VerificationSample:
    def __init__(self):
        self.rows: dict[str, list[tuple[str, bool]]] = {}

    def add(self, predicate_key: str, prompt: str, answer: bool) -> None:
        sample = self.rows.setdefault(predicate_key, [])
        if len(sample) < VERIFY_PER_PREDICATE:
            sample.append((prompt, answer))

    def run(self, judge: ModelJudge) -> dict:
        flat = [(key, prompt, answer)
                for key, values in sorted(self.rows.items())
                for prompt, answer in values]
        reversed_flat = list(reversed(flat))
        rerun = judge.answer([prompt for _, prompt, _ in reversed_flat])
        flips = sum(
            got != expected
            for (_, _, expected), (got, _token) in zip(reversed_flat, rerun))
        return {"compared": len(flat), "answer_differences": flips,
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
                if spec.kind == "filter":
                    prompt = render_filter_prompt(
                        spec, left[spec.left_column])
                else:
                    right = rows_by_table[spec.right_table][saved["right_id"]]
                    prompt = render_join_prompt(
                        spec, left[spec.left_column], right[spec.right_column])
                verification.add(spec.key, prompt, bool(saved["answer"]))
                if (len(verification.rows[spec.key])
                        == VERIFY_PER_PREDICATE):
                    break
            if (len(verification.rows.get(spec.key, []))
                    == VERIFY_PER_PREDICATE):
                break
    return verification


def _write_filter_parts(judge: ModelJudge, verification: VerificationSample,
                        rows: list[dict], specs: list[PredicateSpec],
                        identities: dict[str, dict], corpus_id: str,
                        batch_rows: int) -> None:
    for start in range(0, len(rows), batch_rows):
        end = min(start + batch_rows, len(rows))
        missing = [spec for spec in specs
                   if not _part_path(spec, identities[spec.key],
                                     start, end).exists()]
        if not missing:
            continue
        prompts = []
        cases = []
        for row in rows[start:end]:
            for spec in missing:
                prompt = render_filter_prompt(spec, row[spec.left_column])
                prompts.append(prompt)
                cases.append((spec, row, prompt))
        answers = judge.answer(prompts)
        by_predicate = {spec.key: [] for spec in missing}
        for (spec, row, prompt), (answer, token_id) in zip(cases, answers):
            by_predicate[spec.key].append(_answer_row(
                spec, identities[spec.key], corpus_id, row, None,
                answer, MODEL_NAME, token_id))
            verification.add(spec.key, prompt, answer)
        for spec in missing:
            _atomic_parquet(
                _part_path(spec, identities[spec.key], start, end),
                by_predicate[spec.key])
        results_vol.commit()
        print(f"[judge] filters {specs[0].workload} rows {start}:{end}",
              flush=True)


def _write_qwen_join_parts(judge: ModelJudge,
                           verification: VerificationSample,
                           spec: PredicateSpec, left_rows: list[dict],
                           right_rows: list[dict], identity: dict,
                           corpus_id: str, source_label=None) -> None:
    anchor_batch = rows_per_call(len(right_rows))
    for start in range(0, len(left_rows), anchor_batch):
        end = min(start + anchor_batch, len(left_rows))
        part = _part_path(spec, identity, start, end)
        if part.exists():
            continue
        prompts = []
        qwen_cases = []
        source_rows = []
        for left in left_rows[start:end]:
            for right in right_rows:
                known = source_label(left, right) if source_label else None
                if known is not None:
                    label, source = known
                    source_rows.append(_answer_row(
                        spec, identity, corpus_id, left, right,
                        label, source, None))
                    continue
                prompt = render_join_prompt(
                    spec, left[spec.left_column], right[spec.right_column])
                prompts.append(prompt)
                qwen_cases.append((left, right, prompt))
        answers = judge.answer(prompts)
        output_rows = list(source_rows)
        for (left, right, prompt), (answer, token_id) in zip(
                qwen_cases, answers):
            output_rows.append(_answer_row(
                spec, identity, corpus_id, left, right,
                answer, MODEL_NAME, token_id))
            verification.add(spec.key, prompt, answer)
        output_rows.sort(key=lambda row: (row["left_id"], row["right_id"]))
        _atomic_parquet(part, output_rows)
        results_vol.commit()
        print(f"[judge] join {spec.workload} anchors {start}:{end}, "
              f"qwen={len(qwen_cases)}, source={len(source_rows)}",
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
        results_vol.commit()
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
        VOLUME_ROOT / "corpora" / corpus_id / "active_collection.json",
        {"collection_id": collection_id})


def _required_tables(spec: PredicateSpec) -> tuple[str, ...]:
    tables = {spec.left_table}
    if spec.kind == "join":
        tables.add(spec.right_table)
    return tuple(sorted(tables))


def _label_manifest(label_set_id: str) -> tuple[Path, dict]:
    matches = list((VOLUME_ROOT / "label_sets").glob(
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
        VOLUME_ROOT / "corpora" / str(label_corpus_id) / "manifest.json")
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
    return (VOLUME_ROOT / "label_sets" / spec.workload / spec.slug
            / label_set_id)


@app.function(
    image=data_image, memory=4096, timeout=1200,
    volumes={"/results": results_vol})
def compact_ground_truth(collection_id: str) -> str:
    results_vol.reload()
    collection_path = (VOLUME_ROOT / "collections" / collection_id
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
        identity = {"label_set_id": label_set_id}
        compact_path, rows = _compact_label_parts(
            label_dir, _expected_parts(spec, identity, corpus_rows))
        manifest_path = label_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["compact_path"] = str(compact_path)
        manifest["compact_rows"] = rows
        _atomic_json(manifest_path, manifest)
        compacted[key] = {"path": str(compact_path), "rows": rows}
    results_vol.commit()
    result = {
        "collection_id": collection_id,
        "label_sets": len(compacted),
        "rows": sum(item["rows"] for item in compacted.values()),
        "compacted": compacted,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


def _fever_source_label(claim: dict, passage: dict):
    if claim["evidence_wiki_url"] != passage["id"]:
        return None
    return claim["label"] == "SUPPORTS", "fever_annotation"


@app.function(
    image=corpus_image, memory=4096, timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/results": results_vol})
def prepare_corpus(sf: float = SCALE_FACTOR) -> str:
    """Build the corpus and check its collection on the volume.

    Reports whether the collection the corpus implies is already
    complete.
    """
    if sf != SCALE_FACTOR:
        raise ValueError("the ground-truth pass is fixed at sf=0.1")
    results_vol.reload()
    _, corpus_manifest, _ = _materialize_corpus(sf)
    results_vol.commit()
    identities = {
        spec.key: label_set_identity(
            spec, corpus_manifest["corpus_id"],
            corpus_manifest["corpus_full_hash"])
        for spec in PREDICATES
    }
    collection = _collection_identity(corpus_manifest, identities)
    path = (VOLUME_ROOT / "collections" / collection["collection_id"]
            / "manifest.json")
    complete = False
    if path.exists():
        with open(path) as f:
            complete = json.load(f).get("status") == "complete"
    print(f"[judge] corpus {corpus_manifest['corpus_id']}, collection "
          f"{collection['collection_id']}, complete={complete}", flush=True)
    return json.dumps({"corpus": corpus_manifest,
                       "collection_id": collection["collection_id"],
                       "complete": complete}, sort_keys=True)


@app.function(
    image=image, gpu="H100!", memory=98304, timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/root/.cache/kernels": kernel_cache,
             "/results": results_vol})
def judge_workload(corpus_id: str, workload: str) -> str:
    """Label one workload's predicates on one GPU.

    Every part file is written under a content-addressed path and
    skipped when it already exists, so a container that dies part way
    resumes where it stopped.
    """
    specs = workload_specs(workload)
    if not specs:
        raise ValueError(f"no predicates for workload {workload!r}")
    t_total = time.perf_counter()
    results_vol.reload()
    _, corpus_manifest, rows = _load_corpus(corpus_id)
    identities = {
        spec.key: label_set_identity(
            spec, corpus_manifest["corpus_id"],
            corpus_manifest["corpus_full_hash"])
        for spec in specs
    }
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
    results_vol.commit()

    t_boot = time.perf_counter()
    judge = ModelJudge()
    boot_s = time.perf_counter() - t_boot
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
    results_vol.commit()

    saved = _saved_verification_sample(rows, identities, specs)
    deterministic = saved.run(judge)
    kernel_cache.commit()
    partial = {
        "workload": workload,
        "manifests": manifests,
        "boot_s": round(boot_s, 2),
        "model_wall_s": round(judge.model_wall_s, 2),
        "total_wall_s": round(time.perf_counter() - t_total, 2),
        "prompt_tokens_submitted_this_call": judge.prompt_tokens,
        "model_requests_this_call_including_verification": judge.requests,
        "deterministic_rerun": deterministic,
    }
    print(f"[judge] {workload} done in {partial['total_wall_s']:.1f}s",
          flush=True)
    return json.dumps(partial, sort_keys=True)


@app.function(
    image=data_image, memory=4096, timeout=1200,
    volumes={"/results": results_vol})
def finalize_collection(sf: float, corpus_id: str, partials: str) -> str:
    """Assemble five workloads and activate their ground truth."""
    results_vol.reload()
    _, corpus_manifest, _ = _load_corpus(corpus_id)
    identities = {
        spec.key: label_set_identity(
            spec, corpus_manifest["corpus_id"],
            corpus_manifest["corpus_full_hash"])
        for spec in PREDICATES
    }
    collection = _collection_identity(corpus_manifest, identities)
    collection_dir = (VOLUME_ROOT / "collections"
                      / collection["collection_id"])

    by_workload = json.loads(partials)
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
        "prediction": PREDICTION_TEXT,
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
                "prompt_tokens_submitted_this_call",
                "model_requests_this_call_including_verification",
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
            VOLUME_ROOT / "corpora" / corpus_id / "manifest.json"),
        "summary": summary})
    _atomic_json(collection_dir / "summary.json", summary)
    _activate_collection(corpus_manifest["corpus_id"],
                         collection["collection_id"])
    results_vol.commit()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return json.dumps(summary, sort_keys=True)


@app.function(
    image=data_image, memory=4096, timeout=1200,
    volumes={"/results": results_vol})
def activate_reused_collection(
        sf: float, target_corpus_id: str, source_collection_id: str,
        relabeled_workloads: str) -> str:
    """Build one collection from new labels and verified unchanged tables."""
    results_vol.reload()
    _, target_corpus, _ = _load_corpus(target_corpus_id)
    source_collection_path = (
        VOLUME_ROOT / "collections" / source_collection_id / "manifest.json")
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
    with open(VOLUME_ROOT / "corpora" / source_corpus_id / "manifest.json") as f:
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

    identities = {}
    manifests = {}
    reused = {}
    for spec in PREDICATES:
        if spec.workload in names:
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
        VOLUME_ROOT / "collections" / collection["collection_id"])
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
            VOLUME_ROOT / "corpora" / target_corpus_id / "manifest.json"),
        "reused_label_sets": reused,
        "summary": summary,
    }
    _atomic_json(collection_dir / "manifest.json", collection_manifest)
    _atomic_json(collection_dir / "summary.json", summary)
    _activate_collection(target_corpus["corpus_id"],
                         collection["collection_id"])
    results_vol.commit()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return json.dumps(summary, sort_keys=True)


@app.local_entrypoint()
def main(sf: float = SCALE_FACTOR, compact_collection: str | None = None,
         only: str | None = None, finalize_from: str | None = None,
         reuse_from_collection: str | None = None,
         relabeled_workloads: str = "lepard",
         target_corpus: str | None = None):
    """Run the five workloads side by side, then activate the result.

    ``--only imdb,fever`` restricts the pass to those workloads; the
    finalize step is skipped because a collection needs all label sets.
    """
    if compact_collection:
        call = compact_ground_truth.spawn(compact_collection)
        print(f"function call id: {call.object_id}", flush=True)
        print(call.get(), flush=True)
        return
    prediction = (REUSE_PREDICTION_TEXT if reuse_from_collection
                  else PREDICTION_TEXT)
    print(f"PREDICTION: {prediction}", flush=True)

    if reuse_from_collection and target_corpus:
        corpus_id = target_corpus
        prepared = None
    else:
        call = prepare_corpus.spawn(sf)
        print(f"function call id (prepare_corpus): {call.object_id}",
              flush=True)
        prepared = json.loads(call.get())
        corpus_id = prepared["corpus"]["corpus_id"]
    if reuse_from_collection:
        call = activate_reused_collection.spawn(
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
        call = finalize_collection.spawn(
            sf, corpus_id, json.dumps(partials))
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

    calls = {w: judge_workload.spawn(corpus_id, w) for w in names}
    for w, c in calls.items():
        print(f"function call id (judge_workload {w}): {c.object_id}",
              flush=True)
    partials = {}
    for w, c in calls.items():
        partials[w] = json.loads(c.get())
        print(f"[main] {w} finished in "
              f"{partials[w]['total_wall_s']:.1f}s", flush=True)

    if only:
        print("--only was given, so the collection is not finalized",
              flush=True)
        return
    call = finalize_collection.spawn(sf, corpus_id, json.dumps(partials))
    print(f"function call id (finalize_collection): {call.object_id}",
          flush=True)
    print(call.get(), flush=True)
