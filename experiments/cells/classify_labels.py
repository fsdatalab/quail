"""Label the QUAIL-B classifications with stock vLLM and add them to collections.

    uv run modal run -m experiments.cells.classify_labels::check \
        2>&1 | tee /tmp/classify-check.log
    uv run modal run --detach -m experiments.cells.classify_labels::label \
        2>&1 | tee /tmp/classify-label.log

`label` waits for every shard, then builds the collections. To rebuild
them from finished shards alone:

    uv run modal run --detach -m experiments.cells.classify_labels::finish \
        --calls fc-...,fc-... 2>&1 | tee /tmp/classify-finish.log

A document's reference label is the label with the largest sum of
label-token log probabilities (quail_b.predicates.CLASSIFY_JUDGE_SPEC).
vLLM scores it with one request per shared label prefix: the prompt plus
that prefix, asking for the log probabilities of the tokens that can
follow it. The first request of each document fills the prefix cache, so
the others reuse the document and category list.

Labels are computed at sf=1.0 and copied by document content to sf=0.5
and sf=0.1. Label sets and collections stay on the quail-results volume;
the summary is /results/ablations/classify-reference-labels.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import modal

from quail.bench import labeling as labels
from quail_b.data import PUBLISHED_CORPORA
from quail_b.predicates import (
    CLASSIFY_PREDICATES,
    MAX_MODEL_LEN,
    MODEL_NAME,
    MODEL_REPO,
    MODEL_REVISION,
    PREDICATE_BY_KEY,
    PredicateSpec,
    example_identity,
    judgment_identity,
    predicate_payload,
    render_classify_prompt,
)
from quail_b.predicates import label_set_identity as _label_set_identity
from quail_b.rendering import LABEL_PREFIX

app = labels.app
SPECS = CLASSIFY_PREDICATES
ROWS_PER_PART = 256
# vLLM refuses more requested token ids than this per request
MAX_LOGPROB_TOKEN_IDS = 128
# share of the KV capacity one batch of prompts may occupy, so a
# document's first request stays cached until its other requests run
KV_BATCH_SHARE = 0.5
# containers per predicate; the agent traces average 9,815 tokens
SHARDS = {
    "quailb.imdb.review.sentiment": 2,
    "quailb.imdb.review.genre": 2,
    "quailb.imdb.review.main_complaint": 2,
    "quailb.biodex.reaction.organ_class": 1,
    "quailb.fever.claim.topic": 1,
    "quailb.lepard.excerpt.area_of_law": 1,
    "quailb.agent.trace.outcome": 8,
    "quailb.agent.trace.failure_mode": 8,
}
# the published collections the new collections extend
SOURCES = {
    0.1: "gt_cd3ebdb784f64b9e028e50ea73cdedd0",
    0.5: "gt_68f9ce9439bd7615de92b33d576dff9e",
    1.0: "gt_e87691add604b02c4e43f0ff5bf0cc4f",
}
SUMMARY_PATH = Path("/results/ablations/classify-reference-labels.json")
CHECK_PATH = Path("/results/ablations/classify-reference-labels-check.json")
PREDICTION_TEXT = (
    "At sf=1.0 the eight predicates need 435 million prompt tokens, 364 "
    "million of them agent traces. At the 17,000 fresh tokens per second "
    "the cross-check measured on reviews "
    "(/results/ablations/classify-reference-labels-check.json), that is "
    "about 7 H100 hours: under 30 minutes of scoring per agent-trace "
    "container after a 6-minute boot, across 25 containers. Every label "
    "set gets one label per document, and every classification query "
    "loads its labels at all three scale factors."
)


def label_trie(label_ids: list[tuple[int, ...]]) -> dict:
    """Return each proper label prefix and the tokens that can follow it.

    Args:
        label_ids: The labels' token ids, in label order.

    Returns:
        Prefix to sorted next-token ids, in first-seen order; the empty
        prefix comes first.
    """
    children: dict[tuple[int, ...], set[int]] = {}
    for ids in label_ids:
        if not ids:
            raise ValueError("a label has no tokens")
        for depth in range(len(ids)):
            children.setdefault(tuple(ids[:depth]), set()).add(ids[depth])
    return {prefix: sorted(tokens) for prefix, tokens in children.items()}


def score_labels(label_ids: list[tuple[int, ...]],
                 logprobs: dict) -> tuple[int, list[float]]:
    """Return the winning label index and every label's score.

    Args:
        label_ids: The labels' token ids, in label order.
        logprobs: Prefix to {next token id: log probability}.

    Returns:
        (winner, scores): the smallest index of the largest summed log
        probability, and each label's sum.
    """
    scores = [
        sum(logprobs[tuple(ids[:depth])][ids[depth]]
            for depth in range(len(ids)))
        for ids in label_ids
    ]
    best = max(scores)
    return scores.index(best), scores


def part_bounds(rows: int) -> list[tuple[int, int]]:
    return [(start, min(start + ROWS_PER_PART, rows))
            for start in range(0, rows, ROWS_PER_PART)]


def shard_bounds(rows: int, shards: int) -> list[tuple[int, int]]:
    """Split rows into up to `shards` ranges of whole parts."""
    parts = part_bounds(rows)
    per_shard = -(-len(parts) // shards)
    return [(parts[i][0], parts[min(i + per_shard, len(parts)) - 1][1])
            for i in range(0, len(parts), per_shard)]


def classify_identity(spec: PredicateSpec, corpus: dict) -> dict:
    identity = _label_set_identity(
        spec, corpus["corpus_id"], corpus["corpus_full_hash"])
    return {**identity, "rows_per_part": ROWS_PER_PART}


def corpus_rows(sf: float, spec: PredicateSpec) -> tuple[dict, list[dict]]:
    """Read one saved corpus's manifest and the predicate's table rows."""
    import pyarrow.parquet as pq

    directory = labels.ROOT / "corpora" / PUBLISHED_CORPORA[sf]
    manifest = json.loads((directory / "manifest.json").read_text())
    table = pq.read_table(directory / f"{spec.left_table}.parquet",
                          columns=["id", spec.left_column])
    return manifest, table.to_pylist()


def _label_row(spec, identity, corpus_id, row, label, scores) -> dict:
    operand = labels._operand(spec.left_role, spec.left_table, row,
                              spec.left_column)
    example_id, example_full = example_identity(corpus_id, [operand])
    return {
        "judgment_id": judgment_identity(identity["label_set_id"],
                                           example_full),
        "example_id": example_id,
        "example_full_hash": example_full,
        "label_set_id": identity["label_set_id"],
        "predicate_key": spec.key,
        "predicate_version": identity["predicate_version"],
        "label": label,
        "label_scores": list(scores),
        "label_source": MODEL_NAME,
        "left_role": spec.left_role,
        "left_table": spec.left_table,
        "left_id": str(row["id"]),
        "left_content_sha256": operand["content_sha256"],
        "right_role": None,
        "right_table": None,
        "right_id": None,
        "right_content_sha256": None,
    }


def write_part(path: Path, rows: list[dict]) -> None:
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("judgment_id", pa.string()),
        ("example_id", pa.string()),
        ("example_full_hash", pa.string()),
        ("label_set_id", pa.string()),
        ("predicate_key", pa.string()),
        ("predicate_version", pa.string()),
        ("label", pa.string()),
        ("label_scores", pa.list_(pa.float64())),
        ("label_source", pa.string()),
        ("left_role", pa.string()),
        ("left_table", pa.string()),
        ("left_id", pa.string()),
        ("left_content_sha256", pa.string()),
        ("right_role", pa.string()),
        ("right_table", pa.string()),
        ("right_id", pa.string()),
        ("right_content_sha256", pa.string()),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temp,
                   compression="zstd")
    os.replace(temp, path)


class VLLMJudge:
    """A model on stock vLLM scoring label prefixes; Qwen3 32B fp8 by default."""

    def __init__(self, repo: str = MODEL_REPO, revision: str = MODEL_REVISION):
        from vllm import LLM

        from quail.backends.vllm import _capacity

        started = time.perf_counter()
        self.llm = LLM(
            model=repo, revision=revision,
            tokenizer_revision=revision, max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=True, gpu_memory_utilization=0.9,
            max_logprobs=MAX_LOGPROB_TOKEN_IDS, disable_log_stats=True,
            seed=0)
        self.boot_s = time.perf_counter() - started
        self.capacity = _capacity(self.llm)
        self.tokenizer = self.llm.get_tokenizer()
        self.prompt_tokens = 0
        self.requests = 0
        self.model_wall_s = 0.0

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def label_ids(self, spec: PredicateSpec) -> list[tuple[int, ...]]:
        return [tuple(self.encode(LABEL_PREFIX + label))
                for label in spec.labels]

    def _next_logprobs(self, prompts, prefixes, trie):
        from vllm import SamplingParams

        params = [SamplingParams(
            max_tokens=1, temperature=0.0, detokenize=False,
            logprob_token_ids=trie[prefix]) for prefix in prefixes]
        started = time.perf_counter()
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        self.model_wall_s += time.perf_counter() - started
        self.requests += len(prompts)
        self.prompt_tokens += sum(len(p["prompt_token_ids"]) for p in prompts)
        found = []
        for output, prefix in zip(outputs, prefixes):
            logprobs = output.outputs[0].logprobs[0]
            found.append({token: logprobs[token].logprob
                          for token in trie[prefix]})
        return found

    def label(self, spec: PredicateSpec, documents: list[str]
              ) -> list[tuple[str, list[float]]]:
        """Return each document's label and every label's score."""
        label_ids = self.label_ids(spec)
        trie = label_trie(label_ids)
        if max(map(len, trie.values())) > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(f"{spec.key} has too many labels after a prefix")
        contexts = [self.encode(render_classify_prompt(spec, document))
                    for document in documents]
        longest = max(map(len, label_ids))
        if max(map(len, contexts)) + longest > MAX_MODEL_LEN:
            raise ValueError(f"{spec.key}: a prompt exceeds the context")
        budget = int(self.capacity["kv_cache_size_tokens"] * KV_BATCH_SHARE)
        results = []
        start = 0
        while start < len(contexts):
            end, used = start, 0
            while end < len(contexts) and (
                    end == start or used + len(contexts[end]) <= budget):
                used += len(contexts[end])
                end += 1
            batch = contexts[start:end]
            tables = [{} for _ in batch]
            # the empty prefix first: it fills the cache with each context
            for table, found in zip(tables, self._next_logprobs(
                    [{"prompt_token_ids": context} for context in batch],
                    [()] * len(batch), trie)):
                table[()] = found
            rest = [prefix for prefix in trie if prefix]
            if rest:
                prompts, prefixes, owners = [], [], []
                for index, context in enumerate(batch):
                    for prefix in rest:
                        prompts.append(
                            {"prompt_token_ids": context + list(prefix)})
                        prefixes.append(prefix)
                        owners.append(index)
                for index, prefix, found in zip(
                        owners, prefixes,
                        self._next_logprobs(prompts, prefixes, trie)):
                    tables[index][prefix] = found
            for table in tables:
                winner, scores = score_labels(label_ids, table)
                results.append((spec.labels[winner], scores))
            start = end
        return results

    def sequence_scores(self, spec: PredicateSpec, document: str
                        ) -> list[float]:
        """Every label's score from full-sequence prompt log probabilities."""
        from vllm import SamplingParams

        context = self.encode(render_classify_prompt(spec, document))
        label_ids = self.label_ids(spec)
        prompts = [{"prompt_token_ids": context + list(ids)}
                   for ids in label_ids]
        outputs = self.llm.generate(
            prompts, SamplingParams(max_tokens=1, prompt_logprobs=0,
                                    detokenize=False), use_tqdm=False)
        scores = []
        for ids, output in zip(label_ids, outputs):
            rows = output.prompt_logprobs[len(context):]
            scores.append(sum(row[token].logprob
                              for row, token in zip(rows, ids)))
        return scores


def _volumes() -> dict:
    return {"/root/.cache/huggingface": labels.hf_cache,
            "/root/.cache/kernels": labels.kernel_cache,
            "/results": labels.results_vol}


@app.function(image=labels.image, gpu="H100!", memory=98304, timeout=3600,
              volumes=_volumes())
def check_run(documents_per_predicate: int = 12,
              throughput_documents: int = 256) -> dict:
    """Compare prefix scoring with full-sequence scoring and time a batch."""
    labels._mount()
    judge = VLLMJudge()
    checks = {}
    for key in ("quailb.imdb.review.main_complaint",
                "quailb.agent.trace.failure_mode",
                "quailb.biodex.reaction.organ_class"):
        spec = PREDICATE_BY_KEY[key]
        _, rows = corpus_rows(1.0, spec)
        documents = [row[spec.left_column] for row in rows[:500]]
        if spec.left_table == "agent_traces":
            # full-sequence prompt logprobs hold every row's vocabulary
            # scores, so the long traces are checked on the shortest few
            documents = sorted(documents, key=len)[:4]
        documents = documents[:documents_per_predicate]
        labeled = judge.label(spec, documents)
        differences, agree = [], 0
        for document, (label, scores) in zip(documents, labeled):
            reference = judge.sequence_scores(spec, document)
            differences.append(max(abs(a - b)
                                   for a, b in zip(scores, reference)))
            best = max(reference)
            agree += spec.labels[reference.index(best)] == label
        checks[key] = {
            "documents": len(documents),
            "max_abs_score_difference": max(differences),
            "same_label": agree,
            "labels": [label for label, _ in labeled],
        }
    spec = PREDICATE_BY_KEY["quailb.imdb.review.main_complaint"]
    _, rows = corpus_rows(1.0, spec)
    documents = [row[spec.left_column] for row in rows[:throughput_documents]]
    tokens, wall = judge.prompt_tokens, judge.model_wall_s
    judge.label(spec, documents)
    result = {
        "prediction": PREDICTION_TEXT,
        "model": MODEL_NAME,
        "boot_s": judge.boot_s,
        "capacity": judge.capacity,
        "checks": checks,
        "throughput": {
            "predicate": spec.key,
            "documents": len(documents),
            "requested_prompt_tokens": judge.prompt_tokens - tokens,
            "model_wall_s": judge.model_wall_s - wall,
        },
    }
    labels._atomic_json(CHECK_PATH, result)
    labels.results_vol.commit()
    return result


QUAIL_CHECK_PATH = Path("/results/ablations/classify-quail-same-model.json")
QUAIL_CHECK_PREDICTION_TEXT = (
    "Quail and stock vLLM run the same Qwen3 4B checkpoint, prompt text, "
    "and label scoring. Quail tokenizes the preamble, document, and "
    "question separately and vLLM tokenizes the prompt as one string, and "
    "both compute bf16 logits, so a document whose two best labels are "
    "within about 0.1 nats can differ. Expect at least 99% of labels to "
    "agree, every disagreement inside that margin."
)


@app.function(image=labels.image, gpu="H100!", memory=98304, timeout=3600,
              volumes=_volumes())
def quail_check_run(run_dir: str, query_ids: str, sf: float = 0.1) -> dict:
    """Label a Quail run's classified documents with vLLM and the same model.

    Args:
        run_dir: The QUAIL-B run directory on the volume, under /results.
        query_ids: Comma-separated query ids the run classified.
        sf: The run's scale factor.
    """
    import pyarrow.parquet as pq

    import quail_b
    from quail.specs import MODELS

    labels._mount()
    model = MODELS["qwen3-4b-fp8"]
    judge = VLLMJudge(model.hf_name, model.revision)
    by_template = {spec.template: spec for spec in SPECS}
    checks = {}
    for query_id in query_ids.split(","):
        classifies = quail_b.get_query(query_id)._info.classifies
        for index, operator in enumerate(classifies):
            (path,) = Path(run_dir, "quail").glob(
                f"*/{query_id}/classifications-{index}.parquet")
            quail_labels = pq.read_table(path).to_pylist()
            spec = by_template[operator.prompt]
            _, rows = corpus_rows(sf, spec)
            text = {row["id"]: row[spec.left_column] for row in rows}
            alias = operator.relation
            ids = [row[alias] for row in quail_labels]
            labeled = judge.label(spec, [text[doc] for doc in ids])
            disagreements = []
            for doc, row, (label, scores) in zip(ids, quail_labels, labeled):
                if row["label"] == label:
                    continue
                disagreements.append({
                    "id": doc, "quail": row["label"], "vllm": label,
                    "vllm_margin": max(scores)
                    - scores[spec.labels.index(row["label"])],
                })
            checks[f"{query_id}:{operator.id}"] = {
                "predicate": spec.key,
                "documents": len(ids),
                "same_label": len(ids) - len(disagreements),
                "largest_margin": max(
                    (item["vllm_margin"] for item in disagreements),
                    default=0.0),
                "disagreements": disagreements,
            }
    result = {"prediction": QUAIL_CHECK_PREDICTION_TEXT, "run_dir": run_dir,
              "model": model.name, "sf": sf, "checks": checks}
    labels._atomic_json(QUAIL_CHECK_PATH, result)
    labels.results_vol.commit()
    return result


@app.function(image=labels.image, gpu="H100!", memory=98304, timeout=86400,
              volumes=_volumes())
def judge_shard(key: str, start: int, end: int) -> dict:
    """Write the label parts of one predicate's rows [start, end) at sf=1.0."""
    labels._mount()
    spec = PREDICATE_BY_KEY[key]
    manifest, rows = corpus_rows(1.0, spec)
    identity = classify_identity(spec, manifest)
    started = time.perf_counter()
    judge = VLLMJudge()
    written = skipped = 0
    for part_start, part_end in part_bounds(len(rows)):
        if part_start < start or part_end > end:
            continue
        path = labels._part_path(spec, identity, part_start, part_end)
        if path.exists():
            skipped += 1
            continue
        part = rows[part_start:part_end]
        labeled = judge.label(spec, [row[spec.left_column] for row in part])
        write_part(path, [
            _label_row(spec, identity, manifest["corpus_id"], row, label,
                       scores)
            for row, (label, scores) in zip(part, labeled)])
        labels.results_vol.commit()
        written += 1
        print(f"[classify] {key} {part_start}-{part_end}: "
              f"{judge.prompt_tokens / max(judge.model_wall_s, 1e-9):,.0f} "
              "requested tokens/s", flush=True)
    return {
        "key": key, "start": start, "end": end,
        "label_set_id": identity["label_set_id"],
        "parts_written": written, "parts_skipped": skipped,
        "requests": judge.requests,
        "requested_prompt_tokens": judge.prompt_tokens,
        "model_wall_s": judge.model_wall_s,
        "boot_s": judge.boot_s,
        "total_wall_s": time.perf_counter() - started,
    }


def complete_manifest(spec: PredicateSpec, identity: dict,
                      rows: int) -> dict:
    """Compact one label set's parts and write its manifest."""
    import pyarrow.parquet as pq

    label_dir = labels._label_dir(spec, identity)
    parts = [labels._part_path(spec, identity, start, end)
             for start, end in part_bounds(rows)]
    compact_path, compact_rows = labels._compact_label_parts(label_dir, parts)
    if compact_rows != rows:
        raise ValueError(f"{spec.key}: saved {compact_rows} rows, expected {rows}")
    counts = {label: 0 for label in spec.labels}
    for label in pq.read_table(compact_path, columns=["label"])[
            "label"].to_pylist():
        counts[label] += 1
    manifest = {
        **identity,
        "status": "complete",
        "predicate": asdict(spec),
        "predicate_payload": predicate_payload(spec),
        "expected_rows": rows,
        "compact_path": str(compact_path),
        "compact_rows": compact_rows,
        "rows": rows,
        "label_rows": counts,
        "source_rows": {MODEL_NAME: rows},
    }
    labels._atomic_json(label_dir / "manifest.json", manifest)
    return manifest


def copy_labels(sf: float, spec: PredicateSpec, source: dict) -> dict:
    """Copy sf=1.0 labels by document content into a smaller corpus."""
    import pyarrow.parquet as pq

    manifest, rows = corpus_rows(sf, spec)
    identity = classify_identity(spec, manifest)
    table = pq.read_table(source["compact_path"], columns=[
        "left_content_sha256", "label", "label_scores"])
    by_content = dict(zip(table["left_content_sha256"].to_pylist(),
                          zip(table["label"].to_pylist(),
                              table["label_scores"].to_pylist())))
    for start, end in part_bounds(len(rows)):
        path = labels._part_path(spec, identity, start, end)
        if path.exists():
            continue
        output = []
        for row in rows[start:end]:
            label, scores = by_content[
                labels._content_hash(row, spec.left_column)]
            output.append(_label_row(spec, identity, manifest["corpus_id"],
                                     row, label, scores))
        write_part(path, output)
    return complete_manifest(spec, identity, len(rows))


def check_collection(sf: float, collection_id: str) -> dict:
    """Load each classification query's labels and count its reference rows."""
    import quail_b as benchmark
    from quail_b.queries import queries
    from quail_b.scoring import expected_rows

    counts = {}
    for query_id in (query_id for query_id, spec in queries().items()
                     if spec._info.classifies):
        suite = benchmark.load_benchmark(
            [query_id], scale_factor=sf, collection_id=collection_id,
            root="/results")
        spec = suite.queries[0]
        counts[query_id] = {
            "reference_output_rows": expected_rows(
                spec, suite.ground_truth, suite.tables).num_rows,
            "input_documents": {
                relation.alias: suite.tables[relation.table].num_rows
                for relation in spec._info.relations},
        }
    return counts


@app.function(image=labels.publish_image, memory=65536, timeout=4 * 3600,
              volumes={"/results": labels.results_vol})
def finish_run(calls: str) -> dict:
    """Compact, copy to smaller corpora, and build extended collections."""
    shard_results = [modal.FunctionCall.from_id(call.strip()).get()
                     for call in calls.split(",") if call.strip()]
    labels._mount()
    source = {}
    for spec in SPECS:
        manifest, rows = corpus_rows(1.0, spec)
        source[spec.key] = complete_manifest(
            spec, classify_identity(spec, manifest), len(rows))
    labels.results_vol.commit()
    collections = {}
    for sf in (1.0, 0.5, 0.1):
        if sf != 1.0:
            for spec in SPECS:
                copy_labels(sf, spec, source[spec.key])
            labels.results_vol.commit()
        summary = labels.activate_reused_collection(
            sf, PUBLISHED_CORPORA[sf], SOURCES[sf], "", new_specs=SPECS)
        labels.results_vol.commit()
        summary["classify_queries"] = check_collection(
            sf, summary["collection_id"])
        collections[str(sf)] = summary
    result = {
        "prediction": PREDICTION_TEXT,
        "shards": shard_results,
        "label_sets": {key: {"label_set_id": m["label_set_id"],
                             "label_rows": m["label_rows"]}
                       for key, m in source.items()},
        "collections": collections,
        "result_volume_path": str(SUMMARY_PATH),
    }
    labels._atomic_json(SUMMARY_PATH, result)
    labels.results_vol.commit()
    return result


@app.local_entrypoint()
def check():
    """Run the cross-check and throughput cell."""
    print(PREDICTION_TEXT, flush=True)
    call = check_run.spawn()
    print(f"[classify] check function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)


@app.local_entrypoint()
def quail_check(run_dir: str, query_ids: str, sf: float = 0.1):
    """Compare a Quail run's labels with stock vLLM on the same model."""
    print(QUAIL_CHECK_PREDICTION_TEXT, flush=True)
    call = quail_check_run.spawn(run_dir, query_ids, sf)
    print(f"[classify] quail check function call id: {call.object_id}",
          flush=True)
    result = call.get()
    for name, check in result["checks"].items():
        print(f"[classify] {name}: {check['same_label']}/{check['documents']} "
              f"same, largest vLLM margin {check['largest_margin']:.4f}",
              flush=True)
    print(f"[classify] saved {QUAIL_CHECK_PATH}", flush=True)


@app.local_entrypoint()
def label():
    """Label every shard, one container each, then build the collections."""
    print(PREDICTION_TEXT, flush=True)
    rows = {"reviews": 50_000, "terms": 4_144, "claims": 5_000,
            "citation_contexts": 4_972, "agent_traces": 17_711}
    calls = []
    for spec in SPECS:
        for start, end in shard_bounds(rows[spec.left_table], SHARDS[spec.key]):
            call = judge_shard.spawn(spec.key, start, end)
            calls.append(call.object_id)
            print(f"[classify] {spec.key} [{start}, {end}): {call.object_id}",
                  flush=True)
    print(f"[classify] calls: {','.join(calls)}", flush=True)
    for call_id in calls:
        result = modal.FunctionCall.from_id(call_id).get()
        print(f"[classify] done {call_id}: {json.dumps(result)}", flush=True)
    call = finish_run.spawn(",".join(calls))
    print(f"[classify] finish function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)


@app.local_entrypoint()
def finish(calls: str):
    """Build the collections once every shard has finished."""
    call = finish_run.spawn(calls)
    print(f"[classify] finish function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
