"""Compact recorded OpenHands traces with a Quail semantic join.

Run from the repository root. The demo uses DiffusionGemma on Modal H100 GPUs:

    uv run modal run --detach demos/agent_trace_compaction.py \
      --limit 100 --seed 42 \
      2>&1 | tee /tmp/quail-agent-compaction.log

Sample one trajectory per issue with a fixed seed. --limit counts issues.
Each selected complete trajectory is the conversation at compaction time.
The Python adaptation of fast-jev-compaction prepares the state and questions
and reconstructs messages. Quail replaces Jev's probabilities with Boolean
retention decisions. These are not calibrated probabilities.
The original tool outputs are omitted from the model's decision context.

Inputs, decisions, compacted messages, and timing are saved under
/results/demos/agent-compaction/<run-id> on the quail-results volume.
The source ID index, sampled trajectories, and model weights stay on
quail-hf-cache. No Jev key is needed.
This measures compaction, not whether an agent can finish after compaction.

Reference: https://github.com/tamaratran/fast-jev-compaction (MIT).
Dataset: https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories
(CC BY 4.0; retain the source repository and license in saved records).
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from demos._trace_compaction import (
    MAX_STATE_TOKENS,
    PRESERVE_RECENT,
    REFERENCE_REVISION,
    RESULT_HEAD_CHARS,
    build_state,
    collect_tool_calls,
    json_text,
    message_chars,
    reconstruct_trace,
    retention_questions,
)
from quail.bench.images import cpu_image, gpu_image

DATASET = "nvidia/SWE-Zero-openhands-trajectories"
DATASET_REVISION = "7b3cd106d00f60918e722d33a1d74bc67072a7ea"
MODEL = "diffusion-gemma-26b-a4b-fp8"
DEVICE = "h100-sxm"

SQL = """
SELECT c.id, q.id, q.tool_call_id, q.kind
FROM conversations c
JOIN tool_questions q
  ON c.id = q.conversation_id
 AND AI.IF(
     PROMPT(
         'Using the compaction state in DOCUMENT {0}, evaluate whether
          the retention statement in DOCUMENT {1} is true.',
         c.state, q.statement
     ),
     {'anchor': 'c'}
 )
"""

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)

preparation_image = cpu_image().add_local_python_source("demos")
inference_image = gpu_image().add_local_python_source("demos")


def sample_trajectories(index: pa.Table, limit: int, seed: int) -> pa.Table:
    """Choose one attempt per issue, then sample distinct issues reproducibly."""
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = index.sort_by([("instance_id", "ascending"),
                             ("trajectory_id", "ascending")])
    groups = pc.run_end_encode(ordered["instance_id"].combine_chunks())
    ends = groups.run_ends.to_numpy()
    starts = np.concatenate(([0], ends[:-1]))
    if limit > len(ends):
        raise ValueError(f"requested {limit} issues, but only {len(ends)} exist")
    rng = np.random.default_rng(seed)
    attempts = starts + rng.integers(ends - starts)
    issues = rng.choice(len(ends), size=limit, replace=False)
    return ordered.take(attempts[issues])


def cache_table(table: pa.Table, path: Path) -> None:
    """Publish a complete cached Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def source_index(filesystem, cache: Path) -> pa.Table:
    """Index all attempts using only ID columns, without reading trace bodies."""
    from huggingface_hub import HfApi

    path = cache / "index.parquet"
    if path.exists():
        return pq.read_table(path)
    files = sorted(name for name in HfApi().list_repo_files(
        DATASET, repo_type="dataset", revision=DATASET_REVISION)
        if name.startswith("data/") and name.endswith(".parquet"))
    parts = []
    for name in files:
        remote = f"datasets/{DATASET}@{DATASET_REVISION}/{name}"
        with filesystem.open(remote, block_size=1 << 20) as source:
            parquet = pq.ParquetFile(source)
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(
                    group, columns=["instance_id", "trajectory_id"])
                parts.append(ids
                             .append_column("shard", pa.array([name] * len(ids)))
                             .append_column("row_group", pa.array([group] * len(ids)))
                             .append_column("row_index", pa.array(range(len(ids)))))
        print(f"indexed {name}", flush=True)
    index = pa.concat_tables(parts)
    cache_table(index, path)
    return index


def read_source_group(filesystem, cache: Path, entries: pa.Table) -> pa.Table:
    """Read one record group and cache only the sampled trajectories."""
    shard = entries["shard"][0].as_py()
    group = entries["row_group"][0].as_py()
    path = cache / "selected" / Path(shard).stem / f"{group}.parquet"
    cached = pq.read_table(path) if path.exists() else None
    positions = (pc.index_in(entries["trajectory_id"],
                             value_set=cached["trajectory_id"])
                 if cached is not None else None)
    if positions is not None and positions.null_count == 0:
        selected = cached.take(positions)
    else:
        full_group = cache / Path(shard).stem / f"{group}.parquet"
        if full_group.exists():
            table = pq.read_table(full_group)
        else:
            remote = f"datasets/{DATASET}@{DATASET_REVISION}/{shard}"
            with filesystem.open(remote, block_size=1 << 20) as source:
                table = pq.ParquetFile(source).read_row_group(group, columns=[
                    "instance_id", "trajectory_id", "repo", "license", "trajectory"])
        selected = table.take(entries["row_index"])
        if cached is not None:
            cached = cached.filter(pc.invert(pc.is_in(
                cached["trajectory_id"], value_set=selected["trajectory_id"])))
        cache_table(pa.concat_tables([cached, selected]) if cached is not None
                    else selected, path)
    for column in ("instance_id", "trajectory_id"):
        if not selected[column].equals(entries[column]):
            raise ValueError("cached trajectory does not match the source index")
    print(f"loaded {len(selected)} sampled trajectories from {shard}, "
          f"row group {group}", flush=True)
    return selected


def source_rows(limit: int, seed: int):
    """Sample with Arrow and fetch required record groups four at a time."""
    from huggingface_hub import HfFileSystem
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE) / "quail-agent-compaction" / DATASET_REVISION
    filesystem = HfFileSystem()
    started = time.perf_counter()
    index = source_index(filesystem, cache)
    print(f"source index: {len(index)} attempts, "
          f"{time.perf_counter() - started:.2f} s", flush=True)
    started = time.perf_counter()
    selected = sample_trajectories(index, limit, seed).sort_by([
        ("shard", "ascending"), ("row_group", "ascending"),
        ("row_index", "ascending")])
    print(f"sampled {len(selected)} issues in "
          f"{time.perf_counter() - started:.3f} s", flush=True)
    same_group = pc.and_(
        pc.equal(selected["shard"].slice(1), selected["shard"].slice(0, limit - 1)),
        pc.equal(selected["row_group"].slice(1),
                 selected["row_group"].slice(0, limit - 1)))
    boundaries = np.concatenate(([0], np.flatnonzero(~same_group.to_numpy()) + 1,
                                 [limit]))
    groups = [selected.slice(int(start), int(end - start))
              for start, end in itertools.pairwise(boundaries)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in itertools.batched(groups, 4):
            pending = [pool.submit(read_source_group, filesystem, cache, entries)
                       for entries in batch]
            for future in pending:
                yield from future.result().to_pylist()


def prepare_tables(rows, directory: Path) -> dict:
    """Write bounded Parquet batches of conversations, questions, and messages."""
    for name in ("conversations", "tool_questions", "messages"):
        (directory / name).mkdir(parents=True)
    counts = {"conversations": 0, "questions": 0, "pinned_calls": 0}
    seen, seen_issues = set(), set()
    for batch_id, batch in enumerate(itertools.batched(rows, 100)):
        conversations, questions, originals = [], [], []
        for row in batch:
            conv_id = row["trajectory_id"]
            if conv_id in seen:
                raise ValueError(f"duplicate trajectory ID: {conv_id}")
            if row["instance_id"] in seen_issues:
                raise ValueError(f"duplicate issue: {row['instance_id']}")
            seen.add(conv_id)
            seen_issues.add(row["instance_id"])
            messages = row["trajectory"]
            try:
                calls = collect_tool_calls(messages)
                candidates = [call for call in calls if not call.pinned]
                state, tokens, stage = (build_state(messages, calls) if candidates
                                        else (None, 0, "no candidates"))
            except ValueError as error:
                raise ValueError(f"cannot prepare {conv_id}: {error}") from error
            conversations.append({"id": conv_id, "instance_id": row["instance_id"],
                                  "state": json_text(state)})
            originals.append({
                "id": conv_id, "instance_id": row["instance_id"],
                "repo": row["repo"], "license": row["license"],
                "messages": json.dumps(messages),
                "state_tokens_estimated": tokens,
                "state_fitting_stage": stage,
            })
            for question in itertools.chain.from_iterable(
                    retention_questions(call) for call in candidates):
                questions.append({"id": f"{conv_id}:{question['key']}",
                                  "conversation_id": conv_id, **question})
            counts["conversations"] += 1
            counts["questions"] += 2 * len(candidates)
            counts["pinned_calls"] += len(calls) - len(candidates)
        filename = f"{batch_id:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(conversations),
                       directory / "conversations" / filename)
        pq.write_table(pa.Table.from_pylist(originals),
                       directory / "messages" / filename)
        schema = pa.schema([(name, pa.string()) for name in (
            "id", "conversation_id", "key", "tool_call_id", "kind", "statement")])
        pq.write_table(pa.Table.from_pylist(questions, schema=schema),
                       directory / "tool_questions" / filename)
        print(f"prepared {counts['conversations']} conversations, "
              f"{counts['questions']} questions", flush=True)
    return counts


@app.function(image=preparation_image, timeout=86_400, memory=16_384,
              volumes={"/results": results_volume,
                       "/root/.cache/huggingface": hf_cache})
def prepare(limit: int, seed: int) -> str:
    """Save the original library's inputs without GPU inference."""
    if limit < 1:
        raise ValueError("limit must be positive")
    directory = Path("/results/demos/agent-compaction") / uuid.uuid4().hex
    started = time.perf_counter()
    counts = prepare_tables(source_rows(limit, seed), directory)
    metadata = {**counts, "dataset": DATASET, "dataset_revision": DATASET_REVISION,
                "reference_revision": REFERENCE_REVISION,
                "seed": seed, "issue_limit": limit,
                "sampling": "one random trajectory per sampled instance_id",
                "preserve_recent_messages": PRESERVE_RECENT,
                "max_state_tokens_estimated": MAX_STATE_TOKENS,
                "truncate_result_chars": RESULT_HEAD_CHARS,
                "sql": SQL, "prepare_s": time.perf_counter() - started}
    (directory / "inputs.json").write_text(json.dumps(metadata, indent=2))
    results_volume.commit()
    hf_cache.commit()
    print(f"result volume path: {directory}", flush=True)
    return str(directory)


def complete_answers(result, conversations, questions):
    """Validate all pairs and return answers in question-table order."""
    answers = pa.concat_tables(list(result.answer_tables["joins"].values()))
    ordered = answers.sort_by([("q", "ascending")])
    if not np.array_equal(ordered["q"].to_numpy(), np.arange(len(questions))):
        raise ValueError("query did not answer every question exactly once")
    actual = conversations["id"].take(ordered["c"])
    if actual.to_pylist() != questions["conversation_id"].to_pylist():
        raise ValueError("query evaluated a question against the wrong conversation")
    return ordered


def evaluate_tables(directory: Path, gpus: int) -> dict:
    """Run the join and save every Boolean decision and its execution report."""
    import quail
    from quail.specs import MODAL_GPU_USD_PER_HOUR

    conversations = pq.read_table(directory / "conversations", columns=["id"])
    questions = pq.read_table(directory / "tool_questions", columns=[
        "id", "conversation_id", "key", "tool_call_id", "kind"])
    report = {"wall_s": 0.0, "boot_s": 0.0, "fresh_tokens": 0}
    input_tokens = 0
    if len(questions):
        with quail.Session(config=quail.EngineConfig(
                model=MODEL, device=DEVICE, gpus=gpus)) as session:
            session.register("conversations", quail.DocumentProvider.from_parquet(
                str(directory / "conversations"), id_col="id"))
            session.register("tool_questions", quail.DocumentProvider.from_parquet(
                str(directory / "tool_questions"), id_col="id"))
            query = session.sql(SQL, dialect="bq")
            explanation = query.explain()
            print(explanation, flush=True)
            (directory / "plan.txt").write_text(explanation)
            result = query.run()
            answers = complete_answers(result, conversations, questions)
            pq.write_table(result.collect(), directory / "retained.parquet")
            prompt = query.logical.operators().joins[0].prompt
            pieces = {alias: (label, frame)
                      for alias, label, frame in prompt.label_token_ids}
            overhead = (len(prompt.preamble_token_ids) + len(pieces["c"][1])
                        + len(pieces["q"][0]) + len(prompt.tail_token_ids))
            c_lengths = np.asarray(session.token_lengths("conversations", "state"))
            q_lengths = np.asarray(session.token_lengths("tool_questions", "statement"))
            input_tokens = int(c_lengths[answers["c"].to_numpy()].sum()
                               + q_lengths.sum() + overhead * len(questions))
            decisions = questions.append_column("answer", answers["answer"])
            report = dict(result.report)
    else:
        decisions = questions.append_column("answer", pa.array([], type=pa.bool_()))
    pq.write_table(decisions, directory / "decisions.parquet")
    seconds = report["wall_s"]
    report.update(
        model=MODEL, device=DEVICE, gpus=gpus, input_tokens=input_tokens,
        input_tokens_per_second=input_tokens / seconds if seconds else None,
        evaluated_pairs=len(questions),
        gpu_cost_usd=seconds * gpus * MODAL_GPU_USD_PER_HOUR[DEVICE] / 3600,
        gpu_startup_cost_usd=report.get("boot_s", 0) * gpus
        * MODAL_GPU_USD_PER_HOUR[DEVICE] / 3600,
    )
    return report


@app.function(image=inference_image, gpu="H100!", timeout=86_400, memory=65_536,
              volumes={"/results": results_volume,
                       "/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def evaluate(directory: str, gpus: int) -> dict:
    """Evaluate the prepared questions on a Modal GPU."""
    results_volume.reload()
    started = time.perf_counter()
    report = evaluate_tables(Path(directory), gpus)
    report["evaluate_total_s"] = time.perf_counter() - started
    (Path(directory) / "execution.json").write_text(json.dumps(report, indent=2))
    results_volume.commit()
    return report


@app.function(image=preparation_image, timeout=86_400, memory=32_768,
              volumes={"/results": results_volume})
def reconstruct(directory: str) -> dict:
    """Apply the original retention rules and save the compacted conversations."""
    results_volume.reload()
    path = Path(directory)
    started = time.perf_counter()
    decisions = {}
    for row in pq.read_table(path / "decisions.parquet").to_pylist():
        decisions.setdefault(row["conversation_id"], {})[row["key"]] = row["answer"]
    totals = {"characters_before": 0, "characters_after": 0,
              "keep": 0, "truncate": 0, "drop": 0, "pinned": 0}
    output = path / "compacted"
    output.mkdir()
    for filename in sorted((path / "messages").glob("*.parquet")):
        rows = []
        for original in pq.read_table(filename).to_pylist():
            messages = json.loads(original["messages"])
            calls = collect_tool_calls(messages)
            compacted, actions = reconstruct_trace(
                messages, calls, decisions.get(original["id"], {}))
            totals["characters_before"] += message_chars(messages)
            totals["characters_after"] += message_chars(compacted)
            for action in actions:
                totals[action["action"]] += 1
            rows.append({"id": original["id"], "instance_id": original["instance_id"],
                         "repo": original["repo"],
                         "license": original["license"],
                         "messages": json.dumps(compacted),
                         "decisions": json.dumps(actions)})
        pq.write_table(pa.Table.from_pylist(rows), output / filename.name)
    summary = {**json.loads((path / "inputs.json").read_text()),
               **json.loads((path / "execution.json").read_text()), **totals,
               "reconstruct_s": time.perf_counter() - started,
               "result_volume_path": directory}
    before = totals["characters_before"]
    summary["character_reduction_fraction"] = (
        1 - totals["characters_after"] / before if before else 0)
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
    results_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(limit: int = 100, seed: int = 42, gpus: int = 1):
    """Compact complete traces using DiffusionGemma on one or more H100s."""
    if limit < 1 or gpus not in (1, 2, 4, 8):
        raise ValueError("limit must be positive; gpus must be 1, 2, 4, or 8")
    call = prepare.spawn(limit, seed)
    print(f"function call id (prepare): {call.object_id}", flush=True)
    directory = call.get()
    call = evaluate.with_options(gpu=f"H100!:{gpus}").spawn(directory, gpus)
    print(f"function call id (evaluate): {call.object_id}", flush=True)
    call.get()
    call = reconstruct.spawn(directory)
    print(f"function call id (reconstruct): {call.object_id}", flush=True)
    call.get()
