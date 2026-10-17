r"""Filters on AI.EXTRACT answers over quail-b agent traces, on one H100.

Runs three queries on the first AGENT_DOCS rows of the sf0.1
agent_traces corpus (/results/quailb_data/sf0.1/agent_traces.parquet),
each mixing an AI.IF or AI.CLASSIFY condition with tests on extracted
answers, and prints each plan, its row count, its survivors per node,
and its node seconds. Run from the repository root as a module:

    uv run modal run --detach -m experiments.cells.extract_filters_agent \
        2>&1 | tee results/extract_filters_agent.log

Reports go to /results/extract-filters/<id>.json on quail-results.
"""

import json
import uuid
from pathlib import Path

import modal

import quail
from quail.deploy.images import gpu_image
from quail.planner.plan import EngineConfig
from quail_b.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_PROGRESS,
    AGENT_PROGRESS_LABELS,
    AGENT_RECOVERED,
)

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/results": results_vol,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": kernel_cache,
}

AGENT_DOCS = 128
CORPUS = "/results/quailb_data/sf0.1/agent_traces.parquet"

EDITED_FILE = "Which file did the agent edit last?"
LAST_COMMAND = "Which command did the agent run last?"
ERROR_TEXT = "Which error message did the agent see last?"


def _prompt(template: str) -> str:
    """Return a quail-b filter template as a SQL string literal body."""
    return template.replace("'", "''").replace("\n", "\\n")


LABELS = ", ".join(f"'{label}'" for label in AGENT_PROGRESS_LABELS)
CHANGED = ", ".join(f"'{label}'" for label in AGENT_PROGRESS_LABELS[2:])

# AI_FILTER plus two tests on one extraction field: the ask should
# run first, the extraction over the documents it keeps
FIX_PYTHON_FILE_QUERY = f"""
    SELECT t.id,
           AI.EXTRACT(t.trace, {{'edited': '{EDITED_FILE}',
                                'command': '{LAST_COMMAND}'}}) AS last
    FROM traces t
    WHERE AI_FILTER(PROMPT('{_prompt(AGENT_IMPLEMENTED_FIX)}', t.trace),
                    {{'selectivity': 0.3}})
      AND HINT(last.edited IS NOT NULL, {{'selectivity': 0.8}})
      AND HINT(last.edited LIKE '%.py', {{'selectivity': 0.7}})
"""

# a classification with a label test, and an extraction tested with
# ILIKE and LENGTH, and a span test on where the error sits
CHANGED_CODE_WITH_ERROR_QUERY = f"""
    SELECT t.id,
           AI.CLASSIFY(PROMPT('{_prompt(AGENT_PROGRESS)}', t.trace),
                       ARRAY[{LABELS}]) AS progress,
           AI.EXTRACT(t.trace, {{'error': '{ERROR_TEXT}'}}) AS seen
    FROM traces t
    WHERE HINT(progress IN ({CHANGED}), {{'selectivity': 0.4}})
      AND HINT(seen.error ILIKE '%error%', {{'selectivity': 0.5}})
      AND HINT(LENGTH(seen.error) > 20, {{'selectivity': 0.8}})
      AND HINT(seen_span.error.start > 2000, {{'selectivity': 0.7}})
"""

# the extraction repeated in WHERE, with a regex and a span test, and
# no other condition: the extraction runs alone
COMMAND_REGEX_QUERY = f"""
    SELECT t.id,
           AI.EXTRACT(t.trace, {{'command': '{LAST_COMMAND}'}}) AS last
    FROM traces t
    WHERE HINT(REGEXP_LIKE(
                   AI.EXTRACT(t.trace, {{'command': '{LAST_COMMAND}'}}).command,
                   '(python|pytest|git).*'),
               {{'selectivity': 0.5}})
      AND last_span.command.end > 1000
"""

RAN_TESTS = (
    "Judge strictly from the agent trace above whether the agent ran a "
    "test suite or a test command at any point.\n\n{0}\n\nInstruction: "
    "answer TRUE if the agent ran tests, FALSE otherwise."
)

# three asks, a classification with a label test, and two extractions
# with four tests, all on one table: one pipeline of nine operators
MANY_OPERATORS_QUERY = f"""
    SELECT t.id,
           AI.CLASSIFY(PROMPT('{_prompt(AGENT_PROGRESS)}', t.trace),
                       ARRAY[{LABELS}]) AS progress,
           AI.EXTRACT(t.trace, {{'edited': '{EDITED_FILE}'}}) AS edit,
           AI.EXTRACT(t.trace, {{'command': '{LAST_COMMAND}',
                                'error': '{ERROR_TEXT}'}}) AS last
    FROM traces t
    WHERE AI_FILTER(PROMPT('{_prompt(AGENT_IMPLEMENTED_FIX)}', t.trace),
                    {{'selectivity': 0.3}})
      AND AI_FILTER(PROMPT('{_prompt(AGENT_RECOVERED)}', t.trace),
                    {{'selectivity': 0.3}})
      AND AI_FILTER(PROMPT('{_prompt(RAN_TESTS)}', t.trace),
                    {{'selectivity': 0.7}})
      AND HINT(progress IN ({CHANGED}), {{'selectivity': 0.4}})
      AND HINT(edit.edited LIKE '%.py', {{'selectivity': 0.6}})
      AND HINT(last.command IS NOT NULL, {{'selectivity': 0.9}})
      AND HINT(LENGTH(last.command) < 80, {{'selectivity': 0.8}})
      AND HINT(last_span.error.start > 1000, {{'selectivity': 0.7}})
"""

# a join on a small table of issue kinds, with an extraction of the
# anchor tested in WHERE
ISSUE_KIND = ("Does the agent in DOCUMENT {0} work on the kind of issue "
              "described in DOCUMENT {1}?")
JOIN_WITH_ANSWER_TEST_QUERY = f"""
    SELECT t.id, k.kind,
           AI.EXTRACT(t.trace, {{'edited': '{EDITED_FILE}'}}) AS edit
    FROM traces t
    JOIN kinds k ON AI_FILTER(PROMPT('{ISSUE_KIND}', t.trace, k.kind),
                              {{'selectivity': 0.3}})
    WHERE HINT(edit.edited LIKE '%.py', {{'selectivity': 0.6}})
"""

KINDS = (
    "a missing function, attribute, or import",
    "a wrong result, value, or calculation",
    "a crash, exception, or error message",
    "a build, packaging, or test configuration problem",
)

# name -> (SQL, order rule); the nine-operator query runs in its written
# order as well, to measure what the cost-based order saves
def python_api_query(session):
    """The same shape as MANY_OPERATORS_QUERY, built with the Python API."""
    col, prompt = quail.col, quail.prompt
    return (
        session.docs("traces").alias("t")
        .ai_filter(prompt(AGENT_IMPLEMENTED_FIX, col("t.trace")), selectivity=0.3)
        .ai_classify(prompt(AGENT_PROGRESS, col("t.trace")),
                     list(AGENT_PROGRESS_LABELS), name="progress")
        .where(col("progress").isin(list(AGENT_PROGRESS_LABELS[2:]))
               .hint(selectivity=0.4))
        .ai_extract("t.trace", {"edited": EDITED_FILE, "command": LAST_COMMAND,
                                "error": ERROR_TEXT}, name="last")
        # the middle field is tested, the others only returned
        .where(col("last.command").is_not_null().hint(selectivity=0.9)
               & ~col("last.command").like("%str_replace%").hint(selectivity=0.8)
               & col("last.command").rlike("(python|pytest|git).*")
               .hint(selectivity=0.5)
               & (col("last_span.command").field("end") > 1000))
        .select("t.id", "progress", "last")
    )


# LIMIT stops the pipeline once five documents pass every test; the
# whole-line answers come from trim off
LIMITED_QUERY = f"""
    SELECT t.id,
           AI.EXTRACT(t.trace, {{'command': '{LAST_COMMAND}'}},
                      {{'trim': false}}) AS last
    FROM traces t
    WHERE AI_FILTER(PROMPT('{_prompt(AGENT_RECOVERED)}', t.trace),
                    {{'selectivity': 0.3}})
      AND HINT(last.command LIKE '%python%', {{'selectivity': 0.5}})
    LIMIT 5
"""

QUERIES = {
    "fix_python_file": (FIX_PYTHON_FILE_QUERY, None),
    "limited": (LIMITED_QUERY, None),
    "changed_code_with_error": (CHANGED_CODE_WITH_ERROR_QUERY, None),
    "command_regex": (COMMAND_REGEX_QUERY, None),
    "many_operators": (MANY_OPERATORS_QUERY, None),
    "many_operators_as_written": (MANY_OPERATORS_QUERY, "as_written"),
    "join_with_answer_test": (JOIN_WITH_ANSWER_TEST_QUERY, None),
    # the first query again, after the session is warm
    "fix_python_file_again": (FIX_PYTHON_FILE_QUERY, None),
}


@app.function(
    image=gpu_image(),
    gpu="H100!", memory=98304, volumes=volumes, timeout=3600,
)
def run_queries() -> None:
    """Run the queries and save each report to the results volume."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    docs = pq.read_table(CORPUS).slice(0, AGENT_DOCS)
    subset = "/tmp/agent_traces.parquet"
    pq.write_table(docs, subset)
    session = quail.Session(EngineConfig(
        gpus=1, model="qwen3-4b-fp8", backend="quail", device="h100-sxm",
        gpu_timing=True))
    session.register("traces", quail.DocumentProvider.from_parquet(
        subset, id_col="id"))
    kinds = "/tmp/kinds.parquet"
    pq.write_table(pa.table({"id": list(range(len(KINDS))),
                             "kind": list(KINDS)}), kinds)
    session.register("kinds", quail.DocumentProvider.from_parquet(
        kinds, id_col="id"))
    run_id = uuid.uuid4().hex
    folder = Path("/results/extract-filters") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    print(f"run id: {run_id}", flush=True)
    queries = {**{name: session.sql(text, order=order)
                  for name, (text, order) in QUERIES.items()},
               "python_api": python_api_query(session)}
    for name, query in queries.items():
        print(f"== {name}\n{query.explain()}", flush=True)
        result = query.run()
        rows = result.to_rows()
        report = result.report
        report["result_volume_path"] = str(folder / f"{name}.json")
        report["rows"] = [list(row) for row in rows]
        (folder / f"{name}.json").write_text(json.dumps(report, default=str))
        results_vol.commit()
        print(f"{name}: {len(rows)} rows of {docs.num_rows}", flush=True)
        for row in rows[:5]:
            print("  ", row, flush=True)
        print("node seconds:", json.dumps({
            node_id: round(metrics.wall_s, 2)
            for node_id, metrics in result.node_metrics.items()}), flush=True)
        print("node gpu seconds, chunks, fresh tokens:", json.dumps({
            node_id: (round(metrics.gpu_s, 2), metrics.chunks, metrics.fresh_tokens)
            for node_id, metrics in result.node_metrics.items()
            if metrics.chunks}), flush=True)
        print("seconds: model wall", round(report["model_wall_s"], 2),
              "planning", round(report["planning_s"], 2),
              "plan estimate", round(query.plan().estimated_seconds, 2),
              flush=True)
    session.close()
    kernel_cache.commit()


@app.local_entrypoint()
def main() -> None:
    """Start the run and print its function call id."""
    call = run_queries.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
