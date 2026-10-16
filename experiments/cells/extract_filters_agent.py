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

AGENT_DOCS = 64
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

QUERIES = {
    "fix_python_file": FIX_PYTHON_FILE_QUERY,
    "changed_code_with_error": CHANGED_CODE_WITH_ERROR_QUERY,
    "command_regex": COMMAND_REGEX_QUERY,
}


@app.function(
    image=gpu_image(),
    gpu="H100!", memory=98304, volumes=volumes, timeout=3600,
)
def run_queries() -> None:
    """Run the queries and save each report to the results volume."""
    import pyarrow.parquet as pq

    docs = pq.read_table(CORPUS).slice(0, AGENT_DOCS)
    subset = "/tmp/agent_traces.parquet"
    pq.write_table(docs, subset)
    session = quail.Session(EngineConfig(
        gpus=1, model="qwen3-4b-fp8", backend="quail", device="h100-sxm"))
    session.register("traces", quail.DocumentProvider.from_parquet(
        subset, id_col="id"))
    run_id = uuid.uuid4().hex
    folder = Path("/results/extract-filters") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    print(f"run id: {run_id}", flush=True)
    for name, text in QUERIES.items():
        query = session.sql(text)
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
        print("query seconds:", report.get("query_seconds"), flush=True)
    session.close()
    kernel_cache.commit()


@app.local_entrypoint()
def main() -> None:
    """Start the run and print its function call id."""
    call = run_queries.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
