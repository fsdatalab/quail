"""Build a table from local traces with DuckDB and run a Quail query on Modal.

DuckDB reads the traces on this machine. The table goes to the GPU
function as its input and the answers come back as its output, so no
rows are written to a Modal volume or printed to the logs. The table
query needs DuckDB 2.0 or newer; 1.5.5 fails on its filters.

    uv run --with "duckdb>=2" modal run demos/traces/run_query.py \
      --traces "$HOME/.claude/projects/*/*.jsonl" \
      --table-sql demos/traces/user_messages.sql \
      --query-sql demos/traces/steering_open.sql \
      --out /tmp/steering_open.parquet \
      2>&1 | tee /tmp/steering_open.log
"""

from __future__ import annotations

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)


@app.function(image=gpu_image(), gpu="H100!", timeout=3_600, memory=65_536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def answer(table_ipc: bytes, sql: str, model: str, table_name: str) -> dict:
    """Run one Quail query over a table sent as Arrow IPC bytes.

    Args:
        table_ipc: The table, one row per document, with an `id` column.
        sql: The Quail query, in BigQuery dialect.
        model: The model name, such as "qwen3-4b-fp8".
        table_name: The name the query uses for the table.

    Returns:
        The answers as Arrow IPC bytes, the query report, and the plan.
    """
    import pyarrow as pa

    import quail

    table = pa.ipc.open_stream(table_ipc).read_all()
    config = quail.EngineConfig(model=model, device="h100-sxm")
    with quail.Session(config=config) as session:
        session.register(table_name,
                         quail.DocumentProvider.from_table(table, id_col="id"))
        result = session.sql(sql, dialect="bq").run()
        answers = result.collect()
        report = {key: value for key, value in result.report.items()
                  if isinstance(value, (int, float, str))}
        plan = str(result.explain())
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, answers.schema) as writer:
        writer.write_table(answers)
    return {"answers": sink.getvalue().to_pybytes(), "report": report,
            "plan": plan}


@app.local_entrypoint()
def main(traces: str, table_sql: str, query_sql: str, out: str,
         model: str = "qwen3-4b-fp8", table_name: str = "messages"):
    """Build the table locally, answer the query on Modal, save the answers."""
    import json
    from pathlib import Path

    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    with duckdb.connect() as connection:
        connection.execute("SET VARIABLE traces = ?", [traces])
        table = connection.sql(Path(table_sql).read_text()).to_arrow_table()
    print(f"table: {table.num_rows} rows, {table.nbytes / 1e6:.1f} MB", flush=True)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    call = answer.spawn(sink.getvalue().to_pybytes(),
                        Path(query_sql).read_text(), model, table_name)
    print(f"function call id: {call.object_id}", flush=True)
    output = call.get()
    answers = pa.ipc.open_stream(output["answers"]).read_all()
    pq.write_table(answers, out)
    print(output["plan"])
    print(json.dumps(output["report"], indent=2))
    print(f"answers: {answers.num_rows} rows saved to {out}", flush=True)
