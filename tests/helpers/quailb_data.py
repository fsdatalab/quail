"""Small Parquet stand-ins for the QUAIL-B tables, for CPU planning tests."""

import pyarrow as pa
import pyarrow.parquet as pq

from quail_b import data
from quail_b.data import ASPECTS, SCENARIOS


def standin_sets(tmp_path):
    """Write small parquet files with the benchmark table schemas."""
    six = range(6)
    tables = {
        "reviews": {"body": [f"review text {i}" for i in range(12)]},
        "aspects": {"aspect": ASPECTS},
        "reports": {"report": [f"medical report {i}" for i in range(8)]},
        "terms": {"term": [f"reaction {i}" for i in six]},
        "claims": {
            "claim": [f"claim {i}" for i in six],
            "label": ["SUPPORTS" if i % 2 == 0 else "REFUTES" for i in six],
            # a claim's page is an evidence id, as in the real corpus
            "evidence_wiki_url": [f"evidence{i % 4}" for i in six],
        },
        "evidence": {"text": [f"Wikipedia passage {i}" for i in six]},
        "citation_contexts": {
            "destination_context": [f"citation excerpt {i}" for i in six],
            "cited_passage_ids": [[f"passage-{i}"] for i in six],
        },
        "citation_passages": {
            "passage_text": [f"cited passage {i}" for i in six],
            "passage_ids": [[f"passage-{i}"] for i in six],
        },
        "agent_traces": {
            "trace": [f"agent trace {i}" for i in six],
            "trajectory_id": [f"at{i}" for i in six],
            "turn_index": [5] * 6,
            "token_count": [3] * 6,
        },
        "policies": {"policy_text": [f"privacy policy text {i}" for i in range(8)]},
        "scenarios": {"scenario": SCENARIOS},
    }
    for name, columns in tables.items():
        rows = len(next(iter(columns.values())))
        ids = [f"{name}{i}" for i in range(rows)]
        pq.write_table(pa.table({"id": ids, **columns}), tmp_path / f"{name}.parquet")
    # the trace, sales, and contract tables, with every column of their
    # published schema; a row's key columns name other rows of the same
    # stand-in, as in the real corpus
    schemas = {
        "support_traces": data.SUPPORT_TRACE_SCHEMA,
        "support_messages": data.MESSAGE_SCHEMA,
        "issue_runs": data.ISSUE_RUN_SCHEMA,
        "issue_messages": data.MESSAGE_SCHEMA,
        "wrench_runs": data.WRENCH_RUN_SCHEMA,
        "wrench_steps": data.WRENCH_STEP_SCHEMA,
        "sales_calls": data.SALES_CALL_SCHEMA,
        "contracts": data.CONTRACT_SCHEMA,
        "clause_types": data.CLAUSE_TYPE_SCHEMA,
    }
    for name, schema in schemas.items():
        columns = {}
        for field in schema:
            if field.name == "id":
                columns["id"] = [f"{name}{i}" for i in six]
            elif pa.types.is_string(field.type):
                columns[field.name] = [f"{name} {field.name} {i}" for i in six]
            elif pa.types.is_floating(field.type):
                columns[field.name] = [float(i) for i in six]
            else:
                columns[field.name] = [i % 2 for i in six]
        pq.write_table(pa.table(columns, schema=schema),
                       tmp_path / f"{name}.parquet")
