"""Every QUAIL-B query plans to the same physical plan as the saved one.

The queries plan on the CPU over stand-in tables whose documents have
varied lengths, so join order, anchors, and prompt lowering all take
part. A planner change that moves a plan fails here; set
QUAIL_UPDATE_PLAN_SNAPSHOTS=1 to save the new plans after checking
them.
"""

import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import letter_tokens
from test_quailb import _standin_sets

import quail
from bench.quailb import queries, register_tables
from quail.planner.plan import EngineConfig, Refusal

SNAPSHOTS = Path(__file__).parent / "fixtures" / "plans"
BACKENDS = ("quail", "stock_vllm")
# document columns lengthened so the stand-in corpora have unequal
# token counts; key and label columns stay as written
DOCUMENT_COLUMNS = {
    "reviews": "body", "reports": "report", "claims": "claim",
    "evidence": "text", "citation_contexts": "destination_context",
    "citation_passages": "passage_text", "agent_traces": "trace",
    "policies": "policy_text", "support_messages": "content",
    "issue_messages": "content", "wrench_steps": "content",
    "sales_calls": "transcript", "contracts": "text",
}


def _varied_sets(tmp_path):
    _standin_sets(tmp_path)
    for name, column in DOCUMENT_COLUMNS.items():
        path = tmp_path / f"{name}.parquet"
        table = pq.read_table(path)
        if column not in table.column_names:
            continue
        texts = table.column(column).to_pylist()
        longer = [None if text is None else " ".join([text] * (1 + (i * 7) % 11))
                  for i, text in enumerate(texts)]
        index = table.schema.get_field_index(column)
        table = table.set_column(index, column, pa.array(longer, pa.string()))
        pq.write_table(table, path)


def _compact(value):
    """Replace long token id lists by their length and digest."""
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 8 and all(isinstance(item, int) for item in value):
            digest = hashlib.sha256(json.dumps(value).encode()).hexdigest()
            return {"tokens": len(value), "sha256": digest[:16]}
        return [_compact(item) for item in value]
    return value


def _plans(backend, data_dir) -> dict:
    config = EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                          device="h100-sxm")
    plans = {}
    with quail.Session(config, tokenizer=letter_tokens) as session:
        register_tables(session, data_dir)
        listed = sorted(queries(session).items())
        # plan once to tokenize every table, so the saved plans are
        # made on exact lengths rather than on a sample
        for _, (_, build) in listed:
            query = build()
            query.plan()
            query.wait_for_tokens()
        for qid, (_, build) in listed:
            plan = build().plan()
            plans[qid] = ({"refusal": plan.constraint}
                          if isinstance(plan, Refusal)
                          else _compact(plan.to_envelope(session.registry.codecs)))
    return plans


def _differences(saved, current) -> list[str]:
    if saved.keys() != current.keys():
        return [f"queries differ: {sorted(saved.keys() ^ current.keys())}"]
    out = []
    for qid in saved:
        if saved[qid] != current[qid]:
            out.append(qid)
    return out


@pytest.mark.parametrize("backend", BACKENDS)
def test_benchmark_plans_match_the_snapshots(backend, tmp_path):
    _varied_sets(tmp_path)
    current = json.loads(json.dumps(_plans(backend, tmp_path), sort_keys=True))
    path = SNAPSHOTS / f"{backend}.json"
    if os.environ.get("QUAIL_UPDATE_PLAN_SNAPSHOTS"):
        path.write_text(json.dumps(current, sort_keys=True, indent=1) + "\n")
    assert path.exists(), f"no snapshot at {path}; set QUAIL_UPDATE_PLAN_SNAPSHOTS=1"
    saved = json.loads(path.read_text())
    changed = _differences(saved, current)
    assert not changed, (
        f"{len(changed)} {backend} plans changed: {changed[:10]}; review them, "
        "then set QUAIL_UPDATE_PLAN_SNAPSHOTS=1 to save the new plans")
