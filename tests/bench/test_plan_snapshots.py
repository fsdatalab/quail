"""Every QUAIL-B query plans as its saved snapshot says.

A snapshot is the query's explain() text: the logical plan and the
physical plan with its estimates. A change that alters a plan shows
here as a diff. The chunk and admission budgets are left out of the
comparison: they follow the model's and device's memory constants, not
the plan. To accept new plans, run the test with
QUAIL_UPDATE_SNAPSHOTS=1 and commit the files it writes.
"""

import os
import re
from pathlib import Path

from helpers.quailb_data import standin_sets
from helpers.sessions import TOKENIZER

import quail
from bench.quailb import queries, register_tables
from quail.api.config import EngineConfig

SNAPSHOTS = Path(__file__).resolve().parents[1] / "fixtures" / "plan_snapshots"
BUDGETS = re.compile(r"(chunk|admission) budget=[\d,]+ tokens")


def _plan_text(text: str) -> str:
    """Return explain() text without the token budgets."""
    return BUDGETS.sub(r"\1 budget=N tokens", text)


def test_every_quailb_query_plans_as_its_snapshot(tmp_path):
    standin_sets(tmp_path)
    update = os.environ.get("QUAIL_UPDATE_SNAPSHOTS") == "1"
    changed = []
    with quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER,
    ) as session:
        register_tables(session, tmp_path)
        for query_id, (_, build) in sorted(queries(session).items()):
            query = build()
            # a plan made before the tokens are counted uses a sample's
            # estimate; waiting re-plans it on the exact counts
            query.plan()
            query.wait_for_tokens()
            text = query.explain() + "\n"
            path = SNAPSHOTS / f"{query_id}.txt"
            if update:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            elif (not path.exists()
                  or _plan_text(path.read_text()) != _plan_text(text)):
                changed.append(query_id)
    assert not changed, f"plans differ from their snapshots: {changed}"
