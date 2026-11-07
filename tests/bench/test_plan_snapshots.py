"""Every QUAIL-B query plans as its saved snapshot says.

A snapshot is the query's explain() text: the logical half, DataFusion's
plan of the region, and the physical half, the physical plan with its
estimates or the refusal. A change that alters a plan shows here as a
diff of one half or the other. The chunk and admission budgets are left
out of the comparison: they follow the model's and device's memory
constants, not the plan. To accept new plans, run the test with
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
# the first line of the physical half
PHYSICAL = re.compile(r"^(physical|refusal):", re.MULTILINE)


def _halves(text: str) -> tuple[str, str]:
    """Return explain() text's logical and physical halves, without budgets."""
    text = BUDGETS.sub(r"\1 budget=N tokens", text)
    found = PHYSICAL.search(text)
    split = len(text) if found is None else found.start()
    return text[:split], text[split:]


def test_every_quailb_query_plans_as_its_snapshot(tmp_path):
    standin_sets(tmp_path)
    update = os.environ.get("QUAIL_UPDATE_SNAPSHOTS") == "1"
    changed = {"logical": [], "physical": []}
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
            text = query.explain() + "\n"
            path = SNAPSHOTS / f"{query_id}.txt"
            if update:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                continue
            saved = _halves(path.read_text()) if path.exists() else ("", "")
            for half, now, before in zip(changed, _halves(text), saved, strict=True):
                if now != before:
                    changed[half].append(query_id)
    assert not changed["physical"], (
        f"physical plans differ from their snapshots: {changed['physical']}")
    assert not changed["logical"], (
        f"logical plans differ from their snapshots: {changed['logical']}")
