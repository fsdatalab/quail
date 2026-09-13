"""Score saved FEV-9 answers without running inference.

    uv run modal volume get quail-results \
      /ablations/shared-kv-retention-20260906T054932Z /tmp
    uv run python reports/score_shared_kv_retention.py \
      /tmp/shared-kv-retention-20260906T054932Z

Reference labels come from QUAIL-B's public bucket. Corpus rows come from
quail-results. The derived accuracy.json is saved beside the original results.
"""

import io
import json
import sys
from pathlib import Path

import modal
import pyarrow as pa
import pyarrow.parquet as pq

from quail.bench.substrait import read_plan
from quail_b.data import CORPUS_COLUMNS, _ids, corpus_identity
from quail_b.labels import _load_ground_truth_collection
from quail_b.queries import get_query
from quail_b.scoring import RunOutput, evaluate, rows_from_answers

COLLECTION = "gt_77bb8b128743a79aedddaa24c808c3f8"
ROOT = "ground_truth/quailb/schema_v1"


def compare_answers(root):
    """Compare every saved predicate answer table."""
    names = {path.name for path in (root / "first_anchor").glob("*.parquet")}
    assert names == {path.name for path in (root / "shared").glob("*.parquet")}
    assert len(names) == 7
    for name in sorted(names):
        tables = [pq.read_table(root / label / name)
                  for label in ("first_anchor", "shared")]
        columns = sorted(tables[0].column_names)
        order = [(name, "ascending") for name in columns]
        ordered = [table.select(columns).sort_by(order) for table in tables]
        assert ordered[0].equals(ordered[1]), name
        print(f"{name}: {len(ordered[0]):,} identical answers")


def main(workdir):
    """Validate corpus identity and score the two saved configurations."""
    root = Path(workdir)
    compare_answers(root)
    volume = modal.Volume.from_name("quail-results")
    collection = json.loads(b"".join(volume.read_file(
        f"{ROOT}/collections/{COLLECTION}/manifest.json")))
    collection["label_sets"] = {
        key: value for key, value in collection["label_sets"].items()
        if key.startswith("quailb.fever.") and not key.endswith("contains_date")
    }
    truth = _load_ground_truth_collection(None, collection)
    corpus = {
        name: pq.read_table(
            io.BytesIO(b"".join(volume.read_file(
                f"quailb_data/sf0.1/{name}.parquet"))),
            columns=list(CORPUS_COLUMNS[name]))
        for name in ("claims", "evidence")
    }
    manifest = json.loads(b"".join(volume.read_file(
        f"{ROOT}/corpora/{truth.corpus_id}/manifest.json")))
    actual = corpus_identity(corpus, 0.1, 0, {})["tables"]
    assert actual == {name: manifest["tables"][name] for name in corpus}
    spec = get_query("FEV-9")
    plan = read_plan(spec.plan)
    ids = {relation.alias: _ids(corpus[relation.table])
           for relation in plan.relations}

    def with_ids(table, aliases):
        # the saved answer tables hold row indices; scoring wants ids
        return pa.table({
            **{alias: [ids[alias][index]
                       for index in table.column(alias).to_pylist()]
               for alias in aliases},
            "answer": table.column("answer"),
        })

    scores = {}
    for label in ("first_anchor", "shared"):
        saved = root / label
        summary = json.loads((saved / "summary.json").read_text())
        assert (summary["query"], summary["sf"], summary["lf"]) == ("FEV-9", 0.1, 1)
        # the saved files are named by alias and written join position;
        # scoring keys answers by the plan's operator ids
        filters = {
            plan.filter_id(alias, 0): with_ids(
                pq.read_table(saved / f"filters-{alias}-0.parquet"), [alias])
            for alias in ("c1", "e1", "c2", "e2")
        }
        joins = {
            plan.join_id(index): with_ids(
                pq.read_table(saved / f"joins-{index}.parquet"),
                plan.joins[index].aliases)
            for index in range(3)
        }
        # the saved run kept its answers, not its rows
        rows = rows_from_answers(spec, filters, joins)
        assert rows.num_rows == summary["rows"], (rows.num_rows, summary)
        scores[label] = evaluate(
            spec, RunOutput(filters, joins, rows), truth, corpus)
        print(label, json.dumps(scores[label], indent=2), flush=True)
    assert scores["first_anchor"] == scores["shared"]
    destination = f"ablations/{root.name}/accuracy.json"
    payload = {
        "query": "FEV-9", "configurations": scores,
        "source_volume_path": f"/results/ablations/{root.name}",
        "collection_id": COLLECTION, "corpus_tables": actual,
        "inference_rerun": False,
    }
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(payload, indent=2).encode()), destination)
    print(f"Saved /results/{destination}")


if __name__ == "__main__":
    main(sys.argv[1])
