"""Score saved FEV-9 answers without running inference.

    uv run modal volume get quail-results \
      /ablations/shared-kv-retention-20260906T054932Z /tmp
    uv run python reports/score_shared_kv_retention.py \
      /tmp/shared-kv-retention-20260906T054932Z

Reference labels and corpus rows are read from quail-results. The derived
accuracy.json is saved beside the original results on the volume.
"""

import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quail.runtime.volumes import ModalVolumeFiles
from quail_b.data import CORPUS_COLUMNS, _ids, corpus_identity
from quail_b.labels import _load_ground_truth_collection
from quail_b.queries import queries
from quail_b.scoring import Evaluator, RunOutput, rows_from_answers

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
    files = ModalVolumeFiles()
    collection = json.loads(files.read_bytes(
        f"{ROOT}/collections/{COLLECTION}/manifest.json"))
    collection["label_sets"] = {
        key: value for key, value in collection["label_sets"].items()
        if key.startswith("quailb.fever.") and not key.endswith("contains_date")
    }
    truth = _load_ground_truth_collection(files, collection)
    corpus = {
        name: pq.read_table(
            io.BytesIO(files.read_bytes(f"quailb_data/sf0.1/{name}.parquet")),
            columns=list(CORPUS_COLUMNS[name]))
        for name in ("claims", "evidence")
    }
    manifest = json.loads(files.read_bytes(
        f"{ROOT}/corpora/{truth.corpus_id}/manifest.json"))
    actual = corpus_identity(corpus, 0.1, 0, {})["tables"]
    assert actual == {name: manifest["tables"][name] for name in corpus}
    evaluator = Evaluator(truth, corpus)
    spec = queries()["FEV-9"]
    ids = {alias.alias: _ids(corpus[alias.table]) for alias in spec.aliases}

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
        filters = {
            (alias, 0): with_ids(
                pq.read_table(saved / f"filters-{alias}-0.parquet"), [alias])
            for alias in ("c1", "e1", "c2", "e2")
        }
        joins = {
            index: with_ids(pq.read_table(saved / f"joins-{index}.parquet"),
                            spec.joins[index].aliases)
            for index in range(3)
        }
        # the saved run kept its answers, not its rows
        rows = rows_from_answers(spec, filters, joins)
        assert rows.num_rows == summary["rows"], (rows.num_rows, summary)
        scores[label] = evaluator.evaluate(
            spec, RunOutput(filters, joins, rows))
        print(label, json.dumps(scores[label], indent=2), flush=True)
    assert scores["first_anchor"] == scores["shared"]
    destination = f"ablations/{root.name}/accuracy.json"
    files.write_json(destination, {
        "query": "FEV-9", "configurations": scores,
        "source_volume_path": f"/results/ablations/{root.name}",
        "collection_id": COLLECTION, "corpus_tables": actual,
        "inference_rerun": False,
    })
    print(f"Saved /results/{destination}")


if __name__ == "__main__":
    main(sys.argv[1])
