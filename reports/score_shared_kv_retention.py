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
from types import SimpleNamespace

import pyarrow.parquet as pq

import quail
from quail.bench.evaluate import (
    CORPUS_COLUMNS,
    BenchmarkEvaluator,
    ModalVolumeFiles,
    _load_ground_truth_collection,
    corpus_identity,
)
from quail.bench.quailb import queries
from quail.catalog import DocumentProvider
from quail.runtime.result import true_answer_rows

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
    evaluator = BenchmarkEvaluator(truth, corpus)
    scores = {}
    with quail.Session() as session:
        for name, table in corpus.items():
            session.register(name, DocumentProvider.from_table(table, id_col="id"))
        query = queries(session)["FEV-9"][1]()
        for label in ("first_anchor", "shared"):
            saved = root / label
            summary = json.loads((saved / "summary.json").read_text())
            assert (summary["query"], summary["sf"], summary["lf"]) == ("FEV-9", 0.1, 1)
            filters = {
                (alias, 0): pq.read_table(saved / f"filters-{alias}-0.parquet")
                for alias in ("c1", "e1", "c2", "e2")
            }
            joins = {index: pq.read_table(saved / f"joins-{index}.parquet")
                     for index in range(3)}
            result = SimpleNamespace(
                answer_tables={"filters": filters, "joins": joins},
                survivor_indices={alias: true_answer_rows(table)[alias]
                                  for (alias, _), table in filters.items()},
                true_join_tables={index: true_answer_rows(table)
                                  for index, table in joins.items()},
                count=lambda: summary["rows"],
            )
            scores[label] = evaluator.evaluate(query, result)
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
