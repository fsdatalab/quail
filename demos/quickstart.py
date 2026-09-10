"""Run QUAIL-B IMDB-1 on 100 reviews in the current Python process.

Model execution requires a CUDA GPU and the backend runtime.
Run on a GPU host with: uv run python demos/quickstart.py
"""

import quail
import quail_b as benchmark
from quail.bench.quailb import build_query


def run_query():
    """Find reviews mentioning a positive aspect of the movie."""
    reviews = benchmark.load_table("reviews", limit=100)
    spec = benchmark.get_query("IMDB-1")
    with quail.Session() as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            reviews, id_col="id"))
        query = build_query(session, spec)
        print(f"{spec.id}: {spec.description} ({reviews.num_rows} reviews)")
        result = query.run()
        return result.collect(), result.report


if __name__ == "__main__":
    rows, report = run_query()
    print(rows.to_pylist())
    print(report)
