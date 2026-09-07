"""Filter all 100,000 IMDB reviews for ones that discuss the ending.

Runs on the GPU in this process with the default compute provider.
Loads the three IMDB splits (train, test, unsupervised) from the
pinned Hugging Face revision, registers them as one Arrow dataset,
runs one AI.IF filter, and prints the plan, the row count, and the
timing.

Prediction, from the QUAIL-B IMDB-1 result (one filter over 5,000
reviews: 14.25 s, 351 reviews/s, 352 fresh tokens per review): about
285 s for the query, about 35 M fresh tokens, about $0.31 of H100 time.

    uv run python demos/imdb_ending_filter.py 2>&1 | tee imdb_ending_filter.log
"""

import pyarrow as pa
import pyarrow.dataset as ds
from datasets import concatenate_datasets, load_dataset

import quail
from quail.bench.evaluate import H100_USD_PER_HOUR

IMDB_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"

SQL = """
    SELECT r.review_id
    FROM reviews AS r
    WHERE AI.IF(
        PROMPT(
            'Does this review discuss the ending of the movie?\\n\\n{0}',
            r.review
        ),
        {'selectivity': 0.25}
    )
"""


def load_reviews() -> ds.Dataset:
    """Return every IMDB review as an Arrow dataset with one id column."""
    imdb = load_dataset("stanfordnlp/imdb", revision=IMDB_REVISION)
    all_reviews = concatenate_datasets([
        imdb["train"],
        imdb["test"],
        imdb["unsupervised"],
    ])
    return ds.dataset(pa.table({
        "review_id": [f"review-{i}" for i in range(len(all_reviews))],
        "review": all_reviews["text"],
    }))


def main() -> None:
    reviews = load_reviews()
    n_docs = reviews.count_rows()
    print(f"reviews: {n_docs}", flush=True)

    with quail.Session() as session:
        session.register(
            "reviews",
            quail.DocumentProvider.from_dataset(reviews, id_col="review_id"),
        )
        query = session.sql(SQL, dialect="bq")
        print(query.explain(), flush=True)

        result = query.run()
        table = result.collect()
        report = result.report
        wall_s = report["wall_s"]
        print(f"matching reviews: {table.num_rows} of {n_docs}")
        print(f"boot_s: {report.get('boot_s')} ({report.get('boot_kind')})")
        print(f"wall_s: {wall_s}  (predicted about 285 s)")
        print(f"fresh_tokens: {report.get('fresh_tokens')}")
        print(f"documents/second: {n_docs / wall_s:.1f}")
        print(f"$/query: {wall_s / 3600 * H100_USD_PER_HOUR:.4f}")


if __name__ == "__main__":
    main()
