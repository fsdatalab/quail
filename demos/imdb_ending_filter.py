"""Filter all 100,000 IMDB reviews with two questions, both required."""

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
    AND AI.IF(
        PROMPT(
            'Does the reviewer recommend watching the movie?\\n\\n{0}',
            r.review
        ),
        {'selectivity': 0.5}
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
        boot_s = report.get("boot_s", 0.0)
        total_s = wall_s + boot_s
        cost = total_s / 3600 * H100_USD_PER_HOUR
        print(f"matching reviews: {table.num_rows} of {n_docs}")
        for stage in report.get("stages", ()):
            if stage.get("op") == "filter":
                print(f"  stage evaluated {stage['evaluated']} reviews, "
                      f"{stage['observed_selectivity']:.3f} passed")
        print(f"boot_s: {boot_s} ({report.get('boot_kind')})")
        print(f"token_wait_s: {report.get('token_wait_s')}")
        print(f"wall_s: {wall_s}")
        print(f"total_s: {total_s:.2f} (boot + query)")
        print(f"fresh_tokens: {report.get('fresh_tokens')}")
        print(f"documents/second: {n_docs / wall_s:.1f}")
        print(f"GPU cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
