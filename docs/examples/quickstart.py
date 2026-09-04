"""Run the documentation quickstart on real IMDB reviews.

Reads eight reviews from the pinned stanfordnlp/imdb revision QUAIL-B
uses, registers them with the twelve QUAIL-B movie aspects, and runs
the two quickstart queries on Modal. The printed output is what the
user guide shows.

    uv run python docs/examples/quickstart.py 2>&1 | tee results/docs_quickstart.log
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import quail
from quail.bench.quailb import ASPECTS, DISCUSS_ASPECT, F1, SOURCE_REVISIONS


def load_reviews(count: int = 8) -> pa.Table:
    """Return short, tag-free reviews from the pinned IMDB train split."""
    path = hf_hub_download(
        "stanfordnlp/imdb",
        "plain_text/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=SOURCE_REVISIONS["stanfordnlp/imdb"],
    )
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    picked = [t for t in texts if 250 < len(t) < 600 and "<br" not in t]
    picked = picked[:count]
    return pa.table({
        "id": [f"rv{i}" for i in range(len(picked))],
        "body": picked,
    })


def section(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


def main() -> None:
    reviews = load_reviews()
    aspects = pa.table({
        "id": [f"as{i}" for i in range(len(ASPECTS))],
        "aspect": list(ASPECTS),
    })
    section("reviews")
    for row in reviews.to_pylist():
        print(f"{row['id']}: {row['body']}")
    section("aspects")
    print(aspects.to_pydict())

    with quail.Session() as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            reviews, id_col="id"))
        session.register("aspects", quail.DocumentProvider.from_table(
            aspects, id_col="id"))

        filter_sql = f"""
            SELECT r.id
            FROM reviews r
            WHERE AI_FILTER(PROMPT('{F1}', r.body))
        """
        section("filter sql")
        print(filter_sql)
        query = session.sql(filter_sql)
        section("filter explain")
        print(query.explain())
        section("filter run")
        result = query.run()
        print(result.to_rows())
        section("filter report")
        print(json.dumps(result.report, indent=2, default=str))

        join_sql = f"""
            SELECT r.id, a.aspect
            FROM reviews r
            JOIN aspects a
              ON AI_FILTER(PROMPT('{DISCUSS_ASPECT}', r.body, a.aspect),
                           {{'selectivity': 0.15}})
            WHERE AI_FILTER(PROMPT('{F1}', r.body), {{'selectivity': 0.6}})
        """
        section("join sql")
        print(join_sql)
        join = session.sql(join_sql)
        section("join explain")
        print(join.explain())
        section("join run")
        pairs = join.run()
        print(pairs.collect().to_pandas().to_string(index=False))
        section("join report")
        print(json.dumps(pairs.report, indent=2, default=str))


if __name__ == "__main__":
    main()
