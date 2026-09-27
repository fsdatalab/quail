"""Find positive aspects in eight published IMDB reviews.

Model execution requires a CUDA GPU and the backend runtime.
Run on a GPU host with: uv run python demos/quickstart.py
"""

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import quail

IMDB_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
FILTER_PROMPT = (
    "Judge strictly from the review above whether it mentions at least one "
    "positive aspect of the movie.\n\n{0}\n\nInstruction: answer TRUE if the "
    "review mentions at least one positive aspect of the movie, FALSE "
    "otherwise."
)
FILTER_SQL = f"""
    SELECT r.id
    FROM reviews r
    WHERE AI.IF(PROMPT('{FILTER_PROMPT}', r.body))
"""


def load_reviews(count: int = 8) -> pa.Table:
    """Load short reviews from the public IMDB train split."""
    path = hf_hub_download(
        "stanfordnlp/imdb",
        "plain_text/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=IMDB_REVISION,
    )
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    picked = [
        text for text in texts
        if 250 < len(text) < 600 and "<br" not in text
    ][:count]
    return pa.table({
        "id": [f"rv{index}" for index in range(len(picked))],
        "body": picked,
    })


def run_query():
    """Return reviews that mention at least one positive movie aspect."""
    reviews = load_reviews()
    config = quail.EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    )
    with quail.Session(config) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            reviews, id_col="id"))
        query = session.sql(FILTER_SQL, dialect="bq")
        print(f"positive-aspect filter ({reviews.num_rows} reviews)")
        result = query.run()
        return result.collect(), result.report


if __name__ == "__main__":
    rows, report = run_query()
    print(rows.to_pylist())
    print(report)
