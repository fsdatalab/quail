"""Download the Stanford IMDB reviews and write reviews.arrow."""

import pyarrow as pa
from datasets import load_dataset

imdb = load_dataset(
    "stanfordnlp/imdb",
    split="train+test+unsupervised",
    revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
)
reviews = pa.table(
    {
        "id": [f"review-{index}" for index in range(len(imdb))],
        "body": imdb.data.table.column("text"),
    }
)

with pa.OSFile("reviews.arrow", "wb") as sink:
    with pa.ipc.new_file(sink, reviews.schema) as writer:
        writer.write_table(reviews)
