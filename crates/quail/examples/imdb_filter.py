"""Filter IMDB reviews from an Arrow IPC file with a language model."""

import sys

import pyarrow as pa

import quail

SQL = """
SELECT r.id
FROM reviews r
WHERE AI.IF(PROMPT(
    'Does this review mention a positive aspect of the movie? {0}',
    r.body
))
"""

path = sys.argv[1] if len(sys.argv) > 1 else "reviews.arrow"
with pa.memory_map(path, "r") as source:
    reviews = pa.ipc.open_file(source).read_all()

config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
provider = quail.DocumentProvider.from_table(reviews, id_col="id")

with quail.Session(config=config) as session:
    session.register("reviews", provider)
    table = session.sql(SQL, dialect="bq").collect()

print(f"Matching reviews: {table.num_rows}")
print(table.slice(0, 10).to_pandas().to_markdown(index=False))
