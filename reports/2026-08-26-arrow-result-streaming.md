# Arrow result streaming for large joins

## Setup

The check ran IMDB-9 and BIO-7 from QUAIL-B at scale factor 0.1. Each
query used Qwen3 4B fp8 on one H100!. The two queries ran in separate
Modal containers attached to the existing `quail-milestone1` app.

The SQL stayed unchanged. Both queries still select their final ID
columns. Quail stored the model's TRUE and FALSE answers in Arrow
tables. Acero joined the TRUE answers and counted the final rows without
creating Python row tuples.

The aggregate result is
`/results/benchmarks/quailb/runs/qb_20260827T065151Z_46683ed6/20260827T065151Z-quailb-sf0.1-lf1-qwen3-4b-fp8-parallel2.json`
on the `quail-results` Modal volume. The query calls were
`fc-01M10ZTN6YQWKEHA17NHJ685AE` and
`fc-01M10ZTNBXDGNF3JHNFA9BWVHM`.

## Prediction

Both queries would finish without exhausting host memory. Acero would
count the final rows without materializing them in Python. Each raw JSON
record would remain small because the Arrow answer tables would be saved
as separate Parquet files.

## Result

Both queries finished.

| Query | Model calls | Final rows | GPU query time | Raw JSON | Answer Parquet files |
|---|---:|---:|---:|---:|---:|
| IMDB-9 | 158,040 | 71,362,343 | 49.24 seconds | 5.2 KiB | 71.5 KiB |
| BIO-7 | 356,948 | 386,693,560 | 85.95 seconds | 5.2 KiB | 75.0 KiB |

Figure: plots/arrow_result_streaming.png

The earlier IMDB-9 run saved 71,085,448 Python tuples in
`/results/benchmarks/quailb/runs/qb_20260827T053433Z_29c0709d/single/IMDB-9.json`.
The file is 4.6 GiB on the volume. The same container later exhausted
host memory while constructing BIO-7. The new run produced both counts
and did not exhaust host memory.

## Meaning

The number of model calls and the number of final rows are separate.
The model evaluated 356,948 BIO-7 document tuples. The ordinary joins
between the TRUE answer tables produced 386,693,560 final rows.

Quail now keeps that CPU work in Arrow and Acero. `execute_stream()`
returns bounded Arrow record batches. `collect(limit=...)` explicitly
materializes no more than the requested number of rows in an Arrow
table. `count()` computes the exact output count inside Acero.
