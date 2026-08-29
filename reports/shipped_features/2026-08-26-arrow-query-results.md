# Arrow query results

Quail no longer builds the final result as a Python list of tuples. The
runtime stores model answers in Arrow tables and uses Acero for the
ordinary joins on shared SQL aliases.

`QueryResult.execute_stream()` returns bounded Arrow record batches.
`QueryResult.collect(limit=...)` explicitly materializes an Arrow table.
`QueryResult.count()` counts the final rows inside Acero without creating
Python row tuples.

The old IMDB-9 result contained 71,085,448 Python tuples and occupied
4.6 GiB on the `quail-results` volume. BIO-7 later exhausted host memory
in the same result assembly code. With Arrow results, IMDB-9 counted
71,362,343 rows and BIO-7 counted 386,693,560 rows without exhausting
host memory.
