# Direct benchmark loading

Callers can load a published document table with
`quail_b.load_table("reviews", limit=100)` and get a query definition with
`quail_b.get_query("IMDB-1")`. Data reads default to the public S3 bucket.
An optional root selects a local directory or another public S3 location.

The S3Files and LocalFiles classes are removed. Label loaders now accept a
root path instead of a file-store object. Arrow handles file reads and S3
listing. Applications write their own run records with normal file operations.

Query definitions, sampling, source revisions, corpus ids, and labels are
unchanged. A row limit selects the first rows of an existing published table.
No model execution or benchmark rerun is part of this change.
