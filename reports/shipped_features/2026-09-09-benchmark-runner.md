# Shared benchmark execution and reporting

QUAIL-B now provides `run(run_query, ...)` and `quail-b report <run-dir>`.
The engine callback returns final document IDs, predicate answers, and timings.
QUAIL-B validates inputs, loads reference labels from S3, saves answers,
computes metrics, and writes a Markdown report.

Runs record exact query definitions and label collection IDs. Saved answers
can be scored again without inference, including after copying the directory.
Query failures retain completed outputs and record an error. Existing runs
cannot be overwritten.

Quail's runner now only translates queries and converts results.
The separate evaluator class, old metric-summary helpers, and Quail-specific
answer-saving helper are removed. Modal scripts still own resources and volume
commits. All query families resolve the same label collection before execution.

CPU tests cover filters, joins, all published scale factors, missing measurements,
saved results, failure handling, and report generation. No GPU benchmark was run.
