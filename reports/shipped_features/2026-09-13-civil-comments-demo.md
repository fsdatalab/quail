# 2026-09-13: Civil Comments demo, a toxicity filter and one join over 31 fields

This PR adds `demos/civil_comments_join.py`, a second runnable demo in
the shape of `demos/imdb_ending_filter.py` but over a larger corpus with
labels for every question it asks. No GPU run was made for this PR; the
numbers below are the planner's estimates and the label counts, and the
first GPU run should be reported against them.

## What changed

- `demos/civil_comments_join.py` loads the Jigsaw Civil Comments release
  from the Hugging Face mirror `pietrolesci/civilcomments-wilds`, config
  `raw`, pinned to revision `c227534c`. The file keeps every column of
  the Kaggle train file: 448,000 comments with the text, seven toxicity
  scores, twenty-four identity scores, the moderator rating, five reader
  reaction counts, annotator counts, and thread ids. The
  `google/civil_comments` mirror keeps only the seven toxicity scores, so
  it cannot answer the identity questions. Each score is the fraction of
  annotators who applied the label; a score of at least 0.5 is the label.
- The demo runs one query: keep the toxic comments, then join each with a
  31-row table of statements, one per remaining semantic field (six
  toxicity subtypes, twenty-four identity groups, and moderator
  rejection). The output is one row per (comment, field) the model says
  is true. The comment is the join anchor, so each comment is read once
  and the filter question and the 31 statements are short suffixes on
  its KV.
- After the run it prints precision and recall of the output pairs
  against the labeled pairs, where a labeled pair has both the toxicity
  score and the field's score at least 0.5.
- `--limit N` takes a fixed-seed random sample. The mirror's file is not
  shuffled: 74 percent of its first 20,000 rows are toxic, compared with
  11.3 percent overall.
- `--parquet` runs the same query on any other file with the Kaggle
  columns.

## Data, measured on the CPU

| Quantity | Value |
|---|---:|
| Comments with text | 447,998 |
| Toxic comments (toxicity score at least 0.5) | 50,794, 11.3 percent |
| Mean rate of a joined field among toxic comments | 6.7 percent |
| Labeled (comment, field) pairs | 105,655 |
| Comment tokens, Qwen3 tokenizer | 34,755,649, mean 77.6 per comment |
| Statement tokens, 31 rows | 348 |

The selectivities in the SQL, 0.113 for the filter and 0.067 for the
join, are these measured rates.

## Prediction for the first GPU run

One H100, Qwen3 4B FP8, all 447,998 comments. The planner's estimate on
the CPU:

| Stage | Estimated evaluations | Estimated seconds |
|---|---:|---:|
| Filter over comments | 447,998 | 200.2 |
| Join over surviving pairs | 1,569,337 | 132.3 |

The two stages share forward passes, so the query time should be below
their sum of 332 seconds, about 5 minutes; at $3.9492 per H100 hour that
is at most $0.36. With the filter passing 11.3 percent and the join
passing 6.7 percent of its pairs, the output should be about 105,000
pairs. Precision and recall against the labels are unknown until the
run; the join question is one generic prompt over 31 different
statements, so recall on the rare fields (severe toxicity, the small
identity groups) is the number to watch.

The `--limit 20000` sample plans at 8.99 seconds for the filter and 5.89
seconds for the join, with 4,757 labeled pairs.

## The other sizes

The Kaggle train file has 1,804,874 comments, four times this file, but
identity scores for only about 405,000 of them; the rest would score as
unlabeled. Scaling the estimate above by comment count gives about 22
minutes for the full Kaggle file on one H100. I do not know of a
16-million-comment release of Civil Comments; if one exists with the
same columns, `--parquet` runs it and the estimate scales to about 3.3
hours on one H100.
