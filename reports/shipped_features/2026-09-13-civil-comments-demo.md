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

## First GPU run, 20,000 comments, local H100

Run by hand on one H100 with `--limit 20000`; no Modal function call id.
Predicted from the plan: 8.99 seconds of filter and 5.89 seconds of
join, 70,060 join pairs, about 4,700 output pairs.

| Quantity | Predicted | Measured |
|---|---:|---:|
| Filter pass rate | 11.3 percent | 96.6 percent |
| Join pairs evaluated | 70,060 | 599,168 |
| Join pass rate | 6.7 percent | 65.4 percent |
| Query time, s | under 14.9 | 135.65 |
| Document pairs/s | | 4,417 |
| Fresh tokens | | 15,716,441 |
| $/query | | 0.1488 |
| Model startup, s (cold) | | 58.86 |
| Output precision against the labels | | 0.012 |
| Output recall against the labels | | 0.990 |

The throughput was normal: 4,417 pairs per second, compared with 4,814
for FEV-9 and 2,339 for IMDB-3 in the 2026-09-11 report. The time was
off because the model answered TRUE to almost every question, so the
join evaluated 8.5 times the planned pairs, and 8.5 times the planned
time is 128 seconds, close to the 136 measured.

The prompts were the cause. The filter prompt had only the trailing
instruction; every QUAIL-B filter prompt states the criterion before
the document ("Judge strictly from the review above whether it ...")
and repeats it in the instruction. The join asked a generic "is the
statement true of the document" question. Both now follow the QUAIL-B
pattern. Prediction for the rerun with `--limit 20000`: the filter
passes 10 to 20 percent, the join evaluates 60,000 to 130,000 pairs,
and the query takes 15 to 30 seconds. Recall should drop from 0.99 and
precision should rise well above 0.012; the exact values are unknown
until the run.

## Second GPU run, 2,000 comments, local H100

Same machine, prompts in the QUAIL-B pattern, `--limit 2000`.
Predicted: filter passes 10 to 20 percent, 6,000 to 13,000 join pairs,
under 5 seconds of query time.

| Quantity | Predicted | Measured |
|---|---:|---:|
| Filter pass rate | 10 to 20 percent | 91.0 percent |
| Join pairs evaluated | 6,000 to 13,000 | 56,420 |
| Join pass rate | 6.7 percent | 28.4 percent |
| Query time, s | under 5 | 13.34 |
| Document pairs/s | | 4,229 |
| Output precision against the labels | | 0.026 |
| Output recall against the labels | | 0.924 |

The join improved (65 to 28 percent passing) but the filter did not.
The per-field counts say the model reads the statements loosely
rather than answering TRUE at random: "heterosexual" and every
"other ..." group were near the top, which a comment satisfies if it
mentions any people at all, and "rejected" was true of 98 percent.
So the wording is still too loose for Qwen3 4B on news comments. This
commit makes the identity statements require an explicit mention,
defines the "other" groups by exclusion, defines toxic against
ordinary disagreement, and tells the join to answer FALSE unless the
comment clearly matches. `--criterion` overrides the filter wording
from the command line so a control can be run: a criterion that is
never true of these comments ("is written entirely in French") should
pass close to 0 percent; if it passes most comments, the answer
reading is at fault, not the wording. Prediction for the next
`--limit 2000` run: the control passes under 2 percent; the toxicity
filter passes 10 to 30 percent.

## Opt-in GPU busy time

`EngineConfig(gpu_timing=True)` makes the Quail backend sum the CUDA
event pair each forward chunk already records into a `gpu_s` field of
the result report, per model node and in total. `wall_s` minus `gpu_s`
is the time the GPU sat idle during the query, which is the direct
measure of scheduling gaps on the streamed filter-to-join path. It is
off by default; the demo turns it on with `--gpu-timing`. The only
measurement of idle time so far (2026-09-07 report) came from the
Modal profiler and predates that path.

Prediction for the 2,000-comment run with `--gpu-timing`: idle time
under 10 percent of `wall_s`. The full filter and join chunks each run
for about a second while the next chunk packs in under 0.1 seconds, so
the gaps come only from partial chunks at the end of the query.

## Third GPU run, 2,000 comments, local H100

Stricter wording, `--limit 2000`, without `--gpu-timing`. Predicted:
filter passes 10 to 30 percent, join pass rate below 28 percent, query
time under 5 seconds.

| Quantity | Predicted | Measured |
|---|---:|---:|
| Filter pass rate | 10 to 30 percent | 27.1 percent |
| Join pairs evaluated | | 16,771 |
| Join pass rate | below 28 percent | 1.0 percent |
| Query time, s | under 5 | 5.44 |
| Document pairs/s | | 3,083 |
| Output precision against the labels | | 0.129 |
| Output recall against the labels | | 0.047 |

The filter is now in range, though still above the 11.6 percent the
labels give this sample. The join over-corrected: the labels give a
toxic comment 2.1 true fields on average and the model found 0.3, so
the sentence "Answer FALSE unless the comment clearly matches" is
removed again. The demo now also prints the filter's own precision
and recall against the toxicity labels, so filter errors and join
errors can be told apart. Prediction for the next run: the join passes
3 to 10 percent of pairs and output recall rises above 0.2; the filter
is unchanged at about 27 percent.

## Fourth GPU run, 2,000 comments, local H100

Join sentence removed, `--limit 2000`. Predicted: filter unchanged at
about 27 percent, join passes 3 to 10 percent, output recall above 0.2.

| Quantity | Predicted | Measured |
|---|---:|---:|
| Filter pass rate | about 27 percent | 83.7 percent |
| Filter precision against the toxicity labels | 0.3 to 0.45 | 0.124 |
| Filter recall against the toxicity labels | | 0.954 |
| Join pairs evaluated | | 51,894 |
| Join pass rate | 3 to 10 percent | 13.7 percent |
| Query time, s | | 13.1 |
| Document pairs/s | | 3,961 |
| Output precision against the labels | | 0.054 |
| Output recall against the labels | 0.2 or more | 0.861 |

The filter prompt was identical in the third and fourth runs, and the
plans are identical in shape (checked on the CPU: both stream the
filter's survivors into the join with the same estimates). Yet the
filter passed 27.1 percent in the third run and 83.7 percent in the
fourth. The only thing that changed on the filter's side is the number
of extra KV rows a held survivor reserves for the join's frame, which
changes how documents are grouped into chunks, not what the model
reads. Either the third run used a different command, or the streamed
path's filter answers depend on chunk composition, which would be an
engine bug. `--filter-only` runs the same filter prompt through the
plain filter path with no join, so the two paths can be compared on
the same 2,000 comments. Prediction: if the engine is sound, the
filter-only pass rate equals the streamed run's 83.7 percent to within
one percentage point.

`gpu_s` did not appear in the fourth run's output because the session
dropped it when assembling the report; that is fixed in this commit.
