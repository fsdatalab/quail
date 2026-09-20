# Civil Comments semantic query

This demo compares Quail and Jev on a sample of 10,000 public comments from
[Jigsaw Civil Comments][dataset].

Human annotators rated each comment for toxicity, six toxicity subtypes, and
24 identity mentions; each score is the fraction of annotators who selected
that label; this demo calls a label positive at 50% or higher.

The labels are subjective and noisy; perfect accuracy is impossible, and
strong agreement is hard; the query is still interesting, it combines a
semantic filter with a semantic join, and it shows the effect of planning,
KV reuse, and API cost.

[dataset]: https://huggingface.co/datasets/pietrolesci/civilcomments-wilds

## Query in English

Select comments judged toxic. For each selected comment, return every
applicable semantic field.

The 30 fields comprise:

- Six toxicity subtypes: severe toxicity, obscene, threat, insult, identity
  attack, and sexual explicit.
- Four gender identities.
- Four sexual orientations.
- Seven religions.
- Five races or ethnicities.
- Four disability categories.

Each field has a definition plus one TRUE and one FALSE calibration example.
Historical `rejected` moderation metadata is excluded because it is not a
semantic property of the comment text.

## SQL

```sql
SELECT
    c.comment_id,
    f.field
FROM comments AS c
JOIN fields AS f
  ON AI_FILTER(
      PROMPT(:FIELD_PROMPT, c.text, f.statement)
  )
WHERE AI_FILTER(
    PROMPT(:TOXICITY_PROMPT, c.text)
);
```

The query has no selectivity hints.

## Execution strategies

Quail pushes the toxicity predicate below the join.

Jev supports two configurations:

- **No query planning:** Send one API request per comment. Each request has 31
  `noul` questions: one toxicity question and 30 semantic-field questions.
  Apply the toxicity answer after the request.
- **Push down predicate:** Send one toxicity question per comment. For each
  survivor, send one request containing all 30 semantic-field questions.

The Jev client never sends one request per comment-field pair.

## Results

All primary time and cost numbers exclude model startup.

| Backend | Query time | Tokens/s | Cost | Cost vs. Quail | Filter F1 | Join F1 |
|---|---:|---:|---:|---:|---:|---:|
| Quail, 1 H100 | **85.53 s** | 284,022 | **$0.0938** | **1.00×** | 0.413 | **0.287** |
| Jev, num_threads = 256 (no query planning) | 244.87 s | **313,379** | $1.8892 | **20.13×** | 0.425 | 0.260 |
| Jev, num_threads = 256 (push down predicate) | 100.20 s | 253,935 | $0.7705 | **8.21×** | **0.430** | 0.263 |

## Run

```bash
uv run modal run --detach demos/civil_comments/quail_backend.py \
  --limit 10000 --gpus 1

read -s TYPESAFE_API_KEY
export TYPESAFE_API_KEY

uv run python demos/civil_comments/jev_backend.py \
  --limit 10000 --concurrency 256 --plan-order sql

uv run python demos/civil_comments/jev_backend.py \
  --limit 10000 --concurrency 256 --plan-order pushdown
```
