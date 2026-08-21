# QUAIL-B query catalog

Every query currently defined in `quail/quail/bench/quailb.py`, as
written in code today. This replaces the old B1-B16 queries (planted
YES/NO flags on padded, concatenated text). All 16 queries below run
over real, unpadded, un-concatenated text.

Three base document tables, each scaling with `--sf`:

| Table | Real source | Column |
|---|---|---|
| `reviews` | IMDB movie reviews | `body` |
| `reports` | BioDEX medical case reports | `report` |
| `claims` | FEVER claims | `claim` |

Two fixed partner tables (do not scale with `--sf`):

| Table | Contents | Column |
|---|---|---|
| `aspects` | 12 movie aspects (acting, plot, ...) | `aspect` |
| `terms` | up to 2,560 BioDEX reaction terms, real vocabulary | `term` |

Plus `evidence` - the Wikipedia passages FEVER claims reference,
scaled by whichever claims get sampled, not by `--sf` directly.

`aspects` and `terms` are short words or phrases, not documents - not
worth filtering on their own content. `evidence` is real Wikipedia
text, so it's the one partner table a filter makes sense on (see
FEV-6 below).

## Shapes

Each dataset gets the same five shapes: filter alone, join alone,
then a filter chain of depth 1, 2, and 3 all feeding the *same single*
join - not a second join. A document survives every filter in the
chain, then joins against the partner table.

Deeper filter chains stand in for a second, dependent join
(filter -> filter -> join -> join). Two joins in one query, where the
second join only runs over whatever the first join kept, is a real
shape, but it's a much less exercised path in the engine today
(gating, dedup, and replay all have to hold across the join
boundary), so it's left out here rather than folded in silently.

FEVER's filter chain stops at depth 2 (FEV-1..FEV-4), not depth 3:
F11 -> F12 -> F14 (person, date, place) returned 0 rows at both
sf=0.1 (0/10) and sf=0.2 (0/18) - see the F14 note below for why -
and since the goal of this benchmark is testing joins, not stacking
filters, that query was dropped rather than patched again. FEVER gets
two queries in its place instead: FEV-6, a two-sided filter -> join
(one filter on each table before the join runs, not just the anchor
side), and FEV-7, the same shape with a second filter on the claims
side first. This is the only dataset where the two-sided shape is
meaningful, since `evidence` is the only partner table with real,
substantial text.

Selectivity hints are omitted everywhere - none of these predicates
have been through a judge pass yet, so there's no measured number to
give the planner. The planner falls back to running stages in the
order they're written.

---

## IMDB (reviews, aspects)

| ID | Shape | Tables | Selects |
|---|---|---|---|
| IMDB-1 | filter: F1 (positive opinion) | reviews | r.id |
| IMDB-2 | join: J1 (reviews x aspects) | reviews, aspects | r.id, a.id |
| IMDB-3 | F1 -> J1, dependent | reviews, aspects | r.id, a.id |
| IMDB-4 | F1 -> F4 -> J1, 2 filters then 1 join | reviews, aspects | r.id, a.id |
| IMDB-5 | F1 -> F4 -> F5 -> J1, 3 filters then 1 join | reviews, aspects | r.id, a.id |

**F1** (filter, reviews):
```
Judge strictly from the review above whether it expresses an overall
positive opinion of the movie.

{0}

Instruction: answer YES if the review expresses an overall positive
opinion of the movie, NO otherwise.
ANSWER=
```

**F4** (filter, reviews):
```
Judge strictly from the review above whether it discusses the ending
of the movie.

{0}

Instruction: answer YES if the review discusses the ending of the
movie, NO otherwise.
ANSWER=
```

**F5** (filter, reviews):
```
Judge strictly from the review above whether it mentions any specific
actor or actress by name.

{0}

Instruction: answer YES if the review mentions a specific actor or
actress by name, NO otherwise.
ANSWER=
```

**DISCUSS_ASPECT** (join, reviews x aspects):
```
Candidate movie aspects follow, one at a time. For each, judge
strictly from the review above whether it discusses that aspect of
the movie.

{0}

ASPECT: {1}
Instruction: answer YES if the review above discusses this aspect, NO
otherwise.
ANSWER=
```

---

## BioDEX (reports, terms)

| ID | Shape | Tables | Selects |
|---|---|---|---|
| BIO-1 | filter: F7 (female patient) | reports | r.id |
| BIO-2 | join: J1 (reports x terms) | reports, terms | r.id, m.id |
| BIO-3 | F7 -> J1, dependent | reports, terms | r.id, m.id |
| BIO-4 | F7 -> F8 -> J1, 2 filters then 1 join | reports, terms | r.id, m.id |
| BIO-5 | F7 -> F8 -> F9 -> J1, 3 filters then 1 join | reports, terms | r.id, m.id |

BIO-2 is the direct replacement for the old B5 (the join that crashed
twice during the 32B confirm run, on a Modal client bug, not this
query) - same shape, reports x terms, full cross product.

**F7** (filter, reports):
```
Judge strictly from the report above whether it describes a case
involving a female patient.

{0}

Instruction: answer YES if the report describes a case involving a
female patient, NO otherwise.
ANSWER=
```

**F8** (filter, reports):
```
Judge strictly from the report above whether it describes combination
drug therapy.

{0}

Instruction: answer YES if the report describes combination drug
therapy, NO otherwise.
ANSWER=
```

**F9** (filter, reports):
```
Judge strictly from the report above whether it describes a serious
or life-threatening adverse event.

{0}

Instruction: answer YES if the report describes a serious or
life-threatening adverse event, NO otherwise.
ANSWER=
```

**REACTION** (join, reports x terms):
```
Candidate medical reaction terms follow, one at a time. For each,
judge strictly from the report above whether it describes that
reaction as something the patient experienced.

{0}

CANDIDATE REACTION: {1}
Instruction: answer YES if the report above describes this reaction,
NO otherwise.
ANSWER=
```

---

## FEVER (claims, evidence)

| ID | Shape | Tables | Selects |
|---|---|---|---|
| FEV-1 | filter: F11 (about a person) | claims | c.id |
| FEV-2 | join: J3 (claims x evidence) | claims, evidence | c.id, e.id |
| FEV-3 | F11 -> J3, dependent | claims, evidence | c.id, e.id |
| FEV-4 | F11 -> F12 -> J3, 2 filters then 1 join | claims, evidence | c.id, e.id |
| FEV-6 | two-sided filter -> join: F11 on claims, F13 on evidence, then J3 | claims, evidence | c.id, e.id |
| FEV-7 | two-sided filter -> join, deeper: F11 -> F12 on claims, F13 on evidence, then J3 | claims, evidence | c.id, e.id |

FEV-2 is the direct replacement for the old B16 (`fever_query.py`).

FEV-6 is the two-sided-filter join: instead of filtering only the
anchor (claims) before the join, both sides are filtered
independently - claims by F11, evidence by F13 - before the join
predicate (SUPPORT) runs on whatever survives both filters. FEV-7 is
the same shape, one filter deeper on the claims side: F11 and F12
both filter claims, F13 still filters evidence alone, then the join
runs on whatever survives all three.

**F11** (filter, claims):
```
Judge strictly from the claim above whether it asserts something
about a person, rather than an organization, place, or event.

{0}

Instruction: answer YES if the claim asserts something about a
person, NO otherwise.
ANSWER=
```

**F12** (filter, claims):
```
Judge strictly from the claim above whether it contains a specific
date or year.

{0}

Instruction: answer YES if the claim contains a specific date or
year, NO otherwise.
ANSWER=
```

**F14** (filter, claims):
```
Judge strictly from the claim above whether it references a specific
place (a city, country, or other named location).

{0}

Instruction: answer YES if the claim references a specific place, NO
otherwise.
ANSWER=
```

F14 is used standalone by `judge_pass.py`'s per-predicate agreement
check (over the full claims population). It was also tried as the
third stage of a FEV-5 filter chain (F11 -> F12 -> F14: person, date,
place) but that returned 0 rows at both sf=0.1 (0/10) and sf=0.2
(0/18) - real FEVER claims are single-fact sentences, so "X was born
on [date]" and "X was born in [place]" are separate claims, not
combined, and the F11+F12 population essentially never also names a
place. That's a structural non-overlap, not a sample-size problem, so
scaling sf further wouldn't have fixed it. Since this benchmark's
primary goal is testing joins rather than deep filter chains, FEV-5
was removed rather than patched with a different third predicate.

**F13** (filter, evidence - FEV-6 only):
```
Judge strictly from the Wikipedia passage above whether it primarily
describes a specific person (their life, actions, or role), rather
than an organization, place, or event.

{0}

Instruction: answer YES if the passage primarily describes a specific
person, NO otherwise.
ANSWER=
```

**SUPPORT** (join, claims x evidence):
```
Wikipedia passages follow, one at a time. For each, judge strictly
from the claim above whether it supports that claim.

{0}

PASSAGE:
{1}
Instruction: answer YES if the passage above supports this claim, NO
otherwise.
ANSWER=
```

---

## Open items

- None of these predicates have been through a judge pass (an offline
  accuracy check against a stronger model) yet. Wording may need to
  change if a predicate misses a 90% agreement floor or clusters
  selectivity with another predicate on the same table.
- Two shapes from the original design are not implemented: join-first
  (filter a join's output) and an F-J-F-J interleaved chain. Both need
  a filter to run after a join, which the query builder can't express
  today - see the module docstring in `quailb.py` for why.
- A genuine second, dependent join (filter -> filter -> join -> join)
  is also not implemented here, on purpose - see the "Shapes" section
  above. If a query needs it, that's a deliberate addition, not an
  automatic extension of this catalog.
- `reports/query-design.md`, referenced in `quailb.py`'s docstring,
  does not exist yet in this repo. This catalog is the query list and
  prompts only, not the full design rationale.
