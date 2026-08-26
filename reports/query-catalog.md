# QUAIL-B query catalog

Every query currently defined in `quail/bench/quailb.py`'s `queries()`
function, as written in code today. A living reference doc, not a
per-PR report - update it in place when the catalog changes.

Four base document tables, each scaling with `--sf`:

| Table | Real source | Column |
|---|---|---|
| `reviews` | IMDB movie reviews | `body` |
| `reports` | BioDEX medical case reports | `report` |
| `claims` | FEVER claims | `claim` |
| `citations` | LePaRD legal citation events | `destination_context` (anchor), `passage_text` (partner) |

Three fixed partner tables (do not scale with `--sf`):

| Table | Contents | Column |
|---|---|---|
| `aspects` | 12 movie aspects (acting, plot, ...) | `aspect` |
| `terms` | up to 2,560 BioDEX reaction terms, real vocabulary | `term` |
| `severe_terms` | the 64 most common of `terms`, BIO-6's second join only | `term` |

Plus `evidence` - the Wikipedia passages FEVER claims reference,
scaled by whichever claims get sampled, not by `--sf` directly.
`citations` is self-joined: the anchor alias reads
`destination_context` (the citing excerpt), the partner alias reads
`passage_text` (the cited passage), both from the same registered
provider.

35 queries total: IMDB x10, BioDEX x8, FEVER x9, LePaRD x8.

## Shapes

Each dataset gets the same five base shapes: filter alone, join
alone, then a filter chain of depth 1, 2, and 3 (5 for LePaRD) all
feeding the *same single* join - a document survives every filter in
the chain, then joins against the partner table. On top of that:

- **The star shape** (IMDB-8, BIO-6, FEV-7): two joins anchored on the
  same table, a barrier between the two stages but no anchor switch.
- **The forced-switch chain** (IMDB-9/11, BIO-C/D, FEV-C/D): a real
  3-join path across 4 table positions where no table spans all 3
  edges, so the free anchor search cannot collapse it to one group -
  a barrier is structurally unavoidable regardless of which anchors
  the cost search picks. The "/11"/"/D" variant adds a leading filter
  on the first table, so the forced switch and a filter chain are
  tested together, not separately. IMDB and FEVER chain directly
  through their own scaling tables; BioDEX alternates back through
  `reports` instead of `terms` (a fixed ~2,560-row table) to avoid a
  cardinality blowup.
- **Two-sided pushdown** (FEV-5/6, LEP-7): both sides of the join are
  filtered independently before the join predicate runs, not just the
  anchor side. FEVER's `evidence` and LePaRD's self-join partner are
  the only partner tables with real, substantial text worth filtering
  on their own.

Deeper filter chains stand in for a second, *dependent* join (filter
-> filter -> join -> join, one join's output feeding another). That
shape is not implemented in this catalog on purpose - see "Open
items" below.

FEVER's filter chain stops at depth 2 (FEV-1..4), not depth 3: F11 ->
F12 -> F14 (person, date, place) returned 0 rows at both sf=0.1 (0/10)
and sf=0.2 (0/18) - real FEVER claims are single-fact sentences, so
"X was born on [date]" and "X was born in [place]" are separate
claims, not combined, and the F11+F12 population essentially never
also names a place. That's a structural non-overlap, not a
sample-size problem. F14 is still used standalone by
`judge_pass.py`'s per-predicate agreement check.

Selectivity hints are omitted everywhere - none of these predicates
have been through a judge pass yet, so there's no measured number to
give the planner. The planner falls back to running stages in the
order they're written.

---

## IMDB (reviews, aspects)

| ID | Shape | Tables | Selects |
|---|---|---|---|
| IMDB-1 | filter: F1 (at least one positive aspect) | reviews | r.id |
| IMDB-2 | join: J1 (DISCUSS_ASPECT), reviews x aspects | reviews, aspects | r.id, a.id |
| IMDB-3 | F1 -> J1, dependent | reviews, aspects | r.id, a.id |
| IMDB-4 | F1 -> F4 -> J1, 2 filters then 1 join | reviews, aspects | r.id, a.id |
| IMDB-5 | F1 -> F4 -> F5 -> J1, 3 filters then 1 join | reviews, aspects | r.id, a.id |
| IMDB-6 | F1 -> F4, 2 filters, no join | reviews | r.id |
| IMDB-7 | F1 -> F4 -> F5, 3 filters, no join | reviews | r.id |
| IMDB-8 | 2J, same anchor (star): J1 (DISCUSS_ASPECT) -> J2 (ASPECT_SENTIMENT) | reviews, aspects, aspects | r.id, a.id, a2.id |
| IMDB-9 | engine stress: 3J real chain r-a1-a2-a3, forced anchor switch | reviews, aspects x3 | r.id, a1.id, a2.id, a3.id |
| IMDB-11 | engine stress: F1 filter -> IMDB-9's chain | reviews, aspects x3 | r.id, a1.id, a2.id, a3.id |

**F1** (filter, reviews):
```
Judge strictly from the review above whether it mentions at least one
positive aspect of the movie.

{0}

Instruction: answer TRUE if the review mentions at least one positive
aspect of the movie, FALSE otherwise.
```

**F4** (filter, reviews):
```
Judge strictly from the review above whether it discusses the ending
of the movie.

{0}

Instruction: answer TRUE if the review discusses the ending of the
movie, FALSE otherwise.
```

**F5** (filter, reviews):
```
Judge strictly from the review above whether it mentions any specific
actor or actress by name.

{0}

Instruction: answer TRUE if the review mentions a specific actor or
actress by name, FALSE otherwise.
```

**DISCUSS_ASPECT** (join, reviews x aspects):
```
Does the review in DOCUMENT {0} discuss the movie aspect in DOCUMENT
{1}?
```

**ASPECT_SENTIMENT** (join, reviews x aspects - IMDB-8's second stage
only):
```
Does the review in DOCUMENT {0} express positive sentiment about the
movie aspect in DOCUMENT {1}?
```

**ASPECT_RELATED** (join, aspects x aspects - IMDB-9/11 only): an
engine stress test, not a meaningful accuracy query - `{0}`/`{1}` here
are short aspect phrases, not documents worth judging on their own.
```
Are the movie aspects in DOCUMENT {0} and DOCUMENT {1} commonly
discussed together in the same review?
```

---

## BioDEX (reports, terms)

| ID | Shape | Tables | Selects |
|---|---|---|---|
| BIO-1 | filter: F7 (female patient) | reports | r.id |
| BIO-2 | join: J1 (REACTION), reports x terms | reports, terms | r.id, m.id |
| BIO-3 | F7 -> J1, dependent | reports, terms | r.id, m.id |
| BIO-4 | F7 -> F8 -> J1, 2 filters then 1 join | reports, terms | r.id, m.id |
| BIO-5 | F7 -> F8 -> F9 -> J1, 3 filters then 1 join | reports, terms | r.id, m.id |
| BIO-6 | 2J, same anchor (star): J1 (REACTION) -> J2 (REACTION_SEVERE) | reports, terms, severe_terms | r.id, m.id, m2.id |
| BIO-C | engine stress: 3J real chain r1-m1-r2-m2, forced anchor switch | reports, terms, reports, terms | r1.id, m1.id, r2.id, m2.id |
| BIO-D | engine stress: F7 filter -> BIO-C's chain | reports, terms, reports, terms | r1.id, m1.id, r2.id, m2.id |

BIO-2 is the direct replacement for the old B5 (the join that crashed
twice during the 32B confirm run, on a Modal client bug, not this
query) - same shape, reports x terms, full cross product. BIO-C/D
alternate back through `reports` instead of chaining straight through
`terms` a second time, since `terms` is a fixed ~2,560-row table (not
scaled by `--sf`) and a direct term-to-term chain risks a cardinality
blowup; `reports` does scale with `--sf`, so the chain stays a real
3-edge path with no table spanning all 3 edges either way.

**F7** (filter, reports):
```
Judge strictly from the report above whether it describes a case
involving a female patient.

{0}

Instruction: answer TRUE if the report describes a case involving a
female patient, FALSE otherwise.
```

**F8** (filter, reports):
```
Judge strictly from the report above whether it describes combination
drug therapy.

{0}

Instruction: answer TRUE if the report describes combination drug
therapy, FALSE otherwise.
```

**F9** (filter, reports):
```
Judge strictly from the report above whether it describes a serious
or life-threatening adverse event.

{0}

Instruction: answer TRUE if the report describes a serious or
life-threatening adverse event, FALSE otherwise.
```

**REACTION** (join, reports x terms):
```
Does the medical report in DOCUMENT {0} describe the reaction in
DOCUMENT {1} as something the patient experienced?
```

**REACTION_SEVERE** (join, reports x severe_terms - BIO-6's second
stage only):
```
Does the medical report in DOCUMENT {0} describe the reaction in
DOCUMENT {1} as serious or life threatening for the patient?
```

---

## FEVER (claims, evidence)

| ID | Shape | Tables | Selects |
|---|---|---|---|
| FEV-1 | filter: F11 (about a person) | claims | c.id |
| FEV-2 | join: J3 (SUPPORT), claims x evidence | claims, evidence | c.id, e.id |
| FEV-3 | F11 -> J3, dependent | claims, evidence | c.id, e.id |
| FEV-4 | F11 -> F12 -> J3, 2 filters then 1 join | claims, evidence | c.id, e.id |
| FEV-5 | two-sided pushdown: F11 on claims, F13 on evidence, then J3 | claims, evidence | c.id, e.id |
| FEV-6 | two-sided pushdown, deeper: F11 -> F12 on claims, F13 on evidence, then J3 | claims, evidence | c.id, e.id |
| FEV-7 | 2J, same anchor (star): J1 (SUPPORT) -> J2 (REFUTE) | claims, evidence, evidence | c.id, e.id, e2.id |
| FEV-C | engine stress: 3J real chain c1-e1-c2-e2, forced anchor switch | claims, evidence, claims, evidence | c1.id, e1.id, c2.id, e2.id |
| FEV-D | engine stress: F11 filter -> FEV-C's chain | claims, evidence, claims, evidence | c1.id, e1.id, c2.id, e2.id |

FEV-2 is the direct replacement for the old B16 (`fever_query.py`).
FEV-7 is meaningful, not synthetic: FEVER's real labels are
support/refute/not-enough-info, so asking whether a second passage
refutes the claim is a real question. FEV-C/D chain directly through
`claims`/`evidence` (both scale with `--sf`, no fixed-large table like
BioDEX's `terms`), so no cardinality-blowup workaround is needed
there.

**F11** (filter, claims):
```
Judge strictly from the claim above whether it asserts something
about a person, rather than an organization, place, or event.

{0}

Instruction: answer TRUE if the claim asserts something about a
person, FALSE otherwise.
```

**F12** (filter, claims):
```
Judge strictly from the claim above whether it contains a specific
date or year.

{0}

Instruction: answer TRUE if the claim contains a specific date or
year, FALSE otherwise.
```

**F14** (filter, claims - not part of any query in this catalog;
used standalone by `judge_pass.py`'s per-predicate agreement check):
```
Judge strictly from the claim above whether it references a specific
place (a city, country, or other named location).

{0}

Instruction: answer TRUE if the claim references a specific place,
FALSE otherwise.
```

**F13** (filter, evidence - FEV-5/6's two-sided pushdown only):
```
Judge strictly from the Wikipedia passage above whether it primarily
describes a specific person (their life, actions, or role), rather
than an organization, place, or event.

{0}

Instruction: answer TRUE if the passage primarily describes a
specific person, FALSE otherwise.
```

**SUPPORT** (join, claims x evidence):
```
Does the Wikipedia passage in DOCUMENT {1} support the claim in
DOCUMENT {0}?
```

**REFUTE** (join, claims x evidence - FEV-7's second stage only):
```
Does the Wikipedia passage in DOCUMENT {1} refute or contradict the
claim in DOCUMENT {0}?
```

---

## LePaRD (citations, self-joined)

One table, `citations`, self-joined: the anchor alias ("d") reads
`destination_context`, the partner alias ("s") reads `passage_text`,
both from the same registered provider. "Excerpt" is used throughout
for `destination_context`, to avoid colliding with FEVER's own use of
"passage" for quoted/cited text.

| ID | Shape | Tables | Selects |
|---|---|---|---|
| LEP-1 | filter: LEP1 (reasoning does not apply) | citations | d.id |
| LEP-2 | join: self-join (LEPJOIN), citations x citations | citations | d.id, s.id |
| LEP-3 | LEP1 -> join, dependent | citations | d.id, s.id |
| LEP-4 | LEP1 -> LEP2 -> join, 2 filters then 1 join | citations | d.id, s.id |
| LEP-5 | LEP1 -> LEP2 -> LEP3 -> join, 3 filters then 1 join | citations | d.id, s.id |
| LEP-6 | LEP1..LEP5 -> join, 5 filters then 1 join | citations | d.id, s.id |
| LEP-7 | two-sided pushdown: LEP1+LEP2 on excerpts, LEPS1 on passages, then the self-join | citations | d.id, s.id |
| LEP-8 | LEP1..LEP5, 5 filters, no join | citations | d.id |

LEP-2..LEP-7's join predicate has real ground truth (`passage_id`,
from the dataset itself, not a judge pass) - see `judge_pass.py`'s
LEP probe. LEP-6/LEP-8 are the deepest filter chains in the suite -
LEP-8 is five stages of KV reuse with no join work mixed in.

**LEP1** (filter, citations - destination_context):
```
Judge strictly from the excerpt above whether it argues that the
cited case's reasoning does not apply here.

{0}

Instruction: answer TRUE if the excerpt argues the cited case's
reasoning does not apply here, FALSE otherwise.
```

**LEP2** (filter, citations - destination_context):
```
Judge strictly from the excerpt above whether it discusses a
procedural or jurisdictional issue.

{0}

Instruction: answer TRUE if the excerpt discusses a procedural or
jurisdictional issue, FALSE otherwise.
```

**LEP3** (filter, citations - destination_context):
```
Judge strictly from the excerpt above whether it treats the cited
passage as binding precedent.

{0}

Instruction: answer TRUE if the excerpt treats the cited passage as
binding precedent, FALSE otherwise.
```

**LEP4** (filter, citations - destination_context):
```
Judge strictly from the excerpt above whether it cites the passage to
support a conclusion about a party's liability or guilt.

{0}

Instruction: answer TRUE if the excerpt cites the passage to support
a conclusion about a party's liability or guilt, FALSE otherwise.
```

**LEP5** (filter, citations - destination_context):
```
Judge strictly from the excerpt above whether it acknowledges
disagreement between courts on the issue.

{0}

Instruction: answer TRUE if the excerpt acknowledges disagreement
between courts on the issue, FALSE otherwise.
```

**LEPS1** (filter, citations - passage_text; LEP-7's two-sided
pushdown only):
```
Judge strictly from the passage above whether it states a general
legal rule.

{0}

Instruction: answer TRUE if the passage states a general legal rule,
FALSE otherwise.
```

**LEPJOIN** (join, citations x citations - the self-join predicate):
```
Is the passage in DOCUMENT {1} cited by the legal excerpt in DOCUMENT
{0}?
```

---

## Open items

- None of these predicates have been through a judge pass (an offline
  accuracy check against a stronger model) yet. Wording may need to
  change if a predicate misses an agreement floor or clusters
  selectivity with another predicate on the same table.
- Join-first (filtering a join's output) and a second, *dependent*
  join (filter -> filter -> join -> join, one join's output feeding
  another) are not implemented in this catalog. If a query needs
  either shape, that's a deliberate addition, not an automatic
  extension of this catalog.