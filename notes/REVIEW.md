# Review of paper.md (Scheduling n-Stage AI Filters Under KV-Cache Constraints)

Five reviewer agents read the full paper source independently, each with a
different focus (mathematics, hardware facts, scheduling semantics,
algorithms and complexity, and reproducibility). They produced 28 raw
findings, which were merged into 9 after removing duplicates. A second set
of agents then tried to refute each finding. Four findings survived as
stated, three survived in a weaker form, and two were refuted. The refuted
ones are listed too, because they mark places where the paper is right and a
careless reader gets it wrong. Every finding was also checked against the
working solver in this repo, which implements the paper's model end to end.

The paper's framework is sound and can be implemented, and the solvers in
this repo reproduce its propositions on real data. Before any reported
number can be trusted for the named models, the paper needs two formula
fixes, one repaired proof, one added convergence argument, and two
accounting definitions pinned down.

## Confirmed errors

### 1. The attention cost formula uses the wrong width, undercounting by 1.6 times for both models (critical)

Equation (20) computes attention floating point operations as 4·L·h·A(B),
where h is the model's hidden width. The paper's own derivation, one dot
product plus one accumulation per allowed token pair, actually gives
4·L·n_q·d_h, where n_q is the number of query heads and d_h is the width of
each head. Grouped query attention shares stored keys and values across
heads, but it does not reduce compute on the query side. The two widths
agree for older models, but Qwen3 fixes d_h at 128 regardless of hidden
width, so for Qwen3-4B the true width is 4,096 while h is 2,560, and for
Qwen3-32B the true width is 8,192 while h is 5,120. Both are off by exactly
1.6 times, and the paper's own Table 3 lists the head counts that contradict
the formula. The neighboring formula for stored bytes per token (equation
18) uses heads times head width correctly, which isolates the bug to
equation (20). The error is largest for long documents. At the maximum
context length on Qwen3-4B the corrected attention work exceeds the dense
work, so the 60 percent error dominates exactly where the length scaling
plots live. The three policies also have different attention totals, so the
error can flip break-even conclusions rather than rescale them.

The fix is to define an attention width equal to n_q·d_h, use it in
equations (20), (25), and (55), and add the query head count to the
parameter tables and to the recorded configuration list, which currently
omits it. The solver already computes attention this way (convention C1 in
notes/PLAN.md).

### 2. The hardness proof does not cover the model it claims to (major)

Proposition 7.2 claims that computing the offline optimum is strongly
NP-hard, by a reduction from the 3-PARTITION problem that packs indivisible
documents into batches of a fixed size. But the paper's offline optimum
allows splitting documents into chunks, and the paper itself proves that
splitting can only help (Proposition 5.4). On the instances the proof
constructs, splitting lets every batch be filled exactly, so the schedule
cost no longer depends on whether a partition exists, and the reduction
proves hardness only for the restricted class with indivisible documents.
The proof also assumes a constant cost per batch and zero attention cost,
and neither is a primitive of the cost model. They hold only in a regime the
proof does not state.

The fix is either to restate the proposition for indivisible documents with
the cost regime stated, or to repair the reduction so splitting cannot help.
The repair is possible by routing capacity through the memory limit instead
of a token cap. A document's entire stored state must sit on the card in the
batch where its filter prompts run, no matter how the reading was split, so
prompt placement forces the partition and the stronger claim comes back.

### 3. The online algorithm is not well defined as written (critical)

The paper handles loops in the schedule graph for the offline case by using
Dijkstra's algorithm (line 873), because evicting and recomputing a document
can return the scheduler to a state it has already visited. The online case
has the same loops, but the paper says the online answer is obtained by
"enumerating the same states and evaluating this recurrence" (equation 54).
With loops, equation 54 is a fixed point equation rather than a recursion,
so evaluating it is not a terminating procedure, and the optimality proofs
assume an evaluation order that does not exist.

The fix is to state that equation 54 defines a stochastic shortest path
problem, meaning a shortest path problem in which each move has random
outcomes. Positive batch costs, plus the existence of at least one policy
that always finishes, give a unique fixed point, and value iteration
computes it. The solver does exactly this (convention C4).

### 4. The lower bound is built from quantities that depend on the schedule (major)

Equation 55 builds the bound from total tokens, total attention pairs, total
stored bytes, and a minimum batch count, and treats them as properties of
the instance. Under the paper's own rules they are not fixed. Eviction
forces recomputation, adaptive speculation changes which prompts run, and
splitting changes how much stored data is reread. As written, a schedule
could be compared against a bound computed from a different schedule's
totals. The minimum batch count also has no derivation rule anywhere in the
paper.

The fix is to define each total as the smallest value any feasible schedule
of the policy could have, which is the ledger with no recomputation and
maximal sharing for the realized outcomes, and to derive the minimum batch
count from the memory limit, since every new document token's stored state
must sit on the card during its batch. The solver's bound is built this way
(convention C8), and with it the certificates at 10,000 documents are
legitimate.

## Findings that survived in weaker form

### 5. The stored data accounting is ambiguous for fused batches (major)

K_W counts the new tokens whose stored state must be written to card memory
"for use by a later chunk, branch, or batch" (line 426), and the notation
table gives a second, different definition ("new KV token positions
materialized"). Neither says what happens when a document is read and all
its filter prompts run inside one batch. The fused case therefore admits
three values: zero bytes, one write of the document, or a write plus a read
back. The three differ by about 147 megabytes per 2,000 token document on
the 4B model, enough to change which term of the cost is binding. The
manifest format also cannot record which prompts shared one physical copy of
a document, so a checker cannot tell sharing from reloading. The fix is to
pick one convention and state it. The solver counts document and prompt
block tokens as written and filter prompt tokens as never written, counts
reads only for blocks that were on the card when the batch started
(convention C3), and its manifest records which blocks each operation read.

### 6. One equation invites a wrong reading of when eviction is chosen (minor after verification)

Equation 30 minimizes over a single action before the random outcomes are
drawn, which reads as if the scheduler commits to its evictions before
seeing the filter results. Section 5.2 defines the joint action only for the
offline case, and the online protocol, which is to choose a batch, observe
the results, and then evict, is stated in the next sentence and encoded
correctly in equation 54. So there is one optimum, not two, but three of the
five reviewers misread it. The fix is one sentence at equation 30 saying the
online action is the batch alone, with eviction chosen in the state after
outcomes are seen.

### 7. Some run parameters still need pinning (minor)

The claim that the model is not computable was refuted. The per batch weight
traffic is defined operationally at line 386 as the recorded bytes of the
repeated transformer blocks of the fixed checkpoint, which is a static file
inspection. What genuinely remains open, and must be recorded per run, is
the following: the attention speed ceiling (a factor of 2 swing if set from
measurement instead of the dense ceiling), the stored data precision (a
factor of 2 on capacity), the chunk quantum for the large runs, the
selectivity grid and the number of random repetitions, the exact prompt
texts, which IMDb pool "the source dataset" means (the standard release has
25,000 labeled training reviews, 25,000 labeled test reviews, and 50,000
unlabeled ones, with file names unique only within a folder), and the
tokenizer revision. The repo pins all of these, and the two released
checkpoints ship tokenizer files that are identical byte for byte.

## Refuted findings, where the paper is right

- **Gate timing for the offline scheduler.** The suspicion was that the
  paper never says whether the offline scheduler, which knows all outcomes
  in advance, may run filter j and filter j+1 on the same document in one
  batch. Section 3.4 does say it, at the execution level and before any
  information model. Outcomes become visible only at batch boundaries, and a
  batch that does not speculate cannot contain work that needs an outcome
  still unresolved when the batch starts. The rule binds the offline
  scheduler too, so knowing the future helps only with packing, retention,
  and eviction. The solver enforces this (convention C2).
- **Partial speculative blocks.** The suspicion was that the state cannot
  represent a speculative block split across batches. It does not need to,
  because outcomes are revealed at every batch boundary, so a block split
  across batches is by definition just sequential execution. Speculation is
  within one batch by construction.

## Minor notes

- Equation 2 has a LaTeX bug. A missing backslash renders the letters
  "quad" inside the survival definition.
- Line 93 says decisions are read "from the logits at the final prompt
  position", but in the task-first layout the document comes last, so the
  final position is a document token. Reword to the final position of the
  sequence.
- Counting one byte per stored element ignores the scale factors that 8 bit
  storage needs, which add 0 to 3 percent depending on granularity. One
  sentence would cover it.
- T_init in equation 28 has no cost formula and an unstated relation to the
  first batch's memory check and to the lower bound. The solver sets it to
  zero and loads prompt blocks inside ordinary batches (convention C11), and
  the text should either adopt that or price it.

## What the implementation shows about the fixed model

With findings 1, 3, and 4 fixed as above, the model is solvable, not just
consistent. Exact optima on small instances reproduce every proposition,
including strict improvement from chunking, outcome independence of forced
speculation, and the value of information inequality. The schedules built
for 10,000 documents meet their lower bounds to within 0.07 percent. The
exact optimizer also shows the scheduling behavior the framework exists to
capture, which is holding documents back so that filter answers are waited
out inside useful work. See notes/RESULTS.md.
