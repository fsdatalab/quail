# Ground truth label reuse

## What changed

The ground truth collection manifest can now reuse a label set from an older
corpus when every table read by that predicate is unchanged. Each reuse record
contains the source collection, source corpus, required tables, and the exact
table manifests that were checked.

The ground truth loader repeats these checks. It rejects a reused label set if
the source collection does not contain it, a required table changed, or the
saved table manifest is wrong.

## Why

The LePaRD construction changed without changing the IMDB, BioDEX, or FEVER
tables. Rejudging the 15 predicates on those unchanged tables would produce
duplicate label data.

## Current collection

The active sf0.1 collection is
`gt_363b5ab570635c33894e1a030c21f57e` for corpus
`c_3bd14ed0758287cba9d88fb68de8b7b8`. It reuses 15 predicates from collection
`gt_02ffa2a5720006e8236aa993760e9e29` and uses seven new LePaRD label sets.
The loader read all 22 predicates and 1,771,220 labels from the combined
collection.

The manifest is at
`/results/ground_truth/quailb/schema_v1/collections/gt_363b5ab570635c33894e1a030c21f57e/manifest.json`
on the `quail-results` volume.
