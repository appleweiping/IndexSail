# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

- Added block-max WAND as an opt-in pruning strategy (`--strategy block-max-wand`, or `bmw`).
  Plain WAND bounds a term by the largest impact anywhere in its postings, so one outlier keeps
  that bound high for every other document the term touches; a maximum per 64-posting block lets a
  run of low-impact postings be skipped whole. Block maxima are computed in the same pass that
  scores a term, so they cost no index-time work and no on-disk format change.
- The ranking is unchanged, and that is checked rather than argued. Both strategies share one code
  path, so they cannot drift apart; the exactness tests compare block-max against exhaustive scoring
  over 64 query, operator and cutoff combinations plus field filters and phrase constraints; and the
  benchmark now runs a third verified pass whose hits must match exhaustive results bit for bit,
  score patterns included, across 200 queries over 100,000 documents.
- A separate test asserts the invariant the strategy rests on directly: every block maximum is at or
  above every posting score inside its block, and at or below the global bound. A ranking test alone
  would only notice a broken bound when a query happened to hit it.
- The gain depends entirely on how much impact varies inside a posting list. On a collection with
  skewed impacts it removes 58% to 72% of the documents WAND still scores at `k=10`; on the
  near-uniform synthetic benchmark corpus it removes 0.1% and is marginally slower, because the
  extra bound arithmetic buys nothing there. Both numbers are in the README rather than only the
  favourable one.
- The benchmark JSON gains `block_max_elapsed_micros` and a `block_max_wand` counter block, so its
  `schema_version` is now 2.

## [0.2.0] - 2026-08-31

- Added a streaming adapter for the documented TREC SGML collection subset.
- Added classic/TSV topics, four-column qrels, six-column run output, and JSON batch reports.
- Added MAP@k, MRR@k, nDCG@k, and Recall@k with per-query counters and latency.
- Added optional bit-exact per-query WAND/exhaustive verification.
- Added delta/variable-byte posting statistics and reproducible 100k-document benchmark scripts.
- Upgraded persistence to checksummed, compressed format version 2 while retaining version 1 reads.
- Hardened persistence and codec allocation paths against forged lengths, truncation, and overflow.

## [0.1.0] - 2026-08-31

- Added field-aware positional indexing, BM25, Boolean and phrase queries, and exact filters.
- Added exhaustive and exact WAND execution with stable top-k results and explanations.
- Added versioned binary persistence, CLI workflows, reproducible benchmarks, examples, and CI.
