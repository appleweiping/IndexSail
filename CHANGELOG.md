# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

## [0.5.0] - 2026-09-07

- Added deterministic round-robin physical sharding with collection-wide BM25
  document count, field totals, and per-field document frequencies. Every
  shard is searched independently and its local top-k is mapped back to the
  original global document IDs before a stable exact merge.
- Added `ShardedIndex`, `ShardedIndexBuilder`, shard diagnostics, a checksummed
  v1 container of independently validated v3 indexes, and embedded v1/v2/v3
  index read compatibility. Loading rejects analyzer mismatches, duplicate
  external IDs across shards, non-canonical shard populations, corruption,
  truncation, and trailing bytes.
- Added a seekable two-pass persistence path that verifies the complete outer
  checksum before parsing and bounds raw-container memory to one embedded
  shard. Diagnostics retain and report actual mixed legacy versions.
- Hardened untrusted cutoffs: `usize::MAX` and cutoffs above the collection no
  longer drive eager heap/vector reservation, including 4,096 mostly empty
  shards. A hand-calculated cross-shard BM25 oracle freezes global statistics.
- Added `shard-index`, `shard-search`, `shard-batch`, and `shard-inspect` CLI
  workflows. TSV and TREC ingestion streams each record directly to its
  physical builder instead of retaining a second monolithic index.
  `evaluate_batch` now accepts monolithic and sharded retrieval backends
  through one public trait.
- Extended the correctness benchmark with a configurable sharded block-max
  pass checked bit-for-bit against the monolithic exhaustive oracle; benchmark
  JSON schema version 5 records sharded build/search time, serialized bytes,
  counters, and physical shard count.

## [0.4.0] - 2026-09-07

- Added an exact `MaxScore` term-at-a-time executor for disjunctive queries.
  It uses essential-list upper-bound pruning, preserves complete-term scoring,
  post-filter and phrase semantics, deterministic ties, and bit-exact
  exhaustive-oracle verification. The CLI and benchmark report expose the
  strategy as `maxscore`; conjunctive queries retain the exact intersection
  path.
- Bumped the benchmark JSON schema to version 4 with MaxScore timing and
  counters, and added deterministic regression coverage across cutoffs,
  filters, phrases, ties, and a 600-document synthetic corpus.

## [0.3.0] - 2026-09-07

- Added tag-gated source and Linux/Windows binary releases with a dependency
  SBOM, SHA-256 manifest, and GitHub build-provenance attestations.
- Added block-max WAND as an opt-in pruning strategy (`--strategy block-max-wand`, or `bmw`).
  Plain WAND bounds a term by the largest impact anywhere in its postings, so one outlier keeps
  that bound high for every other document the term touches; a maximum per 64-posting block lets a
  run of low-impact postings be skipped whole.
- Added checksummed persistence format version 3. It stores tight default-BM25 bounds for every
  field-qualified and all-field term stream. New indexes
  compute them once at build completion; version 1 and 2 files remain readable and rebuild them once
  during load. Default queries load one bound per block rather than rescanning scored postings to derive
  block maxima; custom parameters derive conservative bounds from their materialized exact scores and
  report that work.
- Pinned the pure-Rust `libm` logarithm so BM25 IDF and exact v3 bound validation have identical bits on
  Linux and Windows. CI carries frozen IDF bit patterns in addition to cross-platform persistence tests.
- Version 3 loading recomputes the block table and requires exact IEEE-754 bit equality, in addition
  to the existing checksum and structural validation. Tests cover v1/v2 reads, deterministic v3
  round trips, truncation, checksum corruption, forged bounds with a valid checksum, bound
  invariants, frozen legacy-encoding checksums, custom BM25 parameters and bit-exact
  exhaustive/WAND/block-max results.
- The ranking is unchanged, and that is checked rather than argued. All strategies share exact scoring
  and top-k maintenance, while exactness tests compare the distinct block-max pruning control flow
  against exhaustive scoring over 64 query, operator and cutoff combinations plus field filters and phrase constraints; the
  benchmark now runs a third verified pass whose hits must match exhaustive results bit for bit,
  score patterns included, across 500 queries over 100,000 documents.
- A separate test asserts the invariant the strategy rests on directly: every block maximum is at or
  above every posting score inside its block, and at or below the global bound. A ranking test alone
  would only notice a broken bound when a query happened to hit it.
- The gain depends entirely on how much impact varies inside a posting list. On a focused collection
  with skewed impacts it removes 58% to 72% of the documents WAND still scores at `k=10`; on the
  formal near-uniform 100k/500-query benchmark it removes only 0.074%. Repeated timing ranges overlap
  and reverse order, so no latency win or loss is claimed.
- The benchmark JSON schema is now version 3. It records complete/base index bytes, block metadata
  bytes and counts, precomputed bounds loaded, postings covered by those bounds, and postings inspected for
  custom-bound derivation. The formal default-parameter run reports zero for that final counter.
- Kept buffered, owned loading rather than adding a nominal mmap path: safe Rust's standard library has no
  cross-platform mmap API, and an mmap dependency would not make the current owned representation zero-copy.

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
