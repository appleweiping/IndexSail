# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

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
