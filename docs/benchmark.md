# Reproducible benchmark baselines

These are correctness-backed observations for deterministic synthetic workloads, not general throughput
claims. The benchmark constructs one in-memory index, runs exhaustive retrieval, runs WAND, runs block-max
WAND, compares every ranked document and score against the exhaustive results, and only then emits
`verified=true` and a checksum.

## Version 3 storage/query-work measurement

Two consecutive runs measure the current persisted-bound implementation. They are observations from a
non-isolated workstation, not a cross-engine benchmark or a latency claim.

- Date: 2026-09-07
- CPU: Intel Core i5-1240P, 12 cores / 16 logical processors
- Runtime: native Windows 10.0.26200, x86-64 MSVC
- Rust: rustc 1.85.0
- Build: `release`, thin LTO, one codegen unit
- Source: the working tree that became [`v0.3.0`](https://github.com/appleweiping/IndexSail/tree/v0.3.0),
  based on commit `32cdd2c4c7a9bbc30fe0c34ff628d458ddc742d6`

Both passes used:

```shell
cargo run --locked --release -- benchmark \
  --documents 100000 \
  --queries 500 \
  --top-k 10 \
  --seed 42 \
  --json target/benchmark-100k/report.json
```

The retrieval phase queries the just-built resident index. The
`precomputed_bounds_loaded` counter therefore counts values copied from the same default-BM25 table that
format v3 persists; it does not imply that this benchmark reloaded the file before searching. Persistence
sizes are measured by serializing that table after the retrieval passes.

| Pass | Repetition 1 | Repetition 2 |
|---|---:|---:|
| Index build | 10.89 s | 16.70 s |
| Exhaustive | 10.42 s | 9.65 s |
| WAND | 8.26 s | 6.56 s |
| Block-max WAND | 17.35 s | 8.16 s |

The deterministic part matched in both repetitions:

```text
index terms=776 postings=5592575
exhaustive evaluated=12927028
wand evaluated=1136956 advanced=14273218 skipped=8725466
block-max-wand evaluated=1136119 advanced=14273218 skipped=8727041 \
  precomputed_bounds_loaded=223772 postings_covered_by_bounds=14275721 postings_scanned_for_bounds=0
verified=true checksum=1310686fefd0b451
posting_codec raw_bytes=76340600 encoded_bytes=19097862 ratio=0.2502
persistence format=3 serialized_bytes=70139763 base_index_bytes=68716795 \
  block_max_metadata_bytes=1422968 streams=1296 blocks=174308
```

The block section adds 1,422,968 bytes to a 68,716,795-byte base layout: 2.07%. The reported base is the
same serialized documents/postings/header byte count with the v3 block section removed, not an estimate from
a different run. For the block-max query pass, 223,772 bound values summarize 14,275,721 scored postings.
Before v3, deriving bounds visited those scored postings; v3 performs zero postings visits specifically for
that derivation. “98.43% fewer values consumed for bound preparation” describes this implementation counter,
not CPU instructions or end-to-end speed.

The near-uniform generator is unfavorable to block-max pruning: it removed only 837 candidates beyond WAND
(0.074%). The wide, overlapping timing ranges even reverse the WAND/block-max order, so they establish no
latency win. The result is retained because storage and preparation savings must not be converted into a
speed claim. Both generated schema-version-3 JSON reports were parsed after their runs.

## Historical WSL2 baseline

The run below predates the persisted block-max pass, so its output has no `block-max-wand` line and its JSON
is `schema_version` 1. It is left as originally measured because timings from its WSL2 environment are not
comparable with the native-Windows v3 run above.

## Environment

- Date: 2026-08-31
- CPU: Intel Core i5-1240P, 12 cores / 16 logical processors
- Runtime: WSL2, Linux 6.18.33.2-microsoft-standard-WSL2, x86-64
- Rust: rustc 1.96.1
- Build: `release`, thin LTO, one codegen unit
- Repository location during measurement: mounted Windows filesystem

## 100k-document workload

```shell
cargo run --release -- benchmark \
  --documents 100000 \
  --queries 500 \
  --top-k 10 \
  --seed 42 \
  --json target/benchmark-100k/report.json
```

The generator produced:

- 100,000 documents, each with six title tokens, 72 body tokens, and one category;
- 500 three-term disjunctive body queries;
- 776 field/term dictionary keys;
- 5,592,575 posting entries;
- 7,900,000 analyzed tokens.

Observed result:

```text
index elapsed_ms=12223.230
exhaustive elapsed_ms=21722.173 evaluated=12927028
wand elapsed_ms=13612.570 evaluated=1136956 advanced=14273218 skipped=8725466
verified=true checksum=70f92ad0827240fd
posting_codec raw_bytes=76340600 encoded_bytes=19097862 ratio=0.2502
```

| Executor | Elapsed | Exact candidates | Relative candidates |
|---|---:|---:|---:|
| Exhaustive | 21.72 s | 12,927,028 | 100% |
| WAND | 13.61 s | 1,136,956 | 8.80% |

On this workload, WAND evaluated 91.20% fewer exact candidates. End-to-end query time was 37.3% lower on
this one run. Candidate reduction is a deterministic property of this workload and implementation; elapsed
time is not.

The codec comparison counts fixed-width logical posting values (`doc_id`, term frequency, positions) versus
their delta/varbyte representation. It excludes block-length prefixes, dictionary strings, stored document
text, field lengths, and file headers. The 0.2502 ratio must therefore not be presented as whole-index size.

Run [`examples/benchmark_100k.sh`](../examples/benchmark_100k.sh) or
[`examples/benchmark_100k.ps1`](../examples/benchmark_100k.ps1) to reproduce the workload and create the
machine-readable report.

## Original 10k-document baseline

Before the persistence and experiment upgrade, a 10,000-document/200-query run with the same generator and
seed produced 776 dictionary keys and 559,275 postings:

```text
exhaustive evaluated=525333
wand       evaluated=60323 advanced=577733 skipped=332019
verified=true checksum=6293b8828e85f6ad
```

Two timings ranged from 370–1,044 ms exhaustive and 249–465 ms WAND after an 818–1,140 ms index build. The
spread illustrates why a single wall-clock number should not be treated as a stable project property.

## What is reproducible

For the same seed and numeric configuration:

- generated fields and query terms;
- dictionary/posting/token counts;
- executor work counters;
- ranked documents and scores;
- the final FNV-style result checksum;
- posting codec byte counts.

The following are not promised to match across environments:

- index, exhaustive, WAND, filesystem, or serialization duration;
- peak resident memory;
- compiler optimization decisions across Rust releases;
- CPU counters or thermal behavior.

## Reporting protocol

For a meaningful comparison, archive the generated JSON and report:

1. repository revision and whether the tree was clean;
2. CPU, memory, OS/kernel and native versus virtualized runtime;
3. Rust version and target triple;
4. build profile and compiler flags;
5. document/query/cutoff/seed configuration;
6. exact verification status and checksum;
7. at least one warm-up and multiple isolated measured runs;
8. whether files are on a native or mounted/network filesystem.

Do not compare IndexSail's `evaluated_candidates` or `postings_advanced` directly with another engine unless
both counters are operationally defined in the same way.
