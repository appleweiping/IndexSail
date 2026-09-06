# Reproducible benchmark baselines

These are correctness-backed observations for deterministic synthetic workloads, not general throughput
claims. The benchmark constructs one in-memory index, runs exhaustive retrieval, runs WAND, runs block-max
WAND, compares every ranked document and score against the exhaustive results, and only then emits
`verified=true` and a checksum.

The run recorded below predates the block-max pass, so its output has no `block-max-wand` line and its JSON
is `schema_version` 1. It is left as it was measured rather than re-run here, because the environment it
records is not this one. Block-max measurements, and the conditions under which the strategy helps or does
not, are in the README.

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
