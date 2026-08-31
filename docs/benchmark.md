# Reproducible benchmark baseline

This baseline is a correctness-backed performance observation, not a universal performance claim.

## Environment

- Date: 2026-08-31
- CPU: Intel Core i5-1240P, 12 cores / 16 logical processors
- Runtime: WSL2, Linux 6.18.33.2-microsoft-standard-WSL2, x86-64
- Rust: rustc 1.96.1
- Build: `release`, thin LTO, one codegen unit

## Command

```shell
cargo run --release -- benchmark \
  --documents 10000 \
  --queries 200 \
  --top-k 10 \
  --seed 42
```

The deterministic workload produced 776 field/term dictionary keys and 559,275 postings. Both measured
runs returned:

```text
exhaustive evaluated=525333
wand       evaluated=60323 advanced=577733 skipped=332019
verified=true checksum=6293b8828e85f6ad
```

Candidate scoring fell by 88.5%. Timings varied with warm-up and system activity:

| Run | Index build | Exhaustive queries | WAND queries |
|---|---:|---:|---:|
| 1 | 1139.637 ms | 1043.567 ms | 465.056 ms |
| 2 | 817.908 ms | 370.237 ms | 249.308 ms |

The repeated checksum and candidate statistics demonstrate workload reproducibility. The timing spread is
why performance comparisons should use multiple isolated runs and report the full environment.
