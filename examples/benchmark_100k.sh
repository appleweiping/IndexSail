#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$root/target/benchmark-100k"
mkdir -p "$output"

cargo run --release --manifest-path "$root/Cargo.toml" -- benchmark \
  --documents 100000 \
  --queries 500 \
  --top-k 10 \
  --seed 42 \
  --json "$output/report.json"

printf 'report: %s\n' "$output/report.json"
