#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$root/target/ciff-demo"
native="$output/collection.idx"
ciff="$output/collection.ciff"
mkdir -p "$output"

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- index \
  --input "$root/examples/collection.trec" \
  --output "$native" \
  --format trec

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- ciff-export \
  --index "$native" \
  --output "$ciff" \
  --description "IndexSail committed CIFF demo"

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- ciff-inspect \
  --index "$ciff" \
  --term search

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- ciff-search \
  --index "$ciff" \
  --query "local search" \
  --top-k 10

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- ciff-batch \
  --index "$ciff" \
  --topics "$root/examples/topics.tsv" \
  --qrels "$root/examples/qrels.txt" \
  --run "$output/indexsail-ciff.run" \
  --report "$output/ciff-report.json" \
  --top-k 10

printf 'ciff: %s\nrun: %s\nreport: %s\n' \
  "$ciff" "$output/indexsail-ciff.run" "$output/ciff-report.json"
