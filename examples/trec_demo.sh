#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$root/target/trec-demo"
mkdir -p "$output"

cargo run --release --manifest-path "$root/Cargo.toml" -- index \
  --input "$root/examples/collection.trec" \
  --output "$output/collection.idx" \
  --format trec

cargo run --release --manifest-path "$root/Cargo.toml" -- batch \
  --index "$output/collection.idx" \
  --topics "$root/examples/topics.tsv" \
  --qrels "$root/examples/qrels.txt" \
  --run "$output/indexsail.run" \
  --report "$output/report.json" \
  --field body \
  --top-k 10 \
  --verify

printf 'run: %s\nreport: %s\n' "$output/indexsail.run" "$output/report.json"
