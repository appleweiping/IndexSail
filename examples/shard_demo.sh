#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$root/target/shard-demo"
index="$output/collection.shards.idx"
mkdir -p "$output"

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- shard-index \
  --input "$root/examples/collection.trec" \
  --output "$index" \
  --format trec \
  --shards 3

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- shard-inspect \
  --index "$index" \
  --field body \
  --term local

cargo run --locked --release --manifest-path "$root/Cargo.toml" -- shard-batch \
  --index "$index" \
  --topics "$root/examples/topics.tsv" \
  --qrels "$root/examples/qrels.txt" \
  --run "$output/indexsail-sharded.run" \
  --report "$output/sharded-report.json" \
  --field body \
  --strategy block-max-wand \
  --top-k 10 \
  --verify

printf 'run: %s\nreport: %s\n' "$output/indexsail-sharded.run" "$output/sharded-report.json"
