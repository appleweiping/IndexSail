#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output="$project_root/target/basic-demo"
index_path="$output/corpus.idx"

cd "$project_root"
mkdir -p "$output"
cargo run --release -- index --input examples/corpus.tsv --output "$index_path"
cargo run --release -- inspect --index "$index_path" --field body --term search
cargo run --release -- search --index "$index_path" --query "local search" --field body --operator and --phrase "local search" --phrase-field body --explain
