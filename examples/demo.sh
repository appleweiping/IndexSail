#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
index_path="$project_root/examples/corpus.idx"

cd "$project_root"
cargo run --release -- index --input examples/corpus.tsv --output "$index_path"
cargo run --release -- inspect --index "$index_path" --field body --term search
cargo run --release -- search --index "$index_path" --query "local search" --field body --operator and --phrase "local search" --phrase-field body --explain
