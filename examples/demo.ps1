$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IndexPath = Join-Path $PSScriptRoot "corpus.idx"

Push-Location $ProjectRoot
try {
    cargo run --release -- index --input examples/corpus.tsv --output $IndexPath
    cargo run --release -- inspect --index $IndexPath --field body --term search
    cargo run --release -- search --index $IndexPath --query "local search" --field body --operator and --phrase "local search" --phrase-field body --explain
}
finally {
    Pop-Location
}
