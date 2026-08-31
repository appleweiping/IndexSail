$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $ProjectRoot "target/basic-demo"
$IndexPath = Join-Path $Output "corpus.idx"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Push-Location $ProjectRoot
try {
    cargo run --release -- index --input examples/corpus.tsv --output $IndexPath
    cargo run --release -- inspect --index $IndexPath --field body --term search
    cargo run --release -- search --index $IndexPath --query "local search" --field body --operator and --phrase "local search" --phrase-field body --explain
}
finally {
    Pop-Location
}
