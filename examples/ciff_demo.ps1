$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $projectRoot "target/ciff-demo"
$native = Join-Path $output "collection.idx"
$ciff = Join-Path $output "collection.ciff"
New-Item -ItemType Directory -Force -Path $output | Out-Null

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- index `
  --input (Join-Path $projectRoot "examples/collection.trec") `
  --output $native `
  --format trec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- ciff-export `
  --index $native `
  --output $ciff `
  --description "IndexSail committed CIFF demo"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- ciff-inspect `
  --index $ciff `
  --term search
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- ciff-search `
  --index $ciff `
  --query "local search" `
  --top-k 10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- ciff-batch `
  --index $ciff `
  --topics (Join-Path $projectRoot "examples/topics.tsv") `
  --qrels (Join-Path $projectRoot "examples/qrels.txt") `
  --run (Join-Path $output "indexsail-ciff.run") `
  --report (Join-Path $output "ciff-report.json") `
  --top-k 10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "ciff: $ciff"
Write-Host "run: $(Join-Path $output 'indexsail-ciff.run')"
Write-Host "report: $(Join-Path $output 'ciff-report.json')"
