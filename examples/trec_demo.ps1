$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $projectRoot "target/trec-demo"
New-Item -ItemType Directory -Force -Path $output | Out-Null

cargo run --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- index `
  --input (Join-Path $projectRoot "examples/collection.trec") `
  --output (Join-Path $output "collection.idx") `
  --format trec

cargo run --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- batch `
  --index (Join-Path $output "collection.idx") `
  --topics (Join-Path $projectRoot "examples/topics.tsv") `
  --qrels (Join-Path $projectRoot "examples/qrels.txt") `
  --run (Join-Path $output "indexsail.run") `
  --report (Join-Path $output "report.json") `
  --field body `
  --top-k 10 `
  --verify

Write-Host "run: $(Join-Path $output 'indexsail.run')"
Write-Host "report: $(Join-Path $output 'report.json')"
