$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $projectRoot "target/shard-demo"
$index = Join-Path $output "collection.shards.idx"
New-Item -ItemType Directory -Force -Path $output | Out-Null

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- shard-index `
  --input (Join-Path $projectRoot "examples/collection.trec") `
  --output $index `
  --format trec `
  --shards 3
if ($LASTEXITCODE -ne 0) { throw "shard-index failed with exit code $LASTEXITCODE" }

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- shard-inspect `
  --index $index `
  --field body `
  --term local
if ($LASTEXITCODE -ne 0) { throw "shard-inspect failed with exit code $LASTEXITCODE" }

cargo run --locked --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- shard-batch `
  --index $index `
  --topics (Join-Path $projectRoot "examples/topics.tsv") `
  --qrels (Join-Path $projectRoot "examples/qrels.txt") `
  --run (Join-Path $output "indexsail-sharded.run") `
  --report (Join-Path $output "sharded-report.json") `
  --field body `
  --strategy block-max-wand `
  --top-k 10 `
  --verify
if ($LASTEXITCODE -ne 0) { throw "shard-batch failed with exit code $LASTEXITCODE" }

Write-Host "run: $(Join-Path $output 'indexsail-sharded.run')"
Write-Host "report: $(Join-Path $output 'sharded-report.json')"
