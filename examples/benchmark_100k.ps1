$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $projectRoot "target/benchmark-100k"
New-Item -ItemType Directory -Force -Path $output | Out-Null

cargo run --release --manifest-path (Join-Path $projectRoot "Cargo.toml") -- benchmark `
  --documents 100000 `
  --queries 500 `
  --top-k 10 `
  --seed 42 `
  --json (Join-Path $output "report.json")

Write-Host "report: $(Join-Path $output 'report.json')"
