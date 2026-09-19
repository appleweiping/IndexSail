$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    New-Item -ItemType Directory -Force target | Out-Null
    cargo run --locked --release -- forward-build `
        --input examples/collection.jsonl `
        --output target/demo.fwd `
        --format jsonl
    cargo run --locked --release -- forward-inspect `
        --index target/demo.fwd `
        --document 0 `
        --limit 8
    cargo run --locked --release -- lexicon `
        --index target/demo.fwd `
        --field body `
        --term retrieval
    cargo run --locked --release -- forward-invert `
        --input target/demo.fwd `
        --output target/demo-forward.idx
    cargo run --locked --release -- search `
        --index target/demo-forward.idx `
        --query "exact retrieval" `
        --field body `
        --strategy block-max-wand `
        --explain
}
finally {
    Pop-Location
}
