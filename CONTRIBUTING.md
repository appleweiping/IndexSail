# Contributing to IndexSail

Thank you for improving IndexSail. Contributions should keep the engine small, deterministic, safe, and
understandable.

## Before implementation

For behavioral changes, open an issue describing the user-visible problem, a minimal example, expected
semantics, and compatibility considerations. Security-sensitive parser or persistence changes should also
state the untrusted-input model. Small documentation and test corrections may go directly to a pull request.

## Local quality gate

Use Rust 1.85 or later and run:

```shell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
cargo run --release -- benchmark --documents 1000 --queries 30 --top-k 10 --seed 42
```

The benchmark must end with `verified=true`. Performance claims must include the command, build profile,
hardware, OS, Rust version, checksum, and several runs; do not treat a single wall-clock result as proof.

## Design expectations

- Prefer the standard library and justify every proposed dependency.
- Keep indexing and query analysis identical.
- Preserve deterministic serialization and ranking tie-breaking.
- Any pruning optimization must be tested against exhaustive search, including ties and post-filters.
- Binary reader changes require malformed/truncated/oversized-input tests.
- Public behavior needs focused tests and README or architecture updates.
- Do not silently change analyzer or BM25 semantics in a patch release.
- No unsafe Rust is accepted without a separately reviewed design justification; the crate currently forbids
  unsafe code.

## Pull requests

Keep changes focused. Include a summary, motivation, test commands actually run, benchmark impact where
relevant, format compatibility, and remaining risks. Generated index files and build output must not be
committed. Contributions are accepted under the repository's MIT License.
