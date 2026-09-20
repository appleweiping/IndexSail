# Release process

A push of a semantic-version tag such as `v0.10.0` invokes the release
workflow. The workflow refuses a tag that differs from `Cargo.toml`, runs the
release-mode test suite with the minimum supported Rust toolchain, builds
Linux and Windows binaries, and creates the source `.crate` from the committed
lockfile.

The released v0.8.0 scope is the native seeded-random, by-feature, and
explicit-mapping reordering workflow. Its release predates the subsequent
recursive-graph-bisection work, and does not close the broader PISA reordering
roadmap item.

The CI workflow also enforces the fail-closed source-branch report checker
in its pinned Ubuntu nightly job, separate from stable compiler tests.
With `cargo-llvm-cov` 0.9.1 and Rust nightly 2026-09-18 plus
`llvm-tools-preview`, run:

```bash
cargo +nightly-2026-09-18 llvm-cov --branch --locked --all-targets --json --output-path target/branch.json
python scripts/check_branch_coverage.py target/branch.json
```

The checker requires at least 90% actual covered/total source branches and
rejects incomplete/malformed reports. Verify that the same source SHA passes
the remote `branch-coverage` job before signing a release tag; a local pass
alone is not release evidence.

The GitHub Release contains platform archives, the source crate, a CycloneDX
1.5 SBOM, and `SHA256SUMS`. The SBOM is generated from locked Cargo metadata
and records every direct and transitive package, including the pinned `libm`
and `same-file` dependencies and platform support crates. GitHub records
build-provenance attestations for every asset. Third-party workflow actions
are pinned to full commit hashes and each job receives only its required
permissions.

Verify a downloaded file with:

```bash
sha256sum --check SHA256SUMS
gh attestation verify indexsail-v0.10.0-x86_64-unknown-linux-gnu.tar.gz \
  --repo appleweiping/IndexSail
```

crates.io publication is intentionally not claimed or automated until a
project owner configures and tests trusted publication. The GitHub Release is
the authoritative distribution channel meanwhile.
