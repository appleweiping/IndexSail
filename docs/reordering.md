# Document-ID reordering

`reorder` starts from an IndexSail `IDXFW001` forward snapshot. It reassigns
zero-based internal document IDs, writes a reordered forward snapshot, rebuilds
the native positional inverted index, and emits both directions of the ID
permutation. External document IDs and stored fields do not change. Query
scores and matching documents, keyed by external ID, are preserved; an exact
score tie can change rank or top-*k* membership because native tie-breaking
uses internal ID.

```shell
cargo run --release -- forward-build \
  --input examples/collection.jsonl --format jsonl --output target/original.fwd

cargo run --release -- reorder \
  --input target/original.fwd \
  --forward-output target/random.fwd \
  --index-output target/random.idx \
  --old-to-new target/random.old-to-new \
  --new-to-old target/random.new-to-old \
  --random --seed 7

cargo run --release -- search \
  --index target/random.idx --query "local retrieval"
```

Exactly one method is required:

- `--random [--seed N]` runs Fisher–Yates with a portable SplitMix64 stream and
  rejection sampling. The default seed is `0`. The same input and seed produce
  byte-identical outputs across repeated runs. This RNG is **not** PISA's RNG.
- `--by-feature FILE` sorts one UTF-8 line per **old** document ID by UTF-8
  bytes, breaking equal-feature ties by old ID. Blank features are valid.
  For the four-document example, a feature file containing `z`, `a`, `m`, `b`
  on four lines yields new→old IDs `1, 3, 2, 0`.
- `--from-mapping FILE` accepts two whitespace-delimited decimal columns per
  row: `<old ID> <new ID>`. Rows may appear in any order, but every old ID must
  appear exactly once and new IDs must form a bijection over `0..N`.
- `--bp [--depth N] [--iterations N]` performs deterministic recursive graph
  bisection. It splits each ID-sorted range in half, scores moving each document
  using distinct field-qualified terms, sorts candidate gains (ties by old ID),
  and swaps pairs only when their joint gain strictly improves the degree-based
  compression proxy. It then recurses into each half. Depth and iteration count
  default to at most eight each and are capped at 20. This is the PISA/RBG
  algorithm family, but uses native floating-point logs, serial execution,
  deterministic ties, and an exact pair-gain guard; its maps need not match
  PISA's approximate-log, parallel-sort output. The objective is a proxy for
  delta-coded space, not a promise that every codec or corpus will shrink.

```shell
cargo run --release -- reorder \
  --input target/original.fwd \
  --forward-output target/bp.fwd --index-output target/bp.idx \
  --old-to-new target/bp.old-to-new --new-to-old target/bp.new-to-old \
  --bp --depth 4 --iterations 8
```

Both output map files also have two columns and no header. `--old-to-new`
contains `<old ID> <new ID>`; `--new-to-old` contains `<new ID> <old ID>`.
They are numeric internal-ID maps, not external-ID lexicons. The reordered
forward and inverted files retain each document's external ID. A produced
old→new file can be fed back through `--from-mapping` for another run from the
**original** snapshot.

The four output paths must be distinct from one another and from every input
path, including hard-link aliases. Output files are completely staged before
installation; ordinary write/install errors roll back the group. As with
other local multi-file workflows, this is not a power-loss-proof database
transaction. Inputs are capped at 256 MiB of forward payload, 1,000,000
documents, and 20,000,000 token occurrences; feature and mapping files are
each capped at 64 MiB. All counts and permutation ranges are validated before
rebuilding the inverted index. Bisection further caps input at 100,000
documents and 2,000,000 token occurrences, estimates at most 200,000,000
work units accounting for graph-term sorting and per-level ID/gain sorting as
well as term visits. The two degree arrays are allocated once and only terms
touched by a partition are reset before descending to its children; no full
term-count initialization occurs per recursive node. Larger requests are
rejected before writing outputs. Lower `--depth` or `--iterations` to fit a
larger permitted input.

This native workflow was informed by the reordering guide and `old→new`
permutation semantics in PISA commit `e88b09f`. The forward snapshot, inverted
index, map files, and seeded random stream are IndexSail formats/algorithms;
none claims byte compatibility with PISA. The bisection gain formula follows
PISA's frozen `recursive_graph_bisection.hpp`: for a term with degree `d` in a
side of `n` documents, the proxy contribution is
`d × (log₂(n) − log₂(d + 1))`. Only distinct document-term incidence contributes;
token multiplicity, field values, and positions remain unchanged in the output.
Small-case tests independently recompute this objective for every two-term
incidence pattern on four documents and seeded graphs of two to eight documents,
alongside map, resource-boundary, field-qualified-term, and exact-score checks.
This slice does not close the frozen PISA reordering catalog or the wider
whole-repository alignment gate: PISA's node-configuration and forward-cache
controls, its exact parallel/approximate-log execution, and index-size
benchmark evidence remain outside this native implementation.
