# CIFF v1 interoperability

IndexSail reads and writes the Common Index File Format (CIFF) version 1. CIFF
is an index interchange contract: it moves normalized terms, posting payloads,
document identifiers, document lengths, and collection statistics between
retrieval systems. On conventional indexes, `Posting.tf` is term frequency.
Some learned-sparse producers use the same positive integer field for a
quantized impact. The normative schema is the OSIRRC
[`CommonIndexFileFormat.proto`](https://github.com/osirrc/ciff/blob/master/src/main/protobuf/CommonIndexFileFormat.proto).
The design and cross-engine motivation are described in
[*Supporting Interoperability Between Open-Source Search Engines with the Common Index File Format*](https://arxiv.org/abs/2003.08276).

## End-to-end workflow

Build a native fielded index and export its bag-of-words view:

```shell
cargo run --release -- index \
  --input examples/corpus.tsv \
  --output target/corpus.idx

cargo run --release -- ciff-export \
  --index target/corpus.idx \
  --output target/corpus.ciff \
  --description "IndexSail example corpus"
```

Inspect and query the portable index:

```shell
cargo run --release -- ciff-inspect \
  --index target/corpus.ciff \
  --term search

cargo run --release -- ciff-search \
  --index target/corpus.ciff \
  --query "local search" \
  --operator or \
  --top-k 10
```

Run conventional topics and qrels directly against CIFF:

```shell
cargo run --release -- ciff-batch \
  --index target/corpus.ciff \
  --topics examples/topics.tsv \
  --qrels examples/qrels.txt \
  --run target/ciff.run \
  --report target/ciff-report.json \
  --top-k 10
```

The committed PowerShell and POSIX examples execute this entire path. They do
not download data or require credentials.

Interoperability was also checked against the PISA CIFF repository's 337-byte
[`toy-complete-20200309.ciff`](https://github.com/pisa-engine/ciff/blob/0689e2a7536be4306c4eab5472e2bbe6a6696b58/tests/test_data/toy-complete-20200309.ciff) fixture at commit
`0689e2a7536be4306c4eab5472e2bbe6a6696b58`. Its SHA-256 is
`2fce5061fe994f08ae8911d69ff010969892a53b2c85b0461ce87c9ad37867c4`;
IndexSail decodes 3 documents, 9 terms, 14 postings, 16 collection tokens, and
the three documented external IDs. The third-party binary is not
redistributed here. The unit suite separately carries a hand-transcribed
wire-level fixture and a 71-byte learned-sparse fixture produced by PISA's
`jsonl2ciff` at that pinned commit. The latter deliberately has `cf = 2` and
`total_terms_in_collection = 1`, freezing the distinction between an impact
sum and document-vector length. Normal CI remains offline and deterministic.

## Wire contract

A raw `.ciff` file is a sequence of protobuf messages. Every message is
prefixed with its unsigned base-128 varint byte length:

```text
Header
exactly Header.num_postings_lists PostingsList messages
exactly Header.num_docs DocRecord messages
EOF
```

Each `PostingsList` contains nested `Posting` messages. Its `docid` values are
d-gaps, not absolute identifiers. The first absolute ID is the first gap and
every later ID is a checked prefix sum. `DocRecord.docid` is absolute. Scalar
integer fields are signed protobuf `int32`/`int64` in the official schema, so
IndexSail accepts only their non-negative ranges. Proto3 zero values may be
absent from the wire; this is necessary for document zero and a first d-gap of
zero.

The writer emits fields in numeric order, omits scalar defaults, orders terms
lexicographically and DocRecords by integer ID, and produces identical bytes
for identical logical state. Readers do not depend on field order. Unknown
varint, fixed-width, and length-delimited fields are skipped for protobuf
forward compatibility; deprecated group wire types and wrong known-field wire
types fail closed.

## Validation and limits

`CiffLimits` governs parsing and writing. The default policy includes bounds
for:

- one delimited frame;
- contained posting lists and DocRecords;
- total decoded postings;
- term, external-ID, and description byte lengths.

Frame length is checked before allocating its payload. Header counts are
checked before any collection-sized loop. Postings are counted against the
remaining aggregate budget as they are decoded, so no later list receives the
original full budget. Each posting list and DocRecord is validated immediately,
including duplicate keys, before the next declared record is read;
empty/default frames therefore fail at the first record instead of amplifying
header counts. The reader then probes one byte for exact EOF and validates all
cross-record invariants:

1. the contained counts equal the header declarations;
2. contained counts do not exceed the header totals;
3. terms, integer IDs, and external IDs are unique;
4. every list has one or more postings and exactly `df` strictly increasing document IDs;
5. every `tf` payload is positive and its sum equals `cf`;
6. every referenced document has one DocRecord;
7. document IDs are below `total_docs`;
8. each `cf` and `total_terms_in_collection` fits the schema's non-negative `int64` range;
9. zero/non-zero total terms and average document length agree;
10. all floating-point values used by scoring are finite and non-negative.

There is intentionally no `sum(cf) <= total_terms_in_collection` rule. That
relationship holds for ordinary term-frequency payloads, but not for CIFF
learned-sparse indexes whose `tf` values are quantized impacts while document
lengths count non-zero vector entries.

Malformed varints, invalid UTF-8, prefix-sum overflow, duplicated known scalar
fields, truncated frames, and undeclared trailing frames are errors. The
reader is safe Rust and the crate continues to forbid unsafe code.

## Scoring semantics

CIFF records the collection-wide document count and average document length.
For a frequency-valued CIFF index, IndexSail uses those exact header values for
Robertson BM25:

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

score(t,d) = idf(t) * tf(t,d) * (k1 + 1)
             / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
```

The logarithm is the same pinned `libm` operation used by native retrieval.
CIFF search is currently exhaustive because the interchange schema has no
persisted safe-impact bounds. Results sort by score descending and then CIFF
integer document ID ascending. The library exposes `CiffRetrieval` so the same
TREC evaluator, run writer, metrics, and JSON report contract can operate over
a CIFF index.

BM25 commands interpret `Posting.tf` as a term count. They are therefore not a
learned-sparse impact scorer: an impact-valued CIFF remains valid for import,
inspection, canonical save/load, and exchange with other CIFF tools, but its
BM25 scores are not meaningful. IndexSail never rejects such a file merely
because summed impacts exceed the collection's vector-length statistic.

The query analyzer is intentionally supplied by the caller. CIFF describes an
exporter's analysis in free text but does not standardize tokenization. A
consumer must select the analyzer matching the exporting system. The CLI uses
IndexSail Unicode analysis by default and `--ascii` when requested.
Repeated analyzed tokens are aggregated deterministically and scale that
term's contribution. The `CiffRetrieval` adapter performs the same
normalization and sums explicit `QueryTerm` boosts; Boolean AND counts unique
normalized terms.

## Native export is explicitly lossy

A native IndexSail index has named fields, stored text, token positions, exact
field lengths, and per-field statistics. CIFF v1 has one term stream and no
positions or stored text. `CiffIndex::from_native` therefore:

- combines the same normalized term across fields by summing frequency per
  document;
- sums all field lengths into one document length;
- preserves insertion IDs and external collection IDs;
- records the flattening and analyzer mode in `Header.description`.

This portable representation supports BM25 and Boolean AND/OR. It cannot
recover phrase search, field-qualified terms, stored-field filters, original
text, or native WAND/block-max metadata. The API keeps `CiffIndex` separate
from `InvertedIndex` so callers cannot accidentally assume those features
survived export.

Document lengths in third-party CIFF exports may be approximate; this is an
explicit property of some source engines and is why the header carries an
exporter-selected average. IndexSail preserves and scores with the supplied
values instead of silently recomputing different global statistics.

CIFF files are not authenticated. Transport them with an independently
verified digest or signed release when provenance matters. Compression such as
`.ciff.gz` is an outer transport layer; decompress it before using these
commands.

CLI export and batch commands compare existing files by filesystem identity,
not just spelling, so hard-linked input/output aliases are rejected. Batch run
and JSON content is completely serialized and validated before either target
is changed. Both files are staged and synced in their destination directories,
then committed as one recoverable transaction; an I/O error or unwinding panic
during either install restores both previous outputs. CIFF export uses the same
staging discipline, so validation, serialization, and ordinary write failures
cannot expose a partial index or truncate an existing destination. Symbolic-link
outputs are rejected; replacing one hard-link name does not modify its sibling.

The writer calculates each protobuf message's checked encoded size before
streaming the frame directly to its destination. `max_frame_bytes` therefore
bounds both accepted input frames and generated output frames without requiring
the writer to allocate a frame-sized buffer.
