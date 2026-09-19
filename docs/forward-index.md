# Forward index, lexicon, and collection pipeline

IndexSail has two collection paths. `index` and `shard-index` stream records
straight into searchable positional indexes. `forward-build` instead stores a
portable intermediate representation: the original named fields, every
normalized token occurrence in field order, and a canonical field-qualified
lexicon. `forward-invert` deterministically turns that snapshot into the same
native positional structure used by `search` and `batch`.

This is useful when collection parsing and inversion need separate, inspectable
experiment stages. It is inspired by the conventional forward-to-inverted
workflow used by research engines, but the `IDXFW001` file is an IndexSail
format and is not byte-compatible with PISA's binary-sequence files.

## Complete offline lifecycle

```shell
cargo run --release -- forward-build \
  --input examples/collection.jsonl \
  --output target/collection.fwd \
  --format jsonl

cargo run --release -- forward-inspect \
  --index target/collection.fwd \
  --document 0 \
  --limit 8

cargo run --release -- lexicon \
  --index target/collection.fwd \
  --field body \
  --term retrieval

cargo run --release -- forward-invert \
  --input target/collection.fwd \
  --output target/collection.idx

cargo run --release -- search \
  --index target/collection.idx \
  --query "exact retrieval" \
  --field body
```

The committed `examples/forward_demo.sh` and `examples/forward_demo.ps1` run
that chain without network access.

## Collection protocols

All inputs must be regular UTF-8 files. The default safety policy allows at
most 20,000,000 documents, 64 fields per parsed document, 64 MiB per record,
and 4 GiB for the input file. Library callers can lower those four values with
`CollectionLimits`; zero limits are invalid. The input ceiling meters bytes
actually read, including a concurrent append. The opened file is checked
against its starting size, modification time, and path identity after parsing;
an append or replacement during the read is rejected. Concurrent in-place
rewrites that preserve all of those attributes are outside this single-writer
contract. Limits are checked before a record is passed to an index builder.

### Header-led TSV

The first column must be `id`; every later header is a field name. Rows must
have exactly the header's column count. Blank rows are ignored. Tabs and line
breaks inside values are intentionally unsupported.

```text
id<TAB>title<TAB>body
D1<TAB>Local Search<TAB>BM25 ranks local documents
```

### TREC subset

The existing TREC adapter accepts line-separated `<DOC>` blocks with exactly
one `<DOCNO>`. `<TITLE>`/`<HEADLINE>` become `title`, while repeated
`<TEXT>`/`<BODY>` sections become `body`. The caller's record limit applies to
the entire raw block, including markers and line terminators. The parser
consumes at most one byte beyond the remaining block budget before rejection;
the fixed-size buffered reader may prefetch bytes but cannot bypass the
aggregate input ceiling.
This is a deterministic subset, not a general SGML parser.

### JSON Lines

Each nonblank line must be exactly one object in one of two shapes. The native
shape retains arbitrary valid IndexSail field names:

```json
{"id":"D1","fields":{"title":"Local Search","body":"BM25 ranks documents"}}
```

The PISA-style interchange shape uses `title` as the stable external document
identifier, `content` as `body`, and retains a nonempty optional `url` as a
searchable `url` field:

```json
{"title":"D1","content":"BM25 ranks documents","url":"https://example.test/D1"}
```

Shapes cannot be mixed within one record. Unknown keys, duplicate record keys,
duplicate field names, non-string IDs/fields/content/URLs, empty native field
maps, invalid UTF-8, trailing JSON, and excessive JSON nesting are rejected.
Different records in the same file may use different documented shapes.

## Lexicon and inversion invariants

Every lexicon key is `(field, normalized_term)`. Keys are strictly sorted, so a
term ID is independent of source document order and hash-map iteration. The
same surface token in `title` and `body` deliberately receives two IDs. A
document stores one term-ID sequence per original field, including duplicate
occurrences; its ordinal positions become posting positions during inversion.

Loading does more than check binary structure. IndexSail re-analyzes every
stored field and requires each saved ID to resolve to that exact field and
normalized token. It rejects unknown IDs, missing or extra sequences,
unsorted/duplicate/unused lexicon terms, duplicate external IDs, and occurrence
or field-count disagreement. This semantic closure prevents a recomputed
non-cryptographic checksum from turning arbitrary finite IDs into a different
searchable index while leaving the stored source text unchanged.

The independent unit oracle freezes this example:

```text
document 0 body = "blue sea blue"
body:blue -> [(doc=0, tf=2, positions=[0,2])]
body:sea  -> [(doc=0, tf=1, positions=[1]),
              (doc=1, tf=1, positions=[0])]
```

CLI tests then build JSONL, inspect a document, perform both lexicon lookup
directions, invert, and execute a real BM25 query.

## Library surface

- `CollectionFormat` and `CollectionLimits` select and bound parsing.
- `load_collection` materializes validated `Document` values.
- `index_collection` and `index_collection_sharded` stream directly into native
  builders.
- `ForwardIndex::from_documents` and `ForwardIndex::from_collection` build a
  canonical lexicon and occurrence sequences.
- `ForwardIndex::term`, `term_id`, `documents`, and `stats` expose inspection.
- `ForwardIndex::invert` builds exact positional postings.
- `ForwardIndex::save` and `load` implement the checked intermediate format.

`from_collection` necessarily retains all source documents because the forward
snapshot owns them. Use direct `index`/`shard-index` when that extra resident
representation is not needed.

## `IDXFW001` format version 1

All integers are little-endian. A string is a `u32` UTF-8 byte length followed
by its bytes.

```text
8 bytes  magic: IDXFW001
u32      version: 1
u64      payload byte length
u64      FNV-1a checksum of payload
payload:
  u8     analyzer mode
  u32    lexicon term count
  repeat term count:
    string field
    string normalized term
  u32    document count
  repeat document count:
    string external id
    u32 field count
    repeat field count:
      string field
      string original value
      u32 occurrence count
      repeat occurrence count: u32 term id
EOF required
```

The current limits are 4 GiB per payload, 64 MiB per string, 20,000,000
documents, 20,000,000 lexicon terms, 20,000,000 total fields, 4,096 fields per
library-created document, and 1,000,000,000 total occurrences. A loader checks
the real file length against the declared length before payload allocation,
checks collection counts against the minimum remaining structural bytes before
bulk allocation, verifies checksum and EOF, and finally applies the semantic
invariants above.

Saving counts the deterministic payload before allocation, writes a synced
same-directory staging file, and replaces the requested destination only after
serialization succeeds. The checksum detects accidental corruption; it is not
an authenticity mechanism. The format stores source text as well as token IDs,
so it favors auditability and exact reconstruction over minimum disk size.
