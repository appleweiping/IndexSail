# Retrieval experiment formats and metrics

## Topic inputs

The compact format is UTF-8 TSV with no header:

```text
topic-id<TAB>query text
```

Blank lines and lines whose first non-whitespace character is `#` are ignored. A query may contain tabs
after the first separator; they are treated as part of its text and normalized as whitespace by the
analyzer. Topic IDs must be unique and contain no whitespace.

The classic adapter recognizes line-separated `<top>` and `</top>` records and reads:

```text
<num> Number: 301
<title> International Organized Crime
```

Description and narrative sections are intentionally ignored. Closing `</num>` and `</title>` tags on the
same line are accepted. Other historical topic dialects should be converted to TSV.

## Qrels

Qrels use exactly four whitespace-separated columns:

```text
topic-id iteration document-id relevance
```

The iteration token is accepted but not interpreted. Relevance is a signed integer. Values at or below
zero are nonrelevant; values 1 through 31 are relevant and graded for nDCG. A topic/document pair may occur
only once. Topics absent from qrels have zero known relevant documents and therefore zero metrics.

## TREC run output

Each hit is emitted as:

```text
topic-id Q0 document-id one-based-rank score run-tag
```

Topics retain input order. Results retain the search engine's deterministic score-descending,
internal-ID-ascending order. Scores have twelve digits after the decimal point. Topic IDs, document IDs,
and run tags are validated so each line always has six whitespace-separated columns.

## Cutoff and metrics

All metrics are computed at the configured `top_k` cutoff.

### Average precision and MAP

For a topic with `R` known relevant documents:

```text
AP@k = sum(precision@r for every relevant hit at rank r <= k) / R
MAP@k = arithmetic mean of AP@k over all supplied topics
```

The denominator is the total known relevant count, not the number that happened to be retrieved. A topic
with no relevant judgments contributes zero.

### Reciprocal rank and MRR

`RR@k` is `1/r` for the first relevant hit at rank `r <= k`, otherwise zero. `MRR@k` is the arithmetic mean
over all supplied topics.

### Graded nDCG

```text
gain(rel) = 2^rel - 1                    when rel > 0, otherwise 0
DCG@k     = sum(gain(rel_r) / log2(r+1)) for one-based ranks r
nDCG@k    = DCG@k / ideal_DCG@k
```

The ideal list sorts all positive topic judgments by relevance descending. A topic with no positive gain
has nDCG zero.

### Recall

`Recall@k` is relevant retrieved by rank `k` divided by the total known relevant count. A topic with no
known relevant documents contributes zero.

## Search work counters

- `evaluated_candidates`: documents whose exact aggregate query score was computed.
- `postings_advanced`: cursor steps performed by the executor.
- `postings_skipped`: steps beyond the first during an `advance_to` operation.

Counters characterize this implementation's work and are deterministic for an identical index/query
configuration. They should not be interpreted as CPU instructions or compared directly across unrelated
engines.

## Exact executor verification

With `--verify`, the configured executor is timed first. The alternative executor then runs with identical
query, cutoff, field, operator, and BM25 parameters. Topic result lengths, internal document IDs, rank order,
and IEEE-754 score bits must all match. A difference aborts the batch and no successful report is claimed.

Verification time is recorded separately and is not included in `total_search_micros`.

## JSON schema version 1

The report top level contains:

- `schema_version`, strategy, operator, cutoff, field and verification setting;
- query count and total selected/verification microseconds;
- aggregate search counters;
- aggregate metrics or `null` when no qrels were supplied;
- ordered query objects with text, timing, counters, metrics and ranked hits.

The schema is deliberately emitted without a JSON dependency. Strings escape quotes, reverse solidus,
standard whitespace controls, and remaining U+0000 through U+001F characters. Search scores and metrics are
finite by construction.

## Fair experiment checklist

1. Record collection/topic/qrels provenance and exact checksums.
2. Keep analyzer, field, BM25 parameters, Boolean operator, and cutoff fixed across systems.
3. Use `--verify` when changing retrieval or codec code.
4. Archive the six-column run, JSON report, binary revision, compiler version, and command.
5. Treat timing as environment-dependent; warm up and repeat when performance is the research question.
6. Report metric definitions and cutoff explicitly, including the treatment of topics with no relevant
   judgments.
