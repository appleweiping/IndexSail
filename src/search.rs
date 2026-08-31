use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::error::{Error, Result};
use crate::index::{InternalDocId, InvertedIndex, Posting};
use crate::query::{BooleanOperator, PhraseFilter, SearchQuery};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25Params {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Params {
    pub fn validate(self) -> Result<Self> {
        if !self.k1.is_finite() || self.k1 <= 0.0 {
            return Err(Error::InvalidArgument(
                "BM25 k1 must be finite and greater than zero".into(),
            ));
        }
        if !self.b.is_finite() || !(0.0..=1.0).contains(&self.b) {
            return Err(Error::InvalidArgument(
                "BM25 b must be finite and between zero and one".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PruningStrategy {
    Exhaustive,
    #[default]
    Wand,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchOptions {
    pub top_k: usize,
    pub pruning: PruningStrategy,
    pub explain: bool,
    pub bm25: Bm25Params,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            pruning: PruningStrategy::Wand,
            explain: false,
            bm25: Bm25Params::default(),
        }
    }
}

impl SearchOptions {
    fn validate(self) -> Result<Self> {
        if self.top_k == 0 {
            return Err(Error::InvalidArgument(
                "top_k must be greater than zero".into(),
            ));
        }
        self.bm25.validate()?;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TermContribution {
    pub term: String,
    pub field: String,
    pub term_frequency: u32,
    pub document_frequency: usize,
    pub document_length: u32,
    pub average_document_length: f64,
    pub inverse_document_frequency: f64,
    pub boost: f64,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    pub total_score: f64,
    pub terms: Vec<TermContribution>,
    pub phrase_filters_matched: usize,
    pub exact_filters_matched: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub doc_id: InternalDocId,
    pub external_id: String,
    pub score: f64,
    pub explanation: Option<Explanation>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchStats {
    pub evaluated_candidates: usize,
    pub postings_advanced: usize,
    pub postings_skipped: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchOutcome {
    pub hits: Vec<SearchHit>,
    pub stats: SearchStats,
}

#[derive(Clone, Debug)]
struct PreparedTerm {
    normalized: String,
    field: Option<String>,
    boost: f64,
}

#[derive(Clone, Copy, Debug)]
struct ScoredPosting {
    doc_id: InternalDocId,
    score: f64,
}

#[derive(Clone, Debug)]
struct TermScorer {
    ordinal: usize,
    entries: Vec<ScoredPosting>,
    upper_bound: f64,
}

#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    doc_id: InternalDocId,
    score: f64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && self.score.total_cmp(&other.score) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Ordering is deliberately reversed by score so the worst retained hit is on top.
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

#[derive(Debug)]
struct TopK {
    limit: usize,
    heap: BinaryHeap<HeapEntry>,
}

impl TopK {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit),
        }
    }

    fn is_full(&self) -> bool {
        self.heap.len() == self.limit
    }

    fn threshold(&self) -> Option<f64> {
        self.is_full()
            .then(|| self.heap.peek().expect("full heap").score)
    }

    fn consider(&mut self, candidate: HeapEntry) {
        if !self.is_full() {
            self.heap.push(candidate);
            return;
        }
        let worst = *self.heap.peek().expect("full heap");
        let is_better = candidate.score > worst.score
            || (candidate.score.total_cmp(&worst.score) == Ordering::Equal
                && candidate.doc_id < worst.doc_id);
        if is_better {
            self.heap.pop();
            self.heap.push(candidate);
        }
    }

    fn into_sorted(mut self) -> Vec<HeapEntry> {
        let mut entries = self.heap.drain().collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        entries
    }
}

impl InvertedIndex {
    pub fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
        let options = options.validate()?;
        let prepared_terms = self.prepare_terms(query)?;
        let mut scorers = prepared_terms
            .iter()
            .enumerate()
            .map(|(ordinal, term)| self.build_term_scorer(ordinal, term, options.bm25))
            .collect::<Result<Vec<_>>>()?;

        if query.operator() == BooleanOperator::And
            && scorers.iter().any(|scorer| scorer.entries.is_empty())
        {
            return Ok(SearchOutcome {
                hits: Vec::new(),
                stats: SearchStats::default(),
            });
        }
        scorers.retain(|scorer| !scorer.entries.is_empty());
        if scorers.is_empty() {
            return Ok(SearchOutcome {
                hits: Vec::new(),
                stats: SearchStats::default(),
            });
        }

        let (top_k, stats) = match options.pruning {
            PruningStrategy::Exhaustive => self.search_exhaustive(query, &scorers, options.top_k),
            PruningStrategy::Wand => self.search_wand(query, scorers, options.top_k),
        };

        let hits = top_k
            .into_sorted()
            .into_iter()
            .map(|entry| {
                let document = self
                    .document(entry.doc_id)
                    .expect("search candidates always reference existing documents");
                SearchHit {
                    doc_id: entry.doc_id,
                    external_id: document.external_id().to_owned(),
                    score: entry.score,
                    explanation: options
                        .explain
                        .then(|| self.explain(entry.doc_id, &prepared_terms, options.bm25, query)),
                }
            })
            .collect();

        Ok(SearchOutcome { hits, stats })
    }

    fn prepare_terms(&self, query: &SearchQuery) -> Result<Vec<PreparedTerm>> {
        let mut unique = BTreeMap::<(Option<String>, String), f64>::new();
        for term in query.terms() {
            let normalized = self
                .analyzer()
                .normalize_single(term.text())
                .ok_or_else(|| {
                    Error::InvalidQuery(format!(
                        "query term '{}' must analyze to exactly one token",
                        term.text()
                    ))
                })?;
            let key = (term.field().map(str::to_owned), normalized);
            let boost = unique.entry(key).or_default();
            *boost += term.boost();
            if !boost.is_finite() {
                return Err(Error::InvalidQuery(
                    "combined term boost exceeds finite range".into(),
                ));
            }
        }
        Ok(unique
            .into_iter()
            .map(|((field, normalized), boost)| PreparedTerm {
                normalized,
                field,
                boost,
            })
            .collect())
    }

    fn build_term_scorer(
        &self,
        ordinal: usize,
        term: &PreparedTerm,
        params: Bm25Params,
    ) -> Result<TermScorer> {
        let fields: Vec<&str> = match term.field.as_deref() {
            Some(field) => vec![field],
            None => self.fields().into_iter().collect(),
        };
        let mut merged = BTreeMap::<InternalDocId, f64>::new();
        for field in fields {
            let Some(postings) = self.postings(field, &term.normalized) else {
                continue;
            };
            let document_frequency = postings.len();
            let average_length = self.average_field_length(field);
            for posting in postings {
                let contribution = bm25_score(
                    posting.term_frequency,
                    document_frequency,
                    self.documents().len(),
                    self.field_length(posting.doc_id, field),
                    average_length,
                    params,
                ) * term.boost;
                if !contribution.is_finite() {
                    return Err(Error::InvalidArgument(format!(
                        "BM25 score for term '{}' is not finite; reduce boost or k1",
                        term.normalized
                    )));
                }
                let score = merged.entry(posting.doc_id).or_default();
                *score += contribution;
                if !score.is_finite() {
                    return Err(Error::InvalidArgument(format!(
                        "combined BM25 score for term '{}' is not finite",
                        term.normalized
                    )));
                }
            }
        }
        let entries = merged
            .into_iter()
            .map(|(doc_id, score)| ScoredPosting { doc_id, score })
            .collect::<Vec<_>>();
        let upper_bound =
            conservative_next_up(entries.iter().map(|entry| entry.score).fold(0.0, f64::max));
        Ok(TermScorer {
            ordinal,
            entries,
            upper_bound,
        })
    }

    fn search_exhaustive(
        &self,
        query: &SearchQuery,
        scorers: &[TermScorer],
        top_k: usize,
    ) -> (TopK, SearchStats) {
        let mut candidates = BTreeSet::new();
        match query.operator() {
            BooleanOperator::Or => {
                for scorer in scorers {
                    candidates.extend(scorer.entries.iter().map(|entry| entry.doc_id));
                }
            }
            BooleanOperator::And => {
                candidates.extend(scorers[0].entries.iter().map(|entry| entry.doc_id));
                for scorer in &scorers[1..] {
                    let present = scorer
                        .entries
                        .iter()
                        .map(|entry| entry.doc_id)
                        .collect::<BTreeSet<_>>();
                    candidates.retain(|doc_id| present.contains(doc_id));
                }
            }
        }

        let mut heap = TopK::new(top_k);
        let mut stats = SearchStats::default();
        for doc_id in candidates {
            stats.evaluated_candidates += 1;
            let score = scorers
                .iter()
                .filter_map(|scorer| score_at(&scorer.entries, doc_id))
                .sum();
            if self.matches_constraints(doc_id, query) {
                heap.consider(HeapEntry { doc_id, score });
            }
        }
        (heap, stats)
    }

    fn search_wand(
        &self,
        query: &SearchQuery,
        scorers: Vec<TermScorer>,
        top_k: usize,
    ) -> (TopK, SearchStats) {
        #[derive(Debug)]
        struct Cursor {
            scorer: TermScorer,
            position: usize,
        }

        impl Cursor {
            fn current_doc(&self) -> InternalDocId {
                self.scorer.entries[self.position].doc_id
            }

            fn current_score(&self) -> f64 {
                self.scorer.entries[self.position].score
            }

            fn advance_one(&mut self) -> usize {
                self.position += 1;
                1
            }

            fn advance_to(&mut self, target: InternalDocId) -> usize {
                let old = self.position;
                let offset = self.scorer.entries[self.position..]
                    .partition_point(|entry| entry.doc_id < target);
                self.position += offset;
                self.position - old
            }

            fn exhausted(&self) -> bool {
                self.position >= self.scorer.entries.len()
            }
        }

        let required_terms = scorers.len();
        let mut cursors = scorers
            .into_iter()
            .map(|scorer| Cursor {
                scorer,
                position: 0,
            })
            .collect::<Vec<_>>();
        let mut heap = TopK::new(top_k);
        let mut stats = SearchStats::default();

        loop {
            cursors.retain(|cursor| !cursor.exhausted());
            if cursors.is_empty()
                || (query.operator() == BooleanOperator::And && cursors.len() < required_terms)
            {
                break;
            }
            cursors.sort_by_key(Cursor::current_doc);

            let threshold = heap.threshold();
            let mut upper_bound = 0.0;
            let pivot_position = cursors.iter().position(|cursor| {
                upper_bound = conservative_next_up(upper_bound + cursor.scorer.upper_bound);
                threshold.is_none_or(|value| upper_bound >= value)
            });
            let Some(pivot_position) = pivot_position else {
                break;
            };
            let pivot_doc = cursors[pivot_position].current_doc();

            if cursors[0].current_doc() == pivot_doc {
                let matching_count = cursors
                    .iter()
                    .take_while(|cursor| cursor.current_doc() == pivot_doc)
                    .count();
                if query.operator() == BooleanOperator::Or || matching_count == required_terms {
                    stats.evaluated_candidates += 1;
                    let mut pieces = cursors[..matching_count]
                        .iter()
                        .map(|cursor| (cursor.scorer.ordinal, cursor.current_score()))
                        .collect::<Vec<_>>();
                    pieces.sort_by_key(|(ordinal, _)| *ordinal);
                    let score = pieces.into_iter().map(|(_, score)| score).sum();
                    if self.matches_constraints(pivot_doc, query) {
                        heap.consider(HeapEntry {
                            doc_id: pivot_doc,
                            score,
                        });
                    }
                }
                for cursor in &mut cursors[..matching_count] {
                    stats.postings_advanced += cursor.advance_one();
                }
            } else {
                for cursor in &mut cursors[..pivot_position] {
                    let advanced = cursor.advance_to(pivot_doc);
                    stats.postings_advanced += advanced;
                    stats.postings_skipped += advanced.saturating_sub(1);
                }
            }
        }
        (heap, stats)
    }

    fn matches_constraints(&self, doc_id: InternalDocId, query: &SearchQuery) -> bool {
        let Some(document) = self.document(doc_id) else {
            return false;
        };
        query
            .filters()
            .iter()
            .all(|filter| document.field(filter.field()) == Some(filter.value()))
            && query
                .phrases()
                .iter()
                .all(|phrase| self.matches_phrase(doc_id, phrase))
    }

    fn matches_phrase(&self, doc_id: InternalDocId, phrase: &PhraseFilter) -> bool {
        let fields: Vec<&str> = match phrase.field() {
            Some(field) => vec![field],
            None => self.fields().into_iter().collect(),
        };
        fields.into_iter().any(|field| {
            let posting_lists = phrase
                .terms()
                .iter()
                .map(|term| {
                    self.postings(field, term)
                        .and_then(|list| posting_for_doc(list, doc_id))
                })
                .collect::<Option<Vec<_>>>();
            let Some(postings) = posting_lists else {
                return false;
            };
            postings[0].positions.iter().any(|start| {
                postings
                    .iter()
                    .enumerate()
                    .skip(1)
                    .all(|(offset, posting)| {
                        let Ok(offset) = u32::try_from(offset) else {
                            return false;
                        };
                        start.checked_add(offset).is_some_and(|position| {
                            posting.positions.binary_search(&position).is_ok()
                        })
                    })
            })
        })
    }

    fn explain(
        &self,
        doc_id: InternalDocId,
        terms: &[PreparedTerm],
        params: Bm25Params,
        query: &SearchQuery,
    ) -> Explanation {
        let mut contributions = Vec::new();
        for term in terms {
            let fields: Vec<&str> = match term.field.as_deref() {
                Some(field) => vec![field],
                None => self.fields().into_iter().collect(),
            };
            for field in fields {
                let Some(postings) = self.postings(field, &term.normalized) else {
                    continue;
                };
                let Some(posting) = posting_for_doc(postings, doc_id) else {
                    continue;
                };
                let document_frequency = postings.len();
                let average_document_length = self.average_field_length(field);
                let inverse_document_frequency =
                    bm25_idf(self.documents().len(), document_frequency);
                let score = bm25_score(
                    posting.term_frequency,
                    document_frequency,
                    self.documents().len(),
                    self.field_length(doc_id, field),
                    average_document_length,
                    params,
                ) * term.boost;
                contributions.push(TermContribution {
                    term: term.normalized.clone(),
                    field: field.to_owned(),
                    term_frequency: posting.term_frequency,
                    document_frequency,
                    document_length: self.field_length(doc_id, field),
                    average_document_length,
                    inverse_document_frequency,
                    boost: term.boost,
                    score,
                });
            }
        }
        Explanation {
            total_score: contributions.iter().map(|term| term.score).sum(),
            terms: contributions,
            phrase_filters_matched: query.phrases().len(),
            exact_filters_matched: query.filters().len(),
        }
    }
}

fn score_at(entries: &[ScoredPosting], doc_id: InternalDocId) -> Option<f64> {
    entries
        .binary_search_by_key(&doc_id, |entry| entry.doc_id)
        .ok()
        .map(|index| entries[index].score)
}

fn posting_for_doc(postings: &[Posting], doc_id: InternalDocId) -> Option<&Posting> {
    postings
        .binary_search_by_key(&doc_id, |posting| posting.doc_id)
        .ok()
        .map(|index| &postings[index])
}

/// Return the next representable positive float so `WAND` bounds round outward.
fn conservative_next_up(value: f64) -> f64 {
    if value.is_nan() || (value.is_infinite() && value.is_sign_positive()) {
        return value;
    }
    if value.to_bits() == (-0.0_f64).to_bits() {
        return f64::from_bits(1);
    }
    if value >= 0.0 {
        return f64::from_bits(value.to_bits() + 1);
    }
    f64::from_bits(value.to_bits() - 1)
}

#[allow(clippy::cast_precision_loss)]
pub fn bm25_idf(document_count: usize, document_frequency: usize) -> f64 {
    if document_count == 0 || document_frequency == 0 {
        return 0.0;
    }
    let documents = document_count as f64;
    let frequency = document_frequency as f64;
    (1.0 + (documents - frequency + 0.5) / (frequency + 0.5)).ln()
}

fn bm25_score(
    term_frequency: u32,
    document_frequency: usize,
    document_count: usize,
    document_length: u32,
    average_document_length: f64,
    params: Bm25Params,
) -> f64 {
    if term_frequency == 0 || average_document_length <= 0.0 {
        return 0.0;
    }
    let frequency = f64::from(term_frequency);
    let normalized_length = f64::from(document_length) / average_document_length;
    let denominator = frequency + params.k1 * (1.0 - params.b + params.b * normalized_length);
    bm25_idf(document_count, document_frequency) * frequency * (params.k1 + 1.0) / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Analyzer;
    use crate::document::Document;
    use crate::index::IndexBuilder;
    use crate::query::{FieldFilter, PhraseFilter, QueryTerm};

    fn test_index() -> InvertedIndex {
        let documents = [
            ("a", "Rust search", "fast local search engine", "guide"),
            ("b", "Power grid", "grid model and solver", "reference"),
            ("c", "Search ranking", "search search bm25 ranking", "guide"),
            ("d", "Rust model", "local model with rust", "note"),
            ("e", "Other", "unrelated words", "note"),
        ];
        let mut builder = IndexBuilder::new(Analyzer::default());
        for (id, title, body, category) in documents {
            builder
                .add_document(
                    Document::from_fields(
                        id,
                        [("title", title), ("body", body), ("category", category)],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        builder.finish()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn search(
        index: &InvertedIndex,
        query: SearchQuery,
        strategy: PruningStrategy,
    ) -> SearchOutcome {
        index
            .search(
                &query,
                SearchOptions {
                    top_k: 10,
                    pruning: strategy,
                    explain: false,
                    bm25: Bm25Params::default(),
                },
            )
            .unwrap()
    }

    #[test]
    fn idf_is_higher_for_rare_terms() {
        assert!(bm25_idf(100, 2) > bm25_idf(100, 50));
        assert!(bm25_idf(0, 0).abs() < f64::EPSILON);
    }

    #[test]
    fn wand_upper_bounds_round_outward() {
        assert!(conservative_next_up(1.0) > 1.0);
        assert!(conservative_next_up(0.0) > 0.0);
        assert!(conservative_next_up(f64::INFINITY).is_infinite());
    }

    #[test]
    fn or_query_returns_documents_matching_any_term() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "rust grid", Some("body")).unwrap();
        let ids: Vec<_> = search(&index, query, PruningStrategy::Exhaustive)
            .hits
            .into_iter()
            .map(|hit| hit.external_id)
            .collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"b".into()));
        assert!(ids.contains(&"d".into()));
    }

    #[test]
    fn and_query_requires_every_unique_term() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "rust local", None)
            .unwrap()
            .with_operator(BooleanOperator::And);
        let mut ids: Vec<_> = search(&index, query, PruningStrategy::Exhaustive)
            .hits
            .into_iter()
            .map(|hit| hit.external_id)
            .collect();
        ids.sort();
        assert_eq!(ids, ["a", "d"]);
    }

    #[test]
    fn fielded_query_does_not_match_other_fields() {
        let index = test_index();
        let title_query =
            SearchQuery::from_text(index.analyzer(), "ranking", Some("title")).unwrap();
        let body_query = SearchQuery::from_text(index.analyzer(), "ranking", Some("body")).unwrap();
        assert_eq!(
            search(&index, title_query, PruningStrategy::Exhaustive)
                .hits
                .len(),
            1
        );
        assert_eq!(
            search(&index, body_query, PruningStrategy::Exhaustive)
                .hits
                .len(),
            1
        );
    }

    #[test]
    fn unfielded_query_combines_scores_across_fields() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "search", None).unwrap();
        let outcome = search(&index, query, PruningStrategy::Exhaustive);
        assert_eq!(outcome.hits[0].external_id, "c");
        assert!(outcome.hits[0].score > outcome.hits[1].score);
    }

    #[test]
    fn exact_phrase_filter_checks_order_and_adjacency() {
        let index = test_index();
        let phrase =
            PhraseFilter::from_text(index.analyzer(), "local search", Some("body".into())).unwrap();
        let query = SearchQuery::from_text(index.analyzer(), "search", Some("body"))
            .unwrap()
            .with_phrase(phrase);
        let ids: Vec<_> = search(&index, query, PruningStrategy::Exhaustive)
            .hits
            .into_iter()
            .map(|hit| hit.external_id)
            .collect();
        assert_eq!(ids, ["a"]);
    }

    #[test]
    fn phrase_without_field_can_match_any_single_field() {
        let index = test_index();
        let phrase = PhraseFilter::from_text(index.analyzer(), "rust search", None).unwrap();
        let query = SearchQuery::from_text(index.analyzer(), "rust", None)
            .unwrap()
            .with_phrase(phrase);
        let ids: Vec<_> = search(&index, query, PruningStrategy::Exhaustive)
            .hits
            .into_iter()
            .map(|hit| hit.external_id)
            .collect();
        assert_eq!(ids, ["a"]);
    }

    #[test]
    fn exact_field_filter_is_not_tokenized() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "model", None)
            .unwrap()
            .with_filter(FieldFilter::exact("category", "reference").unwrap());
        let ids: Vec<_> = search(&index, query, PruningStrategy::Exhaustive)
            .hits
            .into_iter()
            .map(|hit| hit.external_id)
            .collect();
        assert_eq!(ids, ["b"]);
    }

    #[test]
    fn deterministic_tie_break_prefers_lower_internal_id() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        for id in ["first", "second", "third"] {
            builder
                .add_document(Document::from_fields(id, [("body", "same")]).unwrap())
                .unwrap();
        }
        let index = builder.finish();
        let query = SearchQuery::from_text(index.analyzer(), "same", Some("body")).unwrap();
        let outcome = index
            .search(
                &query,
                SearchOptions {
                    top_k: 2,
                    pruning: PruningStrategy::Wand,
                    ..SearchOptions::default()
                },
            )
            .unwrap();
        assert_eq!(
            outcome
                .hits
                .iter()
                .map(|hit| hit.external_id.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
    }

    #[test]
    fn wand_and_exhaustive_return_identical_rankings() {
        let index = test_index();
        for (text, operator) in [
            ("rust search model", BooleanOperator::Or),
            ("rust local", BooleanOperator::And),
            ("grid missing", BooleanOperator::Or),
        ] {
            let query = SearchQuery::from_text(index.analyzer(), text, None)
                .unwrap()
                .with_operator(operator);
            let exhaustive = search(&index, query.clone(), PruningStrategy::Exhaustive);
            let wand = search(&index, query, PruningStrategy::Wand);
            assert_eq!(
                exhaustive
                    .hits
                    .iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>(),
                wand.hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>()
            );
            for (left, right) in exhaustive.hits.iter().zip(&wand.hits) {
                assert!((left.score - right.score).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn wand_remains_exact_with_post_filters() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "search model rust", None)
            .unwrap()
            .with_filter(FieldFilter::exact("category", "guide").unwrap());
        let exhaustive = search(&index, query.clone(), PruningStrategy::Exhaustive);
        let wand = search(&index, query, PruningStrategy::Wand);
        assert_eq!(exhaustive.hits, wand.hits);
    }

    #[test]
    fn explanations_reconstruct_total_score() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "rust search", None).unwrap();
        let outcome = index
            .search(
                &query,
                SearchOptions {
                    explain: true,
                    pruning: PruningStrategy::Exhaustive,
                    ..SearchOptions::default()
                },
            )
            .unwrap();
        let hit = &outcome.hits[0];
        let explanation = hit.explanation.as_ref().unwrap();
        assert!((explanation.total_score - hit.score).abs() < 1e-12);
        assert!(!explanation.terms.is_empty());
        assert!(explanation.terms.iter().all(|term| term.score > 0.0));
    }

    #[test]
    fn missing_term_is_empty_for_and_and_ignored_for_or() {
        let index = test_index();
        let and_query = SearchQuery::from_text(index.analyzer(), "rust zzzmissing", None)
            .unwrap()
            .with_operator(BooleanOperator::And);
        let or_query = SearchQuery::from_text(index.analyzer(), "rust zzzmissing", None).unwrap();
        assert!(
            search(&index, and_query, PruningStrategy::Wand)
                .hits
                .is_empty()
        );
        assert!(
            !search(&index, or_query, PruningStrategy::Wand)
                .hits
                .is_empty()
        );
    }

    #[test]
    fn duplicate_terms_combine_their_boosts_without_changing_and_semantics() {
        let index = test_index();
        let query = SearchQuery::from_terms(vec![
            QueryTerm::new("rust", None, 1.0).unwrap(),
            QueryTerm::new("rust", None, 2.0).unwrap(),
        ])
        .unwrap()
        .with_operator(BooleanOperator::And);
        let outcome = search(&index, query, PruningStrategy::Exhaustive);
        assert_eq!(outcome.hits.len(), 2);
        let single = SearchQuery::from_text(index.analyzer(), "rust", None).unwrap();
        let single_outcome = search(&index, single, PruningStrategy::Exhaustive);
        assert!((outcome.hits[0].score - single_outcome.hits[0].score * 3.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_search_options_are_rejected() {
        let index = test_index();
        let query = SearchQuery::from_text(index.analyzer(), "rust", None).unwrap();
        assert!(
            index
                .search(
                    &query,
                    SearchOptions {
                        top_k: 0,
                        ..SearchOptions::default()
                    }
                )
                .is_err()
        );
        assert!(
            index
                .search(
                    &query,
                    SearchOptions {
                        bm25: Bm25Params { k1: 1.2, b: 1.1 },
                        ..SearchOptions::default()
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn non_finite_computed_score_is_rejected() {
        let index = test_index();
        let query =
            SearchQuery::from_terms(vec![QueryTerm::new("search", None, f64::MAX).unwrap()])
                .unwrap();
        assert!(index.search(&query, SearchOptions::default()).is_err());
    }

    #[test]
    fn custom_term_boost_changes_ranking() {
        let index = test_index();
        let query = SearchQuery::from_terms(vec![
            QueryTerm::new("grid", Some("body".into()), 8.0).unwrap(),
            QueryTerm::new("search", Some("body".into()), 1.0).unwrap(),
        ])
        .unwrap();
        let outcome = search(&index, query, PruningStrategy::Wand);
        assert_eq!(outcome.hits[0].external_id, "b");
    }
}
