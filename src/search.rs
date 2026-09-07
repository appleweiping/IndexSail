use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::error::{Error, Result};
use crate::index::{
    BLOCK_POSTINGS, BlockMaxKey, BlockMaxMetadata, InternalDocId, InvertedIndex, Posting,
};
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
    /// `WAND` refined by per-block maximum impacts.
    ///
    /// Plain `WAND` bounds a term by the largest impact anywhere in its
    /// postings, so a single outlier document keeps that bound high for every
    /// other document the term touches. Remembering a maximum per block lets a
    /// run of low-impact postings be skipped whole rather than one document at
    /// a time. The ranking is unchanged: a block maximum is a true upper bound
    /// inside its block, so nothing that could enter the top-k is skipped.
    BlockMaxWand,
    /// `MaxScore` term-at-a-time pruning using exact per-term upper bounds.
    ///
    /// Terms are ordered by their maximum possible contribution.  Once the
    /// top-k heap has a threshold, documents occurring only in the low-impact
    /// prefix can be skipped safely; candidates from the remaining essential
    /// terms are scored against the full term set.  This is the same family
    /// of algorithm as PISA's `MaxScore` executor, while retaining `IndexSail`'s
    /// deterministic tie-breaking and post-filter semantics.
    MaxScore,
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
    /// Precomputed default-BM25 block bounds loaded into query scorers.
    ///
    /// Format v3 persists this resident table; a just-built index can consume
    /// it before it has been saved.
    pub block_max_bounds_loaded: usize,
    /// Scored postings summarized by those precomputed bounds.
    pub block_max_postings_covered: usize,
    /// Scored postings inspected to derive conservative bounds from exact
    /// scores for custom BM25 or non-unit boosts. Default BM25 with unit boost
    /// keeps this at zero.
    pub block_max_postings_scanned: usize,
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
    /// Largest impact within each `BLOCK_POSTINGS`-sized run of `entries`.
    ///
    /// Loaded from index metadata for default BM25, or derived from the exact
    /// materialized scores for custom parameters.
    block_max: Vec<f64>,
    precomputed_block_bounds_loaded: usize,
    postings_scanned_for_block_bounds: usize,
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

/// One term's position in a `WAND`-style walk over its scored postings.
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
        let offset =
            self.scorer.entries[self.position..].partition_point(|entry| entry.doc_id < target);
        self.position += offset;
        self.position - old
    }

    fn exhausted(&self) -> bool {
        self.position >= self.scorer.entries.len()
    }

    /// The largest impact this term can contribute anywhere inside the block
    /// the cursor currently sits in.
    fn block_max(&self) -> f64 {
        self.scorer.block_max[self.position / BLOCK_POSTINGS]
    }

    /// The last document the current block covers, which is how far its bound
    /// stays valid.
    fn block_last_doc(&self) -> InternalDocId {
        let block = self.position / BLOCK_POSTINGS;
        let end = ((block + 1) * BLOCK_POSTINGS).min(self.scorer.entries.len()) - 1;
        self.scorer.entries[end].doc_id
    }
}

/// Skip a run of documents no block can lift above the threshold.
///
/// Returns whether the cursors moved. The bound covers every cursor that could
/// contribute at the pivot, including one past the pivot position that happens
/// to sit on the same document: leaving that one out would understate a score
/// about to be computed, and could skip a document belonging in the top-k.
fn block_skip(
    cursors: &mut [Cursor],
    pivot_position: usize,
    pivot_doc: InternalDocId,
    threshold: f64,
    stats: &mut SearchStats,
) -> bool {
    let bound_end = cursors
        .iter()
        .position(|cursor| cursor.current_doc() > pivot_doc)
        .unwrap_or(cursors.len())
        .max(pivot_position + 1);
    let mut bound = 0.0;
    for cursor in &cursors[..bound_end] {
        bound = conservative_next_up(bound + cursor.block_max());
    }
    if bound >= threshold {
        return false;
    }
    // The bound holds until the earliest block ends, and no cursor beyond it
    // reaches back before its own current document.
    let mut next = cursors[..bound_end]
        .iter()
        .map(Cursor::block_last_doc)
        .min()
        .expect("a pivot always has at least one cursor")
        .saturating_add(1);
    if let Some(after) = cursors.get(bound_end) {
        next = next.min(after.current_doc());
    }
    // The pivot cannot reach the threshold either, so stepping past it is both
    // safe and what guarantees the loop advances.
    next = next.max(pivot_doc.saturating_add(1));
    for cursor in cursors.iter_mut() {
        if cursor.current_doc() < next {
            let advanced = cursor.advance_to(next);
            stats.postings_advanced += advanced;
            stats.postings_skipped += advanced.saturating_sub(1);
        }
    }
    true
}

impl InvertedIndex {
    /// Build the complete field-qualified and all-field block-bound table.
    ///
    /// New indexes run this once when the builder is finalized. Legacy indexes
    /// run it once while loading; format version 3 reads and validates the
    /// persisted table instead. Default-BM25 requests use these bounds without
    /// rescanning scores; custom parameters derive conservative bounds from
    /// exact scores at query time.
    pub(crate) fn compute_block_max_metadata(&self) -> BTreeMap<BlockMaxKey, BlockMaxMetadata> {
        let mut metadata = BTreeMap::new();
        let mut fields_by_term = BTreeMap::<&str, Vec<(&str, &[Posting])>>::new();

        for (key, postings) in &self.postings {
            let scores = self.bound_scores(key.field.as_str(), postings);
            metadata.insert(
                BlockMaxKey {
                    field: Some(key.field.clone()),
                    term: key.term.clone(),
                },
                block_metadata(&scores),
            );
            fields_by_term
                .entry(key.term.as_str())
                .or_default()
                .push((key.field.as_str(), postings.as_slice()));
        }

        for (term, fields) in fields_by_term {
            let mut merged = BTreeMap::<InternalDocId, f64>::new();
            for (field, postings) in fields {
                for (doc_id, default_score) in self.bound_scores(field, postings) {
                    let total = merged.entry(doc_id).or_default();
                    // This is the same deterministic field order and addition
                    // used by query scoring for a boost of one.
                    *total += default_score;
                }
            }
            let scores = merged.into_iter().collect::<Vec<_>>();
            metadata.insert(
                BlockMaxKey {
                    field: None,
                    term: term.to_owned(),
                },
                block_metadata(&scores),
            );
        }
        metadata
    }

    fn bound_scores(&self, field: &str, postings: &[Posting]) -> Vec<(InternalDocId, f64)> {
        let document_frequency = postings.len();
        let document_count = self.documents().len();
        let average_length = self.average_field_length(field);
        postings
            .iter()
            .map(|posting| {
                let document_length = self.field_length(posting.doc_id, field);
                let default_score = bm25_score(
                    posting.term_frequency,
                    document_frequency,
                    document_count,
                    document_length,
                    average_length,
                    Bm25Params::default(),
                );
                (posting.doc_id, default_score)
            })
            .collect()
    }

    pub fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
        let options = options.validate()?;
        let prepared_terms = self.prepare_terms(query)?;
        let mut scorers = prepared_terms
            .iter()
            .enumerate()
            .map(|(ordinal, term)| {
                self.build_term_scorer(
                    ordinal,
                    term,
                    options.bm25,
                    options.pruning == PruningStrategy::BlockMaxWand,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let block_max_bounds_loaded = scorers
            .iter()
            .map(|scorer| scorer.precomputed_block_bounds_loaded)
            .sum();
        let block_max_postings_covered = scorers
            .iter()
            .filter(|scorer| scorer.precomputed_block_bounds_loaded > 0)
            .map(|scorer| scorer.entries.len())
            .sum();
        let block_max_postings_scanned = scorers
            .iter()
            .map(|scorer| scorer.postings_scanned_for_block_bounds)
            .sum();
        let preparation_stats = SearchStats {
            block_max_bounds_loaded,
            block_max_postings_covered,
            block_max_postings_scanned,
            ..SearchStats::default()
        };

        if query.operator() == BooleanOperator::And
            && scorers.iter().any(|scorer| scorer.entries.is_empty())
        {
            return Ok(SearchOutcome {
                hits: Vec::new(),
                stats: preparation_stats,
            });
        }
        scorers.retain(|scorer| !scorer.entries.is_empty());
        if scorers.is_empty() {
            return Ok(SearchOutcome {
                hits: Vec::new(),
                stats: preparation_stats,
            });
        }

        let (top_k, mut stats) = match options.pruning {
            PruningStrategy::Exhaustive => self.search_exhaustive(query, &scorers, options.top_k),
            PruningStrategy::Wand => self.search_wand(query, scorers, options.top_k, false),
            PruningStrategy::BlockMaxWand => self.search_wand(query, scorers, options.top_k, true),
            PruningStrategy::MaxScore => self.search_max_score(query, scorers, options.top_k),
        };
        stats.block_max_bounds_loaded = preparation_stats.block_max_bounds_loaded;
        stats.block_max_postings_covered = preparation_stats.block_max_postings_covered;
        stats.block_max_postings_scanned = preparation_stats.block_max_postings_scanned;

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
        load_block_max: bool,
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
        let (block_max, precomputed_block_bounds_loaded, postings_scanned_for_block_bounds) =
            if entries.is_empty() || !load_block_max {
                (Vec::new(), 0, 0)
            } else {
                let key = BlockMaxKey {
                    field: term.field.clone(),
                    term: term.normalized.clone(),
                };
                let stored = self.block_max.get(&key).ok_or_else(|| {
                    Error::CorruptIndex(format!(
                        "missing block-max metadata for term '{}'",
                        term.normalized
                    ))
                })?;
                if stored.posting_count != entries.len() {
                    return Err(Error::CorruptIndex(format!(
                        "block-max posting count differs for term '{}'",
                        term.normalized
                    )));
                }
                // Default, unit-boost queries use tight persisted bounds. All other
                // valid options derive conservative bounds from their exact scores.
                let defaults = Bm25Params::default();
                let use_tight_bounds = params.k1.to_bits() == defaults.k1.to_bits()
                    && params.b.to_bits() == defaults.b.to_bits()
                    && term.boost.to_bits() == 1.0_f64.to_bits();
                if use_tight_bounds {
                    (
                        stored
                            .default_bounds
                            .iter()
                            .map(|bits| f64::from_bits(*bits))
                            .collect(),
                        stored.default_bounds.len(),
                        0,
                    )
                } else {
                    // The public API accepts every finite positive k1 and boost.
                    // No fixed ULP allowance can turn a parameter-independent
                    // floating-point formula into a proof over that full range.
                    // The exact scores are already materialized, so custom
                    // requests derive outward-rounded bounds directly and report
                    // the work instead of risking an unsafe pruning bound.
                    (conservative_block_bounds(&entries), 0, entries.len())
                }
            };
        Ok(TermScorer {
            ordinal,
            entries,
            upper_bound,
            block_max,
            precomputed_block_bounds_loaded,
            postings_scanned_for_block_bounds,
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

    /// `WAND`, optionally refined by per-block maximum impacts.
    ///
    /// Both strategies share this body deliberately. The block bound only ever
    /// decides to skip documents that the shared pivot logic has already shown
    /// cannot reach the threshold, so the two cannot drift into ranking
    /// differently -- which is the property the tests pin.
    fn search_wand(
        &self,
        query: &SearchQuery,
        scorers: Vec<TermScorer>,
        top_k: usize,
        use_block_max: bool,
    ) -> (TopK, SearchStats) {
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

            if let (true, Some(value)) = (use_block_max, threshold) {
                if block_skip(&mut cursors, pivot_position, pivot_doc, value, &mut stats) {
                    continue;
                }
            }

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

    /// `MaxScore`'s essential-list traversal for disjunctive queries.
    ///
    /// The low-bound terms are deliberately excluded only from candidate
    /// generation.  Every candidate is still scored with every term, so the
    /// returned ranking is bit-identical to the exhaustive oracle.  For
    /// conjunctive queries the intersection walk in `search_exhaustive` is
    /// already the most direct exact executor; delegating there preserves the
    /// Boolean semantics instead of applying an OR-oriented optimization.
    fn search_max_score(
        &self,
        query: &SearchQuery,
        scorers: Vec<TermScorer>,
        top_k: usize,
    ) -> (TopK, SearchStats) {
        if query.operator() == BooleanOperator::And {
            return self.search_exhaustive(query, &scorers, top_k);
        }

        let mut cursors = scorers
            .into_iter()
            .map(|scorer| Cursor {
                scorer,
                position: 0,
            })
            .collect::<Vec<_>>();
        cursors.sort_by(|left, right| {
            left.scorer
                .upper_bound
                .total_cmp(&right.scorer.upper_bound)
                .then_with(|| left.scorer.ordinal.cmp(&right.scorer.ordinal))
        });

        let mut heap = TopK::new(top_k);
        let mut stats = SearchStats::default();

        loop {
            cursors.retain(|cursor| !cursor.exhausted());
            if cursors.is_empty() {
                break;
            }

            // The ascending prefix is non-essential when its *whole* upper
            // bound is strictly below the current threshold.  Strictness is
            // required: equal scores can still win the deterministic
            // lower-doc-id tie-break.
            let essential_start = if let Some(threshold) = heap.threshold() {
                let mut prefix_bound = 0.0;
                let mut cut = 0;
                while cut < cursors.len() {
                    let next = conservative_next_up(prefix_bound + cursors[cut].scorer.upper_bound);
                    if next < threshold {
                        prefix_bound = next;
                        cut += 1;
                    } else {
                        break;
                    }
                }
                if cut == cursors.len() {
                    // No unseen document can improve the heap.  If the total
                    // bound is exactly the threshold, retain all terms for
                    // tie correctness; otherwise the search is complete.
                    if prefix_bound < threshold {
                        break;
                    }
                    0
                } else {
                    cut
                }
            } else {
                0
            };

            let candidate_doc = cursors[essential_start..]
                .iter()
                .map(Cursor::current_doc)
                .min()
                .expect("essential list is non-empty");

            // Bring lower-bound cursors up to the candidate.  Documents that
            // occur only in the non-essential prefix are skipped here; their
            // total possible score is below the threshold by construction.
            for cursor in &mut cursors[..essential_start] {
                if cursor.current_doc() < candidate_doc {
                    let advanced = cursor.advance_to(candidate_doc);
                    stats.postings_advanced += advanced;
                    stats.postings_skipped += advanced.saturating_sub(1);
                }
            }

            stats.evaluated_candidates += 1;
            let mut pieces = cursors
                .iter()
                .filter_map(|cursor| {
                    score_at(&cursor.scorer.entries, candidate_doc)
                        .map(|score| (cursor.scorer.ordinal, score))
                })
                .collect::<Vec<_>>();
            pieces.sort_by_key(|(ordinal, _)| *ordinal);
            let score = pieces.into_iter().map(|(_, score)| score).sum();
            if self.matches_constraints(candidate_doc, query) {
                heap.consider(HeapEntry {
                    doc_id: candidate_doc,
                    score,
                });
            }

            for cursor in &mut cursors {
                if !cursor.exhausted() && cursor.current_doc() == candidate_doc {
                    stats.postings_advanced += cursor.advance_one();
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

fn block_metadata(scores: &[(InternalDocId, f64)]) -> BlockMaxMetadata {
    let default_bounds = scores
        .chunks(BLOCK_POSTINGS)
        .map(|block| {
            conservative_next_up(block.iter().map(|(_, score)| *score).fold(0.0, f64::max))
                .to_bits()
        })
        .collect();

    BlockMaxMetadata {
        posting_count: scores.len(),
        default_bounds,
    }
}

fn conservative_block_bounds(entries: &[ScoredPosting]) -> Vec<f64> {
    entries
        .chunks(BLOCK_POSTINGS)
        .map(|block| {
            conservative_next_up(block.iter().map(|entry| entry.score).fold(0.0, f64::max))
        })
        .collect()
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
    libm::log(1.0 + (documents - frequency + 0.5) / (frequency + 0.5))
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
        // The pure-Rust logarithm makes persisted default-score bounds
        // bit-stable across the Linux/Windows CI matrix.
        assert_eq!(bm25_idf(5, 5).to_bits(), 0x3fb6_4660_aa8c_e621);
        assert_eq!(bm25_idf(747, 183).to_bits(), 0x3ff6_7ba6_bce3_b827);
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
    fn maxscore_matches_exhaustive_across_queries_and_cutoffs() {
        let index = block_index(600);
        for text in [
            "common",
            "rare",
            "common mid",
            "common rare",
            "mid rare filler",
            "common mid rare filler padding",
            "absent",
            "common absent",
        ] {
            for top_k in [1, 3, 10, 50, 1_000] {
                let query = SearchQuery::from_text(index.analyzer(), text, None).unwrap();
                let context = format!("{text:?} top_k={top_k}");
                let exhaustive = search_with(&index, &query, PruningStrategy::Exhaustive, top_k);
                let maxscore = search_with(&index, &query, PruningStrategy::MaxScore, top_k);
                assert_same_ranking(&exhaustive, &maxscore, &context);
            }
        }
    }

    #[test]
    fn maxscore_keeps_filters_phrases_and_ties_exact() {
        let index = block_index(400);
        let base = SearchQuery::from_text(index.analyzer(), "common mid rare", None).unwrap();
        let filtered = base
            .clone()
            .with_filter(FieldFilter::exact("category", "even").unwrap());
        let phrased = base
            .clone()
            .with_phrase(PhraseFilter::from_text(index.analyzer(), "common common", None).unwrap());
        for (query, context) in [(base, "plain"), (filtered, "filtered"), (phrased, "phrase")] {
            let exhaustive = search_with(&index, &query, PruningStrategy::Exhaustive, 10);
            let maxscore = search_with(&index, &query, PruningStrategy::MaxScore, 10);
            assert_same_ranking(&exhaustive, &maxscore, context);
        }

        let mut builder = IndexBuilder::new(Analyzer::default());
        for id in ["first", "second", "third"] {
            builder
                .add_document(Document::from_fields(id, [("body", "same")]).unwrap())
                .unwrap();
        }
        let ties = builder.finish();
        let query = SearchQuery::from_text(ties.analyzer(), "same", Some("body")).unwrap();
        let exhaustive = search_with(&ties, &query, PruningStrategy::Exhaustive, 2);
        let maxscore = search_with(&ties, &query, PruningStrategy::MaxScore, 2);
        assert_same_ranking(&exhaustive, &maxscore, "ties");
    }

    #[test]
    fn maxscore_skips_documents_only_in_low_bound_terms() {
        let index = block_index(1_500);
        let query = SearchQuery::from_text(index.analyzer(), "common mid rare", None).unwrap();
        let exhaustive = search_with(&index, &query, PruningStrategy::Exhaustive, 10);
        let maxscore = search_with(&index, &query, PruningStrategy::MaxScore, 10);
        assert_same_ranking(&exhaustive, &maxscore, "maxscore pruning");
        assert!(
            maxscore.stats.evaluated_candidates <= exhaustive.stats.evaluated_candidates,
            "MaxScore evaluated more candidates than exhaustive: {} vs {}",
            maxscore.stats.evaluated_candidates,
            exhaustive.stats.evaluated_candidates
        );
        assert!(maxscore.stats.postings_skipped > 0);
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

    /// A corpus large enough that block maxima matter.
    ///
    /// Terms appear at very different rates and repeat at very different
    /// frequencies, so impacts vary widely inside a single posting list. That
    /// is the situation block-max pruning exists for: a plain `WAND` bound is
    /// set by one outlier and stays high for every other document the term
    /// touches.
    fn block_index(document_count: usize) -> InvertedIndex {
        let mut builder = IndexBuilder::new(Analyzer::default());
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for ordinal in 0..document_count {
            let mut body = String::new();
            // `common` is in almost every document, `mid` in about a third,
            // `rare` in a few, and each carries a different repeat count.
            let repeats = 1 + usize::try_from(next() % 9).unwrap();
            for _ in 0..repeats {
                body.push_str("common ");
            }
            if ordinal % 3 == 0 {
                let repeats = 1 + usize::try_from(next() % 5).unwrap();
                for _ in 0..repeats {
                    body.push_str("mid ");
                }
            }
            if ordinal % 37 == 0 {
                let repeats = 1 + usize::try_from(next() % 11).unwrap();
                for _ in 0..repeats {
                    body.push_str("rare ");
                }
            }
            if ordinal % 7 == 0 {
                body.push_str("filler ");
            }
            // Varying length changes the BM25 normalization, so two documents
            // with the same term frequency still score differently.
            for _ in 0..(next() % 13) {
                body.push_str("padding ");
            }
            let category = if ordinal % 2 == 0 { "even" } else { "odd" };
            builder
                .add_document(
                    Document::from_fields(
                        format!("d{ordinal}"),
                        [
                            ("title", format!("document {ordinal}")),
                            ("body", body),
                            ("category", category.to_owned()),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        builder.finish()
    }

    fn search_with(
        index: &InvertedIndex,
        query: &SearchQuery,
        strategy: PruningStrategy,
        top_k: usize,
    ) -> SearchOutcome {
        index
            .search(
                query,
                SearchOptions {
                    top_k,
                    pruning: strategy,
                    explain: false,
                    bm25: Bm25Params::default(),
                },
            )
            .unwrap()
    }

    fn assert_same_ranking(left: &SearchOutcome, right: &SearchOutcome, context: &str) {
        assert_eq!(
            left.hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
            right.hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
            "document order differs for {context}"
        );
        for (a, b) in left.hits.iter().zip(&right.hits) {
            assert_eq!(
                a.score.to_bits(),
                b.score.to_bits(),
                "score bits differ for {context}: {} vs {}",
                a.score,
                b.score
            );
        }
    }

    #[test]
    fn block_max_wand_matches_exhaustive_across_queries_and_cutoffs() {
        let index = block_index(600);
        for text in [
            "common",
            "rare",
            "common mid",
            "common rare",
            "mid rare filler",
            "common mid rare filler padding",
            "absent",
            "common absent",
        ] {
            for operator in [BooleanOperator::Or, BooleanOperator::And] {
                for top_k in [1, 3, 10, 50] {
                    let query = SearchQuery::from_text(index.analyzer(), text, None)
                        .unwrap()
                        .with_operator(operator);
                    let context = format!("{text:?} {operator:?} top_k={top_k}");
                    let exhaustive =
                        search_with(&index, &query, PruningStrategy::Exhaustive, top_k);
                    let wand = search_with(&index, &query, PruningStrategy::Wand, top_k);
                    let block = search_with(&index, &query, PruningStrategy::BlockMaxWand, top_k);
                    assert_same_ranking(&exhaustive, &wand, &context);
                    assert_same_ranking(&exhaustive, &block, &context);
                }
            }
        }
    }

    #[test]
    fn block_max_wand_stays_exact_with_filters_and_phrases() {
        let index = block_index(400);
        let base = SearchQuery::from_text(index.analyzer(), "common mid rare", None).unwrap();
        let filtered = base
            .clone()
            .with_filter(FieldFilter::exact("category", "even").unwrap());
        let phrased = base
            .clone()
            .with_phrase(PhraseFilter::from_text(index.analyzer(), "common common", None).unwrap());
        for (query, context) in [(base, "plain"), (filtered, "filtered"), (phrased, "phrase")] {
            let exhaustive = search_with(&index, &query, PruningStrategy::Exhaustive, 10);
            let block = search_with(&index, &query, PruningStrategy::BlockMaxWand, 10);
            assert_same_ranking(&exhaustive, &block, context);
        }
    }

    #[test]
    fn block_max_wand_skips_more_than_plain_wand() {
        // The point of the strategy. A term whose impacts vary widely has long
        // runs that a global bound cannot skip but a block bound can.
        let index = block_index(1_500);
        let query = SearchQuery::from_text(index.analyzer(), "common mid rare", None).unwrap();
        let wand = search_with(&index, &query, PruningStrategy::Wand, 10);
        let block = search_with(&index, &query, PruningStrategy::BlockMaxWand, 10);

        assert_same_ranking(&wand, &block, "pruning comparison");
        assert!(block.stats.block_max_bounds_loaded > 0);
        assert!(
            block.stats.evaluated_candidates <= wand.stats.evaluated_candidates,
            "block-max scored more candidates than plain WAND: {} vs {}",
            block.stats.evaluated_candidates,
            wand.stats.evaluated_candidates
        );
        assert!(
            block.stats.postings_skipped >= wand.stats.postings_skipped,
            "block-max skipped fewer postings than plain WAND: {} vs {}",
            block.stats.postings_skipped,
            wand.stats.postings_skipped
        );
    }

    #[test]
    fn block_max_wand_terminates_when_every_document_matches() {
        // Every document carries `common`, so the pivot repeatedly lands on the
        // first cursor. A skip rule that failed to step past the pivot would
        // spin here rather than fail.
        let index = block_index(300);
        let query = SearchQuery::from_text(index.analyzer(), "common", None).unwrap();
        let block = search_with(&index, &query, PruningStrategy::BlockMaxWand, 5);
        assert_eq!(block.hits.len(), 5);
    }

    #[test]
    fn block_max_wand_handles_a_cutoff_larger_than_the_corpus() {
        let index = block_index(20);
        let query = SearchQuery::from_text(index.analyzer(), "common rare", None).unwrap();
        let exhaustive = search_with(&index, &query, PruningStrategy::Exhaustive, 500);
        let block = search_with(&index, &query, PruningStrategy::BlockMaxWand, 500);
        assert_same_ranking(&exhaustive, &block, "cutoff beyond corpus");
    }

    #[test]
    fn block_maxima_bound_every_posting_in_their_block() {
        // The invariant the whole strategy rests on. If a block maximum were
        // ever below a score inside its block, pruning could drop a document
        // that belonged in the top-k, and the ranking tests above would only
        // notice when a query happened to hit it.
        let index = block_index(500);
        let term = PreparedTerm {
            normalized: "common".to_owned(),
            field: None,
            boost: 1.0,
        };
        let scorer = index
            .build_term_scorer(0, &term, Bm25Params::default(), true)
            .unwrap();
        assert!(scorer.entries.len() > BLOCK_POSTINGS, "corpus too small");
        assert_eq!(
            scorer.block_max.len(),
            scorer.entries.len().div_ceil(BLOCK_POSTINGS)
        );
        let stored = index
            .block_max
            .get(&BlockMaxKey {
                field: None,
                term: "common".to_owned(),
            })
            .unwrap();
        assert_eq!(
            scorer
                .block_max
                .iter()
                .map(|bound| bound.to_bits())
                .collect::<Vec<_>>(),
            stored.default_bounds
        );
        for (block, chunk) in scorer.entries.chunks(BLOCK_POSTINGS).enumerate() {
            for entry in chunk {
                assert!(
                    scorer.block_max[block] >= entry.score,
                    "block {block} maximum {} is below a posting score {}",
                    scorer.block_max[block],
                    entry.score
                );
            }
        }
        for bound in &scorer.block_max {
            assert!(
                *bound <= scorer.upper_bound,
                "a block maximum exceeds the global bound"
            );
        }
    }

    #[test]
    fn exact_custom_block_bounds_cover_extreme_bm25_and_boosts() {
        let index = block_index(500);
        for params in [
            Bm25Params {
                k1: f64::MIN_POSITIVE,
                b: f64::from_bits(1),
            },
            Bm25Params { k1: 0.01, b: 0.0 },
            Bm25Params { k1: 0.5, b: 1.0 },
            Bm25Params { k1: 20.0, b: 0.4 },
            Bm25Params { k1: 1.2, b: 0.75 },
            Bm25Params {
                k1: 3.408_216_882_372_321e265,
                b: f64::from_bits(1.0_f64.to_bits() - 1),
            },
        ] {
            for boost in [f64::MIN_POSITIVE, 2.863_342_160_935_730_6e-69, 1.5, 10.0] {
                let term = PreparedTerm {
                    normalized: "common".to_owned(),
                    field: None,
                    boost,
                };
                let scorer = index.build_term_scorer(0, &term, params, true).unwrap();
                assert_eq!(scorer.precomputed_block_bounds_loaded, 0);
                assert_eq!(
                    scorer.postings_scanned_for_block_bounds,
                    scorer.entries.len()
                );
                for (block, chunk) in scorer.entries.chunks(BLOCK_POSTINGS).enumerate() {
                    for entry in chunk {
                        assert!(
                            scorer.block_max[block] >= entry.score,
                            "custom bound {} is below {} for {params:?}, boost={boost}",
                            scorer.block_max[block],
                            entry.score
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn extreme_custom_parameters_remain_bit_exact_against_exhaustive() {
        let index = block_index(500);
        let query = SearchQuery::from_terms(vec![
            QueryTerm::new("common", None, 2.863_342_160_935_730_6e-69).unwrap(),
            QueryTerm::new("mid", None, 1.603_021_237_095_590_6e-70).unwrap(),
        ])
        .unwrap();
        let params = Bm25Params {
            k1: 3.408_216_882_372_321e265,
            b: f64::from_bits(1.0_f64.to_bits() - 1),
        };
        let options = |pruning| SearchOptions {
            top_k: 10,
            pruning,
            explain: false,
            bm25: params,
        };

        let exhaustive = index
            .search(&query, options(PruningStrategy::Exhaustive))
            .unwrap();
        let block = index
            .search(&query, options(PruningStrategy::BlockMaxWand))
            .unwrap();

        assert_same_ranking(
            &exhaustive,
            &block,
            "extreme custom floating-point parameters",
        );
        assert_eq!(block.stats.block_max_bounds_loaded, 0);
        assert!(block.stats.block_max_postings_scanned > 0);
    }

    #[test]
    fn default_block_bounds_are_loaded_without_scanning_scores() {
        let index = block_index(200);
        let scorer = index
            .build_term_scorer(
                0,
                &PreparedTerm {
                    normalized: "common".to_owned(),
                    field: None,
                    boost: 1.0,
                },
                Bm25Params::default(),
                true,
            )
            .unwrap();

        assert_eq!(
            scorer.precomputed_block_bounds_loaded,
            scorer.block_max.len()
        );
        assert_eq!(scorer.postings_scanned_for_block_bounds, 0);
    }

    #[test]
    fn scorer_reads_the_resident_block_table_instead_of_recomputing_it() {
        let mut index = block_index(200);
        let key = BlockMaxKey {
            field: None,
            term: "common".to_owned(),
        };
        let metadata = index.block_max.get_mut(&key).unwrap();
        let altered = conservative_next_up(f64::from_bits(metadata.default_bounds[0]) + 1.0);
        metadata.default_bounds[0] = altered.to_bits();

        let scorer = index
            .build_term_scorer(
                0,
                &PreparedTerm {
                    normalized: "common".to_owned(),
                    field: None,
                    boost: 1.0,
                },
                Bm25Params::default(),
                true,
            )
            .unwrap();
        assert_eq!(scorer.block_max[0].to_bits(), altered.to_bits());
    }

    #[test]
    #[ignore = "measurement, not an assertion"]
    #[allow(clippy::cast_precision_loss)]
    fn report_block_max_pruning_effect() {
        for documents in [1_000, 5_000, 20_000] {
            let index = block_index(documents);
            for text in ["common mid rare", "common mid", "common"] {
                for top_k in [10, 100] {
                    let query = SearchQuery::from_text(index.analyzer(), text, None).unwrap();
                    let w = search_with(&index, &query, PruningStrategy::Wand, top_k);
                    let b = search_with(&index, &query, PruningStrategy::BlockMaxWand, top_k);
                    let e = search_with(&index, &query, PruningStrategy::Exhaustive, top_k);
                    assert_same_ranking(&e, &b, "measurement");
                    let reduction = 100.0
                        - 100.0 * b.stats.evaluated_candidates as f64
                            / w.stats.evaluated_candidates.max(1) as f64;
                    println!(
                        "docs={documents:>6} q={text:<16} k={top_k:>3}  scored: exhaustive={:>6} wand={:>6} bmw={:>6}  ({reduction:>5.1}% fewer than wand)",
                        e.stats.evaluated_candidates,
                        w.stats.evaluated_candidates,
                        b.stats.evaluated_candidates
                    );
                }
            }
        }
    }
}
