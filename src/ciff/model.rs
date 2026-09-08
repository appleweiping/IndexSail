use std::collections::{BTreeMap, BTreeSet};

use crate::analysis::Analyzer;
use crate::error::{Error, Result};
use crate::evaluation::RetrievalBackend;
use crate::index::InvertedIndex;
use crate::query::{BooleanOperator, SearchQuery};
use crate::search::{
    Bm25Params, PruningStrategy, SearchHit, SearchOptions, SearchOutcome, SearchStats, bm25_idf,
};

pub const CIFF_FORMAT_VERSION: u32 = 1;

/// Resource policy applied before any CIFF collection-sized allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CiffLimits {
    pub max_frame_bytes: usize,
    pub max_posting_lists: usize,
    pub max_documents: usize,
    pub max_postings: usize,
    pub max_term_bytes: usize,
    pub max_external_id_bytes: usize,
    pub max_description_bytes: usize,
}

impl Default for CiffLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 256 * 1024 * 1024,
            max_posting_lists: 10_000_000,
            max_documents: 100_000_000,
            max_postings: 100_000_000,
            max_term_bytes: 1024 * 1024,
            max_external_id_bytes: 1024 * 1024,
            max_description_bytes: 64 * 1024,
        }
    }
}

impl CiffLimits {
    pub(crate) fn validate(self) -> Result<Self> {
        for (label, value) in [
            ("max_frame_bytes", self.max_frame_bytes),
            ("max_posting_lists", self.max_posting_lists),
            ("max_documents", self.max_documents),
            ("max_postings", self.max_postings),
            ("max_term_bytes", self.max_term_bytes),
            ("max_external_id_bytes", self.max_external_id_bytes),
            ("max_description_bytes", self.max_description_bytes),
        ] {
            if value == 0 {
                return Err(Error::InvalidArgument(format!(
                    "CIFF {label} must be greater than zero"
                )));
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CiffHeader {
    pub version: u32,
    pub num_posting_lists: u32,
    pub num_documents: u32,
    pub total_posting_lists: u32,
    pub total_documents: u32,
    pub total_terms_in_collection: u64,
    pub average_document_length: f64,
    pub description: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CiffPosting {
    /// Absolute (decoded) CIFF document identifier.
    pub document_id: u32,
    /// The non-negative CIFF `tf` payload.
    ///
    /// Frequency indexes store a term count. Some learned-sparse producers
    /// store a quantized impact instead; loading and round-tripping preserve
    /// that value, while `IndexSail` BM25 search requires frequency semantics.
    pub term_frequency: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiffPostingList {
    pub term: String,
    pub document_frequency: u64,
    pub collection_frequency: u64,
    pub postings: Vec<CiffPosting>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CiffDocumentRecord {
    pub document_id: u32,
    pub external_id: String,
    pub document_length: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CiffStats {
    pub contained_posting_lists: usize,
    pub total_posting_lists: u32,
    pub contained_documents: usize,
    pub total_documents: u32,
    pub postings: usize,
    pub total_terms_in_collection: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CiffSearchOptions {
    pub top_k: usize,
    pub operator: BooleanOperator,
    pub bm25: Bm25Params,
}

impl Default for CiffSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            operator: BooleanOperator::Or,
            bm25: Bm25Params::default(),
        }
    }
}

impl CiffSearchOptions {
    fn validate(self) -> Result<Self> {
        if self.top_k == 0 {
            return Err(Error::InvalidArgument(
                "CIFF top_k must be greater than zero".into(),
            ));
        }
        self.bm25.validate()?;
        Ok(self)
    }
}

/// An interoperable bag-of-words index represented by CIFF v1 structures.
///
/// CIFF contains posting payloads and approximate document lengths, but no
/// source text, fields, or positions. It therefore has a deliberately separate
/// search surface from [`InvertedIndex`]. BM25 and Boolean AND/OR apply when
/// `tf` has frequency semantics; learned-impact payloads remain interoperable
/// for loading, inspection, and round-tripping but are not BM25 frequencies.
/// Phrases and stored-field filters remain native-index features.
#[derive(Clone, Debug, PartialEq)]
pub struct CiffIndex {
    header: CiffHeader,
    posting_lists: Vec<CiffPostingList>,
    documents: Vec<CiffDocumentRecord>,
}

/// Adapter that supplies the query analyzer CIFF itself intentionally does not
/// prescribe, allowing the shared TREC batch evaluator to consume a CIFF file.
#[derive(Clone, Copy, Debug)]
pub struct CiffRetrieval<'a> {
    index: &'a CiffIndex,
    analyzer: Analyzer,
}

impl CiffIndex {
    pub fn from_parts(
        header: CiffHeader,
        posting_lists: Vec<CiffPostingList>,
        documents: Vec<CiffDocumentRecord>,
        limits: CiffLimits,
    ) -> Result<Self> {
        let limits = limits.validate()?;
        let mut index = Self {
            header,
            posting_lists,
            documents,
        };
        index
            .posting_lists
            .sort_by(|left, right| left.term.cmp(&right.term));
        index.documents.sort_by_key(|document| document.document_id);
        index.validate(limits)?;
        Ok(index)
    }

    pub const fn header(&self) -> &CiffHeader {
        &self.header
    }

    pub fn posting_lists(&self) -> &[CiffPostingList] {
        &self.posting_lists
    }

    pub fn documents(&self) -> &[CiffDocumentRecord] {
        &self.documents
    }

    pub fn posting_list(&self, term: &str) -> Option<&CiffPostingList> {
        self.posting_lists
            .binary_search_by(|candidate| candidate.term.as_str().cmp(term))
            .ok()
            .map(|index| &self.posting_lists[index])
    }

    pub fn document(&self, document_id: u32) -> Option<&CiffDocumentRecord> {
        self.documents
            .binary_search_by_key(&document_id, |document| document.document_id)
            .ok()
            .map(|index| &self.documents[index])
    }

    pub fn stats(&self) -> CiffStats {
        CiffStats {
            contained_posting_lists: self.posting_lists.len(),
            total_posting_lists: self.header.total_posting_lists,
            contained_documents: self.documents.len(),
            total_documents: self.header.total_documents,
            postings: self
                .posting_lists
                .iter()
                .map(|list| list.postings.len())
                .sum(),
            total_terms_in_collection: self.header.total_terms_in_collection,
        }
    }

    pub const fn retrieval(&self, analyzer: Analyzer) -> CiffRetrieval<'_> {
        CiffRetrieval {
            index: self,
            analyzer,
        }
    }

    /// Flatten every native field into a CIFF bag-of-words collection.
    ///
    /// Term frequencies for the same normalized term in multiple fields are
    /// summed. Document length is the sum of native field lengths. This loses
    /// field and positional information by design, which is required by CIFF's
    /// single-stream schema and is stated explicitly in the generated
    /// description.
    pub fn from_native(index: &InvertedIndex, description: impl Into<String>) -> Result<Self> {
        let posting_lists = flatten_native_postings(index)?;
        let documents = native_document_records(index)?;
        let total_terms = documents.iter().try_fold(0_u64, |total, document| {
            total
                .checked_add(u64::from(document.document_length))
                .ok_or_else(|| Error::InvalidArgument("CIFF total term count overflows".into()))
        })?;
        let contained_lists = checked_i32_count(posting_lists.len(), "posting-list")?;
        let contained_documents = checked_i32_count(documents.len(), "document")?;
        let average = if documents.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                total_terms as f64 / documents.len() as f64
            }
        };
        let supplied = description.into();
        let description = format!(
            "{supplied}; IndexSail flattened all native fields using {:?} analysis; field and position data are not present in CIFF",
            index.analyzer().mode()
        );
        Self::from_parts(
            CiffHeader {
                version: CIFF_FORMAT_VERSION,
                num_posting_lists: contained_lists,
                num_documents: contained_documents,
                total_posting_lists: contained_lists,
                total_documents: contained_documents,
                total_terms_in_collection: total_terms,
                average_document_length: average,
                description,
            },
            posting_lists,
            documents,
            CiffLimits::default(),
        )
    }

    pub fn search(
        &self,
        analyzer: Analyzer,
        query: &str,
        options: CiffSearchOptions,
    ) -> Result<SearchOutcome> {
        let mut terms = BTreeMap::<String, f64>::new();
        for token in analyzer.analyze(query) {
            let weight = terms.entry(token.text).or_default();
            *weight += 1.0;
            if !weight.is_finite() {
                return Err(Error::InvalidQuery(
                    "combined CIFF query-term weight exceeds finite range".into(),
                ));
            }
        }
        self.search_terms(&terms, options)
    }

    fn search_terms(
        &self,
        terms: &BTreeMap<String, f64>,
        options: CiffSearchOptions,
    ) -> Result<SearchOutcome> {
        let options = options.validate()?;
        if terms.is_empty() {
            return Err(Error::InvalidQuery(
                "CIFF query must contain at least one searchable token".into(),
            ));
        }
        if options.operator == BooleanOperator::And
            && terms.keys().any(|term| self.posting_list(term).is_none())
        {
            return Ok(SearchOutcome {
                hits: Vec::new(),
                stats: SearchStats::default(),
            });
        }

        let mut scores = BTreeMap::<u32, (f64, usize)>::new();
        let mut postings_visited = 0_usize;
        for (term, weight) in terms {
            let Some(list) = self.posting_list(term) else {
                if options.operator == BooleanOperator::And {
                    return Ok(SearchOutcome {
                        hits: Vec::new(),
                        stats: SearchStats::default(),
                    });
                }
                continue;
            };
            let document_frequency = usize::try_from(list.document_frequency)
                .map_err(|_| Error::CorruptIndex("CIFF document frequency exceeds usize".into()))?;
            for posting in &list.postings {
                postings_visited = postings_visited.checked_add(1).ok_or_else(|| {
                    Error::InvalidArgument("CIFF search posting counter overflow".into())
                })?;
                let document = self.document(posting.document_id).ok_or_else(|| {
                    Error::CorruptIndex(format!(
                        "CIFF posting references missing document {}",
                        posting.document_id
                    ))
                })?;
                let contribution = bm25(
                    posting.term_frequency,
                    document_frequency,
                    self.header.total_documents,
                    document.document_length,
                    self.header.average_document_length,
                    options.bm25,
                ) * weight;
                if !contribution.is_finite() || contribution.is_sign_negative() {
                    return Err(Error::InvalidQuery(
                        "CIFF BM25 parameters produced a non-finite score".into(),
                    ));
                }
                let entry = scores.entry(posting.document_id).or_default();
                entry.0 += contribution;
                if !entry.0.is_finite() {
                    return Err(Error::InvalidQuery(
                        "CIFF accumulated BM25 score is not finite".into(),
                    ));
                }
                entry.1 += 1;
            }
        }

        if options.operator == BooleanOperator::And {
            scores.retain(|_, (_, matches)| *matches == terms.len());
        }
        let evaluated = scores.len();
        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|(left_id, (left_score, _)), (right_id, (right_score, _))| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_id.cmp(right_id))
        });
        ranked.truncate(options.top_k.min(ranked.len()));
        let hits = ranked
            .into_iter()
            .map(|(document_id, (score, _))| {
                let document = self
                    .document(document_id)
                    .expect("validated CIFF candidate references a document");
                SearchHit {
                    doc_id: document_id,
                    external_id: document.external_id.clone(),
                    score,
                    explanation: None,
                }
            })
            .collect();
        Ok(SearchOutcome {
            hits,
            stats: SearchStats {
                evaluated_candidates: evaluated,
                postings_advanced: postings_visited,
                ..SearchStats::default()
            },
        })
    }

    pub(crate) fn validate(&self, limits: CiffLimits) -> Result<()> {
        validate_header(&self.header, limits)?;
        if usize::try_from(self.header.num_posting_lists).ok() != Some(self.posting_lists.len()) {
            return Err(corrupt(
                "header num_postings_lists does not match the stream",
            ));
        }
        if usize::try_from(self.header.num_documents).ok() != Some(self.documents.len()) {
            return Err(corrupt("header num_docs does not match the stream"));
        }
        let referenced_documents =
            validate_posting_lists(&self.posting_lists, self.header.total_documents, limits)?;
        validate_documents(
            &self.documents,
            self.header.total_documents,
            limits,
            &referenced_documents,
        )?;
        Ok(())
    }
}

impl RetrievalBackend for CiffRetrieval<'_> {
    fn analyzer(&self) -> Analyzer {
        self.analyzer
    }

    fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
        if options.pruning != PruningStrategy::Exhaustive {
            return Err(Error::InvalidArgument(
                "CIFF retrieval currently supports the exhaustive strategy only".into(),
            ));
        }
        if options.explain {
            return Err(Error::InvalidArgument(
                "CIFF batch retrieval does not expose native field explanations".into(),
            ));
        }
        if !query.phrases().is_empty()
            || !query.filters().is_empty()
            || query.terms().iter().any(|term| term.field().is_some())
        {
            return Err(Error::InvalidQuery(
                "CIFF has no fields, positions, or stored filters".into(),
            ));
        }
        let mut terms = BTreeMap::<String, f64>::new();
        for term in query.terms() {
            let normalized = self.analyzer.normalize_single(term.text()).ok_or_else(|| {
                Error::InvalidQuery(format!(
                    "query term '{}' must analyze to exactly one token",
                    term.text()
                ))
            })?;
            let weight = terms.entry(normalized).or_default();
            *weight += term.boost();
            if !weight.is_finite() {
                return Err(Error::InvalidQuery(
                    "combined CIFF query-term weight exceeds finite range".into(),
                ));
            }
        }
        self.index.search_terms(
            &terms,
            CiffSearchOptions {
                top_k: options.top_k,
                operator: query.operator(),
                bm25: options.bm25,
            },
        )
    }
}

fn flatten_native_postings(index: &InvertedIndex) -> Result<Vec<CiffPostingList>> {
    let mut by_term = BTreeMap::<String, BTreeMap<u32, u32>>::new();
    for (key, postings) in &index.postings {
        let destinations = by_term.entry(key.term.clone()).or_default();
        for posting in postings {
            let frequency = destinations.entry(posting.doc_id).or_default();
            *frequency = frequency
                .checked_add(posting.term_frequency)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "combined CIFF frequency overflows for term '{}' and document {}",
                        key.term, posting.doc_id
                    ))
                })?;
        }
    }

    by_term
        .into_iter()
        .map(|(term, by_document)| {
            let mut collection_frequency = 0_u64;
            let postings = by_document
                .into_iter()
                .map(|(document_id, term_frequency)| {
                    collection_frequency = collection_frequency
                        .checked_add(u64::from(term_frequency))
                        .ok_or_else(|| {
                            Error::InvalidArgument(format!(
                                "CIFF collection frequency overflows for term '{term}'"
                            ))
                        })?;
                    Ok(CiffPosting {
                        document_id,
                        term_frequency,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(CiffPostingList {
                term,
                document_frequency: u64::try_from(postings.len())
                    .map_err(|_| Error::InvalidArgument("CIFF posting count exceeds u64".into()))?,
                collection_frequency,
                postings,
            })
        })
        .collect()
}

fn native_document_records(index: &InvertedIndex) -> Result<Vec<CiffDocumentRecord>> {
    index
        .documents
        .iter()
        .enumerate()
        .map(|(position, document)| {
            let document_id = u32::try_from(position).map_err(|_| {
                Error::InvalidArgument("native document id exceeds CIFF int32".into())
            })?;
            let document_length = index.field_lengths[position]
                .values()
                .try_fold(0_u32, |total, length| total.checked_add(*length))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "combined CIFF document length overflows for '{}'",
                        document.external_id()
                    ))
                })?;
            Ok(CiffDocumentRecord {
                document_id,
                external_id: document.external_id().to_owned(),
                document_length,
            })
        })
        .collect()
}

fn validate_posting_lists(
    lists: &[CiffPostingList],
    total_documents: u32,
    limits: CiffLimits,
) -> Result<BTreeSet<u32>> {
    let mut terms = BTreeSet::new();
    let mut referenced_documents = BTreeSet::new();
    let mut total_postings = 0_usize;
    for list in lists {
        if !terms.insert(list.term.as_str()) {
            return Err(corrupt(format!("duplicate postings term '{}'", list.term)));
        }
        total_postings = total_postings
            .checked_add(list.postings.len())
            .ok_or_else(|| corrupt("total posting count overflows usize"))?;
        if total_postings > limits.max_postings {
            return Err(corrupt(format!(
                "postings exceed the configured {}-entry limit",
                limits.max_postings
            )));
        }
        validate_posting_list_local(list, total_documents, limits)?;
        referenced_documents.extend(list.postings.iter().map(|posting| posting.document_id));
    }
    Ok(referenced_documents)
}

pub(super) fn validate_posting_list_local(
    list: &CiffPostingList,
    total_documents: u32,
    limits: CiffLimits,
) -> Result<()> {
    validate_term(&list.term, limits.max_term_bytes)?;
    if usize::try_from(list.document_frequency).ok() != Some(list.postings.len()) {
        return Err(corrupt(format!(
            "term '{}' df does not match its posting count",
            list.term
        )));
    }
    if list.document_frequency > u64::from(total_documents) {
        return Err(corrupt(format!(
            "term '{}' df exceeds total_docs",
            list.term
        )));
    }
    if i64::try_from(list.collection_frequency).is_err() {
        return Err(corrupt(format!(
            "term '{}' cf exceeds non-negative int64",
            list.term
        )));
    }
    if list.postings.is_empty() {
        return Err(corrupt(format!(
            "term '{}' must contain at least one posting",
            list.term
        )));
    }
    let mut previous = None;
    let mut collection_frequency = 0_u64;
    for posting in &list.postings {
        if i32::try_from(posting.document_id).is_err() {
            return Err(corrupt("posting document id exceeds non-negative int32"));
        }
        if posting.document_id >= total_documents {
            return Err(corrupt(format!(
                "term '{}' references document {} outside total_docs {}",
                list.term, posting.document_id, total_documents
            )));
        }
        if posting.term_frequency == 0 || i32::try_from(posting.term_frequency).is_err() {
            return Err(corrupt(format!(
                "term '{}' has an invalid term frequency",
                list.term
            )));
        }
        if previous.is_some_and(|value| posting.document_id <= value) {
            return Err(corrupt(format!(
                "term '{}' posting document ids are not strictly increasing",
                list.term
            )));
        }
        previous = Some(posting.document_id);
        collection_frequency = collection_frequency
            .checked_add(u64::from(posting.term_frequency))
            .ok_or_else(|| corrupt("collection frequency overflows u64"))?;
    }
    if collection_frequency != list.collection_frequency {
        return Err(corrupt(format!(
            "term '{}' cf does not match summed term frequencies",
            list.term
        )));
    }
    Ok(())
}

fn validate_documents(
    documents: &[CiffDocumentRecord],
    total_documents: u32,
    limits: CiffLimits,
    referenced_documents: &BTreeSet<u32>,
) -> Result<()> {
    let mut document_ids = BTreeSet::new();
    let mut external_ids = BTreeSet::new();
    for document in documents {
        validate_document_local(document, total_documents, limits)?;
        if !document_ids.insert(document.document_id) {
            return Err(corrupt(format!(
                "duplicate document id {}",
                document.document_id
            )));
        }
        if !external_ids.insert(document.external_id.as_str()) {
            return Err(corrupt(format!(
                "duplicate collection document id '{}'",
                document.external_id
            )));
        }
    }
    if let Some(missing) = referenced_documents
        .iter()
        .find(|document_id| !document_ids.contains(document_id))
    {
        return Err(corrupt(format!(
            "posting references document {missing} without a DocRecord"
        )));
    }
    Ok(())
}

pub(super) fn validate_document_local(
    document: &CiffDocumentRecord,
    total_documents: u32,
    limits: CiffLimits,
) -> Result<()> {
    if i32::try_from(document.document_id).is_err() {
        return Err(corrupt("document id exceeds non-negative int32"));
    }
    validate_external_id(&document.external_id, limits.max_external_id_bytes)?;
    if i32::try_from(document.document_length).is_err() {
        return Err(corrupt(format!(
            "document '{}' length exceeds non-negative int32",
            document.external_id
        )));
    }
    if document.document_id >= total_documents {
        return Err(corrupt(format!(
            "document id {} is outside total_docs {}",
            document.document_id, total_documents
        )));
    }
    Ok(())
}

pub(super) fn validate_header(header: &CiffHeader, limits: CiffLimits) -> Result<()> {
    if header.version != CIFF_FORMAT_VERSION {
        return Err(Error::UnsupportedVersion(header.version));
    }
    if header.num_posting_lists > header.total_posting_lists {
        return Err(corrupt("num_postings_lists exceeds total_postings_lists"));
    }
    if header.num_documents > header.total_documents {
        return Err(corrupt("num_docs exceeds total_docs"));
    }
    for (label, value) in [
        ("num_postings_lists", header.num_posting_lists),
        ("num_docs", header.num_documents),
        ("total_postings_lists", header.total_posting_lists),
        ("total_docs", header.total_documents),
    ] {
        if i32::try_from(value).is_err() {
            return Err(corrupt(format!("{label} exceeds non-negative int32")));
        }
    }
    if usize::try_from(header.num_posting_lists)
        .ok()
        .is_none_or(|value| value > limits.max_posting_lists)
    {
        return Err(corrupt(format!(
            "num_postings_lists exceeds the configured {}-entry limit",
            limits.max_posting_lists
        )));
    }
    if usize::try_from(header.num_documents)
        .ok()
        .is_none_or(|value| value > limits.max_documents)
    {
        return Err(corrupt(format!(
            "num_docs exceeds the configured {}-entry limit",
            limits.max_documents
        )));
    }
    if i64::try_from(header.total_terms_in_collection).is_err() {
        return Err(corrupt(
            "total_terms_in_collection exceeds non-negative int64",
        ));
    }
    if !header.average_document_length.is_finite()
        || header.average_document_length.is_sign_negative()
    {
        return Err(corrupt("average_doclength must be finite and non-negative"));
    }
    if header.total_documents == 0
        && (header.total_terms_in_collection != 0 || header.average_document_length != 0.0)
    {
        return Err(corrupt(
            "an empty CIFF collection must have zero total terms and average length",
        ));
    }
    if (header.total_terms_in_collection == 0) != (header.average_document_length == 0.0) {
        return Err(corrupt(
            "total_terms_in_collection and average_doclength must both be zero or both be positive",
        ));
    }
    if header.description.len() > limits.max_description_bytes {
        return Err(corrupt(format!(
            "description exceeds the configured {}-byte limit",
            limits.max_description_bytes
        )));
    }
    if header.description.contains('\0') {
        return Err(corrupt("description contains a NUL character"));
    }
    Ok(())
}

fn validate_term(term: &str, max_bytes: usize) -> Result<()> {
    if term.is_empty() {
        return Err(corrupt("postings term must not be empty"));
    }
    if term.len() > max_bytes {
        return Err(corrupt(format!(
            "postings term exceeds the configured {max_bytes}-byte limit"
        )));
    }
    if term.chars().any(char::is_control) {
        return Err(corrupt("postings term contains a control character"));
    }
    Ok(())
}

fn validate_external_id(external_id: &str, max_bytes: usize) -> Result<()> {
    if external_id.is_empty() {
        return Err(corrupt("collection_docid must not be empty"));
    }
    if external_id.len() > max_bytes {
        return Err(corrupt(format!(
            "collection_docid exceeds the configured {max_bytes}-byte limit"
        )));
    }
    if external_id.chars().any(char::is_control) {
        return Err(corrupt("collection_docid contains a control character"));
    }
    Ok(())
}

fn checked_i32_count(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|value| i32::try_from(*value).is_ok())
        .ok_or_else(|| Error::InvalidArgument(format!("CIFF {label} count exceeds int32")))
}

#[allow(clippy::cast_precision_loss)]
fn bm25(
    term_frequency: u32,
    document_frequency: usize,
    document_count: u32,
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
    bm25_idf(
        usize::try_from(document_count).expect("u32 fits usize on supported targets"),
        document_frequency,
    ) * frequency
        * (params.k1 + 1.0)
        / denominator
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::CorruptIndex(format!("CIFF: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;
    use crate::evaluation::{BatchConfig, RetrievalBackend, evaluate_batch};
    use crate::index::IndexBuilder;
    use crate::query::QueryTerm;
    use crate::trec::Topic;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_path() -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let number = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "indexsail-ciff-model-{}-{number}.ciff",
            std::process::id()
        ))
    }

    fn fixture() -> CiffIndex {
        CiffIndex::from_parts(
            CiffHeader {
                version: 1,
                num_posting_lists: 2,
                num_documents: 3,
                total_posting_lists: 2,
                total_documents: 3,
                total_terms_in_collection: 6,
                average_document_length: 2.0,
                description: "fixture".into(),
            },
            vec![
                CiffPostingList {
                    term: "apple".into(),
                    document_frequency: 2,
                    collection_frequency: 3,
                    postings: vec![
                        CiffPosting {
                            document_id: 0,
                            term_frequency: 2,
                        },
                        CiffPosting {
                            document_id: 2,
                            term_frequency: 1,
                        },
                    ],
                },
                CiffPostingList {
                    term: "banana".into(),
                    document_frequency: 1,
                    collection_frequency: 1,
                    postings: vec![CiffPosting {
                        document_id: 1,
                        term_frequency: 1,
                    }],
                },
            ],
            vec![
                CiffDocumentRecord {
                    document_id: 0,
                    external_id: "A".into(),
                    document_length: 2,
                },
                CiffDocumentRecord {
                    document_id: 1,
                    external_id: "B".into(),
                    document_length: 1,
                },
                CiffDocumentRecord {
                    document_id: 2,
                    external_id: "C".into(),
                    document_length: 3,
                },
            ],
            CiffLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn validates_and_indexes_sorted_parts() {
        let index = fixture();
        assert_eq!(index.stats().postings, 3);
        assert_eq!(index.posting_list("apple").unwrap().postings.len(), 2);
        assert_eq!(index.document(2).unwrap().external_id, "C");
    }

    #[test]
    fn bm25_matches_hand_computed_oracle() {
        let index = fixture();
        let outcome = index
            .search(
                Analyzer::default(),
                "apple banana",
                CiffSearchOptions::default(),
            )
            .unwrap();
        let by_id = outcome
            .hits
            .iter()
            .map(|hit| (hit.external_id.as_str(), hit.score))
            .collect::<BTreeMap<_, _>>();

        // Independent textbook BM25 arithmetic for N=3, avgdl=2, k1=1.2,
        // b=.75. apple has df=2; banana has df=1.
        let apple_idf = libm::log(1.0 + 1.5 / 2.5);
        let banana_idf = libm::log(1.0 + 2.5 / 1.5);
        let expected_a = apple_idf * 2.0 * 2.2 / (2.0 + 1.2);
        let expected_b = banana_idf * 2.2 / (1.0 + 1.2 * (0.25 + 0.75 * 0.5));
        let expected_c = apple_idf * 2.2 / (1.0 + 1.2 * (0.25 + 0.75 * 1.5));
        assert!((by_id["A"] - expected_a).abs() < 1e-12);
        assert!((by_id["B"] - expected_b).abs() < 1e-12);
        assert!((by_id["C"] - expected_c).abs() < 1e-12);
        assert_eq!(outcome.stats.postings_advanced, 3);
    }

    #[test]
    fn partial_index_uses_declared_collection_wide_statistics() {
        let index = CiffIndex::from_parts(
            CiffHeader {
                version: 1,
                num_posting_lists: 1,
                num_documents: 1,
                total_posting_lists: 5,
                total_documents: 3,
                total_terms_in_collection: 30,
                average_document_length: 10.0,
                description: "partial fixture".into(),
            },
            vec![CiffPostingList {
                term: "term".into(),
                document_frequency: 1,
                collection_frequency: 1,
                postings: vec![CiffPosting {
                    document_id: 2,
                    term_frequency: 1,
                }],
            }],
            vec![CiffDocumentRecord {
                document_id: 2,
                external_id: "partial-doc".into(),
                document_length: 1,
            }],
            CiffLimits::default(),
        )
        .unwrap();
        let result = index
            .search(Analyzer::default(), "term", CiffSearchOptions::default())
            .unwrap();
        let expected = libm::log(1.0 + 2.5 / 1.5) * 2.2 / (1.0 + 1.2 * (0.25 + 0.75 * 0.1));
        assert!((result.hits[0].score - expected).abs() < 1e-12);
        assert_eq!(index.stats().contained_documents, 1);
        assert_eq!(index.stats().total_documents, 3);
        assert_eq!(index.stats().contained_posting_lists, 1);
        assert_eq!(index.stats().total_posting_lists, 5);
    }

    #[test]
    fn boolean_and_counts_unique_terms_after_query_aggregation() {
        let index = fixture();
        let outcome = index
            .search(
                Analyzer::default(),
                "apple banana",
                CiffSearchOptions {
                    operator: BooleanOperator::And,
                    ..CiffSearchOptions::default()
                },
            )
            .unwrap();
        assert!(outcome.hits.is_empty());

        let outcome = index
            .search(
                Analyzer::default(),
                "apple apple",
                CiffSearchOptions {
                    top_k: usize::MAX,
                    operator: BooleanOperator::And,
                    ..CiffSearchOptions::default()
                },
            )
            .unwrap();
        assert_eq!(outcome.hits.len(), 2);
    }

    #[test]
    fn repeated_terms_and_query_boosts_match_hand_ranking_and_batch() {
        let index = fixture();
        let options = CiffSearchOptions {
            top_k: 3,
            ..CiffSearchOptions::default()
        };
        let once = index
            .search(Analyzer::default(), "apple banana", options)
            .unwrap();
        let repeated = index
            .search(Analyzer::default(), "apple apple banana", options)
            .unwrap();
        assert_eq!(once.hits[0].external_id, "B");
        assert_eq!(repeated.hits[0].external_id, "A");

        let apple_idf = libm::log(1.0 + 1.5 / 2.5);
        let expected_a = 2.0 * apple_idf * 2.0 * 2.2 / (2.0 + 1.2);
        let banana_idf = libm::log(1.0 + 2.5 / 1.5);
        let expected_b = banana_idf * 2.2 / (1.0 + 1.2 * (0.25 + 0.75 * 0.5));
        assert!((repeated.hits[0].score - expected_a).abs() < 1e-12);
        assert!((repeated.hits[1].score - expected_b).abs() < 1e-12);

        let query = SearchQuery::from_terms(vec![
            QueryTerm::new("APPLE", None, 1.25).unwrap(),
            QueryTerm::new("apple", None, 0.75).unwrap(),
            QueryTerm::new("BANANA", None, 1.0).unwrap(),
        ])
        .unwrap();
        let backend = index.retrieval(Analyzer::default());
        let via_backend = RetrievalBackend::search(
            &backend,
            &query,
            SearchOptions {
                top_k: 3,
                pruning: PruningStrategy::Exhaustive,
                explain: false,
                bm25: Bm25Params::default(),
            },
        )
        .unwrap();
        assert_eq!(via_backend.hits, repeated.hits);

        let report = evaluate_batch(
            &backend,
            &[Topic {
                id: "q1".into(),
                text: "APPLE apple BANANA".into(),
            }],
            None,
            BatchConfig {
                top_k: 3,
                pruning: PruningStrategy::Exhaustive,
                ..BatchConfig::default()
            },
        )
        .unwrap();
        assert_eq!(report.queries[0].hits, repeated.hits);
    }

    #[test]
    fn equal_scores_use_internal_id_before_and_after_save_load() {
        let index = CiffIndex::from_parts(
            CiffHeader {
                version: 1,
                num_posting_lists: 1,
                num_documents: 2,
                total_posting_lists: 1,
                total_documents: 2,
                total_terms_in_collection: 2,
                average_document_length: 1.0,
                description: "tie fixture".into(),
            },
            vec![CiffPostingList {
                term: "tie".into(),
                document_frequency: 2,
                collection_frequency: 2,
                postings: vec![
                    CiffPosting {
                        document_id: 0,
                        term_frequency: 1,
                    },
                    CiffPosting {
                        document_id: 1,
                        term_frequency: 1,
                    },
                ],
            }],
            vec![
                CiffDocumentRecord {
                    document_id: 0,
                    external_id: "Z-last-lexically".into(),
                    document_length: 1,
                },
                CiffDocumentRecord {
                    document_id: 1,
                    external_id: "A-first-lexically".into(),
                    document_length: 1,
                },
            ],
            CiffLimits::default(),
        )
        .unwrap();
        let assert_order = |candidate: &CiffIndex| {
            for _ in 0..3 {
                let result = candidate
                    .search(
                        Analyzer::default(),
                        "tie",
                        CiffSearchOptions {
                            top_k: 2,
                            ..CiffSearchOptions::default()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    result
                        .hits
                        .iter()
                        .map(|hit| (hit.doc_id, hit.external_id.as_str()))
                        .collect::<Vec<_>>(),
                    [(0, "Z-last-lexically"), (1, "A-first-lexically")]
                );
                assert_eq!(
                    result.hits[0].score.to_bits(),
                    result.hits[1].score.to_bits()
                );
            }
        };
        assert_order(&index);
        let path = temp_path();
        index.save(&path).unwrap();
        let loaded = CiffIndex::load(&path).unwrap();
        assert_order(&loaded);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_export_flattens_fields_and_preserves_external_ids() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(
                Document::from_fields("doc-a", [("title", "Sail sail"), ("body", "sail search")])
                    .unwrap(),
            )
            .unwrap();
        builder
            .add_document(Document::from_fields("doc-b", [("body", "search")]).unwrap())
            .unwrap();
        let native = builder.finish();
        let exported = CiffIndex::from_native(&native, "unit test").unwrap();
        assert_eq!(exported.documents[0].document_length, 4);
        assert_eq!(exported.documents[0].external_id, "doc-a");
        let sail = exported.posting_list("sail").unwrap();
        assert_eq!(sail.postings[0].term_frequency, 3);
        assert!(
            exported
                .header
                .description
                .contains("flattened all native fields")
        );
    }

    #[test]
    fn rejects_bad_frequency_order_and_missing_document_references() {
        let mut index = fixture();
        index.posting_lists[0].collection_frequency = 99;
        assert!(index.validate(CiffLimits::default()).is_err());

        let mut index = fixture();
        index.posting_lists[0].collection_frequency = (i64::MAX as u64) + 1;
        let error = index.validate(CiffLimits::default()).unwrap_err();
        assert!(error.to_string().contains("non-negative int64"));

        let mut index = fixture();
        index.posting_lists[0].postings[1].document_id = 1;
        index.posting_lists[1].postings.clear();
        index.posting_lists[1].document_frequency = 0;
        index.posting_lists[1].collection_frequency = 0;
        assert!(index.validate(CiffLimits::default()).is_err());

        let mut index = fixture();
        index.documents.pop();
        index.header.num_documents = 2;
        assert!(index.validate(CiffLimits::default()).is_err());
    }

    #[test]
    fn rejects_duplicate_terms_documents_and_external_ids() {
        let mut index = fixture();
        index.posting_lists[1].term = "apple".into();
        assert!(index.validate(CiffLimits::default()).is_err());

        let mut index = fixture();
        index.documents[1].document_id = 0;
        assert!(index.validate(CiffLimits::default()).is_err());

        let mut index = fixture();
        index.documents[1].external_id = "A".into();
        assert!(index.validate(CiffLimits::default()).is_err());

        let mut index = fixture();
        index.documents[1].external_id = "B\u{1b}[31m".into();
        assert!(index.validate(CiffLimits::default()).is_err());
    }

    #[test]
    fn enforces_limits_and_header_consistency() {
        let mut index = fixture();
        index.header.num_documents = 2;
        assert!(index.validate(CiffLimits::default()).is_err());
        let mut index = fixture();
        index.header.total_terms_in_collection = 0;
        assert!(index.validate(CiffLimits::default()).is_err());
        let mut index = fixture();
        index.header.average_document_length = 0.0;
        assert!(index.validate(CiffLimits::default()).is_err());
        let limits = CiffLimits {
            max_postings: 2,
            ..CiffLimits::default()
        };
        assert!(fixture().validate(limits).is_err());
        let limits = CiffLimits {
            max_documents: 2,
            ..CiffLimits::default()
        };
        assert!(fixture().validate(limits).is_err());
    }

    #[test]
    fn invalid_search_parameters_and_empty_queries_fail() {
        let index = fixture();
        assert!(
            index
                .search(Analyzer::default(), "---", CiffSearchOptions::default())
                .is_err()
        );
        assert!(
            index
                .search(
                    Analyzer::default(),
                    "apple",
                    CiffSearchOptions {
                        top_k: 0,
                        ..CiffSearchOptions::default()
                    }
                )
                .is_err()
        );
        let mut extreme_statistics = index.clone();
        extreme_statistics.header.total_documents = 2_147_483_647;
        assert!(
            extreme_statistics
                .search(
                    Analyzer::default(),
                    "apple",
                    CiffSearchOptions {
                        bm25: Bm25Params {
                            k1: f64::MAX,
                            b: 1.0,
                        },
                        ..CiffSearchOptions::default()
                    }
                )
                .is_err()
        );

        let overflowing = SearchQuery::from_terms(vec![
            QueryTerm::new("apple", None, f64::MAX).unwrap(),
            QueryTerm::new("APPLE", None, f64::MAX).unwrap(),
        ])
        .unwrap();
        let error = RetrievalBackend::search(
            &index.retrieval(Analyzer::default()),
            &overflowing,
            SearchOptions {
                pruning: PruningStrategy::Exhaustive,
                ..SearchOptions::default()
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("combined CIFF query-term weight")
        );
    }
}
