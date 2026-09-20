use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::analysis::Analyzer;
use crate::codec::{PostingCodecStats, encoded_posting_bytes};
use crate::document::{Document, validate_field_name};
use crate::error::{Error, Result};

pub type InternalDocId = u32;

pub(crate) const BLOCK_POSTINGS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Posting {
    pub doc_id: InternalDocId,
    pub term_frequency: u32,
    pub positions: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct TermKey {
    pub field: String,
    pub term: String,
}

/// Identifies the scored posting stream for one normalized query term.
///
/// `field == None` is the deterministic merge across every indexed field;
/// `Some` identifies the field-qualified stream. Both forms are persisted so
/// default-BM25 query construction never has to derive block bounds from
/// scored postings.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct BlockMaxKey {
    pub field: Option<String>,
    pub term: String,
}

/// Wire-stable upper bounds for one scored posting stream.
///
/// Bounds are stored as IEEE-754 bits so format validation can require exact,
/// deterministic equality rather than an epsilon comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BlockMaxMetadata {
    pub posting_count: usize,
    pub default_bounds: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexStats {
    pub documents: usize,
    pub fields: usize,
    pub terms: usize,
    pub postings: usize,
    pub tokens: u64,
}

/// Immutable positional inverted index.
#[derive(Clone, Debug)]
pub struct InvertedIndex {
    pub(crate) analyzer: Analyzer,
    pub(crate) documents: Vec<Document>,
    pub(crate) field_lengths: Vec<BTreeMap<String, u32>>,
    pub(crate) field_totals: BTreeMap<String, u64>,
    pub(crate) postings: BTreeMap<TermKey, Vec<Posting>>,
    pub(crate) block_max: BTreeMap<BlockMaxKey, BlockMaxMetadata>,
}

impl InvertedIndex {
    pub const fn analyzer(&self) -> Analyzer {
        self.analyzer
    }

    pub fn document(&self, doc_id: InternalDocId) -> Option<&Document> {
        usize::try_from(doc_id)
            .ok()
            .and_then(|index| self.documents.get(index))
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }

    pub fn fields(&self) -> BTreeSet<&str> {
        self.field_totals.keys().map(String::as_str).collect()
    }

    /// Return postings for an already-normalized term.
    pub fn postings(&self, field: &str, normalized_term: &str) -> Option<&[Posting]> {
        self.postings
            .get(&TermKey {
                field: field.to_owned(),
                term: normalized_term.to_owned(),
            })
            .map(Vec::as_slice)
    }

    pub fn document_frequency(&self, field: &str, normalized_term: &str) -> usize {
        self.postings(field, normalized_term)
            .map_or(0, <[Posting]>::len)
    }

    pub fn field_length(&self, doc_id: InternalDocId, field: &str) -> u32 {
        usize::try_from(doc_id)
            .ok()
            .and_then(|index| self.field_lengths.get(index))
            .and_then(|lengths| lengths.get(field))
            .copied()
            .unwrap_or(0)
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn average_field_length(&self, field: &str) -> f64 {
        if self.documents.is_empty() {
            return 0.0;
        }
        self.field_totals.get(field).copied().unwrap_or(0) as f64 / self.documents.len() as f64
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            documents: self.documents.len(),
            fields: self.field_totals.len(),
            terms: self.postings.len(),
            postings: self.postings.values().map(Vec::len).sum(),
            tokens: self.field_totals.values().sum(),
        }
    }

    /// Report the size of fixed-width posting values versus the delta/varbyte
    /// representation used by persistence formats version 2 and 3.
    pub fn posting_codec_stats(&self) -> Result<PostingCodecStats> {
        let mut stats = PostingCodecStats::default();
        for postings in self.postings.values() {
            stats.posting_lists += 1;
            stats.postings += postings.len();
            let positions = postings
                .iter()
                .map(|posting| posting.positions.len())
                .sum::<usize>();
            stats.positions += positions;
            let values = postings
                .len()
                .checked_mul(2)
                .and_then(|value| value.checked_add(positions))
                .ok_or_else(|| Error::InvalidArgument("posting statistics overflow".into()))?;
            stats.uncompressed_bytes = stats
                .uncompressed_bytes
                .checked_add(
                    u64::try_from(values)
                        .map_err(|_| Error::InvalidArgument("posting statistics overflow".into()))?
                        .checked_mul(4)
                        .ok_or_else(|| {
                            Error::InvalidArgument("posting statistics overflow".into())
                        })?,
                )
                .ok_or_else(|| Error::InvalidArgument("posting statistics overflow".into()))?;
            stats.encoded_bytes = stats
                .encoded_bytes
                .checked_add(
                    u64::try_from(encoded_posting_bytes(postings)?).map_err(|_| {
                        Error::InvalidArgument("posting statistics overflow".into())
                    })?,
                )
                .ok_or_else(|| Error::InvalidArgument("posting statistics overflow".into()))?;
        }
        Ok(stats)
    }

    pub(crate) fn from_parts(
        analyzer: Analyzer,
        documents: Vec<Document>,
        field_lengths: Vec<BTreeMap<String, u32>>,
        postings: BTreeMap<TermKey, Vec<Posting>>,
    ) -> Result<Self> {
        if documents.len() != field_lengths.len() {
            return Err(Error::CorruptIndex(
                "document and field-length counts differ".into(),
            ));
        }
        if documents.len() > u32::MAX as usize {
            return Err(Error::CorruptIndex("too many documents".into()));
        }

        validate_posting_lists(analyzer, documents.len(), &field_lengths, &postings)?;
        let field_totals =
            validate_documents_and_lengths(analyzer, &documents, &field_lengths, &postings)?;

        let mut index = Self {
            analyzer,
            documents,
            field_lengths,
            field_totals,
            postings,
            block_max: BTreeMap::new(),
        };
        index.block_max = index.compute_block_max_metadata();
        Ok(index)
    }

    pub(crate) fn from_parts_with_block_max(
        analyzer: Analyzer,
        documents: Vec<Document>,
        field_lengths: Vec<BTreeMap<String, u32>>,
        postings: BTreeMap<TermKey, Vec<Posting>>,
        block_max: BTreeMap<BlockMaxKey, BlockMaxMetadata>,
    ) -> Result<Self> {
        let mut index = Self::from_parts(analyzer, documents, field_lengths, postings)?;
        if index.block_max != block_max {
            return Err(Error::CorruptIndex(
                "persisted block-max metadata does not match the postings".into(),
            ));
        }
        index.block_max = block_max;
        Ok(index)
    }
}

fn validate_documents_and_lengths(
    analyzer: Analyzer,
    documents: &[Document],
    field_lengths: &[BTreeMap<String, u32>],
    postings: &BTreeMap<TermKey, Vec<Posting>>,
) -> Result<BTreeMap<String, u64>> {
    let mut external_ids = HashSet::new();
    let mut field_totals = BTreeMap::<String, u64>::new();
    let mut actual_tokens = 0_u64;
    for values in postings.values() {
        for posting in values {
            actual_tokens = actual_tokens
                .checked_add(u64::from(posting.term_frequency))
                .ok_or_else(|| Error::CorruptIndex("posting token count overflow".into()))?;
        }
    }
    let mut expected_tokens = 0_u64;
    for (doc_id, (document, lengths)) in documents.iter().zip(field_lengths).enumerate() {
        let doc_id =
            u32::try_from(doc_id).map_err(|_| Error::CorruptIndex("too many documents".into()))?;
        if !external_ids.insert(document.external_id()) {
            return Err(Error::CorruptIndex(format!(
                "duplicate external id '{}'",
                document.external_id()
            )));
        }
        if document.fields().len() != lengths.len()
            || document
                .fields()
                .keys()
                .any(|field| !lengths.contains_key(field))
        {
            return Err(Error::CorruptIndex(format!(
                "stored field lengths do not match document '{}'",
                document.external_id()
            )));
        }
        for (field, value) in document.fields() {
            validate_field_name(field).map_err(|error| Error::CorruptIndex(error.to_string()))?;
            let tokens = analyzer.analyze(value);
            let expected = u32::try_from(tokens.len())
                .map_err(|_| Error::CorruptIndex("field token count exceeds u32".into()))?;
            if lengths.get(field).copied() != Some(expected) {
                return Err(Error::CorruptIndex(format!(
                    "stored length for '{}:{}' does not match analyzed text",
                    document.external_id(),
                    field
                )));
            }
            let total = field_totals.entry(field.clone()).or_default();
            *total = total
                .checked_add(u64::from(expected))
                .ok_or_else(|| Error::CorruptIndex("field token count overflow".into()))?;
            expected_tokens = expected_tokens
                .checked_add(u64::from(expected))
                .ok_or_else(|| Error::CorruptIndex("field token count overflow".into()))?;
            let mut positions_by_term = BTreeMap::<String, Vec<u32>>::new();
            for token in tokens {
                positions_by_term
                    .entry(token.text)
                    .or_default()
                    .push(token.position);
            }
            for (term, positions) in positions_by_term {
                let key = TermKey {
                    field: field.clone(),
                    term,
                };
                let Some(values) = postings.get(&key) else {
                    return Err(Error::CorruptIndex(format!(
                        "missing posting list for '{}:{}'",
                        key.field, key.term
                    )));
                };
                let Ok(position) = values.binary_search_by_key(&doc_id, |posting| posting.doc_id)
                else {
                    return Err(Error::CorruptIndex(format!(
                        "missing posting for '{}:{}' in document {doc_id}",
                        key.field, key.term
                    )));
                };
                if values[position].positions != positions {
                    return Err(Error::CorruptIndex(format!(
                        "posting positions for '{}:{}' in document {doc_id} do not match analyzed text",
                        key.field, key.term
                    )));
                }
            }
        }
    }
    if actual_tokens != expected_tokens {
        return Err(Error::CorruptIndex(
            "posting token count does not match analyzed documents".into(),
        ));
    }
    Ok(field_totals)
}

fn validate_posting_lists(
    analyzer: Analyzer,
    document_count: usize,
    field_lengths: &[BTreeMap<String, u32>],
    postings: &BTreeMap<TermKey, Vec<Posting>>,
) -> Result<()> {
    for (key, values) in postings {
        validate_field_name(&key.field).map_err(|error| Error::CorruptIndex(error.to_string()))?;
        if analyzer.normalize_single(&key.term).as_deref() != Some(key.term.as_str()) {
            return Err(Error::CorruptIndex(format!(
                "dictionary term '{}:{}' is not normalized",
                key.field, key.term
            )));
        }
        if values.is_empty() {
            return Err(Error::CorruptIndex(format!(
                "term '{}:{}' has an empty posting list",
                key.field, key.term
            )));
        }
        let mut previous_doc = None;
        for posting in values {
            let doc_index = posting.doc_id as usize;
            if doc_index >= document_count {
                return Err(Error::CorruptIndex(format!(
                    "posting references missing document {}",
                    posting.doc_id
                )));
            }
            if previous_doc.is_some_and(|previous| previous >= posting.doc_id) {
                return Err(Error::CorruptIndex(
                    "posting lists must be strictly ordered by document id".into(),
                ));
            }
            previous_doc = Some(posting.doc_id);
            if posting.term_frequency == 0
                || posting.term_frequency as usize != posting.positions.len()
            {
                return Err(Error::CorruptIndex(
                    "term frequency does not match positions".into(),
                ));
            }
            if !posting.positions.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(Error::CorruptIndex(
                    "term positions must be strictly increasing".into(),
                ));
            }
            let field_length = field_lengths[doc_index]
                .get(&key.field)
                .copied()
                .unwrap_or(0);
            if posting
                .positions
                .last()
                .is_some_and(|position| *position >= field_length)
            {
                return Err(Error::CorruptIndex(format!(
                    "term position exceeds field length for document {}",
                    posting.doc_id
                )));
            }
        }
    }
    Ok(())
}

/// Incrementally constructs an immutable index while preserving insertion order.
#[derive(Debug)]
pub struct IndexBuilder {
    analyzer: Analyzer,
    documents: Vec<Document>,
    external_ids: HashSet<String>,
    field_lengths: Vec<BTreeMap<String, u32>>,
    field_totals: BTreeMap<String, u64>,
    postings: BTreeMap<TermKey, Vec<Posting>>,
}

impl IndexBuilder {
    pub fn new(analyzer: Analyzer) -> Self {
        Self {
            analyzer,
            documents: Vec::new(),
            external_ids: HashSet::new(),
            field_lengths: Vec::new(),
            field_totals: BTreeMap::new(),
            postings: BTreeMap::new(),
        }
    }

    pub fn add_document(&mut self, document: Document) -> Result<InternalDocId> {
        if self.documents.len() >= u32::MAX as usize {
            return Err(Error::InvalidDocument(
                "index cannot contain more than u32::MAX documents".into(),
            ));
        }
        if !self.external_ids.insert(document.external_id().to_owned()) {
            return Err(Error::DuplicateDocumentId(
                document.external_id().to_owned(),
            ));
        }

        let doc_id = InternalDocId::try_from(self.documents.len()).map_err(|_| {
            Error::InvalidDocument("index cannot contain more than u32::MAX documents".into())
        })?;
        let mut lengths = BTreeMap::new();
        for (field, value) in document.fields() {
            let tokens = self.analyzer.analyze(value);
            let field_length = u32::try_from(tokens.len()).map_err(|_| {
                Error::InvalidDocument(format!("field '{field}' has more than u32::MAX tokens"))
            })?;
            lengths.insert(field.clone(), field_length);
            *self.field_totals.entry(field.clone()).or_default() += u64::from(field_length);

            let mut positions_by_term = BTreeMap::<String, Vec<u32>>::new();
            for token in tokens {
                positions_by_term
                    .entry(token.text)
                    .or_default()
                    .push(token.position);
            }
            for (term, positions) in positions_by_term {
                self.postings
                    .entry(TermKey {
                        field: field.clone(),
                        term,
                    })
                    .or_default()
                    .push(Posting {
                        doc_id,
                        term_frequency: u32::try_from(positions.len())
                            .expect("field length checked"),
                        positions,
                    });
            }
        }

        self.field_lengths.push(lengths);
        self.documents.push(document);
        Ok(doc_id)
    }

    pub fn finish(self) -> InvertedIndex {
        let mut index = InvertedIndex {
            analyzer: self.analyzer,
            documents: self.documents,
            field_lengths: self.field_lengths,
            field_totals: self.field_totals,
            postings: self.postings,
            block_max: BTreeMap::new(),
        };
        index.block_max = index.compute_block_max_metadata();
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisMode, Analyzer};

    fn document(id: &str, title: &str, body: &str) -> Document {
        Document::from_fields(id, [("title", title), ("body", body)]).unwrap()
    }

    #[test]
    fn builds_dictionary_and_index_statistics() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("a", "Rust search", "small search engine"))
            .unwrap();
        builder
            .add_document(document("b", "Grid model", "small model"))
            .unwrap();
        let index = builder.finish();
        assert_eq!(index.stats().documents, 2);
        assert_eq!(index.stats().fields, 2);
        assert_eq!(index.stats().tokens, 9);
        assert!(index.stats().terms >= 6);
    }

    #[test]
    fn rejects_duplicate_external_ids_without_partial_insertion() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("same", "one", "body"))
            .unwrap();
        assert!(matches!(
            builder.add_document(document("same", "two", "other")),
            Err(Error::DuplicateDocumentId(id)) if id == "same"
        ));
        assert_eq!(builder.finish().stats().documents, 1);
    }

    #[test]
    fn postings_capture_frequency_and_positions() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("a", "", "red blue red red"))
            .unwrap();
        let index = builder.finish();
        let posting = &index.postings("body", "red").unwrap()[0];
        assert_eq!(posting.term_frequency, 3);
        assert_eq!(posting.positions, [0, 2, 3]);
    }

    #[test]
    fn document_frequency_counts_documents_not_occurrences() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("a", "", "red red red"))
            .unwrap();
        builder.add_document(document("b", "", "red blue")).unwrap();
        let index = builder.finish();
        assert_eq!(index.document_frequency("body", "red"), 2);
        assert_eq!(index.document_frequency("body", "missing"), 0);
    }

    #[test]
    fn average_length_uses_all_documents_including_missing_fields() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("a", "", "one two four six"))
            .unwrap();
        builder
            .add_document(Document::from_fields("b", [("title", "only")]).unwrap())
            .unwrap();
        let index = builder.finish();
        assert!((index.average_field_length("body") - 2.0).abs() < f64::EPSILON);
        assert_eq!(index.field_length(1, "body"), 0);
    }

    #[test]
    fn documents_keep_stable_insertion_ids() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        assert_eq!(builder.add_document(document("first", "", "a")).unwrap(), 0);
        assert_eq!(
            builder.add_document(document("second", "", "b")).unwrap(),
            1
        );
        let index = builder.finish();
        assert_eq!(index.document(0).unwrap().external_id(), "first");
        assert_eq!(index.document(1).unwrap().external_id(), "second");
        assert!(index.document(2).is_none());
    }

    #[test]
    fn index_uses_the_configured_ascii_analyzer() {
        let mut builder = IndexBuilder::new(Analyzer::new(AnalysisMode::Ascii));
        builder.add_document(document("a", "", "CAFÉ")).unwrap();
        let index = builder.finish();
        assert!(index.postings("body", "caf").is_some());
        assert!(index.postings("body", "café").is_none());
        assert_eq!(index.analyzer().mode(), AnalysisMode::Ascii);
    }

    #[test]
    fn fields_are_reported_in_deterministic_order() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("a", "title", "body"))
            .unwrap();
        let index = builder.finish();
        assert_eq!(
            index.fields().into_iter().collect::<Vec<_>>(),
            ["body", "title"]
        );
    }

    #[test]
    fn persisted_parts_reject_incorrect_field_length() {
        let document = Document::from_fields("doc", [("body", "one two")]).unwrap();
        let lengths = BTreeMap::from([("body".to_owned(), 1)]);
        assert!(matches!(
            InvertedIndex::from_parts(
                Analyzer::default(),
                vec![document],
                vec![lengths],
                BTreeMap::new()
            ),
            Err(Error::CorruptIndex(_))
        ));
    }

    #[test]
    fn persisted_parts_reject_document_count_field_set_and_frequency_disagreement() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("d0", "sea", "blue blue"))
            .unwrap();
        builder.add_document(document("d1", "sky", "blue")).unwrap();
        let original = builder.finish();
        let validate = |documents, lengths, postings| {
            InvertedIndex::from_parts(original.analyzer, documents, lengths, postings)
        };

        let error = validate(
            original.documents.clone(),
            original.field_lengths[..1].to_vec(),
            original.postings.clone(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("document and field-length counts differ")
        );

        let mut duplicate = original.documents.clone();
        duplicate[1] = document("d0", "sky", "blue");
        let error = validate(
            duplicate,
            original.field_lengths.clone(),
            original.postings.clone(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("duplicate external id"));

        let mut missing_field = original.field_lengths.clone();
        missing_field[0].remove("title");
        let mut postings_without_title = original.postings.clone();
        postings_without_title.retain(|key, _| key.field != "title");
        let error = validate(
            original.documents.clone(),
            missing_field,
            postings_without_title,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stored field lengths do not match")
        );

        let mut wrong_frequency = original.postings.clone();
        wrong_frequency
            .get_mut(&TermKey {
                field: "body".into(),
                term: "blue".into(),
            })
            .unwrap()[0]
            .term_frequency = 1;
        let error = validate(
            original.documents.clone(),
            original.field_lengths.clone(),
            wrong_frequency,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("term frequency does not match positions")
        );
    }

    #[test]
    fn persisted_parts_reject_unsorted_posting_documents() {
        let documents = vec![
            Document::from_fields("a", [("body", "term")]).unwrap(),
            Document::from_fields("b", [("body", "term")]).unwrap(),
        ];
        let lengths = vec![
            BTreeMap::from([("body".to_owned(), 1)]),
            BTreeMap::from([("body".to_owned(), 1)]),
        ];
        let postings = BTreeMap::from([(
            TermKey {
                field: "body".into(),
                term: "term".into(),
            },
            vec![
                Posting {
                    doc_id: 1,
                    term_frequency: 1,
                    positions: vec![0],
                },
                Posting {
                    doc_id: 0,
                    term_frequency: 1,
                    positions: vec![0],
                },
            ],
        )]);
        assert!(matches!(
            InvertedIndex::from_parts(Analyzer::default(), documents, lengths, postings),
            Err(Error::CorruptIndex(_))
        ));
    }

    #[test]
    fn persisted_parts_reject_missing_and_forged_token_postings() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("first", "blue", "red blue red"))
            .unwrap();
        builder
            .add_document(document("second", "green", "blue green"))
            .unwrap();
        let index = builder.finish();
        let validate = |postings| {
            InvertedIndex::from_parts(
                index.analyzer,
                index.documents.clone(),
                index.field_lengths.clone(),
                postings,
            )
        };

        let mut missing_term = index.postings.clone();
        missing_term.remove(&TermKey {
            field: "body".into(),
            term: "red".into(),
        });
        assert!(matches!(
            validate(missing_term),
            Err(Error::CorruptIndex(_))
        ));

        let mut missing_document = index.postings.clone();
        missing_document
            .get_mut(&TermKey {
                field: "body".into(),
                term: "blue".into(),
            })
            .unwrap()
            .pop();
        assert!(matches!(
            validate(missing_document),
            Err(Error::CorruptIndex(_))
        ));

        let mut forged_term = index.postings.clone();
        forged_term
            .get_mut(&TermKey {
                field: "body".into(),
                term: "red".into(),
            })
            .unwrap()[0]
            .positions = vec![0, 1];
        assert!(matches!(validate(forged_term), Err(Error::CorruptIndex(_))));

        let mut extra_term = index.postings.clone();
        extra_term.insert(
            TermKey {
                field: "body".into(),
                term: "phantom".into(),
            },
            vec![Posting {
                doc_id: 0,
                term_frequency: 1,
                positions: vec![0],
            }],
        );
        assert!(matches!(validate(extra_term), Err(Error::CorruptIndex(_))));
    }

    #[test]
    fn persisted_parts_reject_malformed_posting_boundaries() {
        let mut builder = IndexBuilder::new(Analyzer::default());
        builder
            .add_document(document("first", "red", "red red blue"))
            .unwrap();
        builder
            .add_document(document("second", "blue", "red blue"))
            .unwrap();
        let index = builder.finish();
        let key = TermKey {
            field: "body".into(),
            term: "red".into(),
        };
        let validate = |postings| {
            InvertedIndex::from_parts(
                index.analyzer,
                index.documents.clone(),
                index.field_lengths.clone(),
                postings,
            )
        };
        let mut malformed = Vec::new();
        let mut copy = index.postings.clone();
        copy.insert(
            TermKey {
                field: "bad field".into(),
                term: "red".into(),
            },
            copy[&key].clone(),
        );
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.insert(
            TermKey {
                field: "body".into(),
                term: "UPPER".into(),
            },
            copy[&key].clone(),
        );
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.get_mut(&key).unwrap().clear();
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.get_mut(&key).unwrap()[0].doc_id = 99;
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.get_mut(&key).unwrap()[1].doc_id = 0;
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.get_mut(&key).unwrap()[0].term_frequency = 0;
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.get_mut(&key).unwrap()[0].positions = vec![1, 1];
        malformed.push(copy);
        let mut copy = index.postings.clone();
        copy.get_mut(&key).unwrap()[0].positions = vec![0, 3];
        malformed.push(copy);
        for (case, postings) in malformed.into_iter().enumerate() {
            assert!(
                matches!(validate(postings), Err(Error::CorruptIndex(_))),
                "accepted malformed posting case {case}"
            );
        }
    }

    #[test]
    fn empty_index_accessors_have_explicit_missing_semantics() {
        let index = IndexBuilder::new(Analyzer::default()).finish();
        assert_eq!(index.document(0), None);
        assert_eq!(index.documents().len(), 0);
        assert!(index.fields().is_empty());
        assert_eq!(index.postings("body", "missing"), None);
        assert_eq!(index.document_frequency("body", "missing"), 0);
        assert_eq!(index.field_length(u32::MAX, "body"), 0);
        assert_eq!(
            index.average_field_length("body").to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(index.stats().tokens, 0);
        assert_eq!(index.posting_codec_stats().unwrap().posting_lists, 0);
    }

    #[test]
    fn thousand_document_semantic_validation_is_bounded_and_consistent() {
        let analyzer = Analyzer::default();
        let mut builder = IndexBuilder::new(analyzer);
        for ordinal in 0..1_000 {
            builder
                .add_document(document(
                    &format!("doc-{ordinal:04}"),
                    &format!("group {}", ordinal % 17),
                    &format!("red blue red shard {}", ordinal % 29),
                ))
                .unwrap();
        }
        let index = builder.finish();
        let structural_started = std::time::Instant::now();
        validate_posting_lists(
            analyzer,
            index.documents.len(),
            &index.field_lengths,
            &index.postings,
        )
        .unwrap();
        let structural_elapsed = structural_started.elapsed();
        let semantic_started = std::time::Instant::now();
        let totals = validate_documents_and_lengths(
            analyzer,
            &index.documents,
            &index.field_lengths,
            &index.postings,
        )
        .unwrap();
        let semantic_elapsed = semantic_started.elapsed();
        assert_eq!(totals, index.field_totals);
        eprintln!(
            "1000-document validation: posting structure {:.3} ms, document semantics {:.3} ms",
            structural_elapsed.as_secs_f64() * 1_000.0,
            semantic_elapsed.as_secs_f64() * 1_000.0
        );
    }
}
