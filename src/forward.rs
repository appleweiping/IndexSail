//! Deterministic forward indexes and lexicons for collection parsing and inversion.
//!
//! A forward snapshot preserves every normalized token occurrence in document
//! order alongside the original named fields. Its field-qualified lexicon can
//! therefore rebuild the native positional inverted index without reopening or
//! reinterpreting the source collection.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Write};
use std::path::Path;

use crate::analysis::{AnalysisMode, Analyzer};
use crate::atomic::atomic_write_with;
use crate::codec::checksum;
use crate::collection::{CollectionFormat, CollectionLimits, load_collection};
use crate::document::{Document, validate_field_name};
use crate::error::{Error, Result};
use crate::index::{InvertedIndex, Posting, TermKey};
use crate::reorder::{DocIdMap, MAX_REORDER_FORWARD_BYTES, MAX_REORDER_OCCURRENCES};

const MAGIC: &[u8; 8] = b"IDXFW001";
pub const FORWARD_FORMAT_VERSION: u32 = 1;
const MAX_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_DOCUMENTS: usize = 20_000_000;
const MAX_TERMS: usize = 20_000_000;
const MAX_FIELDS: usize = 20_000_000;
const MAX_FIELDS_PER_DOCUMENT: usize = 4_096;
const MAX_OCCURRENCES: u64 = 1_000_000_000;
const HEADER_BYTES: u64 = 28;

#[derive(Debug, Default)]
struct ByteCounter {
    bytes: u64,
}

impl Write for ByteCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(u64::try_from(buffer.len()).map_err(std::io::Error::other)?)
            .ok_or_else(|| std::io::Error::other("forward byte count overflow"))?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub type TermId = u32;

/// One stable, field-qualified term in lexicographic identifier order.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ForwardTerm {
    pub field: String,
    pub term: String,
}

/// One document and its token-id sequences, keyed by field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForwardDocument {
    document: Document,
    term_ids: BTreeMap<String, Vec<TermId>>,
}

impl ForwardDocument {
    pub const fn document(&self) -> &Document {
        &self.document
    }

    pub fn term_ids(&self, field: &str) -> Option<&[TermId]> {
        self.term_ids.get(field).map(Vec::as_slice)
    }
}

/// Summary counts for a forward snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForwardStats {
    pub documents: usize,
    pub fields: usize,
    pub terms: usize,
    pub occurrences: u64,
}

/// Immutable forward index with a canonical field-qualified lexicon.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForwardIndex {
    analyzer: Analyzer,
    terms: Vec<ForwardTerm>,
    documents: Vec<ForwardDocument>,
    occurrences: u64,
}

fn collect_source_documents(
    documents: impl IntoIterator<Item = Document>,
) -> Result<Vec<Document>> {
    let mut bounded = Vec::new();
    let mut external_ids = HashSet::new();
    let mut fields = 0_usize;
    for document in documents {
        if bounded.len() == MAX_DOCUMENTS {
            return Err(Error::InvalidDocument(format!(
                "forward index exceeds {MAX_DOCUMENTS} document limit"
            )));
        }
        if document.external_id().len() > MAX_STRING_BYTES {
            return Err(Error::InvalidDocument(format!(
                "document id exceeds {MAX_STRING_BYTES} byte limit"
            )));
        }
        if document.fields().len() > MAX_FIELDS_PER_DOCUMENT {
            return Err(Error::InvalidDocument(format!(
                "document '{}' exceeds {MAX_FIELDS_PER_DOCUMENT} field limit",
                document.external_id()
            )));
        }
        fields = fields
            .checked_add(document.fields().len())
            .ok_or_else(|| Error::InvalidDocument("forward field count overflow".into()))?;
        if fields > MAX_FIELDS {
            return Err(Error::InvalidDocument(format!(
                "forward index exceeds {MAX_FIELDS} field limit"
            )));
        }
        if !external_ids.insert(document.external_id().to_owned()) {
            return Err(Error::DuplicateDocumentId(
                document.external_id().to_owned(),
            ));
        }
        if let Some((field, _)) = document
            .fields()
            .iter()
            .find(|(_, value)| value.len() > MAX_STRING_BYTES)
        {
            return Err(Error::InvalidDocument(format!(
                "field '{}:{field}' exceeds {MAX_STRING_BYTES} byte limit",
                document.external_id()
            )));
        }
        bounded
            .try_reserve(1)
            .map_err(|_| Error::InvalidDocument("could not allocate forward documents".into()))?;
        bounded.push(document);
    }
    Ok(bounded)
}

fn build_forward_lexicon(analyzer: Analyzer, documents: &[Document]) -> Result<Vec<ForwardTerm>> {
    let mut unique_terms = BTreeSet::new();
    let mut occurrences = 0_u64;
    for document in documents {
        for (field, value) in document.fields() {
            for token in analyzer.analyze(value) {
                occurrences = add_occurrences(occurrences, 1).ok_or_else(|| {
                    Error::InvalidDocument(format!(
                        "forward index exceeds {MAX_OCCURRENCES} occurrence limit"
                    ))
                })?;
                let term = ForwardTerm {
                    field: field.clone(),
                    term: token.text,
                };
                if unique_terms.len() == MAX_TERMS && !unique_terms.contains(&term) {
                    return Err(Error::InvalidDocument(format!(
                        "forward lexicon exceeds {MAX_TERMS} term limit"
                    )));
                }
                unique_terms.insert(term);
            }
        }
    }
    Ok(unique_terms.into_iter().collect())
}

fn encode_forward_documents(
    analyzer: Analyzer,
    documents: Vec<Document>,
    terms: &[ForwardTerm],
) -> Result<Vec<ForwardDocument>> {
    let term_ids = terms
        .iter()
        .enumerate()
        .map(|(index, term)| {
            let id = TermId::try_from(index)
                .map_err(|_| Error::InvalidDocument("forward term id does not fit u32".into()))?;
            Ok(((term.field.clone(), term.term.clone()), id))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve(documents.len())
        .map_err(|_| Error::InvalidDocument("could not allocate forward documents".into()))?;
    for document in documents {
        let mut fields = BTreeMap::new();
        for (field, value) in document.fields() {
            let tokens = analyzer.analyze(value);
            let mut sequence = Vec::new();
            sequence.try_reserve(tokens.len()).map_err(|_| {
                Error::InvalidDocument(format!(
                    "could not allocate term sequence for '{}:{field}'",
                    document.external_id()
                ))
            })?;
            for token in tokens {
                let id = term_ids
                    .get(&(field.clone(), token.text))
                    .copied()
                    .ok_or_else(|| {
                        Error::CorruptIndex("forward lexicon construction mismatch".into())
                    })?;
                sequence.push(id);
            }
            fields.insert(field.clone(), sequence);
        }
        encoded.push(ForwardDocument {
            document,
            term_ids: fields,
        });
    }
    Ok(encoded)
}

impl ForwardIndex {
    /// Analyze documents and assign term identifiers by sorted `(field, term)`.
    pub fn from_documents(
        analyzer: Analyzer,
        documents: impl IntoIterator<Item = Document>,
    ) -> Result<Self> {
        let documents = collect_source_documents(documents)?;

        let terms = build_forward_lexicon(analyzer, &documents)?;
        let forward_documents = encode_forward_documents(analyzer, documents, &terms)?;
        Self::from_parts(analyzer, terms, forward_documents)
    }

    /// Parse a supported collection and build its forward representation.
    pub fn from_collection(
        path: impl AsRef<Path>,
        analyzer: Analyzer,
        format: CollectionFormat,
        limits: CollectionLimits,
    ) -> Result<Self> {
        Self::from_documents(analyzer, load_collection(path, format, limits)?)
    }

    pub const fn analyzer(&self) -> Analyzer {
        self.analyzer
    }

    pub fn terms(&self) -> &[ForwardTerm] {
        &self.terms
    }

    pub fn documents(&self) -> &[ForwardDocument] {
        &self.documents
    }

    /// Reassign internal IDs while preserving each document's external ID,
    /// stored fields, normalized occurrences, and canonical term lexicon.
    pub fn reordered(&self, mapping: &DocIdMap) -> Result<Self> {
        if mapping.len() != self.documents.len() {
            return Err(Error::InvalidArgument(format!(
                "document mapping has {} entries; forward index has {} documents",
                mapping.len(),
                self.documents.len()
            )));
        }
        if self.occurrences > MAX_REORDER_OCCURRENCES {
            return Err(Error::InvalidArgument(format!(
                "reordering exceeds {MAX_REORDER_OCCURRENCES} occurrence limit"
            )));
        }
        let mut counter = ByteCounter::default();
        self.write_payload(&mut counter)?;
        if counter.bytes > MAX_REORDER_FORWARD_BYTES {
            return Err(Error::InvalidArgument(format!(
                "reordering exceeds {MAX_REORDER_FORWARD_BYTES} forward payload byte limit"
            )));
        }
        let mut documents = Vec::new();
        documents.try_reserve_exact(mapping.len()).map_err(|_| {
            Error::InvalidArgument("could not allocate reordered forward documents".into())
        })?;
        for &old in mapping.new_to_old() {
            documents.push(self.documents[old as usize].clone());
        }
        Ok(Self {
            analyzer: self.analyzer,
            terms: self.terms.clone(),
            documents,
            occurrences: self.occurrences,
        })
    }

    pub fn term(&self, id: TermId) -> Option<&ForwardTerm> {
        usize::try_from(id)
            .ok()
            .and_then(|index| self.terms.get(index))
    }

    /// Look up an already-normalized term in the canonical lexicon.
    pub fn term_id(&self, field: &str, normalized_term: &str) -> Option<TermId> {
        let key = ForwardTerm {
            field: field.to_owned(),
            term: normalized_term.to_owned(),
        };
        self.terms
            .binary_search(&key)
            .ok()
            .and_then(|index| TermId::try_from(index).ok())
    }

    pub fn stats(&self) -> ForwardStats {
        ForwardStats {
            documents: self.documents.len(),
            fields: self
                .documents
                .iter()
                .map(|document| document.term_ids.len())
                .sum(),
            terms: self.terms.len(),
            occurrences: self.occurrences,
        }
    }

    /// Invert token occurrences into exact positional posting lists.
    pub fn invert(&self) -> Result<InvertedIndex> {
        let mut documents = Vec::new();
        let mut field_lengths = Vec::new();
        let mut postings = BTreeMap::<TermKey, Vec<Posting>>::new();
        documents
            .try_reserve(self.documents.len())
            .map_err(|_| Error::InvalidDocument("could not allocate inverted documents".into()))?;
        field_lengths
            .try_reserve(self.documents.len())
            .map_err(|_| {
                Error::InvalidDocument("could not allocate inverted field lengths".into())
            })?;

        for (document_index, forward) in self.documents.iter().enumerate() {
            let doc_id = u32::try_from(document_index)
                .map_err(|_| Error::InvalidDocument("document id does not fit u32".into()))?;
            let mut lengths = BTreeMap::new();
            for (field, ids) in &forward.term_ids {
                lengths.insert(
                    field.clone(),
                    u32::try_from(ids.len()).map_err(|_| {
                        Error::InvalidDocument(format!(
                            "field '{}:{field}' exceeds u32 token limit",
                            forward.document.external_id()
                        ))
                    })?,
                );
                let mut positions_by_term = BTreeMap::<TermId, Vec<u32>>::new();
                for (position, &term_id) in ids.iter().enumerate() {
                    positions_by_term.entry(term_id).or_default().push(
                        u32::try_from(position).map_err(|_| {
                            Error::InvalidDocument("term position does not fit u32".into())
                        })?,
                    );
                }
                for (term_id, positions) in positions_by_term {
                    let term = self.term(term_id).ok_or_else(|| {
                        Error::CorruptIndex(format!("unknown forward term id {term_id}"))
                    })?;
                    postings
                        .entry(TermKey {
                            field: term.field.clone(),
                            term: term.term.clone(),
                        })
                        .or_default()
                        .push(Posting {
                            doc_id,
                            term_frequency: u32::try_from(positions.len()).map_err(|_| {
                                Error::InvalidDocument("term frequency does not fit u32".into())
                            })?,
                            positions,
                        });
                }
            }
            documents.push(forward.document.clone());
            field_lengths.push(lengths);
        }
        InvertedIndex::from_parts(self.analyzer, documents, field_lengths, postings)
    }

    /// Atomically persist a checksummed version 1 forward snapshot.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        atomic_write_with(path.as_ref(), |writer| self.write_to(writer))
    }

    /// Serialize the same version 1 forward snapshot to a caller-owned writer.
    pub fn write_to(&self, mut writer: impl Write) -> Result<()> {
        let mut counter = ByteCounter::default();
        self.write_payload(&mut counter)?;
        if counter.bytes > MAX_PAYLOAD_BYTES {
            return Err(Error::InvalidArgument(format!(
                "forward payload exceeds {MAX_PAYLOAD_BYTES} byte safety limit"
            )));
        }
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(usize::try_from(counter.bytes).map_err(|_| {
                Error::InvalidArgument("forward payload does not fit this platform".into())
            })?)
            .map_err(|_| Error::InvalidArgument("could not allocate forward payload".into()))?;
        self.write_payload(&mut payload)?;
        let payload_length = u64::try_from(payload.len())
            .map_err(|_| Error::InvalidArgument("forward payload does not fit u64".into()))?;
        if payload_length > MAX_PAYLOAD_BYTES {
            return Err(Error::InvalidArgument(format!(
                "forward payload exceeds {MAX_PAYLOAD_BYTES} byte safety limit"
            )));
        }
        writer.write_all(MAGIC)?;
        write_u32(&mut writer, FORWARD_FORMAT_VERSION)?;
        write_u64(&mut writer, payload_length)?;
        write_u64(&mut writer, checksum(&payload))?;
        writer.write_all(&payload)?;
        Ok(())
    }

    /// Load and semantically validate a version 1 forward snapshot.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_length = file.metadata()?.len();
        if file_length > HEADER_BYTES + MAX_PAYLOAD_BYTES {
            return Err(Error::CorruptIndex(format!(
                "forward file length {file_length} exceeds safety limit"
            )));
        }
        let mut reader = BufReader::new(file);
        let mut magic = [0_u8; 8];
        read_exact(&mut reader, &mut magic, "forward signature")?;
        if &magic != MAGIC {
            return Err(Error::CorruptIndex(
                "invalid forward index signature".into(),
            ));
        }
        let version = read_u32(&mut reader)?;
        if version != FORWARD_FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let payload_length = read_u64(&mut reader)?;
        if payload_length > MAX_PAYLOAD_BYTES {
            return Err(Error::CorruptIndex(format!(
                "forward payload length {payload_length} exceeds safety limit"
            )));
        }
        let expected_checksum = read_u64(&mut reader)?;
        let expected_file_length = HEADER_BYTES
            .checked_add(payload_length)
            .ok_or_else(|| Error::CorruptIndex("forward file length overflow".into()))?;
        if file_length != expected_file_length {
            return Err(Error::CorruptIndex(format!(
                "forward file length {file_length} does not match declared length {expected_file_length}"
            )));
        }
        let payload_length = usize::try_from(payload_length).map_err(|_| {
            Error::CorruptIndex("forward payload does not fit this platform".into())
        })?;
        let mut payload = allocate_bytes(payload_length, "forward payload")?;
        read_exact(&mut reader, &mut payload, "forward payload")?;
        reject_trailing(&mut reader)?;
        let actual_checksum = checksum(&payload);
        if actual_checksum != expected_checksum {
            return Err(Error::CorruptIndex(format!(
                "forward payload checksum mismatch: expected {expected_checksum:016x}, got {actual_checksum:016x}"
            )));
        }
        Self::read_payload(&payload)
    }

    fn write_payload(&self, writer: &mut impl Write) -> Result<()> {
        write_u8(writer, self.analyzer.mode().wire_value())?;
        write_len(writer, self.terms.len(), "forward terms")?;
        for term in &self.terms {
            write_string(writer, &term.field)?;
            write_string(writer, &term.term)?;
        }
        write_len(writer, self.documents.len(), "forward documents")?;
        for forward in &self.documents {
            write_string(writer, forward.document.external_id())?;
            write_len(writer, forward.document.fields().len(), "forward fields")?;
            for (field, value) in forward.document.fields() {
                write_string(writer, field)?;
                write_string(writer, value)?;
                let ids = forward.term_ids.get(field).ok_or_else(|| {
                    Error::CorruptIndex("forward field lacks term sequence".into())
                })?;
                write_len(writer, ids.len(), "forward term occurrences")?;
                for &id in ids {
                    write_u32(writer, id)?;
                }
            }
        }
        Ok(())
    }

    fn read_payload(payload: &[u8]) -> Result<Self> {
        let mut reader = Cursor::new(payload);
        let mode_value = read_u8(&mut reader)?;
        let mode = AnalysisMode::from_wire(mode_value).ok_or_else(|| {
            Error::CorruptIndex(format!("unknown forward analyzer mode {mode_value}"))
        })?;
        let term_count = read_bounded_len(&mut reader, "forward terms", MAX_TERMS)?;
        ensure_structural_bytes(&reader, term_count, 8, "forward terms")?;
        let mut terms = fallible_vec(term_count, "forward terms")?;
        for _ in 0..term_count {
            terms.push(ForwardTerm {
                field: read_string(&mut reader)?,
                term: read_string(&mut reader)?,
            });
        }
        let document_count = read_bounded_len(&mut reader, "forward documents", MAX_DOCUMENTS)?;
        ensure_structural_bytes(&reader, document_count, 8, "forward documents")?;
        let mut documents = fallible_vec(document_count, "forward documents")?;
        let mut declared_occurrences = 0_u64;
        let mut declared_fields = 0_usize;
        for _ in 0..document_count {
            let external_id = read_string(&mut reader)?;
            let field_count = read_bounded_len(&mut reader, "forward fields", MAX_FIELDS)?;
            declared_fields = declared_fields
                .checked_add(field_count)
                .ok_or_else(|| Error::CorruptIndex("forward field count overflow".into()))?;
            if declared_fields > MAX_FIELDS {
                return Err(Error::CorruptIndex(format!(
                    "forward index exceeds {MAX_FIELDS} field limit"
                )));
            }
            ensure_structural_bytes(&reader, field_count, 12, "forward fields")?;
            let mut fields = fallible_vec(field_count, "forward fields")?;
            let mut sequences = BTreeMap::new();
            for _ in 0..field_count {
                let field = read_string(&mut reader)?;
                let value = read_string(&mut reader)?;
                let occurrence_count = read_bounded_len(
                    &mut reader,
                    "forward term occurrences",
                    usize::try_from(MAX_OCCURRENCES).unwrap_or(usize::MAX),
                )?;
                ensure_structural_bytes(&reader, occurrence_count, 4, "forward term occurrences")?;
                declared_occurrences = add_occurrences(
                    declared_occurrences,
                    u64::try_from(occurrence_count).map_err(|_| {
                        Error::CorruptIndex("forward occurrence count does not fit u64".into())
                    })?,
                )
                .ok_or_else(|| {
                    Error::CorruptIndex(format!(
                        "forward index exceeds {MAX_OCCURRENCES} occurrence limit"
                    ))
                })?;
                let mut ids = fallible_vec(occurrence_count, "forward term occurrences")?;
                for _ in 0..occurrence_count {
                    ids.push(read_u32(&mut reader)?);
                }
                if sequences.insert(field.clone(), ids).is_some() {
                    return Err(Error::CorruptIndex(format!(
                        "duplicate forward field '{field}'"
                    )));
                }
                fields.push((field, value));
            }
            let document = Document::from_fields(external_id, fields)
                .map_err(|error| Error::CorruptIndex(error.to_string()))?;
            documents.push(ForwardDocument {
                document,
                term_ids: sequences,
            });
        }
        reject_trailing(&mut reader)?;
        let index = Self::from_parts(Analyzer::new(mode), terms, documents)?;
        if index.occurrences != declared_occurrences {
            return Err(Error::CorruptIndex(
                "forward occurrence count changed during validation".into(),
            ));
        }
        Ok(index)
    }

    fn from_parts(
        analyzer: Analyzer,
        terms: Vec<ForwardTerm>,
        documents: Vec<ForwardDocument>,
    ) -> Result<Self> {
        if terms.len() > MAX_TERMS || documents.len() > MAX_DOCUMENTS {
            return Err(Error::CorruptIndex(
                "forward collection count exceeds safety limit".into(),
            ));
        }
        validate_forward_lexicon(analyzer, &terms)?;
        let occurrences = validate_forward_documents(analyzer, &terms, &documents)?;
        Ok(Self {
            analyzer,
            terms,
            documents,
            occurrences,
        })
    }
}

fn validate_forward_lexicon(analyzer: Analyzer, terms: &[ForwardTerm]) -> Result<()> {
    if !terms.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(Error::CorruptIndex(
            "forward lexicon must be strictly sorted and unique".into(),
        ));
    }
    for term in terms {
        validate_field_name(&term.field).map_err(|error| Error::CorruptIndex(error.to_string()))?;
        if analyzer.normalize_single(&term.term).as_deref() != Some(term.term.as_str()) {
            return Err(Error::CorruptIndex(format!(
                "forward term '{}:{}' is not normalized",
                term.field, term.term
            )));
        }
    }
    Ok(())
}

fn validate_forward_documents(
    analyzer: Analyzer,
    terms: &[ForwardTerm],
    documents: &[ForwardDocument],
) -> Result<u64> {
    let mut observed_terms = fallible_vec(terms.len(), "forward term usage")?;
    observed_terms.resize(terms.len(), false);
    let mut external_ids = HashSet::new();
    let mut occurrences = 0_u64;
    let mut field_count = 0_usize;
    for forward in documents {
        if !external_ids.insert(forward.document.external_id()) {
            return Err(Error::CorruptIndex(format!(
                "duplicate forward document id '{}'",
                forward.document.external_id()
            )));
        }
        field_count = field_count
            .checked_add(forward.term_ids.len())
            .ok_or_else(|| Error::CorruptIndex("forward field count overflow".into()))?;
        if field_count > MAX_FIELDS {
            return Err(Error::CorruptIndex(format!(
                "forward index exceeds {MAX_FIELDS} field limit"
            )));
        }
        let document_occurrences =
            validate_forward_document(analyzer, terms, forward, &mut observed_terms)?;
        occurrences = add_occurrences(occurrences, document_occurrences).ok_or_else(|| {
            Error::CorruptIndex(format!(
                "forward index exceeds {MAX_OCCURRENCES} occurrence limit"
            ))
        })?;
    }
    if observed_terms.iter().any(|used| !used) {
        return Err(Error::CorruptIndex(
            "forward lexicon contains a term absent from every document".into(),
        ));
    }
    Ok(occurrences)
}

fn validate_forward_document(
    analyzer: Analyzer,
    terms: &[ForwardTerm],
    forward: &ForwardDocument,
    observed_terms: &mut [bool],
) -> Result<u64> {
    if forward.document.external_id().len() > MAX_STRING_BYTES {
        return Err(Error::CorruptIndex(
            "forward document id exceeds string byte limit".into(),
        ));
    }
    if forward.term_ids.len() > MAX_FIELDS_PER_DOCUMENT {
        return Err(Error::CorruptIndex(format!(
            "forward document '{}' exceeds {MAX_FIELDS_PER_DOCUMENT} field limit",
            forward.document.external_id()
        )));
    }
    if forward.document.fields().len() != forward.term_ids.len()
        || forward
            .document
            .fields()
            .keys()
            .any(|field| !forward.term_ids.contains_key(field))
    {
        return Err(Error::CorruptIndex(format!(
            "forward fields and sequences differ for '{}'",
            forward.document.external_id()
        )));
    }
    let mut occurrences = 0_u64;
    for (field, value) in forward.document.fields() {
        if value.len() > MAX_STRING_BYTES {
            return Err(Error::CorruptIndex(format!(
                "forward field '{}:{field}' exceeds string byte limit",
                forward.document.external_id()
            )));
        }
        let ids = &forward.term_ids[field];
        let tokens = analyzer.analyze(value);
        if tokens.len() != ids.len() {
            return Err(Error::CorruptIndex(format!(
                "forward token count differs for '{}:{field}'",
                forward.document.external_id()
            )));
        }
        occurrences = occurrences
            .checked_add(u64::try_from(ids.len()).map_err(|_| {
                Error::CorruptIndex("forward occurrence count does not fit u64".into())
            })?)
            .ok_or_else(|| Error::CorruptIndex("forward occurrence count overflow".into()))?;
        for (token, &id) in tokens.iter().zip(ids) {
            let term = usize::try_from(id)
                .ok()
                .and_then(|index| terms.get(index))
                .ok_or_else(|| Error::CorruptIndex(format!("unknown forward term id {id}")))?;
            observed_terms[id as usize] = true;
            if term.field != *field || term.term != token.text {
                return Err(Error::CorruptIndex(format!(
                    "forward term id {id} does not match '{}:{field}' token '{}'",
                    forward.document.external_id(),
                    token.text
                )));
            }
        }
    }
    Ok(occurrences)
}

const fn add_occurrences(current: u64, additional: u64) -> Option<u64> {
    match current.checked_add(additional) {
        Some(total) if total <= MAX_OCCURRENCES => Some(total),
        _ => None,
    }
}

fn write_u8(writer: &mut impl Write, value: u8) -> Result<()> {
    writer.write_all(&[value])?;
    Ok(())
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_len(writer: &mut impl Write, value: usize, label: &str) -> Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{label} count does not fit u32")))?;
    write_u32(writer, value)
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    if value.len() > MAX_STRING_BYTES {
        return Err(Error::InvalidArgument(format!(
            "forward string exceeds {MAX_STRING_BYTES} byte safety limit"
        )));
    }
    write_len(writer, value.len(), "forward string bytes")?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0_u8; 1];
    read_exact(reader, &mut bytes, "u8")?;
    Ok(bytes[0])
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    read_exact(reader, &mut bytes, "u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    read_exact(reader, &mut bytes, "u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_bounded_len(reader: &mut impl Read, label: &str, maximum: usize) -> Result<usize> {
    let value = read_u32(reader)? as usize;
    if value > maximum {
        return Err(Error::CorruptIndex(format!(
            "{label} count {value} exceeds safety limit {maximum}"
        )));
    }
    Ok(value)
}

fn read_string(reader: &mut Cursor<&[u8]>) -> Result<String> {
    let length = read_bounded_len(reader, "forward string bytes", MAX_STRING_BYTES)?;
    ensure_structural_bytes(reader, length, 1, "forward string")?;
    let mut bytes = allocate_bytes(length, "forward string")?;
    read_exact(reader, &mut bytes, "forward string")?;
    String::from_utf8(bytes)
        .map_err(|_| Error::CorruptIndex("forward string is not valid UTF-8".into()))
}

fn ensure_structural_bytes(
    reader: &Cursor<&[u8]>,
    count: usize,
    minimum_bytes: usize,
    label: &str,
) -> Result<()> {
    let required = count
        .checked_mul(minimum_bytes)
        .ok_or_else(|| Error::CorruptIndex(format!("{label} byte count overflow")))?;
    let position = usize::try_from(reader.position())
        .map_err(|_| Error::CorruptIndex("forward cursor does not fit usize".into()))?;
    let remaining = reader.get_ref().len().saturating_sub(position);
    if required > remaining {
        return Err(Error::CorruptIndex(format!(
            "{label} declares at least {required} bytes with only {remaining} remaining"
        )));
    }
    Ok(())
}

fn allocate_bytes(length: usize, label: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| Error::CorruptIndex(format!("could not allocate {label}")))?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn fallible_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::CorruptIndex(format!("could not allocate {label}")))?;
    Ok(values)
}

fn read_exact(reader: &mut impl Read, bytes: &mut [u8], label: &str) -> Result<()> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::CorruptIndex(format!("truncated {label}"))
        } else {
            Error::Io(error)
        }
    })
}

fn reject_trailing(reader: &mut impl Read) -> Result<()> {
    let mut trailing = [0_u8; 1];
    match reader.read(&mut trailing) {
        Ok(0) => Ok(()),
        Ok(_) => Err(Error::CorruptIndex(
            "trailing bytes after forward payload".into(),
        )),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn documents() -> Vec<Document> {
        vec![
            Document::from_fields("d1", [("title", "Blue sail"), ("body", "blue sea blue")])
                .unwrap(),
            Document::from_fields("d2", [("title", "Red"), ("body", "sea wind")]).unwrap(),
        ]
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "indexsail-forward-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn lexicon_is_sorted_and_independent_of_document_order() {
        let first = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        let mut reversed = documents();
        reversed.reverse();
        let second = ForwardIndex::from_documents(Analyzer::default(), reversed).unwrap();
        assert_eq!(first.terms(), second.terms());
        assert!(first.terms().windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(first.term_id("body", "blue"), Some(0));
        assert_eq!(first.term(0).unwrap().term, "blue");
        assert_eq!(first.stats().occurrences, 8);
    }

    #[test]
    fn inversion_matches_a_hand_computed_positional_oracle() {
        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        let index = forward.invert().unwrap();
        assert_eq!(index.stats().documents, 2);
        assert_eq!(index.field_length(0, "body"), 3);
        assert_eq!(index.field_length(1, "body"), 2);
        assert_eq!(
            index.postings("body", "blue").unwrap(),
            [Posting {
                doc_id: 0,
                term_frequency: 2,
                positions: vec![0, 2],
            }]
        );
        assert_eq!(
            index.postings("body", "sea").unwrap(),
            [
                Posting {
                    doc_id: 0,
                    term_frequency: 1,
                    positions: vec![1],
                },
                Posting {
                    doc_id: 1,
                    term_frequency: 1,
                    positions: vec![0],
                },
            ]
        );
    }

    #[test]
    fn inversion_matches_direct_builder_bit_for_bit() {
        let source = documents();
        let forward = ForwardIndex::from_documents(Analyzer::default(), source.clone()).unwrap();
        let mut direct = crate::index::IndexBuilder::new(Analyzer::default());
        for document in source {
            direct.add_document(document).unwrap();
        }
        let mut inverted_bytes = Vec::new();
        forward
            .invert()
            .unwrap()
            .write_to(&mut inverted_bytes)
            .unwrap();
        let mut direct_bytes = Vec::new();
        direct.finish().write_to(&mut direct_bytes).unwrap();
        assert_eq!(inverted_bytes, direct_bytes);
    }

    #[test]
    fn reordering_preserves_external_id_search_semantics_and_persistence() {
        use crate::query::{FieldFilter, PhraseFilter, SearchQuery};
        use crate::search::{PruningStrategy, SearchOptions};

        let source = vec![
            Document::from_fields("A", [("body", "blue sea blue"), ("kind", "water")]).unwrap(),
            Document::from_fields("B", [("body", "red wind"), ("kind", "air")]).unwrap(),
            Document::from_fields("C", [("body", "blue wind sea"), ("kind", "water")]).unwrap(),
            Document::from_fields("D", [("body", "sea wind wind"), ("kind", "water")]).unwrap(),
        ];
        let forward = ForwardIndex::from_documents(Analyzer::default(), source).unwrap();
        let original = forward.invert().unwrap();
        let mappings = [
            DocIdMap::from_old_to_new(vec![3, 0, 2, 1]).unwrap(),
            DocIdMap::by_feature(4, b"z\na\nc\nb\n".as_slice()).unwrap(),
            DocIdMap::random(4, 17).unwrap(),
            DocIdMap::recursive_graph_bisection(
                &forward,
                crate::reorder::BisectionOptions::for_documents(4),
            )
            .unwrap(),
        ];
        for mapping in mappings {
            let reordered = forward.reordered(&mapping).unwrap();
            assert_eq!(reordered.stats(), forward.stats());
            assert_eq!(reordered.terms(), forward.terms());
            for (old, &new) in mapping.old_to_new().iter().enumerate() {
                assert_eq!(
                    reordered.documents()[new as usize],
                    forward.documents()[old]
                );
            }
            let forward_path = temp_path("reordered.fwd");
            let index_path = temp_path("reordered.idx");
            reordered.save(&forward_path).unwrap();
            assert_eq!(ForwardIndex::load(&forward_path).unwrap(), reordered);
            let rebuilt = reordered.invert().unwrap();
            rebuilt.save(&index_path).unwrap();
            let loaded = InvertedIndex::load(&index_path).unwrap();
            assert_eq!(loaded.documents(), rebuilt.documents());
            for term in ["blue", "sea", "wind", "red"] {
                let mut expected = original
                    .postings("body", term)
                    .unwrap()
                    .iter()
                    .cloned()
                    .map(|mut posting| {
                        posting.doc_id = mapping.old_to_new()[posting.doc_id as usize];
                        posting
                    })
                    .collect::<Vec<_>>();
                expected.sort_by_key(|posting| posting.doc_id);
                assert_eq!(loaded.postings("body", term).unwrap(), expected);
            }
            for text in ["blue", "sea wind", "red wind"] {
                let query =
                    SearchQuery::from_text(original.analyzer(), text, Some("body")).unwrap();
                for pruning in [PruningStrategy::Exhaustive, PruningStrategy::Wand] {
                    let options = SearchOptions {
                        top_k: 4,
                        pruning,
                        ..SearchOptions::default()
                    };
                    let before = original.search(&query, options).unwrap();
                    let after = loaded.search(&query, options).unwrap();
                    let by_external = |hits: &[crate::search::SearchHit]| {
                        hits.iter()
                            .map(|hit| (hit.external_id.clone(), hit.score.to_bits()))
                            .collect::<BTreeMap<_, _>>()
                    };
                    assert_eq!(by_external(&before.hits), by_external(&after.hits));
                }
            }
            let query = SearchQuery::from_text(original.analyzer(), "blue sea", Some("body"))
                .unwrap()
                .with_phrase(
                    PhraseFilter::from_text(original.analyzer(), "blue sea", Some("body".into()))
                        .unwrap(),
                )
                .with_filter(FieldFilter::exact("kind", "water").unwrap());
            let options = SearchOptions {
                top_k: 4,
                pruning: PruningStrategy::Exhaustive,
                ..SearchOptions::default()
            };
            let before = original.search(&query, options).unwrap();
            let after = loaded.search(&query, options).unwrap();
            assert_eq!(before.hits.len(), 1);
            assert_eq!(before.hits[0].external_id, after.hits[0].external_id);
            assert_eq!(
                before.hits[0].score.to_bits(),
                after.hits[0].score.to_bits()
            );
            std::fs::remove_file(forward_path).unwrap();
            std::fs::remove_file(index_path).unwrap();
        }
        assert!(
            forward
                .reordered(&DocIdMap::from_old_to_new(vec![1, 0]).unwrap())
                .is_err()
        );
    }

    #[test]
    fn empty_forward_index_can_be_reordered_and_inverted() {
        let empty = ForwardIndex::from_documents(Analyzer::default(), Vec::new()).unwrap();
        let mapping = DocIdMap::random(0, 3).unwrap();
        let reordered = empty.reordered(&mapping).unwrap();
        assert_eq!(reordered, empty);
        assert_eq!(reordered.invert().unwrap().stats().documents, 0);
    }

    #[test]
    fn terms_are_field_qualified() {
        let forward = ForwardIndex::from_documents(
            Analyzer::default(),
            [Document::from_fields("d", [("title", "same"), ("body", "same")]).unwrap()],
        )
        .unwrap();
        assert_eq!(forward.stats().terms, 2);
        assert_ne!(
            forward.term_id("title", "same"),
            forward.term_id("body", "same")
        );
    }

    #[test]
    fn checksummed_persistence_round_trips() {
        let path = temp_path("roundtrip.fwd");
        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        forward.save(&path).unwrap();
        let stored = std::fs::read(&path).unwrap();
        assert_eq!(&stored[..8], MAGIC);
        assert_eq!(ForwardIndex::load(&path).unwrap(), forward);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn persistence_rejects_corruption_and_trailing_bytes() {
        let path = temp_path("corrupt.fwd");
        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        forward.save(&path).unwrap();
        let mut stored = std::fs::read(&path).unwrap();
        let last = stored.len() - 1;
        stored[last] ^= 0x80;
        std::fs::write(&path, &stored).unwrap();
        assert!(
            ForwardIndex::load(&path)
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );

        forward.save(&path).unwrap();
        let mut stored = std::fs::read(&path).unwrap();
        stored.push(0);
        std::fs::write(&path, stored).unwrap();
        assert!(
            ForwardIndex::load(&path)
                .unwrap_err()
                .to_string()
                .contains("does not match declared length")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn oversized_declared_payload_is_rejected_before_reading_it() {
        let path = temp_path("oversized.fwd");
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        write_u32(&mut header, FORWARD_FORMAT_VERSION).unwrap();
        write_u64(&mut header, MAX_PAYLOAD_BYTES + 1).unwrap();
        write_u64(&mut header, 0).unwrap();
        std::fs::write(&path, header).unwrap();
        assert!(
            ForwardIndex::load(&path)
                .unwrap_err()
                .to_string()
                .contains("exceeds safety limit")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn truncated_declared_payload_and_impossible_counts_fail_before_bulk_allocation() {
        let path = temp_path("truncated.fwd");
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        write_u32(&mut header, FORWARD_FORMAT_VERSION).unwrap();
        write_u64(&mut header, 1_000_000_000).unwrap();
        write_u64(&mut header, 0).unwrap();
        std::fs::write(&path, header).unwrap();
        assert!(
            ForwardIndex::load(&path)
                .unwrap_err()
                .to_string()
                .contains("does not match declared length")
        );
        std::fs::remove_file(path).unwrap();

        let mut payload = vec![AnalysisMode::Unicode.wire_value()];
        write_u32(&mut payload, 1_000_000).unwrap();
        assert!(
            ForwardIndex::read_payload(&payload)
                .unwrap_err()
                .to_string()
                .contains("with only 0 remaining")
        );
    }

    #[test]
    fn semantic_validation_rejects_a_token_id_for_the_wrong_term() {
        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        let mut tampered = forward.documents.clone();
        tampered[0].term_ids.get_mut("body").unwrap()[0] = forward.term_id("body", "sea").unwrap();
        assert!(
            ForwardIndex::from_parts(forward.analyzer, forward.terms, tampered)
                .unwrap_err()
                .to_string()
                .contains("does not match")
        );
    }

    #[test]
    fn duplicate_document_ids_are_rejected_before_indexing() {
        let duplicate = Document::from_fields("same", [("body", "one")]).unwrap();
        assert!(matches!(
            ForwardIndex::from_documents(
                Analyzer::default(),
                [duplicate.clone(), duplicate]
            ),
            Err(Error::DuplicateDocumentId(id)) if id == "same"
        ));
    }

    #[test]
    fn jsonl_collection_builds_a_forward_index_and_inverts() {
        let path = temp_path("collection.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"a\",\"fields\":{\"body\":\"one two one\"}}\n{\"id\":\"b\",\"fields\":{\"body\":\"two\"}}\n",
        )
        .unwrap();
        let forward = ForwardIndex::from_collection(
            &path,
            Analyzer::default(),
            CollectionFormat::Jsonl,
            CollectionLimits::default(),
        )
        .unwrap();
        assert_eq!(forward.stats().documents, 2);
        assert_eq!(forward.stats().occurrences, 4);
        assert_eq!(
            forward.invert().unwrap().postings("body", "one").unwrap()[0].positions,
            [0, 2]
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn ascii_analyzer_state_round_trips() {
        let path = temp_path("ascii.fwd");
        let forward = ForwardIndex::from_documents(
            Analyzer::new(AnalysisMode::Ascii),
            [Document::from_fields("d", [("body", "CAFÉ")]).unwrap()],
        )
        .unwrap();
        forward.save(&path).unwrap();
        let restored = ForwardIndex::load(&path).unwrap();
        assert_eq!(restored.analyzer().mode(), AnalysisMode::Ascii);
        assert_eq!(restored.term_id("body", "caf"), Some(0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_forward_header_rejects_signature_version_and_truncation() {
        let path = temp_path("header-rejections.fwd");
        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        forward.save(&path).unwrap();
        let original = std::fs::read(&path).unwrap();
        let mut wrong = original.clone();
        wrong[0] ^= 1;
        std::fs::write(&path, wrong).unwrap();
        assert!(
            ForwardIndex::load(&path)
                .unwrap_err()
                .to_string()
                .contains("signature")
        );
        let mut wrong = original.clone();
        wrong[8..12].copy_from_slice(&(FORWARD_FORMAT_VERSION + 1).to_le_bytes());
        std::fs::write(&path, wrong).unwrap();
        assert!(matches!(
            ForwardIndex::load(&path),
            Err(Error::UnsupportedVersion(_))
        ));
        for end in [0, 8, 12, 20, 27] {
            std::fs::write(&path, &original[..end]).unwrap();
            assert!(ForwardIndex::load(&path).is_err(), "truncated at {end}");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_forward_payload_rejects_structure_and_duplicate_fields() {
        let mut payload = vec![255];
        write_u32(&mut payload, 0).unwrap();
        write_u32(&mut payload, 0).unwrap();
        assert!(
            ForwardIndex::read_payload(&payload)
                .unwrap_err()
                .to_string()
                .contains("analyzer mode")
        );

        let mut payload = vec![AnalysisMode::Unicode.wire_value()];
        write_u32(&mut payload, u32::try_from(MAX_TERMS + 1).unwrap()).unwrap();
        assert!(
            ForwardIndex::read_payload(&payload)
                .unwrap_err()
                .to_string()
                .contains("term")
        );

        let mut payload = vec![AnalysisMode::Unicode.wire_value()];
        write_u32(&mut payload, 0).unwrap();
        write_u32(&mut payload, u32::try_from(MAX_DOCUMENTS + 1).unwrap()).unwrap();
        assert!(
            ForwardIndex::read_payload(&payload)
                .unwrap_err()
                .to_string()
                .contains("document")
        );

        let mut payload = vec![AnalysisMode::Unicode.wire_value()];
        write_u32(&mut payload, 0).unwrap();
        write_u32(&mut payload, 1).unwrap();
        write_string(&mut payload, "doc").unwrap();
        write_u32(&mut payload, 2).unwrap();
        for _ in 0..2 {
            write_string(&mut payload, "body").unwrap();
            write_string(&mut payload, "").unwrap();
            write_u32(&mut payload, 0).unwrap();
        }
        assert!(
            ForwardIndex::read_payload(&payload)
                .unwrap_err()
                .to_string()
                .contains("duplicate forward field")
        );

        let mut payload = vec![AnalysisMode::Unicode.wire_value()];
        write_u32(&mut payload, 0).unwrap();
        write_u32(&mut payload, 0).unwrap();
        payload.push(1);
        assert!(
            ForwardIndex::read_payload(&payload)
                .unwrap_err()
                .to_string()
                .contains("trailing")
        );
    }

    #[test]
    fn semantic_forward_validation_rejects_inconsistent_lexicon_and_sequences() {
        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        let check = |terms: Vec<ForwardTerm>, rows: Vec<ForwardDocument>, message: &str| {
            let error = ForwardIndex::from_parts(forward.analyzer, terms, rows).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        };
        let mut terms = forward.terms.clone();
        terms.swap(0, 1);
        check(terms, forward.documents.clone(), "strictly sorted");
        let mut terms = forward.terms.clone();
        terms[0].term = "UPPER".into();
        check(terms, forward.documents.clone(), "not normalized");
        let mut terms = forward.terms.clone();
        terms.push(ForwardTerm {
            field: "zzzz".into(),
            term: "unused".into(),
        });
        check(
            terms,
            forward.documents.clone(),
            "absent from every document",
        );
        let mut rows = forward.documents.clone();
        rows[0].term_ids.get_mut("body").unwrap().pop();
        check(forward.terms.clone(), rows, "token count differs");
        let mut rows = forward.documents.clone();
        rows[0].term_ids.get_mut("body").unwrap()[0] = u32::MAX;
        check(forward.terms.clone(), rows, "unknown forward term id");
        let mut rows = forward.documents.clone();
        rows[0].term_ids.remove("body");
        check(forward.terms.clone(), rows, "fields and sequences differ");
        let mut rows = forward.documents.clone();
        rows[0].term_ids.remove("body");
        rows[0].term_ids.insert("other".into(), Vec::new());
        check(forward.terms.clone(), rows, "fields and sequences differ");
        let mut rows = forward.documents.clone();
        rows[0].term_ids.get_mut("body").unwrap()[0] = forward.term_id("title", "blue").unwrap();
        check(forward.terms.clone(), rows, "does not match");
        let mut rows = forward.documents.clone();
        rows[1].document = rows[0].document.clone();
        check(forward.terms.clone(), rows, "duplicate forward document id");
        let mut invalid = forward.clone();
        invalid.occurrences = MAX_REORDER_OCCURRENCES + 1;
        let mapping = DocIdMap::from_old_to_new(vec![0, 1]).unwrap();
        assert!(
            invalid
                .reordered(&mapping)
                .unwrap_err()
                .to_string()
                .contains("occurrence limit")
        );
        let short_map = DocIdMap::from_old_to_new(vec![0]).unwrap();
        assert!(
            forward
                .reordered(&short_map)
                .unwrap_err()
                .to_string()
                .contains("mapping has")
        );
        assert_eq!(add_occurrences(MAX_OCCURRENCES, 0), Some(MAX_OCCURRENCES));
        assert_eq!(add_occurrences(MAX_OCCURRENCES, 1), None);
        assert_eq!(add_occurrences(u64::MAX, 1), None);
    }

    #[test]
    fn forward_reader_distinguishes_io_errors_from_truncation() {
        struct Denied;
        impl Read for Denied {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
            }
        }
        assert!(
            matches!(read_u32(&mut Denied), Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied)
        );
        let mut bad_utf8 = Vec::new();
        write_u32(&mut bad_utf8, 1).unwrap();
        bad_utf8.push(255);
        assert!(
            read_string(&mut Cursor::new(bad_utf8.as_slice()))
                .unwrap_err()
                .to_string()
                .contains("UTF-8")
        );
    }

    #[test]
    fn too_many_fields_are_rejected_before_analyzing_or_revalidating() {
        let fields = (0..=MAX_FIELDS_PER_DOCUMENT)
            .map(|number| (format!("field_{number}"), String::new()))
            .collect::<Vec<_>>();
        let document = Document::from_fields("many", fields).unwrap();
        assert!(
            ForwardIndex::from_documents(Analyzer::default(), [document])
                .unwrap_err()
                .to_string()
                .contains("field limit")
        );

        let forward = ForwardIndex::from_documents(Analyzer::default(), documents()).unwrap();
        let mut rows = forward.documents.clone();
        for number in 0..=MAX_FIELDS_PER_DOCUMENT {
            rows[0]
                .term_ids
                .insert(format!("extra_{number}"), Vec::new());
        }
        assert!(
            ForwardIndex::from_parts(forward.analyzer, forward.terms, rows)
                .unwrap_err()
                .to_string()
                .contains("field limit")
        );
    }
}
