//! Deterministic collection sharding with collection-wide BM25 statistics.
//!
//! Each physical shard owns an ordinary [`InvertedIndex`]. Documents are
//! assigned round-robin in global insertion order, so the mapping between a
//! global document id and `(shard, local id)` is arithmetic and wire-stable.
//! Query execution uses document frequencies, field totals, and document
//! count aggregated across every shard before independently searching each
//! shard. The coordinator then merges the shard-local top-k lists using the
//! same score/doc-id ordering as a monolithic index.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::analysis::Analyzer;
use crate::codec::Checksum;
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::{IndexBuilder, IndexStats, InternalDocId, InvertedIndex, TermKey};
use crate::persistence::{PERSISTENCE_FORMAT_VERSION, format_version_from_header};
use crate::query::SearchQuery;
use crate::search::{SearchOptions, SearchOutcome, SearchStats};

const MAGIC: &[u8; 8] = b"IDXSHD01";
pub const SHARDED_PERSISTENCE_FORMAT_VERSION: u32 = 1;
const MAX_SHARDS: usize = 4_096;
const MAX_CONTAINER_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_SHARD_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const READ_CHUNK_BYTES: usize = 8 * 1024;

/// Collection-wide statistics used to make scores comparable across shards.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlobalStatistics {
    document_count: usize,
    field_totals: BTreeMap<String, u64>,
    document_frequencies: BTreeMap<TermKey, usize>,
}

impl GlobalStatistics {
    fn from_shards(shards: &[InvertedIndex]) -> Result<Self> {
        let mut document_count = 0_usize;
        let mut field_totals = BTreeMap::<String, u64>::new();
        let mut document_frequencies = BTreeMap::<TermKey, usize>::new();
        for shard in shards {
            document_count = document_count
                .checked_add(shard.documents.len())
                .ok_or_else(|| Error::CorruptIndex("global document count overflow".into()))?;
            for (field, total) in &shard.field_totals {
                let value = field_totals.entry(field.clone()).or_default();
                *value = value
                    .checked_add(*total)
                    .ok_or_else(|| Error::CorruptIndex("global field total overflow".into()))?;
            }
            for (term, postings) in &shard.postings {
                let frequency = document_frequencies.entry(term.clone()).or_default();
                *frequency = frequency.checked_add(postings.len()).ok_or_else(|| {
                    Error::CorruptIndex("global document frequency overflow".into())
                })?;
            }
        }
        if document_count > u32::MAX as usize {
            return Err(Error::CorruptIndex(
                "sharded index cannot contain more than u32::MAX documents".into(),
            ));
        }
        Ok(Self {
            document_count,
            field_totals,
            document_frequencies,
        })
    }

    pub(crate) const fn document_count(&self) -> usize {
        self.document_count
    }

    pub(crate) fn document_frequency(&self, field: &str, term: &str) -> usize {
        self.document_frequencies
            .get(&TermKey {
                field: field.to_owned(),
                term: term.to_owned(),
            })
            .copied()
            .unwrap_or(0)
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn average_field_length(&self, field: &str) -> f64 {
        if self.document_count == 0 {
            return 0.0;
        }
        self.field_totals.get(field).copied().unwrap_or(0) as f64 / self.document_count as f64
    }
}

/// Per-shard index counters in physical shard order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalShardStats {
    pub shard_id: usize,
    pub documents: usize,
    pub fields: usize,
    pub terms: usize,
    pub postings: usize,
    pub tokens: u64,
}

/// Immutable set of independently searchable physical shards.
#[derive(Clone, Debug)]
pub struct ShardedIndex {
    shards: Vec<InvertedIndex>,
    global: GlobalStatistics,
    embedded_format_versions: Vec<u32>,
}

impl ShardedIndex {
    /// Rebuild a monolithic index into deterministic round-robin shards.
    pub fn from_index(index: &InvertedIndex, shard_count: usize) -> Result<Self> {
        let mut builder = ShardedIndexBuilder::new(index.analyzer(), shard_count)?;
        for document in index.documents() {
            builder.add_document(document.clone())?;
        }
        Ok(builder.finish())
    }

    fn from_loaded_shards(
        shards: Vec<InvertedIndex>,
        document_count: usize,
        embedded_format_versions: Vec<u32>,
    ) -> Result<Self> {
        validate_shard_count(shards.len())?;
        if embedded_format_versions.len() != shards.len() {
            return Err(Error::CorruptIndex(
                "embedded format-version count differs from physical shard count".into(),
            ));
        }
        let analyzer = shards[0].analyzer();
        let mut external_ids = HashSet::new();
        for (shard_id, shard) in shards.iter().enumerate() {
            if shard.analyzer() != analyzer {
                return Err(Error::CorruptIndex(
                    "physical shards use different analyzers".into(),
                ));
            }
            let expected = documents_in_shard(document_count, shards.len(), shard_id);
            if shard.documents().len() != expected {
                return Err(Error::CorruptIndex(format!(
                    "physical shard {shard_id} has {} documents; expected {expected}",
                    shard.documents().len()
                )));
            }
            for document in shard.documents() {
                if !external_ids.insert(document.external_id()) {
                    return Err(Error::CorruptIndex(format!(
                        "duplicate external id '{}' across physical shards",
                        document.external_id()
                    )));
                }
            }
        }
        let global = GlobalStatistics::from_shards(&shards)?;
        if global.document_count != document_count {
            return Err(Error::CorruptIndex(format!(
                "sharded document count is {}; manifest records {document_count}",
                global.document_count
            )));
        }
        Ok(Self {
            shards,
            global,
            embedded_format_versions,
        })
    }

    pub fn analyzer(&self) -> Analyzer {
        // Construction requires at least one physical shard.
        self.shards[0].analyzer()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Persistence version observed for each embedded physical index.
    pub fn embedded_format_versions(&self) -> &[u32] {
        &self.embedded_format_versions
    }

    pub const fn document_count(&self) -> usize {
        self.global.document_count
    }

    pub fn fields(&self) -> BTreeSet<&str> {
        self.global
            .field_totals
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// Return collection-wide document frequency, not a shard-local count.
    pub fn document_frequency(&self, field: &str, normalized_term: &str) -> usize {
        self.global.document_frequency(field, normalized_term)
    }

    pub fn average_field_length(&self, field: &str) -> f64 {
        self.global.average_field_length(field)
    }

    /// Resolve a global document id without a persisted routing table.
    pub fn document(&self, global_doc_id: InternalDocId) -> Option<&Document> {
        let global_doc_id = global_doc_id as usize;
        if global_doc_id >= self.document_count() {
            return None;
        }
        let shard_id = global_doc_id % self.shard_count();
        let local_doc_id = InternalDocId::try_from(global_doc_id / self.shard_count()).ok()?;
        self.shards[shard_id].document(local_doc_id)
    }

    /// Access one physical shard for diagnostics or independent persistence.
    pub fn shard(&self, shard_id: usize) -> Option<&InvertedIndex> {
        self.shards.get(shard_id)
    }

    /// Aggregate unique dictionary terms and additive posting/token counters.
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            documents: self.document_count(),
            fields: self.global.field_totals.len(),
            terms: self.global.document_frequencies.len(),
            postings: self.shards.iter().map(|shard| shard.stats().postings).sum(),
            tokens: self.global.field_totals.values().sum(),
        }
    }

    pub fn physical_shard_stats(&self) -> Vec<PhysicalShardStats> {
        self.shards
            .iter()
            .enumerate()
            .map(|(shard_id, shard)| {
                let stats = shard.stats();
                PhysicalShardStats {
                    shard_id,
                    documents: stats.documents,
                    fields: stats.fields,
                    terms: stats.terms,
                    postings: stats.postings,
                    tokens: stats.tokens,
                }
            })
            .collect()
    }

    /// Search every shard with one collection-wide BM25 statistics snapshot.
    ///
    /// A global top-k member must be in its shard's local top-k, so retaining
    /// only `top_k` candidates per shard is exact. Scores are ordered
    /// descending and ties by the original global insertion id ascending.
    pub fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
        let options = options.validate()?;
        let candidate_capacity = self
            .document_count()
            .min(options.top_k.saturating_mul(self.shard_count()));
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(candidate_capacity)
            .map_err(|_| {
                Error::InvalidArgument(
                    "sharded result candidates cannot be allocated safely".into(),
                )
            })?;
        let mut total_stats = SearchStats::default();
        for (shard_id, shard) in self.shards.iter().enumerate() {
            if shard.documents().is_empty() {
                continue;
            }
            let shard_options = SearchOptions {
                top_k: options.top_k.min(shard.documents().len()),
                ..options
            };
            let outcome =
                shard.search_with_global_statistics(query, shard_options, &self.global)?;
            add_search_stats(&mut total_stats, outcome.stats)?;
            for mut hit in outcome.hits {
                let global_doc_id = (hit.doc_id as usize)
                    .checked_mul(self.shard_count())
                    .and_then(|value| value.checked_add(shard_id))
                    .ok_or_else(|| Error::CorruptIndex("global document id overflow".into()))?;
                if global_doc_id >= self.document_count() {
                    return Err(Error::CorruptIndex(
                        "physical shard returned a document outside the global collection".into(),
                    ));
                }
                hit.doc_id = InternalDocId::try_from(global_doc_id)
                    .map_err(|_| Error::CorruptIndex("global document id overflow".into()))?;
                candidates.push(hit);
            }
        }
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        candidates.truncate(options.top_k);
        Ok(SearchOutcome {
            hits: candidates,
            stats: total_stats,
        })
    }

    /// Persist the sharded container with a checksum over all embedded v3
    /// index snapshots. Embedded index readers retain v1/v2 compatibility.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_from(BufReader::new(File::open(path)?))
    }

    pub fn write_to(&self, mut writer: impl Write) -> Result<()> {
        let shard_count = u32::try_from(self.shard_count())
            .map_err(|_| Error::InvalidArgument("shard count does not fit u32".into()))?;
        let document_count = u64::try_from(self.document_count())
            .map_err(|_| Error::InvalidArgument("document count does not fit u64".into()))?;
        let mut payload_length = 4_u64 + 8;
        let mut shard_lengths = Vec::new();
        shard_lengths
            .try_reserve_exact(self.shard_count())
            .map_err(|_| {
                Error::InvalidArgument(
                    "physical shard length table cannot be allocated safely".into(),
                )
            })?;
        for shard in &self.shards {
            let length = serialized_length(shard)?;
            if length > MAX_SHARD_PAYLOAD_BYTES {
                return Err(Error::InvalidArgument(format!(
                    "physical shard exceeds {MAX_SHARD_PAYLOAD_BYTES} byte safety limit"
                )));
            }
            payload_length = payload_length
                .checked_add(8)
                .and_then(|value| value.checked_add(length))
                .ok_or_else(|| Error::InvalidArgument("sharded payload length overflow".into()))?;
            if payload_length > MAX_CONTAINER_PAYLOAD_BYTES {
                return Err(Error::InvalidArgument(format!(
                    "sharded payload exceeds {MAX_CONTAINER_PAYLOAD_BYTES} byte safety limit"
                )));
            }
            shard_lengths.push(length);
        }

        // The checksum precedes the payload in format v1. Compute it in a
        // bounded first pass, then emit the same deterministic bytes. This
        // intentionally trades serialization CPU for a peak-memory bound of
        // one embedded index instead of retaining the whole collection plus
        // another copy of the current shard.
        let mut checksum = Checksum::new();
        checksum.update(&shard_count.to_le_bytes());
        checksum.update(&document_count.to_le_bytes());
        for (shard, &length) in self.shards.iter().zip(&shard_lengths) {
            checksum.update(&length.to_le_bytes());
            let written = update_checksum(shard, &mut checksum)?;
            if written != length {
                return Err(Error::InvalidArgument(
                    "physical shard serialization length changed between passes".into(),
                ));
            }
        }

        writer.write_all(MAGIC)?;
        write_u32(&mut writer, SHARDED_PERSISTENCE_FORMAT_VERSION)?;
        write_u64(&mut writer, payload_length)?;
        write_u64(&mut writer, checksum.finish())?;
        write_u32(&mut writer, shard_count)?;
        write_u64(&mut writer, document_count)?;
        for (shard, &length) in self.shards.iter().zip(&shard_lengths) {
            write_u64(&mut writer, length)?;
            shard.write_to(&mut writer)?;
        }
        Ok(())
    }

    /// Read from a seekable source after verifying the entire outer checksum.
    ///
    /// Seeking permits a bounded validation pass before any shard structures
    /// are decoded. The second pass then retains at most one embedded index
    /// payload instead of buffering and copying the full sharded container.
    pub fn read_from(mut reader: impl Read + Seek) -> Result<Self> {
        let mut magic = [0_u8; 8];
        read_exact_corrupt(&mut reader, &mut magic, "sharded file signature")?;
        if &magic != MAGIC {
            return Err(Error::CorruptIndex("invalid sharded file signature".into()));
        }
        let version = read_u32(&mut reader)?;
        if version != SHARDED_PERSISTENCE_FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let payload_length = read_u64(&mut reader)?;
        if payload_length > MAX_CONTAINER_PAYLOAD_BYTES {
            return Err(Error::CorruptIndex(format!(
                "sharded payload length {payload_length} exceeds safety limit"
            )));
        }
        let expected_checksum = read_u64(&mut reader)?;
        let payload_start = reader.stream_position()?;
        let actual_checksum = checksum_reader(&mut reader, payload_length)?;
        reject_trailing(&mut reader)?;
        if actual_checksum != expected_checksum {
            return Err(Error::CorruptIndex(format!(
                "sharded payload checksum mismatch: expected {expected_checksum:016x}, got {actual_checksum:016x}"
            )));
        }
        reader.seek(SeekFrom::Start(payload_start))?;

        let mut payload = (&mut reader).take(payload_length);
        let shard_count = usize::try_from(read_u32(&mut payload)?)
            .map_err(|_| Error::CorruptIndex("shard count does not fit this platform".into()))?;
        validate_shard_count(shard_count)
            .map_err(|error| Error::CorruptIndex(error.to_string()))?;
        let document_count = usize::try_from(read_u64(&mut payload)?)
            .map_err(|_| Error::CorruptIndex("document count does not fit this platform".into()))?;
        if document_count > u32::MAX as usize {
            return Err(Error::CorruptIndex(
                "sharded index cannot contain more than u32::MAX documents".into(),
            ));
        }
        let mut shards = Vec::new();
        let mut embedded_format_versions = Vec::new();
        shards.try_reserve_exact(shard_count).map_err(|_| {
            Error::CorruptIndex("physical shard table cannot be allocated safely".into())
        })?;
        embedded_format_versions
            .try_reserve_exact(shard_count)
            .map_err(|_| {
                Error::CorruptIndex(
                    "embedded format-version table cannot be allocated safely".into(),
                )
            })?;
        for shard_id in 0..shard_count {
            let length = read_u64(&mut payload)?;
            if length > MAX_SHARD_PAYLOAD_BYTES {
                return Err(Error::CorruptIndex(format!(
                    "physical shard {shard_id} length {length} exceeds safety limit"
                )));
            }
            let mut shard_reader = (&mut payload).take(length);
            let mut header = [0_u8; 12];
            read_exact_corrupt(&mut shard_reader, &mut header, "embedded index header")?;
            let magic = <[u8; 8]>::try_from(&header[..8])
                .map_err(|_| Error::CorruptIndex("invalid embedded index header".into()))?;
            let embedded_version =
                format_version_from_header(
                    magic,
                    u32::from_le_bytes(header[8..].try_into().map_err(|_| {
                        Error::CorruptIndex("invalid embedded index version".into())
                    })?),
                )?;
            let shard = {
                let mut complete_reader = Cursor::new(header).chain(&mut shard_reader);
                InvertedIndex::read_from(&mut complete_reader)?
            };
            shards.push(shard);
            if shard_reader.limit() != 0 {
                return Err(Error::CorruptIndex(format!(
                    "truncated physical shard {shard_id}"
                )));
            }
            embedded_format_versions.push(embedded_version);
        }
        if payload.limit() != 0 {
            return Err(Error::CorruptIndex(
                "unexpected trailing bytes in sharded payload".into(),
            ));
        }
        Self::from_loaded_shards(shards, document_count, embedded_format_versions)
    }
}

/// Incrementally constructs balanced, deterministic physical shards.
#[derive(Debug)]
pub struct ShardedIndexBuilder {
    analyzer: Analyzer,
    builders: Vec<IndexBuilder>,
    external_ids: HashSet<String>,
    document_count: usize,
}

impl ShardedIndexBuilder {
    pub fn new(analyzer: Analyzer, shard_count: usize) -> Result<Self> {
        validate_shard_count(shard_count)?;
        let builders = (0..shard_count)
            .map(|_| IndexBuilder::new(analyzer))
            .collect();
        Ok(Self {
            analyzer,
            builders,
            external_ids: HashSet::new(),
            document_count: 0,
        })
    }

    pub fn add_document(&mut self, document: Document) -> Result<InternalDocId> {
        if self.document_count >= u32::MAX as usize {
            return Err(Error::InvalidDocument(
                "sharded index cannot contain more than u32::MAX documents".into(),
            ));
        }
        let external_id = document.external_id().to_owned();
        if self.external_ids.contains(&external_id) {
            return Err(Error::DuplicateDocumentId(external_id));
        }
        let global_doc_id = InternalDocId::try_from(self.document_count)
            .map_err(|_| Error::InvalidDocument("global document id overflow".into()))?;
        let shard_id = self.document_count % self.builders.len();
        self.builders[shard_id].add_document(document)?;
        self.external_ids.insert(external_id);
        self.document_count += 1;
        Ok(global_doc_id)
    }

    pub fn finish(self) -> ShardedIndex {
        let shards = self
            .builders
            .into_iter()
            .map(IndexBuilder::finish)
            .collect::<Vec<_>>();
        let global = GlobalStatistics::from_shards(&shards)
            .expect("validated in-memory shard counters cannot overflow");
        debug_assert_eq!(self.analyzer, shards[0].analyzer());
        debug_assert_eq!(global.document_count, self.document_count);
        let embedded_format_versions = vec![PERSISTENCE_FORMAT_VERSION; shards.len()];
        ShardedIndex {
            shards,
            global,
            embedded_format_versions,
        }
    }
}

/// Read and validate only the sharded container signature/version header.
pub fn persisted_sharded_format_version(path: impl AsRef<Path>) -> Result<u32> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 8];
    read_exact_corrupt(&mut reader, &mut magic, "sharded file signature")?;
    if &magic != MAGIC {
        return Err(Error::CorruptIndex("invalid sharded file signature".into()));
    }
    let version = read_u32(&mut reader)?;
    if version != SHARDED_PERSISTENCE_FORMAT_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    Ok(version)
}

fn validate_shard_count(shard_count: usize) -> Result<()> {
    if shard_count == 0 || shard_count > MAX_SHARDS {
        return Err(Error::InvalidArgument(format!(
            "shard count must be between 1 and {MAX_SHARDS}"
        )));
    }
    Ok(())
}

fn documents_in_shard(document_count: usize, shard_count: usize, shard_id: usize) -> usize {
    if shard_id >= document_count {
        0
    } else {
        (document_count - 1 - shard_id) / shard_count + 1
    }
}

fn add_search_stats(total: &mut SearchStats, current: SearchStats) -> Result<()> {
    total.evaluated_candidates =
        checked_counter(total.evaluated_candidates, current.evaluated_candidates)?;
    total.postings_advanced = checked_counter(total.postings_advanced, current.postings_advanced)?;
    total.postings_skipped = checked_counter(total.postings_skipped, current.postings_skipped)?;
    total.block_max_bounds_loaded = checked_counter(
        total.block_max_bounds_loaded,
        current.block_max_bounds_loaded,
    )?;
    total.block_max_postings_covered = checked_counter(
        total.block_max_postings_covered,
        current.block_max_postings_covered,
    )?;
    total.block_max_postings_scanned = checked_counter(
        total.block_max_postings_scanned,
        current.block_max_postings_scanned,
    )?;
    Ok(())
}

fn checked_counter(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::InvalidArgument("search statistics counter overflow".into()))
}

#[derive(Debug, Default)]
struct CountingWriter {
    bytes: u64,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let length = u64::try_from(bytes.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "write length does not fit u64",
            )
        })?;
        self.bytes = self.bytes.checked_add(length).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "write length overflow")
        })?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialized_length(index: &InvertedIndex) -> Result<u64> {
    let mut counter = CountingWriter::default();
    index.write_to(&mut counter)?;
    Ok(counter.bytes)
}

#[derive(Debug)]
struct ChecksumWriter<'a> {
    checksum: &'a mut Checksum,
    bytes: u64,
}

impl Write for ChecksumWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let length = u64::try_from(bytes.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "write length does not fit u64",
            )
        })?;
        self.bytes = self.bytes.checked_add(length).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "write length overflow")
        })?;
        self.checksum.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn update_checksum(index: &InvertedIndex, checksum: &mut Checksum) -> Result<u64> {
    let mut writer = ChecksumWriter { checksum, bytes: 0 };
    index.write_to(&mut writer)?;
    Ok(writer.bytes)
}

fn checksum_reader(reader: &mut impl Read, length: u64) -> Result<u64> {
    let mut checksum = Checksum::new();
    let mut remaining = length;
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    while remaining > 0 {
        let chunk_length = usize::try_from(remaining.min(READ_CHUNK_BYTES as u64))
            .map_err(|_| Error::CorruptIndex("sharded read length does not fit usize".into()))?;
        read_exact_corrupt(reader, &mut chunk[..chunk_length], "sharded payload")?;
        checksum.update(&chunk[..chunk_length]);
        remaining -= chunk_length as u64;
    }
    Ok(checksum.finish())
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    read_exact_corrupt(reader, &mut bytes, "u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    read_exact_corrupt(reader, &mut bytes, "u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact_corrupt(reader: &mut impl Read, bytes: &mut [u8], label: &str) -> Result<()> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::CorruptIndex(format!("truncated {label}"))
        } else {
            Error::Io(error)
        }
    })
}

fn reject_trailing(reader: &mut impl Read) -> Result<()> {
    let mut byte = [0_u8; 1];
    match reader.read(&mut byte) {
        Ok(0) => Ok(()),
        Ok(_) => Err(Error::CorruptIndex("unexpected trailing bytes".into())),
        Err(error) => Err(Error::Io(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisMode, Analyzer};
    use crate::persistence::{encode_legacy_v1_for_test, encode_v2_for_test};
    use crate::query::{BooleanOperator, FieldFilter, PhraseFilter, SearchQuery};
    use crate::search::{Bm25Params, PruningStrategy, SearchHit};

    fn indexes(document_count: usize, shard_count: usize) -> (InvertedIndex, ShardedIndex) {
        let mut monolithic = IndexBuilder::new(Analyzer::default());
        let mut sharded = ShardedIndexBuilder::new(Analyzer::default(), shard_count).unwrap();
        for doc_id in 0..document_count {
            let common = if doc_id % 2 == 0 {
                "common common"
            } else {
                "common"
            };
            let rare = if doc_id % 7 == 0 { "rare" } else { "other" };
            let phrase = if doc_id % 11 == 0 {
                "blue sail"
            } else {
                "sail blue"
            };
            let document = Document::from_fields(
                format!("doc-{doc_id:04}"),
                [
                    ("title", format!("{rare} title {doc_id}")),
                    ("body", format!("{common} {rare} {phrase}")),
                    ("kind", (doc_id % 3).to_string()),
                ],
            )
            .unwrap();
            monolithic.add_document(document.clone()).unwrap();
            assert_eq!(sharded.add_document(document).unwrap() as usize, doc_id);
        }
        (monolithic.finish(), sharded.finish())
    }

    fn assert_same(left: &[SearchHit], right: &[SearchHit], context: &str) {
        assert_eq!(left.len(), right.len(), "length: {context}");
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.doc_id, right.doc_id, "doc id: {context}");
            assert_eq!(
                left.external_id, right.external_id,
                "external id: {context}"
            );
            assert_eq!(
                left.score.to_bits(),
                right.score.to_bits(),
                "score: {context}"
            );
            assert_eq!(
                left.explanation, right.explanation,
                "explanation: {context}"
            );
        }
    }

    #[test]
    fn round_robin_layout_preserves_global_document_order() {
        let (_, sharded) = indexes(10, 3);
        assert_eq!(sharded.shard_count(), 3);
        assert_eq!(sharded.document_count(), 10);
        assert_eq!(
            sharded
                .physical_shard_stats()
                .iter()
                .map(|stats| stats.documents)
                .collect::<Vec<_>>(),
            [4, 3, 3]
        );
        for doc_id in 0..10 {
            assert_eq!(
                sharded.document(doc_id).unwrap().external_id(),
                format!("doc-{doc_id:04}")
            );
        }
        assert!(sharded.document(10).is_none());
    }

    #[test]
    fn every_strategy_matches_monolithic_exhaustive_bit_for_bit() {
        let (monolithic, sharded) = indexes(173, 5);
        let queries = ["common rare", "title other", "blue sail", "missing common"];
        for text in queries {
            for operator in [BooleanOperator::Or, BooleanOperator::And] {
                for top_k in [1, 3, 10, 250] {
                    let query = SearchQuery::from_text(monolithic.analyzer(), text, None)
                        .unwrap()
                        .with_operator(operator);
                    let oracle = monolithic
                        .search(
                            &query,
                            SearchOptions {
                                top_k,
                                pruning: PruningStrategy::Exhaustive,
                                ..SearchOptions::default()
                            },
                        )
                        .unwrap();
                    for strategy in [
                        PruningStrategy::Exhaustive,
                        PruningStrategy::Wand,
                        PruningStrategy::BlockMaxWand,
                        PruningStrategy::MaxScore,
                    ] {
                        let actual = sharded
                            .search(
                                &query,
                                SearchOptions {
                                    top_k,
                                    pruning: strategy,
                                    ..SearchOptions::default()
                                },
                            )
                            .unwrap();
                        assert_same(
                            &oracle.hits,
                            &actual.hits,
                            &format!("{text}/{operator:?}/{top_k}/{strategy:?}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn global_statistics_cover_custom_bm25_filters_phrases_and_explanations() {
        let (monolithic, sharded) = indexes(191, 7);
        let query = SearchQuery::from_text(monolithic.analyzer(), "common rare", Some("body"))
            .unwrap()
            .with_filter(FieldFilter::exact("kind", "1").unwrap())
            .with_phrase(
                PhraseFilter::from_text(monolithic.analyzer(), "blue sail", Some("body".into()))
                    .unwrap(),
            );
        let options = SearchOptions {
            top_k: 20,
            pruning: PruningStrategy::BlockMaxWand,
            explain: true,
            bm25: Bm25Params { k1: 2.1, b: 0.35 },
        };
        let oracle = monolithic
            .search(
                &query,
                SearchOptions {
                    pruning: PruningStrategy::Exhaustive,
                    ..options
                },
            )
            .unwrap();
        let actual = sharded.search(&query, options).unwrap();
        assert_same(&oracle.hits, &actual.hits, "constrained explained query");
        assert_eq!(actual.stats.block_max_bounds_loaded, 0);
        assert!(actual.stats.block_max_postings_scanned > 0);
        let explanation = actual.hits[0].explanation.as_ref().unwrap();
        assert_eq!(
            explanation.total_score.to_bits(),
            actual.hits[0].score.to_bits()
        );
    }

    #[test]
    fn global_top_k_resolves_equal_scores_by_original_document_id() {
        let mut monolithic = IndexBuilder::new(Analyzer::default());
        let mut sharded = ShardedIndexBuilder::new(Analyzer::default(), 3).unwrap();
        for id in 0..12 {
            let document = Document::from_fields(format!("d{id}"), [("body", "same")]).unwrap();
            monolithic.add_document(document.clone()).unwrap();
            sharded.add_document(document).unwrap();
        }
        let monolithic = monolithic.finish();
        let sharded = sharded.finish();
        let query = SearchQuery::from_text(monolithic.analyzer(), "same", Some("body")).unwrap();
        let expected = monolithic
            .search(
                &query,
                SearchOptions {
                    top_k: 5,
                    pruning: PruningStrategy::Exhaustive,
                    ..SearchOptions::default()
                },
            )
            .unwrap();
        let actual = sharded
            .search(
                &query,
                SearchOptions {
                    top_k: 5,
                    pruning: PruningStrategy::MaxScore,
                    ..SearchOptions::default()
                },
            )
            .unwrap();
        assert_same(&expected.hits, &actual.hits, "cross-shard tie");
        assert_eq!(
            actual.hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn manual_cross_shard_bm25_oracle_uses_collection_statistics() {
        let mut builder = ShardedIndexBuilder::new(Analyzer::default(), 2).unwrap();
        for (id, body) in [
            ("long", "alpha alpha x"),
            ("short", "alpha"),
            ("none", "beta beta"),
        ] {
            builder
                .add_document(Document::from_fields(id, [("body", body)]).unwrap())
                .unwrap();
        }
        let index = builder.finish();
        let query = SearchQuery::from_text(index.analyzer(), "alpha", Some("body")).unwrap();
        let outcome = index
            .search(
                &query,
                SearchOptions {
                    top_k: 3,
                    pruning: PruningStrategy::Exhaustive,
                    explain: true,
                    ..SearchOptions::default()
                },
            )
            .unwrap();

        // Hand oracle, independent from the production scorer:
        // N=3, df=2, avgdl=(3+1+2)/3=2, k1=1.2, b=0.75.
        // The frozen decimal values are the direct Robertson BM25 formula.
        assert_eq!(
            outcome
                .hits
                .iter()
                .map(|hit| hit.external_id.as_str())
                .collect::<Vec<_>>(),
            ["short", "long"]
        );
        let expected = [0.590_861_705_337_496_3, 0.566_579_717_446_914_3];
        for (hit, expected) in outcome.hits.iter().zip(expected) {
            assert!((hit.score - expected).abs() < 1.0e-14);
            let term = &hit.explanation.as_ref().unwrap().terms[0];
            assert_eq!(term.document_frequency, 2);
            assert!((term.average_document_length - 2.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn maximum_cutoff_never_preallocates_from_untrusted_input() {
        let (monolithic, sharded) = indexes(1, MAX_SHARDS);
        let query = SearchQuery::from_text(monolithic.analyzer(), "common", None).unwrap();
        let options = SearchOptions {
            top_k: usize::MAX,
            ..SearchOptions::default()
        };
        assert_eq!(monolithic.search(&query, options).unwrap().hits.len(), 1);
        let actual = sharded.search(&query, options).unwrap();
        assert_eq!(actual.hits.len(), 1);
        assert_eq!(actual.hits[0].external_id, "doc-0000");
    }

    #[test]
    fn stats_deduplicate_terms_present_on_multiple_shards() {
        let (monolithic, sharded) = indexes(80, 4);
        assert_eq!(sharded.stats(), monolithic.stats());
        assert!(
            sharded
                .physical_shard_stats()
                .iter()
                .map(|stats| stats.terms)
                .sum::<usize>()
                > sharded.stats().terms
        );
    }

    #[test]
    fn persistence_is_deterministic_and_preserves_exact_search() {
        let (monolithic, sharded) = indexes(130, 4);
        let mut first = Vec::new();
        sharded.write_to(&mut first).unwrap();
        let restored = ShardedIndex::read_from(Cursor::new(&first)).unwrap();
        assert_eq!(
            restored.embedded_format_versions(),
            vec![PERSISTENCE_FORMAT_VERSION; 4]
        );
        let mut second = Vec::new();
        restored.write_to(&mut second).unwrap();
        assert_eq!(first, second);

        let query = SearchQuery::from_text(monolithic.analyzer(), "common rare", None).unwrap();
        let options = SearchOptions {
            top_k: 23,
            pruning: PruningStrategy::BlockMaxWand,
            explain: true,
            ..SearchOptions::default()
        };
        let expected = monolithic
            .search(
                &query,
                SearchOptions {
                    pruning: PruningStrategy::Exhaustive,
                    ..options
                },
            )
            .unwrap();
        let actual = restored.search(&query, options).unwrap();
        assert_same(&expected.hits, &actual.hits, "persisted shards");
    }

    #[test]
    fn container_records_mixed_legacy_embedded_versions() {
        fn one_document(id: &str, body: &str) -> InvertedIndex {
            let mut builder = IndexBuilder::new(Analyzer::default());
            builder
                .add_document(Document::from_fields(id, [("body", body)]).unwrap())
                .unwrap();
            builder.finish()
        }

        let v1 = encode_legacy_v1_for_test(&one_document("v1", "alpha old")).unwrap();
        let v2 = encode_v2_for_test(&one_document("v2", "beta old")).unwrap();
        let mut v3 = Vec::new();
        one_document("v3", "alpha current")
            .write_to(&mut v3)
            .unwrap();

        let embedded = [v1, v2, v3];
        let mut payload = Vec::new();
        write_u32(&mut payload, 3).unwrap();
        write_u64(&mut payload, 3).unwrap();
        for bytes in &embedded {
            write_u64(&mut payload, bytes.len() as u64).unwrap();
            payload.extend_from_slice(bytes);
        }
        let mut container = Vec::new();
        container.extend_from_slice(MAGIC);
        write_u32(&mut container, SHARDED_PERSISTENCE_FORMAT_VERSION).unwrap();
        write_u64(&mut container, payload.len() as u64).unwrap();
        write_u64(&mut container, crate::codec::checksum(&payload)).unwrap();
        container.extend_from_slice(&payload);

        let restored = ShardedIndex::read_from(Cursor::new(container)).unwrap();
        assert_eq!(restored.embedded_format_versions(), [1, 2, 3]);
        assert_eq!(restored.document(0).unwrap().external_id(), "v1");
        assert_eq!(restored.document(1).unwrap().external_id(), "v2");
        assert_eq!(restored.document(2).unwrap().external_id(), "v3");
        let query = SearchQuery::from_text(restored.analyzer(), "alpha", Some("body")).unwrap();
        assert_eq!(
            restored
                .search(&query, SearchOptions::default())
                .unwrap()
                .hits
                .iter()
                .map(|hit| hit.external_id.as_str())
                .collect::<Vec<_>>(),
            ["v1", "v3"]
        );
    }

    #[test]
    fn persistence_emits_shards_without_one_collection_sized_write() {
        #[derive(Default)]
        struct WriteShape {
            bytes: u64,
            largest_write: usize,
        }

        impl Write for WriteShape {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.bytes += bytes.len() as u64;
                self.largest_write = self.largest_write.max(bytes.len());
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let (_, sharded) = indexes(200, 4);
        let mut shape = WriteShape::default();
        sharded.write_to(&mut shape).unwrap();
        assert!(shape.bytes > 0);
        assert!((shape.largest_write as u64) < shape.bytes / 2);
    }

    #[test]
    fn save_load_and_header_inspection_round_trip() {
        let (_, sharded) = indexes(17, 3);
        let path = std::env::temp_dir().join(format!(
            "indexsail-shards-{}-{}.idx",
            std::process::id(),
            sharded.document_count()
        ));
        sharded.save(&path).unwrap();
        assert_eq!(
            persisted_sharded_format_version(&path).unwrap(),
            SHARDED_PERSISTENCE_FORMAT_VERSION
        );
        let restored = ShardedIndex::load(&path).unwrap();
        assert_eq!(restored.stats(), sharded.stats());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_counts_duplicates_and_cross_shard_analyzers() {
        assert!(ShardedIndexBuilder::new(Analyzer::default(), 0).is_err());
        assert!(ShardedIndexBuilder::new(Analyzer::default(), MAX_SHARDS + 1).is_err());
        assert!(
            ShardedIndex::from_loaded_shards(
                vec![IndexBuilder::new(Analyzer::default()).finish()],
                0,
                Vec::new()
            )
            .is_err()
        );
        let mut builder = ShardedIndexBuilder::new(Analyzer::default(), 2).unwrap();
        let document = Document::from_fields("duplicate", [("body", "one")]).unwrap();
        builder.add_document(document.clone()).unwrap();
        assert!(matches!(
            builder.add_document(document),
            Err(Error::DuplicateDocumentId(id)) if id == "duplicate"
        ));

        let unicode = IndexBuilder::new(Analyzer::default()).finish();
        let ascii = IndexBuilder::new(Analyzer::new(AnalysisMode::Ascii)).finish();
        assert!(
            ShardedIndex::from_loaded_shards(
                vec![unicode, ascii],
                0,
                vec![PERSISTENCE_FORMAT_VERSION; 2]
            )
            .is_err()
        );

        let duplicate = Document::from_fields("same", [("body", "term")]).unwrap();
        let mut left = IndexBuilder::new(Analyzer::default());
        left.add_document(duplicate.clone()).unwrap();
        let mut right = IndexBuilder::new(Analyzer::default());
        right.add_document(duplicate).unwrap();
        assert!(matches!(
            ShardedIndex::from_loaded_shards(
                vec![left.finish(), right.finish()],
                2,
                vec![PERSISTENCE_FORMAT_VERSION; 2]
            ),
            Err(Error::CorruptIndex(message)) if message.contains("duplicate external id")
        ));

        let mut oversized_left = IndexBuilder::new(Analyzer::default());
        oversized_left
            .add_document(Document::from_fields("a", [("body", "one")]).unwrap())
            .unwrap();
        oversized_left
            .add_document(Document::from_fields("b", [("body", "two")]).unwrap())
            .unwrap();
        assert!(matches!(
            ShardedIndex::from_loaded_shards(
                vec![
                    oversized_left.finish(),
                    IndexBuilder::new(Analyzer::default()).finish()
                ],
                2,
                vec![PERSISTENCE_FORMAT_VERSION; 2]
            ),
            Err(Error::CorruptIndex(message)) if message.contains("expected 1")
        ));
    }

    #[test]
    fn empty_and_single_shard_indexes_are_supported() {
        let empty = ShardedIndexBuilder::new(Analyzer::default(), 3)
            .unwrap()
            .finish();
        assert_eq!(empty.stats().documents, 0);
        let query = SearchQuery::from_text(empty.analyzer(), "anything", None).unwrap();
        assert!(
            empty
                .search(&query, SearchOptions::default())
                .unwrap()
                .hits
                .is_empty()
        );

        let (monolithic, single) = indexes(40, 1);
        let query = SearchQuery::from_text(monolithic.analyzer(), "common rare", None).unwrap();
        let expected = monolithic.search(&query, SearchOptions::default()).unwrap();
        let actual = single.search(&query, SearchOptions::default()).unwrap();
        assert_same(&expected.hits, &actual.hits, "one shard");
    }

    #[test]
    fn checksum_version_truncation_and_trailing_data_are_rejected() {
        let (_, sharded) = indexes(8, 2);
        let mut bytes = Vec::new();
        sharded.write_to(&mut bytes).unwrap();

        let mut corrupt = bytes.clone();
        *corrupt.last_mut().unwrap() ^= 0x01;
        assert!(matches!(
            ShardedIndex::read_from(Cursor::new(corrupt)),
            Err(Error::CorruptIndex(message)) if message.contains("checksum")
        ));

        let mut unknown = bytes.clone();
        unknown[8..12].copy_from_slice(&99_u32.to_le_bytes());
        assert!(matches!(
            ShardedIndex::read_from(Cursor::new(unknown)),
            Err(Error::UnsupportedVersion(99))
        ));

        let mut truncated = bytes.clone();
        truncated.truncate(truncated.len() - 1);
        assert!(matches!(
            ShardedIndex::read_from(Cursor::new(truncated)),
            Err(Error::CorruptIndex(_))
        ));

        let mut trailing = bytes;
        trailing.push(0);
        assert!(matches!(
            ShardedIndex::read_from(Cursor::new(trailing)),
            Err(Error::CorruptIndex(message)) if message.contains("trailing")
        ));
    }

    #[test]
    fn truncated_maximum_payload_is_rejected_in_bounded_chunks() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        write_u32(&mut bytes, SHARDED_PERSISTENCE_FORMAT_VERSION).unwrap();
        write_u64(&mut bytes, MAX_CONTAINER_PAYLOAD_BYTES).unwrap();
        write_u64(&mut bytes, 0).unwrap();
        assert!(matches!(
            ShardedIndex::read_from(Cursor::new(bytes)),
            Err(Error::CorruptIndex(message)) if message.contains("truncated")
        ));
    }
}
