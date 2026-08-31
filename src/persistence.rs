//! Versioned binary persistence with checksummed, compressed posting blocks.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::Path;

use crate::analysis::{AnalysisMode, Analyzer};
use crate::codec::{checksum, decode_postings, encode_postings};
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::{InvertedIndex, Posting, TermKey};

const MAGIC_V1: &[u8; 8] = b"IDXSAL01";
const MAGIC_V2: &[u8; 8] = b"IDXSAL02";
const LEGACY_VERSION: u32 = 1;
pub const PERSISTENCE_FORMAT_VERSION: u32 = 2;
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_COLLECTION_ITEMS: usize = 20_000_000;
const MAX_POSTING_BLOCK_BYTES: usize = 512 * 1024 * 1024;
const MAX_INDEX_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const READ_CHUNK_BYTES: usize = 8 * 1024;
const INITIAL_COLLECTION_CAPACITY: usize = 1_024;

/// Read and validate only the persisted signature/version header.
pub fn persisted_format_version(path: impl AsRef<Path>) -> Result<u32> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 8];
    read_exact_corrupt(&mut reader, &mut magic, "file signature")?;
    let version = read_u32(&mut reader)?;
    match &magic {
        value if value == MAGIC_V1 && version == LEGACY_VERSION => Ok(version),
        value if value == MAGIC_V2 && version == PERSISTENCE_FORMAT_VERSION => Ok(version),
        value if value == MAGIC_V1 || value == MAGIC_V2 => Err(Error::UnsupportedVersion(version)),
        _ => Err(Error::CorruptIndex("invalid file signature".into())),
    }
}

impl InvertedIndex {
    /// Persist a snapshot to a new or truncated file.
    ///
    /// Format version 2 checksums the complete payload and delta/varbyte
    /// encodes document IDs, term frequencies, and term positions.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::read_from(reader)
    }

    pub fn write_to(&self, mut writer: impl Write) -> Result<()> {
        let mut payload = Vec::new();
        write_documents(self, &mut payload)?;
        write_collection_len(&mut payload, self.postings.len(), "dictionary terms")?;
        for (key, postings) in &self.postings {
            write_string(&mut payload, &key.field)?;
            write_string(&mut payload, &key.term)?;
            write_collection_len(&mut payload, postings.len(), "postings")?;
            let block = encode_postings(postings)?;
            if block.len() > MAX_POSTING_BLOCK_BYTES {
                return Err(Error::InvalidArgument(format!(
                    "compressed posting block exceeds {MAX_POSTING_BLOCK_BYTES} byte safety limit"
                )));
            }
            write_len(&mut payload, block.len(), "compressed posting bytes")?;
            payload.write_all(&block)?;
        }

        let payload_length = u64::try_from(payload.len())
            .map_err(|_| Error::InvalidArgument("index payload does not fit u64".into()))?;
        if payload_length > MAX_INDEX_PAYLOAD_BYTES {
            return Err(Error::InvalidArgument(format!(
                "index payload exceeds {MAX_INDEX_PAYLOAD_BYTES} byte safety limit"
            )));
        }
        writer.write_all(MAGIC_V2)?;
        write_u32(&mut writer, PERSISTENCE_FORMAT_VERSION)?;
        write_u64(&mut writer, payload_length)?;
        write_u64(&mut writer, checksum(&payload))?;
        writer.write_all(&payload)?;
        Ok(())
    }

    /// Load current version 2 indexes and legacy version 1 indexes.
    pub fn read_from(mut reader: impl Read) -> Result<Self> {
        let mut magic = [0_u8; 8];
        read_exact_corrupt(&mut reader, &mut magic, "file signature")?;
        match &magic {
            value if value == MAGIC_V2 => read_v2(&mut reader),
            value if value == MAGIC_V1 => read_v1(&mut reader),
            _ => Err(Error::CorruptIndex("invalid file signature".into())),
        }
    }
}

fn write_documents(index: &InvertedIndex, writer: &mut impl Write) -> Result<()> {
    write_u8(writer, index.analyzer.mode().wire_value())?;
    write_collection_len(writer, index.documents.len(), "documents")?;
    for (document, lengths) in index.documents.iter().zip(&index.field_lengths) {
        write_string(writer, document.external_id())?;
        write_collection_len(writer, document.fields().len(), "document fields")?;
        for (field, value) in document.fields() {
            write_string(writer, field)?;
            write_string(writer, value)?;
        }
        write_collection_len(writer, lengths.len(), "field lengths")?;
        for (field, length) in lengths {
            write_string(writer, field)?;
            write_u32(writer, *length)?;
        }
    }
    Ok(())
}

fn read_v2(reader: &mut impl Read) -> Result<InvertedIndex> {
    let version = read_u32(reader)?;
    if version != PERSISTENCE_FORMAT_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    let payload_length = read_u64(reader)?;
    if payload_length > MAX_INDEX_PAYLOAD_BYTES {
        return Err(Error::CorruptIndex(format!(
            "index payload length {payload_length} exceeds safety limit"
        )));
    }
    let payload_length = usize::try_from(payload_length)
        .map_err(|_| Error::CorruptIndex("index payload does not fit this platform".into()))?;
    let expected_checksum = read_u64(reader)?;
    let payload = read_bytes_fallible(reader, payload_length, "index payload")?;
    reject_trailing(reader)?;
    let actual_checksum = checksum(&payload);
    if actual_checksum != expected_checksum {
        return Err(Error::CorruptIndex(format!(
            "payload checksum mismatch: expected {expected_checksum:016x}, got {actual_checksum:016x}"
        )));
    }

    let mut payload_reader = Cursor::new(payload.as_slice());
    let (analyzer, documents, all_lengths) = read_documents(&mut payload_reader)?;
    let term_count = read_len(&mut payload_reader, "dictionary terms")?;
    let mut postings_by_term = BTreeMap::new();
    for _ in 0..term_count {
        let key = TermKey {
            field: read_string(&mut payload_reader)?,
            term: read_string(&mut payload_reader)?,
        };
        let posting_count = read_len(&mut payload_reader, "postings")?;
        let block_length = read_bounded_len(
            &mut payload_reader,
            "compressed posting bytes",
            MAX_POSTING_BLOCK_BYTES,
        )?;
        let block_start = usize::try_from(payload_reader.position())
            .map_err(|_| Error::CorruptIndex("posting block offset does not fit usize".into()))?;
        let block_end = block_start
            .checked_add(block_length)
            .ok_or_else(|| Error::CorruptIndex("posting block offset overflow".into()))?;
        let block = payload_reader
            .get_ref()
            .get(block_start..block_end)
            .ok_or_else(|| Error::CorruptIndex("truncated compressed posting block".into()))?;
        let postings = decode_postings(block, posting_count)?;
        payload_reader
            .set_position(u64::try_from(block_end).map_err(|_| {
                Error::CorruptIndex("posting block offset does not fit u64".into())
            })?);
        if postings_by_term.insert(key.clone(), postings).is_some() {
            return Err(Error::CorruptIndex(format!(
                "duplicate dictionary key '{}:{}'",
                key.field, key.term
            )));
        }
    }
    reject_trailing(&mut payload_reader)?;
    InvertedIndex::from_parts(analyzer, documents, all_lengths, postings_by_term)
}

fn read_v1(reader: &mut impl Read) -> Result<InvertedIndex> {
    let version = read_u32(reader)?;
    if version != LEGACY_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    let (analyzer, documents, all_lengths) = read_documents(reader)?;
    let term_count = read_len(reader, "dictionary terms")?;
    let mut postings_by_term = BTreeMap::new();
    for _ in 0..term_count {
        let key = TermKey {
            field: read_string(reader)?,
            term: read_string(reader)?,
        };
        let posting_count = read_len(reader, "postings")?;
        let mut postings = fallible_vec(posting_count, "legacy postings")?;
        for _ in 0..posting_count {
            let doc_id = read_u32(reader)?;
            let term_frequency = read_u32(reader)?;
            let position_count = read_len(reader, "positions")?;
            let mut positions = fallible_vec(position_count, "legacy positions")?;
            for _ in 0..position_count {
                push_fallible(&mut positions, read_u32(reader)?, "legacy positions")?;
            }
            push_fallible(
                &mut postings,
                Posting {
                    doc_id,
                    term_frequency,
                    positions,
                },
                "legacy postings",
            )?;
        }
        if postings_by_term.insert(key.clone(), postings).is_some() {
            return Err(Error::CorruptIndex(format!(
                "duplicate dictionary key '{}:{}'",
                key.field, key.term
            )));
        }
    }
    reject_trailing(reader)?;
    InvertedIndex::from_parts(analyzer, documents, all_lengths, postings_by_term)
}

type StoredDocuments = (Analyzer, Vec<Document>, Vec<BTreeMap<String, u32>>);

fn read_documents(reader: &mut impl Read) -> Result<StoredDocuments> {
    let mode_value = read_u8(reader)?;
    let mode = AnalysisMode::from_wire(mode_value)
        .ok_or_else(|| Error::CorruptIndex(format!("unknown analyzer mode {mode_value}")))?;
    let document_count = read_len(reader, "documents")?;
    let mut documents = fallible_vec(document_count, "documents")?;
    let mut all_lengths = fallible_vec(document_count, "field-length maps")?;
    for _ in 0..document_count {
        let external_id = read_string(reader)?;
        let field_count = read_len(reader, "document fields")?;
        let mut fields = fallible_vec(field_count, "document fields")?;
        for _ in 0..field_count {
            let field = (read_string(reader)?, read_string(reader)?);
            push_fallible(&mut fields, field, "document fields")?;
        }
        let document = Document::from_fields(external_id, fields)
            .map_err(|error| Error::CorruptIndex(error.to_string()))?;
        push_fallible(&mut documents, document, "documents")?;

        let length_count = read_len(reader, "field lengths")?;
        let mut lengths = BTreeMap::new();
        for _ in 0..length_count {
            let field = read_string(reader)?;
            let length = read_u32(reader)?;
            if lengths.insert(field.clone(), length).is_some() {
                return Err(Error::CorruptIndex(format!(
                    "duplicate field length entry '{field}'"
                )));
            }
        }
        push_fallible(&mut all_lengths, lengths, "field-length maps")?;
    }
    Ok((Analyzer::new(mode), documents, all_lengths))
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
    let value = u32::try_from(value).map_err(|_| {
        Error::InvalidArgument(format!("{label} count exceeds binary format limit"))
    })?;
    write_u32(writer, value)
}

fn write_collection_len(writer: &mut impl Write, value: usize, label: &str) -> Result<()> {
    if value > MAX_COLLECTION_ITEMS {
        return Err(Error::InvalidArgument(format!(
            "{label} count exceeds {MAX_COLLECTION_ITEMS} item safety limit"
        )));
    }
    write_len(writer, value, label)
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    if value.len() > MAX_STRING_BYTES {
        return Err(Error::InvalidArgument(format!(
            "string exceeds {MAX_STRING_BYTES} byte safety limit"
        )));
    }
    write_len(writer, value.len(), "string bytes")?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0_u8; 1];
    read_exact_corrupt(reader, &mut bytes, "u8")?;
    Ok(bytes[0])
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

fn read_len(reader: &mut impl Read, label: &str) -> Result<usize> {
    read_bounded_len(reader, label, MAX_COLLECTION_ITEMS)
}

fn read_bounded_len(reader: &mut impl Read, label: &str, limit: usize) -> Result<usize> {
    let length = read_u32(reader)? as usize;
    if length > limit {
        return Err(Error::CorruptIndex(format!(
            "{label} count {length} exceeds safety limit"
        )));
    }
    Ok(length)
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let length = read_bounded_len(reader, "string bytes", MAX_STRING_BYTES)?;
    let bytes = read_bytes_fallible(reader, length, "UTF-8 string")?;
    String::from_utf8(bytes).map_err(|_| Error::CorruptIndex("string is not valid UTF-8".into()))
}

fn read_bytes_fallible(reader: &mut impl Read, length: usize, label: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut remaining = length;
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    while remaining > 0 {
        let chunk_length = remaining.min(chunk.len());
        bytes
            .try_reserve(chunk_length)
            .map_err(|_| Error::CorruptIndex(format!("{label} cannot be allocated safely")))?;
        read_exact_corrupt(reader, &mut chunk[..chunk_length], label)?;
        bytes.extend_from_slice(&chunk[..chunk_length]);
        remaining -= chunk_length;
    }
    Ok(bytes)
}

fn fallible_vec<T>(expected: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve(expected.min(INITIAL_COLLECTION_CAPACITY))
        .map_err(|_| Error::CorruptIndex(format!("{label} cannot be allocated safely")))?;
    Ok(values)
}

fn push_fallible<T>(values: &mut Vec<T>, value: T, label: &str) -> Result<()> {
    values
        .try_reserve(1)
        .map_err(|_| Error::CorruptIndex(format!("{label} cannot be allocated safely")))?;
    values.push(value);
    Ok(())
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
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(Error::CorruptIndex("trailing bytes after index".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::analysis::AnalysisMode;
    use crate::index::IndexBuilder;

    fn sample_index(mode: AnalysisMode) -> InvertedIndex {
        let mut builder = IndexBuilder::new(Analyzer::new(mode));
        builder
            .add_document(
                Document::from_fields("one", [("title", "Café Search"), ("body", "red blue red")])
                    .unwrap(),
            )
            .unwrap();
        builder
            .add_document(Document::from_fields("two", [("body", "blue green")]).unwrap())
            .unwrap();
        builder.finish()
    }

    fn bytes(index: &InvertedIndex) -> Vec<u8> {
        let mut output = Vec::new();
        index.write_to(&mut output).unwrap();
        output
    }

    fn legacy_bytes(index: &InvertedIndex) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(MAGIC_V1);
        write_u32(&mut output, LEGACY_VERSION).unwrap();
        write_documents(index, &mut output).unwrap();
        write_len(&mut output, index.postings.len(), "dictionary terms").unwrap();
        for (key, postings) in &index.postings {
            write_string(&mut output, &key.field).unwrap();
            write_string(&mut output, &key.term).unwrap();
            write_len(&mut output, postings.len(), "postings").unwrap();
            for posting in postings {
                write_u32(&mut output, posting.doc_id).unwrap();
                write_u32(&mut output, posting.term_frequency).unwrap();
                write_len(&mut output, posting.positions.len(), "positions").unwrap();
                for &position in &posting.positions {
                    write_u32(&mut output, position).unwrap();
                }
            }
        }
        output
    }

    #[test]
    fn binary_round_trip_preserves_documents_postings_and_stats() {
        let original = sample_index(AnalysisMode::Unicode);
        let restored = InvertedIndex::read_from(Cursor::new(bytes(&original))).unwrap();
        assert_eq!(restored.documents(), original.documents());
        assert_eq!(restored.stats(), original.stats());
        assert_eq!(
            restored.postings("body", "red"),
            original.postings("body", "red")
        );
    }

    #[test]
    fn writer_uses_version_two_magic_and_is_deterministic() {
        let index = sample_index(AnalysisMode::Unicode);
        let first = bytes(&index);
        assert_eq!(&first[..8], MAGIC_V2);
        assert_eq!(u32::from_le_bytes(first[8..12].try_into().unwrap()), 2);
        assert_eq!(first, bytes(&index));
        assert_eq!(first, bytes(&sample_index(AnalysisMode::Unicode)));
    }

    #[test]
    fn reads_legacy_version_one_indexes() {
        let original = sample_index(AnalysisMode::Ascii);
        let restored = InvertedIndex::read_from(Cursor::new(legacy_bytes(&original))).unwrap();
        assert_eq!(restored.documents(), original.documents());
        assert_eq!(
            restored.postings("body", "red"),
            original.postings("body", "red")
        );
        assert_eq!(restored.analyzer().mode(), AnalysisMode::Ascii);
    }

    #[test]
    fn checksum_rejects_single_byte_corruption_before_parsing() {
        let mut data = bytes(&sample_index(AnalysisMode::Unicode));
        let last = data.len() - 1;
        data[last] ^= 1;
        let error = InvertedIndex::read_from(Cursor::new(data)).unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn analyzer_mode_round_trips() {
        let restored =
            InvertedIndex::read_from(Cursor::new(bytes(&sample_index(AnalysisMode::Ascii))))
                .unwrap();
        assert_eq!(restored.analyzer().mode(), AnalysisMode::Ascii);
        assert!(restored.postings("title", "caf").is_some());
    }

    #[test]
    fn empty_index_round_trips() {
        let index = IndexBuilder::new(Analyzer::default()).finish();
        let restored = InvertedIndex::read_from(Cursor::new(bytes(&index))).unwrap();
        assert_eq!(restored.stats(), index.stats());
    }

    #[test]
    fn rejects_invalid_signature_and_unknown_version() {
        let mut invalid = bytes(&sample_index(AnalysisMode::Unicode));
        invalid[0] ^= 0xff;
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(invalid)),
            Err(Error::CorruptIndex(_))
        ));

        let mut unknown = bytes(&sample_index(AnalysisMode::Unicode));
        unknown[8..12].copy_from_slice(&999_u32.to_le_bytes());
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(unknown)),
            Err(Error::UnsupportedVersion(999))
        ));
    }

    #[test]
    fn rejects_truncation_and_trailing_bytes() {
        let mut truncated = bytes(&sample_index(AnalysisMode::Unicode));
        truncated.truncate(truncated.len() - 2);
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(truncated)),
            Err(Error::CorruptIndex(message)) if message.contains("truncated")
        ));

        let mut trailing = bytes(&sample_index(AnalysisMode::Unicode));
        trailing.push(1);
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(trailing)),
            Err(Error::CorruptIndex(message)) if message.contains("trailing")
        ));
    }

    #[test]
    fn rejects_truncated_large_payload_without_preallocating_declared_length() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC_V2);
        write_u32(&mut data, PERSISTENCE_FORMAT_VERSION).unwrap();
        write_u64(&mut data, MAX_INDEX_PAYLOAD_BYTES).unwrap();
        write_u64(&mut data, 0).unwrap();

        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(data)),
            Err(Error::CorruptIndex(message)) if message.contains("truncated index payload")
        ));
    }

    #[test]
    fn file_api_round_trips_without_external_dependencies() {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("indexsail-{}-{suffix}.idx", std::process::id()));
        let original = sample_index(AnalysisMode::Unicode);
        original.save(&path).unwrap();
        assert_eq!(persisted_format_version(&path).unwrap(), 2);
        let restored = InvertedIndex::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(restored.stats(), original.stats());
        assert_eq!(restored.documents(), original.documents());
    }
}
