use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::analysis::{AnalysisMode, Analyzer};
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::{InvertedIndex, Posting, TermKey};

const MAGIC: &[u8; 8] = b"IDXSAL01";
const VERSION: u32 = 1;
const MAX_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_COLLECTION_ITEMS: usize = 20_000_000;

impl InvertedIndex {
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
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    pub fn write_to(&self, mut writer: impl Write) -> Result<()> {
        writer.write_all(MAGIC)?;
        write_u32(&mut writer, VERSION)?;
        write_u8(&mut writer, self.analyzer.mode().wire_value())?;
        write_len(&mut writer, self.documents.len(), "documents")?;

        for (document, lengths) in self.documents.iter().zip(&self.field_lengths) {
            write_string(&mut writer, document.external_id())?;
            write_len(&mut writer, document.fields().len(), "document fields")?;
            for (field, value) in document.fields() {
                write_string(&mut writer, field)?;
                write_string(&mut writer, value)?;
            }
            write_len(&mut writer, lengths.len(), "field lengths")?;
            for (field, length) in lengths {
                write_string(&mut writer, field)?;
                write_u32(&mut writer, *length)?;
            }
        }

        write_len(&mut writer, self.postings.len(), "dictionary terms")?;
        for (key, postings) in &self.postings {
            write_string(&mut writer, &key.field)?;
            write_string(&mut writer, &key.term)?;
            write_len(&mut writer, postings.len(), "postings")?;
            for posting in postings {
                write_u32(&mut writer, posting.doc_id)?;
                write_u32(&mut writer, posting.term_frequency)?;
                write_len(&mut writer, posting.positions.len(), "positions")?;
                for position in &posting.positions {
                    write_u32(&mut writer, *position)?;
                }
            }
        }
        Ok(())
    }

    pub fn read_from(mut reader: impl Read) -> Result<Self> {
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::CorruptIndex("invalid file signature".into()));
        }
        let version = read_u32(&mut reader)?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let mode_value = read_u8(&mut reader)?;
        let mode = AnalysisMode::from_wire(mode_value)
            .ok_or_else(|| Error::CorruptIndex(format!("unknown analyzer mode {mode_value}")))?;
        let document_count = read_len(&mut reader, "documents")?;
        let mut documents = Vec::with_capacity(document_count);
        let mut all_lengths = Vec::with_capacity(document_count);

        for _ in 0..document_count {
            let external_id = read_string(&mut reader)?;
            let field_count = read_len(&mut reader, "document fields")?;
            let mut fields = Vec::with_capacity(field_count);
            for _ in 0..field_count {
                fields.push((read_string(&mut reader)?, read_string(&mut reader)?));
            }
            let document = Document::from_fields(external_id, fields)
                .map_err(|error| Error::CorruptIndex(error.to_string()))?;
            documents.push(document);

            let length_count = read_len(&mut reader, "field lengths")?;
            let mut lengths = BTreeMap::new();
            for _ in 0..length_count {
                let field = read_string(&mut reader)?;
                let length = read_u32(&mut reader)?;
                if lengths.insert(field.clone(), length).is_some() {
                    return Err(Error::CorruptIndex(format!(
                        "duplicate field length entry '{field}'"
                    )));
                }
            }
            all_lengths.push(lengths);
        }

        let term_count = read_len(&mut reader, "dictionary terms")?;
        let mut postings_by_term = BTreeMap::new();
        for _ in 0..term_count {
            let key = TermKey {
                field: read_string(&mut reader)?,
                term: read_string(&mut reader)?,
            };
            let posting_count = read_len(&mut reader, "postings")?;
            let mut postings = Vec::with_capacity(posting_count);
            for _ in 0..posting_count {
                let doc_id = read_u32(&mut reader)?;
                let term_frequency = read_u32(&mut reader)?;
                let position_count = read_len(&mut reader, "positions")?;
                let mut positions = Vec::with_capacity(position_count);
                for _ in 0..position_count {
                    positions.push(read_u32(&mut reader)?);
                }
                postings.push(Posting {
                    doc_id,
                    term_frequency,
                    positions,
                });
            }
            if postings_by_term.insert(key.clone(), postings).is_some() {
                return Err(Error::CorruptIndex(format!(
                    "duplicate dictionary key '{}:{}'",
                    key.field, key.term
                )));
            }
        }

        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing)? != 0 {
            return Err(Error::CorruptIndex("trailing bytes after index".into()));
        }
        InvertedIndex::from_parts(
            Analyzer::new(mode),
            documents,
            all_lengths,
            postings_by_term,
        )
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

fn write_len(writer: &mut impl Write, value: usize, label: &str) -> Result<()> {
    let value = u32::try_from(value).map_err(|_| {
        Error::InvalidArgument(format!("{label} count exceeds binary format limit"))
    })?;
    write_u32(writer, value)
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    write_len(writer, value.len(), "string bytes")?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0_u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_len(reader: &mut impl Read, label: &str) -> Result<usize> {
    let length = read_u32(reader)? as usize;
    if length > MAX_COLLECTION_ITEMS {
        return Err(Error::CorruptIndex(format!(
            "{label} count {length} exceeds safety limit"
        )));
    }
    Ok(length)
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let length = read_u32(reader)? as usize;
    if length > MAX_STRING_BYTES {
        return Err(Error::CorruptIndex(format!(
            "string length {length} exceeds safety limit"
        )));
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| Error::CorruptIndex("string is not valid UTF-8".into()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
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
    fn analyzer_mode_round_trips() {
        let restored =
            InvertedIndex::read_from(Cursor::new(bytes(&sample_index(AnalysisMode::Ascii))))
                .unwrap();
        assert_eq!(restored.analyzer().mode(), AnalysisMode::Ascii);
        assert!(restored.postings("title", "caf").is_some());
    }

    #[test]
    fn serialization_is_deterministic() {
        let index = sample_index(AnalysisMode::Unicode);
        assert_eq!(bytes(&index), bytes(&index));
        assert_eq!(bytes(&index), bytes(&sample_index(AnalysisMode::Unicode)));
    }

    #[test]
    fn empty_index_round_trips() {
        let index = IndexBuilder::new(Analyzer::default()).finish();
        let restored = InvertedIndex::read_from(Cursor::new(bytes(&index))).unwrap();
        assert_eq!(restored.stats(), index.stats());
    }

    #[test]
    fn rejects_invalid_signature() {
        let mut data = bytes(&sample_index(AnalysisMode::Unicode));
        data[0] ^= 0xff;
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(data)),
            Err(Error::CorruptIndex(_))
        ));
    }

    #[test]
    fn rejects_unknown_version() {
        let mut data = bytes(&sample_index(AnalysisMode::Unicode));
        data[8..12].copy_from_slice(&999_u32.to_le_bytes());
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(data)),
            Err(Error::UnsupportedVersion(999))
        ));
    }

    #[test]
    fn rejects_truncation_and_trailing_bytes() {
        let mut truncated = bytes(&sample_index(AnalysisMode::Unicode));
        truncated.truncate(truncated.len() - 2);
        assert!(InvertedIndex::read_from(Cursor::new(truncated)).is_err());

        let mut trailing = bytes(&sample_index(AnalysisMode::Unicode));
        trailing.push(1);
        assert!(matches!(
            InvertedIndex::read_from(Cursor::new(trailing)),
            Err(Error::CorruptIndex(_))
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
        let restored = InvertedIndex::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(restored.stats(), original.stats());
        assert_eq!(restored.documents(), original.documents());
    }
}
