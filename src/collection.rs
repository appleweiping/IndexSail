//! Bounded, streaming collection adapters shared by the native and forward-index pipelines.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use same_file::Handle;

use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};

use crate::analysis::Analyzer;
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::{IndexBuilder, InvertedIndex};
use crate::shard::{ShardedIndex, ShardedIndexBuilder};
use crate::trec::visit_trec_reader_with_limit;

/// Supported source collection protocols.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionFormat {
    /// A header-led table: `id<TAB>field-name...` followed by one document per line.
    Tsv,
    /// The documented TREC SGML subset used by [`crate::trec`].
    Trec,
    /// Canonical or PISA-compatible JSON Lines records.
    Jsonl,
}

impl CollectionFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "tsv" => Ok(Self::Tsv),
            "trec" => Ok(Self::Trec),
            "jsonl" => Ok(Self::Jsonl),
            _ => Err(Error::InvalidArgument(format!(
                "unknown collection format '{value}', expected tsv, trec, or jsonl"
            ))),
        }
    }
}

/// Explicit resource limits for untrusted collection inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionLimits {
    pub max_documents: usize,
    pub max_fields_per_document: usize,
    pub max_record_bytes: usize,
    pub max_input_bytes: u64,
}

impl Default for CollectionLimits {
    fn default() -> Self {
        Self {
            max_documents: 20_000_000,
            max_fields_per_document: 64,
            max_record_bytes: 64 * 1024 * 1024,
            max_input_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl CollectionLimits {
    fn validate(self) -> Result<Self> {
        if self.max_documents == 0 {
            return Err(Error::InvalidArgument(
                "collection max_documents must be positive".into(),
            ));
        }
        if self.max_fields_per_document == 0 {
            return Err(Error::InvalidArgument(
                "collection max_fields_per_document must be positive".into(),
            ));
        }
        if self.max_record_bytes == 0 {
            return Err(Error::InvalidArgument(
                "collection max_record_bytes must be positive".into(),
            ));
        }
        if self.max_input_bytes == 0 {
            return Err(Error::InvalidArgument(
                "collection max_input_bytes must be positive".into(),
            ));
        }
        Ok(self)
    }
}

/// Limit bytes actually read, including bytes appended after the initial stat.
pub(crate) struct InputLimitReader<R> {
    inner: R,
    maximum: u64,
    read: u64,
}

impl<R> InputLimitReader<R> {
    pub(crate) const fn new(inner: R, maximum: u64) -> Self {
        Self {
            inner,
            maximum,
            read: 0,
        }
    }

    pub(crate) const fn bytes_read(&self) -> u64 {
        self.read
    }

    pub(crate) fn inner(&self) -> &R {
        &self.inner
    }
}

impl<R: Read> Read for InputLimitReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let remaining = self.maximum.saturating_sub(self.read);
        let probe = usize::try_from(remaining.saturating_add(1))
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        let count = self.inner.read(&mut buffer[..probe])?;
        self.read = self.read.checked_add(count as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "input byte count overflow")
        })?;
        if self.read > self.maximum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("input exceeds {} byte limit", self.maximum),
            ));
        }
        Ok(count)
    }
}

/// Materialize a bounded collection. Prefer [`index_collection`] when the
/// documents only need to be indexed, because that path streams records.
pub fn load_collection(
    path: impl AsRef<Path>,
    format: CollectionFormat,
    limits: CollectionLimits,
) -> Result<Vec<Document>> {
    let mut documents = Vec::new();
    visit_collection(path, format, limits, |document| {
        documents.try_reserve(1).map_err(|_| {
            Error::InvalidDocument("could not allocate collection document list".into())
        })?;
        documents.push(document);
        Ok(())
    })?;
    Ok(documents)
}

/// Stream a collection directly into one positional inverted index.
pub fn index_collection(
    path: impl AsRef<Path>,
    analyzer: Analyzer,
    format: CollectionFormat,
    limits: CollectionLimits,
) -> Result<InvertedIndex> {
    let mut builder = IndexBuilder::new(analyzer);
    visit_collection(path, format, limits, |document| {
        builder.add_document(document).map(|_| ())
    })?;
    Ok(builder.finish())
}

/// Stream a collection directly into deterministic physical shards.
pub fn index_collection_sharded(
    path: impl AsRef<Path>,
    analyzer: Analyzer,
    format: CollectionFormat,
    limits: CollectionLimits,
    shard_count: usize,
) -> Result<ShardedIndex> {
    let mut builder = ShardedIndexBuilder::new(analyzer, shard_count)?;
    visit_collection(path, format, limits, |document| {
        builder.add_document(document).map(|_| ())
    })?;
    Ok(builder.finish())
}

pub(crate) fn visit_collection(
    path: impl AsRef<Path>,
    format: CollectionFormat,
    limits: CollectionLimits,
    mut visit: impl FnMut(Document) -> Result<()>,
) -> Result<()> {
    let limits = limits.validate()?;
    let path = path.as_ref();
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(Error::InvalidArgument(format!(
            "collection path '{}' is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > limits.max_input_bytes {
        return Err(Error::InvalidDocument(format!(
            "collection has {} bytes; limit is {}",
            metadata.len(),
            limits.max_input_bytes
        )));
    }

    let identity = Handle::from_file(file.try_clone()?)?;
    if Handle::from_path(path)? != identity {
        return Err(Error::InvalidDocument(
            "collection path changed while it was being opened".into(),
        ));
    }
    let mut reader = BufReader::new(InputLimitReader::new(file, limits.max_input_bytes));
    let mut document_count = 0_usize;
    let mut bounded_visit = |document: Document| {
        if document.fields().len() > limits.max_fields_per_document {
            return Err(Error::InvalidDocument(format!(
                "document '{}' has {} fields; limit is {}",
                document.external_id(),
                document.fields().len(),
                limits.max_fields_per_document
            )));
        }
        document_count = document_count
            .checked_add(1)
            .ok_or_else(|| Error::InvalidDocument("collection document count overflow".into()))?;
        if document_count > limits.max_documents {
            return Err(Error::InvalidDocument(format!(
                "collection exceeds {} document limit",
                limits.max_documents
            )));
        }
        visit(document)
    };

    match format {
        CollectionFormat::Tsv => visit_tsv_reader(&mut reader, limits, &mut bounded_visit)?,
        CollectionFormat::Trec => {
            visit_trec_reader_with_limit(&mut reader, limits.max_record_bytes, &mut bounded_visit)?;
        }
        CollectionFormat::Jsonl => visit_jsonl_reader(&mut reader, limits, &mut bounded_visit)?,
    }
    let final_metadata = reader.get_ref().inner().metadata()?;
    if reader.get_ref().bytes_read() != metadata.len()
        || final_metadata.len() != metadata.len()
        || final_metadata.modified().ok() != metadata.modified().ok()
        || Handle::from_path(path).ok().as_ref() != Some(&identity)
    {
        return Err(Error::InvalidDocument(
            "collection changed while it was being read".into(),
        ));
    }
    if document_count == 0 {
        return Err(Error::InvalidDocument(
            "collection contains no documents".into(),
        ));
    }
    Ok(())
}

fn visit_tsv_reader(
    mut reader: impl BufRead,
    limits: CollectionLimits,
    visit: &mut impl FnMut(Document) -> Result<()>,
) -> Result<()> {
    let Some(header) = read_bounded_line(&mut reader, limits.max_record_bytes, 1)? else {
        return Err(Error::InvalidDocument("TSV input has no header".into()));
    };
    let columns = trim_line_ending(&header).split('\t').collect::<Vec<_>>();
    if columns.len() < 2 || columns[0] != "id" {
        return Err(Error::InvalidDocument(
            "TSV header must start with id and contain at least one field".into(),
        ));
    }
    if columns.len() - 1 > limits.max_fields_per_document {
        return Err(Error::InvalidDocument(format!(
            "TSV header has {} fields; limit is {}",
            columns.len() - 1,
            limits.max_fields_per_document
        )));
    }
    let mut seen_fields = BTreeSet::new();
    for field in &columns[1..] {
        if !seen_fields.insert(*field) {
            return Err(Error::InvalidDocument(format!(
                "duplicate TSV field '{field}'"
            )));
        }
        Document::from_fields("header-validation", [(*field, "")])?;
    }

    let mut line_number = 2_usize;
    while let Some(line) = read_bounded_line(&mut reader, limits.max_record_bytes, line_number)? {
        let line = trim_line_ending(&line);
        if !line.is_empty() {
            let values = line.split('\t').collect::<Vec<_>>();
            if values.len() != columns.len() {
                return Err(Error::InvalidDocument(format!(
                    "TSV line {line_number} has {} columns; expected {}",
                    values.len(),
                    columns.len()
                )));
            }
            let fields = columns[1..]
                .iter()
                .zip(&values[1..])
                .map(|(name, value)| (*name, *value));
            visit(Document::from_fields(values[0], fields)?)?;
        }
        line_number = line_number
            .checked_add(1)
            .ok_or_else(|| Error::InvalidDocument("TSV line count overflow".into()))?;
    }
    Ok(())
}

fn visit_jsonl_reader(
    mut reader: impl BufRead,
    limits: CollectionLimits,
    visit: &mut impl FnMut(Document) -> Result<()>,
) -> Result<()> {
    let mut line_number = 1_usize;
    while let Some(line) = read_bounded_line(&mut reader, limits.max_record_bytes, line_number)? {
        let line = trim_line_ending(&line);
        if !line.trim().is_empty() {
            let record: StrictJsonRecord = serde_json::from_str(line).map_err(|error| {
                Error::InvalidDocument(format!(
                    "invalid JSONL record at line {line_number}: {error}"
                ))
            })?;
            visit(record.into_document(line_number, limits)?)?;
        }
        line_number = line_number
            .checked_add(1)
            .ok_or_else(|| Error::InvalidDocument("JSONL line count overflow".into()))?;
    }
    Ok(())
}

#[derive(Debug, Default)]
struct StrictJsonRecord {
    id: Option<String>,
    fields: Option<BTreeMap<String, String>>,
    title: Option<String>,
    content: Option<String>,
    url: Option<String>,
}

impl StrictJsonRecord {
    fn into_document(self, line: usize, limits: CollectionLimits) -> Result<Document> {
        if let Some(id) = self.id {
            if self.title.is_some() || self.content.is_some() || self.url.is_some() {
                return Err(Error::InvalidDocument(format!(
                    "JSONL record at line {line} mixes canonical and PISA-compatible keys"
                )));
            }
            let fields = self.fields.ok_or_else(|| {
                Error::InvalidDocument(format!(
                    "JSONL record at line {line} requires object-valued 'fields'"
                ))
            })?;
            if fields.is_empty() {
                return Err(Error::InvalidDocument(format!(
                    "JSONL record at line {line} has no fields"
                )));
            }
            if fields.len() > limits.max_fields_per_document {
                return Err(Error::InvalidDocument(format!(
                    "JSONL record at line {line} has {} fields; limit is {}",
                    fields.len(),
                    limits.max_fields_per_document
                )));
            }
            return Document::from_fields(id, fields);
        }
        if self.fields.is_some() {
            return Err(Error::InvalidDocument(format!(
                "canonical JSONL fields at line {line} require string 'id'"
            )));
        }
        let title = self.title.ok_or_else(|| {
            Error::InvalidDocument(format!(
                "PISA-compatible JSONL record at line {line} requires string 'title'"
            ))
        })?;
        let content = self.content.ok_or_else(|| {
            Error::InvalidDocument(format!(
                "PISA-compatible JSONL record at line {line} requires string 'content'"
            ))
        })?;
        let mut fields = vec![("body".to_owned(), content)];
        if let Some(url) = self.url.filter(|url| !url.is_empty()) {
            fields.push(("url".to_owned(), url));
        }
        Document::from_fields(title, fields)
    }
}

impl<'de> Deserialize<'de> for StrictJsonRecord {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RecordVisitor)
    }
}

struct RecordVisitor;

impl<'de> Visitor<'de> for RecordVisitor {
    type Value = StrictJsonRecord;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a canonical or PISA-compatible JSON object")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut record = StrictJsonRecord::default();
        let mut seen = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON record key '{key}'"
                )));
            }
            match key.as_str() {
                "id" => record.id = Some(map.next_value::<String>()?),
                "fields" => record.fields = Some(map.next_value::<StrictStringMap>()?.0),
                "title" => record.title = Some(map.next_value::<String>()?),
                "content" => record.content = Some(map.next_value::<String>()?),
                "url" => record.url = Some(map.next_value::<String>()?),
                _ => {
                    return Err(de::Error::custom(format!("unsupported JSONL key '{key}'")));
                }
            }
        }
        Ok(record)
    }
}

struct StrictStringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for StrictStringMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(StringMapVisitor)
    }
}

struct StringMapVisitor;

impl<'de> Visitor<'de> for StringMapVisitor {
    type Value = StrictStringMap;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("an object mapping unique field names to strings")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, String>()? {
            if fields.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!(
                    "duplicate JSON field key '{key}'"
                )));
            }
        }
        Ok(StrictStringMap(fields))
    }
}

fn read_bounded_line(
    reader: &mut impl BufRead,
    max_bytes: usize,
    line: usize,
) -> Result<Option<String>> {
    let take_limit = u64::try_from(max_bytes)
        .ok()
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| Error::InvalidArgument("record byte limit does not fit u64".into()))?;
    let mut bytes = Vec::new();
    let count = reader.take(take_limit).read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    let content_bytes = bytes
        .strip_suffix(b"\n")
        .unwrap_or(&bytes)
        .strip_suffix(b"\r")
        .unwrap_or_else(|| bytes.strip_suffix(b"\n").unwrap_or(&bytes));
    if content_bytes.len() > max_bytes {
        return Err(Error::InvalidDocument(format!(
            "collection record at line {line} exceeds {max_bytes} bytes"
        )));
    }
    String::from_utf8(bytes).map(Some).map_err(|error| {
        Error::InvalidDocument(format!(
            "collection record at line {line} is not valid UTF-8: {error}"
        ))
    })
}

/// Read no more than `max_bytes` raw bytes, including the line terminator.
/// Unlike `BufRead::lines`, this rejects a long line before allocating it.
pub(crate) fn read_bounded_raw_line(
    reader: &mut impl BufRead,
    max_bytes: usize,
    reported_max_bytes: usize,
    line: usize,
    source: &str,
) -> Result<Option<String>> {
    let probe = u64::try_from(max_bytes)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Error::InvalidArgument("record byte limit does not fit u64".into()))?;
    let mut bytes = Vec::new();
    let count = reader.take(probe).read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    if count > max_bytes {
        return Err(Error::InvalidDocument(format!(
            "{source} at line {line} exceeds {reported_max_bytes} bytes"
        )));
    }
    String::from_utf8(bytes).map(Some).map_err(|error| {
        Error::InvalidDocument(format!(
            "{source} at line {line} is not valid UTF-8: {error}"
        ))
    })
}

fn trim_line_ending(value: &str) -> &str {
    let value = value.strip_suffix('\n').unwrap_or(value);
    value.strip_suffix('\r').unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "indexsail-collection-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn collect_jsonl(input: &str) -> Result<Vec<Document>> {
        let mut documents = Vec::new();
        visit_jsonl_reader(
            Cursor::new(input),
            CollectionLimits::default(),
            &mut |document| {
                documents.push(document);
                Ok(())
            },
        )?;
        Ok(documents)
    }

    #[test]
    fn canonical_jsonl_retains_named_fields() {
        let documents = collect_jsonl(
            "{\"id\":\"d1\",\"fields\":{\"title\":\"Sail\",\"body\":\"blue water\"}}\n",
        )
        .unwrap();
        assert_eq!(documents[0].external_id(), "d1");
        assert_eq!(documents[0].field("title"), Some("Sail"));
        assert_eq!(documents[0].field("body"), Some("blue water"));
    }

    #[test]
    fn pisa_jsonl_retains_content_and_optional_url() {
        let documents = collect_jsonl(
            "{\"title\":\"tiny1\",\"content\":\"lorem ipsum\",\"url\":\"https://tiny1.test\"}\n",
        )
        .unwrap();
        assert_eq!(documents[0].external_id(), "tiny1");
        assert_eq!(documents[0].field("body"), Some("lorem ipsum"));
        assert_eq!(documents[0].field("url"), Some("https://tiny1.test"));
    }

    #[test]
    fn jsonl_rejects_unknown_keys_and_non_string_fields() {
        assert!(collect_jsonl("{\"id\":\"d\",\"fields\":{\"body\":1}}\n").is_err());
        assert!(collect_jsonl("{\"id\":\"d\",\"fields\":{\"body\":\"x\"},\"drop\":1}\n").is_err());
    }

    #[test]
    fn jsonl_rejects_duplicate_keys_and_mixed_protocols() {
        assert!(
            collect_jsonl("{\"id\":\"first\",\"id\":\"second\",\"fields\":{\"body\":\"x\"}}\n")
                .unwrap_err()
                .to_string()
                .contains("duplicate JSON record key 'id'")
        );
        assert!(
            collect_jsonl("{\"id\":\"d\",\"fields\":{\"body\":\"x\",\"body\":\"y\"}}\n")
                .unwrap_err()
                .to_string()
                .contains("duplicate JSON field key 'body'")
        );
        assert!(
            collect_jsonl("{\"id\":\"d\",\"fields\":{\"body\":\"x\"},\"title\":\"legacy\"}\n")
                .unwrap_err()
                .to_string()
                .contains("mixes canonical and PISA-compatible")
        );
    }

    #[test]
    fn bounded_line_reader_handles_crlf_and_rejects_long_or_invalid_utf8() {
        let mut valid = Cursor::new(b"abc\r\nnext\n".to_vec());
        assert_eq!(
            read_bounded_line(&mut valid, 3, 1).unwrap().as_deref(),
            Some("abc\r\n")
        );
        assert_eq!(
            read_bounded_line(&mut valid, 4, 2).unwrap().as_deref(),
            Some("next\n")
        );

        let mut long = Cursor::new(b"abcd\n".to_vec());
        assert!(read_bounded_line(&mut long, 3, 1).is_err());
        let mut invalid = Cursor::new(vec![0xff, b'\n']);
        assert!(
            read_bounded_line(&mut invalid, 3, 1)
                .unwrap_err()
                .to_string()
                .contains("not valid UTF-8")
        );
    }

    #[test]
    fn actual_input_byte_limit_stops_a_multi_page_read() {
        let mut source = Cursor::new(vec![b'x'; 20_000]);
        let mut limited = InputLimitReader::new(&mut source, 9_000);
        let mut output = Vec::new();
        let error = limited.read_to_end(&mut output).unwrap_err();
        assert!(error.to_string().contains("exceeds 9000 byte limit"));
        assert_eq!(limited.bytes_read(), 9_001);
        assert_eq!(source.position(), 9_001);
    }

    #[test]
    fn collection_rejects_an_append_during_the_document_callback() {
        let path = temp_path("append.jsonl");
        std::fs::write(&path, "{\"id\":\"a\",\"fields\":{\"body\":\"x\"}}\n").unwrap();
        let mut called = false;
        let error = visit_collection(
            &path,
            CollectionFormat::Jsonl,
            CollectionLimits::default(),
            |_| {
                if !called {
                    std::fs::OpenOptions::new()
                        .append(true)
                        .open(&path)?
                        .write_all(b"\n")?;
                    called = true;
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("collection changed while it was being read")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn collection_rejects_same_length_in_place_mutation_during_callback() {
        let path = temp_path("same-length-mutation.jsonl");
        let original = b"{\"id\":\"a\",\"fields\":{\"body\":\"x\"}}\n";
        let replacement = b"{\"id\":\"a\",\"fields\":{\"body\":\"y\"}}\n";
        assert_eq!(original.len(), replacement.len());
        std::fs::write(&path, original).unwrap();
        let error = visit_collection(
            &path,
            CollectionFormat::Jsonl,
            CollectionLimits::default(),
            |_| {
                std::fs::write(&path, replacement)?;
                std::fs::File::options()
                    .write(true)
                    .open(&path)?
                    .set_modified(
                        std::time::SystemTime::now() + std::time::Duration::from_secs(60),
                    )?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("collection changed while it was being read")
        );
        assert_eq!(std::fs::read(&path).unwrap(), replacement);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn collection_does_not_treat_a_directory_as_an_empty_input_file() {
        let path = temp_path("directory-as-collection");
        std::fs::create_dir(&path).unwrap();
        let mut visited = false;
        let error = visit_collection(
            &path,
            CollectionFormat::Tsv,
            CollectionLimits::default(),
            |_| {
                visited = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(!visited);
        assert!(matches!(error, Error::InvalidArgument(_) | Error::Io(_)));
        std::fs::remove_dir(path).unwrap();
    }

    #[test]
    fn collection_enforces_its_byte_ceiling_against_an_append() {
        let path = temp_path("append-over-limit.jsonl");
        let bytes = b"{\"id\":\"a\",\"fields\":{\"body\":\"x\"}}\n";
        std::fs::write(&path, bytes).unwrap();
        let error = visit_collection(
            &path,
            CollectionFormat::Jsonl,
            CollectionLimits {
                max_input_bytes: bytes.len() as u64,
                ..CollectionLimits::default()
            },
            |_| {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)?
                    .write_all(b"\n")?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("input exceeds"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn collection_rejects_a_path_swap_during_the_document_callback() {
        let path = temp_path("swap.jsonl");
        let old_path = temp_path("swapped-old.jsonl");
        let bytes = b"{\"id\":\"a\",\"fields\":{\"body\":\"x\"}}\n";
        std::fs::write(&path, bytes).unwrap();
        let error = visit_collection(
            &path,
            CollectionFormat::Jsonl,
            CollectionLimits::default(),
            |_| {
                std::fs::rename(&path, &old_path)?;
                std::fs::write(&path, bytes)?;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("collection changed while it was being read")
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(old_path).unwrap();
    }

    #[test]
    fn limits_apply_before_extra_documents_and_large_trec_records() {
        let jsonl = temp_path("limits.jsonl");
        std::fs::write(
            &jsonl,
            "{\"id\":\"a\",\"fields\":{\"body\":\"x\"}}\n{\"id\":\"b\",\"fields\":{\"body\":\"y\"}}\n",
        )
        .unwrap();
        let error = load_collection(
            &jsonl,
            CollectionFormat::Jsonl,
            CollectionLimits {
                max_documents: 1,
                ..CollectionLimits::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds 1 document limit"));
        std::fs::remove_file(jsonl).unwrap();

        let trec = temp_path("limits.trec");
        std::fs::write(
            &trec,
            "<DOC>\n<DOCNO>D1</DOCNO>\n<TEXT>123456789</TEXT>\n</DOC>\n",
        )
        .unwrap();
        let error = load_collection(
            &trec,
            CollectionFormat::Trec,
            CollectionLimits {
                max_record_bytes: 20,
                ..CollectionLimits::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds 20 bytes"));
        std::fs::remove_file(trec).unwrap();
    }

    #[test]
    fn collection_limits_must_be_positive() {
        for limits in [
            CollectionLimits {
                max_documents: 0,
                ..CollectionLimits::default()
            },
            CollectionLimits {
                max_fields_per_document: 0,
                ..CollectionLimits::default()
            },
            CollectionLimits {
                max_record_bytes: 0,
                ..CollectionLimits::default()
            },
            CollectionLimits {
                max_input_bytes: 0,
                ..CollectionLimits::default()
            },
        ] {
            assert!(limits.validate().is_err());
        }
    }

    #[test]
    fn zero_budget_input_reader_probes_one_byte_without_unbounded_allocation() {
        let mut reader = InputLimitReader::new(Cursor::new(b"abc"), 0);
        assert_eq!(reader.read(&mut []).unwrap(), 0);
        assert_eq!(reader.bytes_read(), 0);
        let error = reader.read(&mut [0; 32]).unwrap_err();
        assert!(error.to_string().contains("exceeds 0 byte limit"));
        assert_eq!(reader.bytes_read(), 1);
    }

    #[test]
    fn tsv_rejects_bad_headers_and_reports_malformed_rows() {
        let limits = CollectionLimits::default();
        for input in [
            "",
            "name\tbody\nA\tx\n",
            "id\nA\n",
            "id\tbody\tbody\nA\tx\ty\n",
            "id\tbad name\nA\tx\n",
        ] {
            assert!(
                visit_tsv_reader(Cursor::new(input), limits, &mut |_| Ok(())).is_err(),
                "{input:?}"
            );
        }
        let error = visit_tsv_reader(
            Cursor::new("id\tbody\ntoo\tmany\tvalues\n"),
            limits,
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("line 2 has 3 columns"));
        let mut ids = Vec::new();
        visit_tsv_reader(
            Cursor::new("id\tbody\r\n\r\nA\tx\r\n"),
            limits,
            &mut |document| {
                ids.push(document.external_id().to_owned());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(ids, ["A"]);
    }

    #[test]
    fn jsonl_requires_one_complete_protocol_and_obeys_field_limit() {
        for input in [
            "{\"id\":\"A\"}\n",
            "{\"id\":\"A\",\"fields\":{}}\n",
            "{\"fields\":{\"body\":\"x\"}}\n",
            "{\"title\":\"A\"}\n",
            "{\"title\":\"A\",\"content\":\"x\",\"id\":\"B\"}\n",
            "{\"id\":\"A\",\"fields\":{\"body\":\"x\"},\"url\":\"x\"}\n",
        ] {
            assert!(collect_jsonl(input).is_err(), "{input}");
        }
        let mut documents = Vec::new();
        visit_jsonl_reader(
            Cursor::new("\n{\"title\":\"A\",\"content\":\"x\",\"url\":\"\"}\n"),
            CollectionLimits::default(),
            &mut |document| {
                documents.push(document);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].field("url"), None);
        let error = visit_jsonl_reader(
            Cursor::new("{\"id\":\"A\",\"fields\":{\"body\":\"x\",\"title\":\"y\"}}\n"),
            CollectionLimits {
                max_fields_per_document: 1,
                ..CollectionLimits::default()
            },
            &mut |_| Ok(()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("limit is 1"));
    }

    #[test]
    fn raw_line_limit_counts_terminator_and_rejects_invalid_utf8() {
        let mut exact = Cursor::new(b"abc\r\nnext".to_vec());
        assert_eq!(
            read_bounded_raw_line(&mut exact, 5, 5, 1, "fixture")
                .unwrap()
                .as_deref(),
            Some("abc\r\n")
        );
        assert_eq!(
            read_bounded_raw_line(&mut exact, 4, 4, 2, "fixture")
                .unwrap()
                .as_deref(),
            Some("next")
        );
        assert!(
            read_bounded_raw_line(&mut Cursor::new(b"abc\r\n"), 4, 4, 1, "fixture")
                .unwrap_err()
                .to_string()
                .contains("exceeds 4 bytes")
        );
        assert!(
            read_bounded_raw_line(&mut Cursor::new([0xff, b'\n']), 2, 2, 1, "fixture")
                .unwrap_err()
                .to_string()
                .contains("not valid UTF-8")
        );
        assert_eq!(
            read_bounded_raw_line(&mut Cursor::new([]), 2, 2, 1, "fixture").unwrap(),
            None
        );
    }

    #[test]
    fn collection_rejects_empty_and_oversize_inputs_before_visiting() {
        let empty = temp_path("empty.tsv");
        std::fs::write(&empty, "id\tbody\n").unwrap();
        assert!(
            load_collection(&empty, CollectionFormat::Tsv, CollectionLimits::default())
                .unwrap_err()
                .to_string()
                .contains("no documents")
        );
        let mut called = false;
        let error = visit_collection(
            &empty,
            CollectionFormat::Tsv,
            CollectionLimits {
                max_input_bytes: 1,
                ..CollectionLimits::default()
            },
            |_| {
                called = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("limit is 1"));
        assert!(!called);
        std::fs::remove_file(empty).unwrap();
    }

    #[test]
    fn both_indexing_adapters_stop_at_the_same_document_limit() {
        let path = temp_path("index-document-limit.tsv");
        std::fs::write(&path, "id\tbody\nA\tfirst\nB\tsecond\n").unwrap();
        let limits = CollectionLimits {
            max_documents: 1,
            ..CollectionLimits::default()
        };

        let single = index_collection(&path, Analyzer::default(), CollectionFormat::Tsv, limits)
            .unwrap_err();
        let sharded =
            index_collection_sharded(&path, Analyzer::default(), CollectionFormat::Tsv, limits, 2)
                .unwrap_err();
        for error in [single, sharded] {
            assert!(
                error.to_string().contains("exceeds 1 document limit"),
                "{error}"
            );
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn format_and_field_limits_reject_untrusted_inputs_before_indexing() {
        assert_eq!(
            CollectionFormat::parse("tsv").unwrap(),
            CollectionFormat::Tsv
        );
        assert_eq!(
            CollectionFormat::parse("trec").unwrap(),
            CollectionFormat::Trec
        );
        assert_eq!(
            CollectionFormat::parse("jsonl").unwrap(),
            CollectionFormat::Jsonl
        );
        assert!(
            CollectionFormat::parse("csv")
                .unwrap_err()
                .to_string()
                .contains("unknown collection format")
        );

        let header = temp_path("many-fields.tsv");
        std::fs::write(&header, "id\ttitle\tbody\nD\tblue\tsea\n").unwrap();
        let limit = CollectionLimits {
            max_fields_per_document: 1,
            ..CollectionLimits::default()
        };
        assert!(
            index_collection(&header, Analyzer::default(), CollectionFormat::Tsv, limit)
                .unwrap_err()
                .to_string()
                .contains("TSV header has 2 fields")
        );
        std::fs::remove_file(header).unwrap();

        // PISA-compatible JSONL makes a body and an optional URL field; the
        // generic document limit must apply after that transformation too.
        let pisa = temp_path("pisa-fields.jsonl");
        std::fs::write(
            &pisa,
            "{\"title\":\"D\",\"content\":\"blue\",\"url\":\"https://example.test\"}\n",
        )
        .unwrap();
        assert!(
            index_collection(&pisa, Analyzer::default(), CollectionFormat::Jsonl, limit)
                .unwrap_err()
                .to_string()
                .contains("has 2 fields; limit is 1")
        );
        std::fs::remove_file(pisa).unwrap();
    }
}
