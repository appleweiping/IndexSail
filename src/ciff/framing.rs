use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::atomic::atomic_write_with;
use crate::error::{Error, Result};

use super::model::{
    CiffDocumentRecord, CiffHeader, CiffIndex, CiffLimits, CiffPosting, CiffPostingList,
    validate_document_local, validate_header, validate_posting_list_local,
};
use super::wire::{
    Decoder, WIRE_FIXED64, WIRE_LENGTH_DELIMITED, WIRE_VARINT, mark_once, read_delimited,
    require_wire, write_varint,
};
#[cfg(test)]
use super::wire::{Encoder, write_delimited};

impl CiffIndex {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_limits(path, CiffLimits::default())
    }

    pub fn load_with_limits(path: impl AsRef<Path>, limits: CiffLimits) -> Result<Self> {
        let limits = limits.validate()?;
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader, limits)
    }

    pub fn read_from(reader: &mut impl Read, limits: CiffLimits) -> Result<Self> {
        let limits = limits.validate()?;
        let header_bytes = read_required_frame(reader, limits.max_frame_bytes, "header")?;
        let header = parse_header(&header_bytes, limits)?;
        validate_header(&header, limits)?;

        let posting_list_count = usize::try_from(header.num_posting_lists)
            .map_err(|_| corrupt("num_postings_lists exceeds usize"))?;
        let mut posting_lists = Vec::new();
        let mut posting_count = 0_usize;
        let mut terms = BTreeSet::new();
        for ordinal in 0..posting_list_count {
            let bytes = read_required_frame(reader, limits.max_frame_bytes, "postings list")?;
            let remaining_postings = limits.max_postings - posting_count;
            let list = parse_posting_list(&bytes, limits, remaining_postings)
                .map_err(|error| contextual(error, &format!("postings list {}", ordinal + 1)))?;
            validate_posting_list_local(&list, header.total_documents, limits)
                .map_err(|error| contextual(error, &format!("postings list {}", ordinal + 1)))?;
            if !terms.insert(list.term.clone()) {
                return Err(corrupt(format!(
                    "duplicate postings term '{}' (postings list {})",
                    list.term,
                    ordinal + 1
                )));
            }
            posting_count = posting_count
                .checked_add(list.postings.len())
                .ok_or_else(|| corrupt("total posting count overflows usize"))?;
            if posting_count > limits.max_postings {
                return Err(corrupt(format!(
                    "postings exceed the configured {}-entry limit",
                    limits.max_postings
                )));
            }
            posting_lists.push(list);
        }
        drop(terms);

        let document_count =
            usize::try_from(header.num_documents).map_err(|_| corrupt("num_docs exceeds usize"))?;
        let mut documents = Vec::new();
        let mut document_ids = BTreeSet::new();
        let mut external_ids = BTreeSet::new();
        for ordinal in 0..document_count {
            let bytes = read_required_frame(reader, limits.max_frame_bytes, "document record")?;
            let document = parse_document_record(&bytes, limits)
                .map_err(|error| contextual(error, &format!("document record {}", ordinal + 1)))?;
            validate_document_local(&document, header.total_documents, limits)
                .map_err(|error| contextual(error, &format!("document record {}", ordinal + 1)))?;
            if !document_ids.insert(document.document_id) {
                return Err(corrupt(format!(
                    "duplicate document id {} (document record {})",
                    document.document_id,
                    ordinal + 1
                )));
            }
            if !external_ids.insert(document.external_id.clone()) {
                return Err(corrupt(format!(
                    "duplicate collection document id '{}' (document record {})",
                    document.external_id,
                    ordinal + 1
                )));
            }
            documents.push(document);
        }
        drop(document_ids);
        drop(external_ids);
        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing)? != 0 {
            return Err(corrupt("unexpected message after the declared DocRecords"));
        }
        Self::from_parts(header, posting_lists, documents, limits)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save_with_limits(path, CiffLimits::default())
    }

    pub fn save_with_limits(&self, path: impl AsRef<Path>, limits: CiffLimits) -> Result<()> {
        let limits = limits.validate()?;
        let path = path.as_ref();
        atomic_write_with(path, |file| {
            let mut writer = BufWriter::new(file);
            self.write_to(&mut writer, limits)?;
            writer.flush()?;
            Ok(())
        })
    }

    pub fn write_to(&self, writer: &mut impl Write, limits: CiffLimits) -> Result<()> {
        let limits = limits.validate()?;
        self.validate(limits)?;
        let header_size = encoded_header_len(self.header())?;
        write_sized_frame(writer, header_size, limits, |writer| {
            write_header_message(writer, self.header())
        })?;
        for list in self.posting_lists() {
            let list_size = encoded_posting_list_len(list)?;
            write_sized_frame(writer, list_size, limits, |writer| {
                write_posting_list_message(writer, list)
            })?;
        }
        for document in self.documents() {
            let document_size = encoded_document_record_len(document)?;
            write_sized_frame(writer, document_size, limits, |writer| {
                write_document_record_message(writer, document)
            })?;
        }
        Ok(())
    }
}

fn read_required_frame(
    reader: &mut impl Read,
    max_frame_bytes: usize,
    label: &str,
) -> Result<Vec<u8>> {
    read_delimited(reader, max_frame_bytes)?
        .ok_or_else(|| corrupt(format!("missing {label} message")))
}

fn ensure_frame_size(encoded_size: usize, limits: CiffLimits) -> Result<()> {
    if encoded_size > limits.max_frame_bytes {
        return Err(Error::InvalidArgument(format!(
            "CIFF output frame has {} bytes, exceeding the configured {}-byte limit",
            encoded_size, limits.max_frame_bytes
        )));
    }
    Ok(())
}

fn write_sized_frame<Writer: Write>(
    writer: &mut Writer,
    encoded_size: usize,
    limits: CiffLimits,
    write_message: impl FnOnce(&mut ExactSizeWriter<'_, Writer>) -> Result<()>,
) -> Result<()> {
    ensure_frame_size(encoded_size, limits)?;
    write_varint(
        writer,
        u64::try_from(encoded_size)
            .map_err(|_| Error::InvalidArgument("CIFF frame length exceeds u64".into()))?,
    )?;
    let mut exact = ExactSizeWriter {
        writer,
        remaining: encoded_size,
    };
    write_message(&mut exact)?;
    if exact.remaining != 0 {
        return Err(Error::InvalidArgument(format!(
            "CIFF encoded-size invariant left {} unwritten bytes",
            exact.remaining
        )));
    }
    Ok(())
}

struct ExactSizeWriter<'a, Writer> {
    writer: &'a mut Writer,
    remaining: usize,
}

impl<Writer: Write> Write for ExactSizeWriter<'_, Writer> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "CIFF encoder exceeded its checked frame size",
            ));
        }
        let written = self.writer.write(bytes)?;
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

fn parse_header(bytes: &[u8], limits: CiffLimits) -> Result<CiffHeader> {
    let mut decoder = Decoder::new(bytes);
    let mut seen = 0_u64;
    let mut header = CiffHeader {
        version: 0,
        num_posting_lists: 0,
        num_documents: 0,
        total_posting_lists: 0,
        total_documents: 0,
        total_terms_in_collection: 0,
        average_document_length: 0.0,
        description: String::new(),
    };
    while let Some(field) = decoder.next_field()? {
        match field.number {
            1 => {
                mark_once(&mut seen, 1, "Header.version")?;
                require_wire(field, WIRE_VARINT, "Header.version")?;
                header.version = decoder.read_i32_nonnegative("Header.version")?;
            }
            2 => {
                mark_once(&mut seen, 2, "Header.num_postings_lists")?;
                require_wire(field, WIRE_VARINT, "Header.num_postings_lists")?;
                header.num_posting_lists =
                    decoder.read_i32_nonnegative("Header.num_postings_lists")?;
            }
            3 => {
                mark_once(&mut seen, 3, "Header.num_docs")?;
                require_wire(field, WIRE_VARINT, "Header.num_docs")?;
                header.num_documents = decoder.read_i32_nonnegative("Header.num_docs")?;
            }
            4 => {
                mark_once(&mut seen, 4, "Header.total_postings_lists")?;
                require_wire(field, WIRE_VARINT, "Header.total_postings_lists")?;
                header.total_posting_lists =
                    decoder.read_i32_nonnegative("Header.total_postings_lists")?;
            }
            5 => {
                mark_once(&mut seen, 5, "Header.total_docs")?;
                require_wire(field, WIRE_VARINT, "Header.total_docs")?;
                header.total_documents = decoder.read_i32_nonnegative("Header.total_docs")?;
            }
            6 => {
                mark_once(&mut seen, 6, "Header.total_terms_in_collection")?;
                require_wire(field, WIRE_VARINT, "Header.total_terms_in_collection")?;
                header.total_terms_in_collection =
                    decoder.read_i64_nonnegative("Header.total_terms_in_collection")?;
            }
            7 => {
                mark_once(&mut seen, 7, "Header.average_doclength")?;
                require_wire(field, WIRE_FIXED64, "Header.average_doclength")?;
                header.average_document_length = f64::from_bits(decoder.read_fixed64()?);
            }
            8 => {
                mark_once(&mut seen, 8, "Header.description")?;
                require_wire(field, WIRE_LENGTH_DELIMITED, "Header.description")?;
                header.description =
                    decoder.read_string("Header.description", limits.max_description_bytes)?;
            }
            _ => decoder.skip(field)?,
        }
    }
    Ok(header)
}

fn parse_posting_list(
    bytes: &[u8],
    limits: CiffLimits,
    remaining_postings: usize,
) -> Result<CiffPostingList> {
    let mut decoder = Decoder::new(bytes);
    let mut seen = 0_u64;
    let mut term = String::new();
    let mut document_frequency = 0_u64;
    let mut collection_frequency = 0_u64;
    let mut postings = Vec::new();
    let mut previous = 0_u32;
    while let Some(field) = decoder.next_field()? {
        match field.number {
            1 => {
                mark_once(&mut seen, 1, "PostingsList.term")?;
                require_wire(field, WIRE_LENGTH_DELIMITED, "PostingsList.term")?;
                term = decoder.read_string("PostingsList.term", limits.max_term_bytes)?;
            }
            2 => {
                mark_once(&mut seen, 2, "PostingsList.df")?;
                require_wire(field, WIRE_VARINT, "PostingsList.df")?;
                document_frequency = decoder.read_i64_nonnegative("PostingsList.df")?;
            }
            3 => {
                mark_once(&mut seen, 3, "PostingsList.cf")?;
                require_wire(field, WIRE_VARINT, "PostingsList.cf")?;
                collection_frequency = decoder.read_i64_nonnegative("PostingsList.cf")?;
            }
            4 => {
                require_wire(field, WIRE_LENGTH_DELIMITED, "PostingsList.postings")?;
                if postings.len() >= remaining_postings {
                    return Err(corrupt(format!(
                        "postings exceed the configured {}-entry limit",
                        limits.max_postings,
                    )));
                }
                let nested = decoder.read_length_delimited()?;
                let (gap, term_frequency) = parse_posting(nested)?;
                let document_id = previous
                    .checked_add(gap)
                    .ok_or_else(|| corrupt("posting d-gap prefix sum overflows u32"))?;
                postings.push(CiffPosting {
                    document_id,
                    term_frequency,
                });
                previous = document_id;
            }
            _ => decoder.skip(field)?,
        }
    }
    Ok(CiffPostingList {
        term,
        document_frequency,
        collection_frequency,
        postings,
    })
}

fn parse_posting(bytes: &[u8]) -> Result<(u32, u32)> {
    let mut decoder = Decoder::new(bytes);
    let mut seen = 0_u64;
    let mut document_gap = 0;
    let mut term_frequency = 0;
    while let Some(field) = decoder.next_field()? {
        match field.number {
            1 => {
                mark_once(&mut seen, 1, "Posting.docid")?;
                require_wire(field, WIRE_VARINT, "Posting.docid")?;
                document_gap = decoder.read_i32_nonnegative("Posting.docid")?;
            }
            2 => {
                mark_once(&mut seen, 2, "Posting.tf")?;
                require_wire(field, WIRE_VARINT, "Posting.tf")?;
                term_frequency = decoder.read_i32_nonnegative("Posting.tf")?;
            }
            _ => decoder.skip(field)?,
        }
    }
    Ok((document_gap, term_frequency))
}

fn parse_document_record(bytes: &[u8], limits: CiffLimits) -> Result<CiffDocumentRecord> {
    let mut decoder = Decoder::new(bytes);
    let mut seen = 0_u64;
    let mut document = CiffDocumentRecord {
        document_id: 0,
        external_id: String::new(),
        document_length: 0,
    };
    while let Some(field) = decoder.next_field()? {
        match field.number {
            1 => {
                mark_once(&mut seen, 1, "DocRecord.docid")?;
                require_wire(field, WIRE_VARINT, "DocRecord.docid")?;
                document.document_id = decoder.read_i32_nonnegative("DocRecord.docid")?;
            }
            2 => {
                mark_once(&mut seen, 2, "DocRecord.collection_docid")?;
                require_wire(field, WIRE_LENGTH_DELIMITED, "DocRecord.collection_docid")?;
                document.external_id = decoder
                    .read_string("DocRecord.collection_docid", limits.max_external_id_bytes)?;
            }
            3 => {
                mark_once(&mut seen, 3, "DocRecord.doclength")?;
                require_wire(field, WIRE_VARINT, "DocRecord.doclength")?;
                document.document_length = decoder.read_i32_nonnegative("DocRecord.doclength")?;
            }
            _ => decoder.skip(field)?,
        }
    }
    Ok(document)
}

fn write_header_message(writer: &mut impl Write, header: &CiffHeader) -> Result<()> {
    write_varint_field(writer, 1, u64::from(header.version))?;
    write_varint_field(writer, 2, u64::from(header.num_posting_lists))?;
    write_varint_field(writer, 3, u64::from(header.num_documents))?;
    write_varint_field(writer, 4, u64::from(header.total_posting_lists))?;
    write_varint_field(writer, 5, u64::from(header.total_documents))?;
    write_varint_field(writer, 6, header.total_terms_in_collection)?;
    write_fixed64_field(writer, 7, header.average_document_length.to_bits())?;
    write_bytes_field(writer, 8, header.description.as_bytes())
}

fn write_posting_list_message(writer: &mut impl Write, list: &CiffPostingList) -> Result<()> {
    #[cfg(test)]
    WRITE_POSTING_LIST_CALLS.with(|calls| calls.set(calls.get() + 1));
    write_bytes_field(writer, 1, list.term.as_bytes())?;
    write_varint_field(writer, 2, list.document_frequency)?;
    write_varint_field(writer, 3, list.collection_frequency)?;
    let mut previous = 0_u32;
    for posting in &list.postings {
        let gap = posting
            .document_id
            .checked_sub(previous)
            .ok_or_else(|| corrupt("posting ids are not increasing while encoding"))?;
        let nested_size = encoded_posting_len(gap, posting.term_frequency)?;
        if nested_size != 0 {
            write_key(writer, 4, WIRE_LENGTH_DELIMITED)?;
            write_varint(
                writer,
                u64::try_from(nested_size).map_err(|_| {
                    Error::InvalidArgument("CIFF nested posting length exceeds u64".into())
                })?,
            )?;
            write_varint_field(writer, 1, u64::from(gap))?;
            write_varint_field(writer, 2, u64::from(posting.term_frequency))?;
        }
        previous = posting.document_id;
    }
    Ok(())
}

fn write_document_record_message(
    writer: &mut impl Write,
    document: &CiffDocumentRecord,
) -> Result<()> {
    write_varint_field(writer, 1, u64::from(document.document_id))?;
    write_bytes_field(writer, 2, document.external_id.as_bytes())?;
    write_varint_field(writer, 3, u64::from(document.document_length))
}

fn write_varint_field(writer: &mut impl Write, field: u32, value: u64) -> Result<()> {
    if value != 0 {
        write_key(writer, field, WIRE_VARINT)?;
        write_varint(writer, value)?;
    }
    Ok(())
}

fn write_fixed64_field(writer: &mut impl Write, field: u32, value: u64) -> Result<()> {
    if value != 0 {
        write_key(writer, field, WIRE_FIXED64)?;
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_bytes_field(writer: &mut impl Write, field: u32, value: &[u8]) -> Result<()> {
    if !value.is_empty() {
        write_key(writer, field, WIRE_LENGTH_DELIMITED)?;
        write_varint(
            writer,
            u64::try_from(value.len())
                .map_err(|_| Error::InvalidArgument("CIFF byte field exceeds u64".into()))?,
        )?;
        writer.write_all(value)?;
    }
    Ok(())
}

fn write_key(writer: &mut impl Write, field: u32, wire_type: u8) -> Result<()> {
    let key = u64::from(field)
        .checked_shl(3)
        .and_then(|value| value.checked_add(u64::from(wire_type)))
        .ok_or_else(|| Error::InvalidArgument("CIFF protobuf field key overflows".into()))?;
    write_varint(writer, key)
}

#[cfg(test)]
fn encode_header(header: &CiffHeader, encoded_size: usize) -> Vec<u8> {
    let mut encoder = Encoder::with_capacity(encoded_size);
    encoder.varint(1, u64::from(header.version));
    encoder.varint(2, u64::from(header.num_posting_lists));
    encoder.varint(3, u64::from(header.num_documents));
    encoder.varint(4, u64::from(header.total_posting_lists));
    encoder.varint(5, u64::from(header.total_documents));
    encoder.varint(6, header.total_terms_in_collection);
    encoder.fixed64(7, header.average_document_length.to_bits());
    encoder.string(8, &header.description);
    encoder.into_bytes()
}

#[cfg(test)]
fn encode_posting_list(list: &CiffPostingList, encoded_size: usize) -> Result<Vec<u8>> {
    let mut encoder = Encoder::with_capacity(encoded_size);
    encoder.string(1, &list.term);
    encoder.varint(2, list.document_frequency);
    encoder.varint(3, list.collection_frequency);
    let mut previous = 0_u32;
    for posting in &list.postings {
        let gap = posting
            .document_id
            .checked_sub(previous)
            .ok_or_else(|| corrupt("posting ids are not increasing while encoding"))?;
        let nested_size = encoded_posting_len(gap, posting.term_frequency)?;
        let mut nested = Encoder::with_capacity(nested_size);
        nested.varint(1, u64::from(gap));
        nested.varint(2, u64::from(posting.term_frequency));
        encoder.bytes(4, &nested.into_bytes());
        previous = posting.document_id;
    }
    Ok(encoder.into_bytes())
}

#[cfg(test)]
fn encode_document_record(document: &CiffDocumentRecord, encoded_size: usize) -> Vec<u8> {
    let mut encoder = Encoder::with_capacity(encoded_size);
    encoder.varint(1, u64::from(document.document_id));
    encoder.string(2, &document.external_id);
    encoder.varint(3, u64::from(document.document_length));
    encoder.into_bytes()
}

fn encoded_header_len(header: &CiffHeader) -> Result<usize> {
    let mut size = 0_usize;
    add_varint_field_len(&mut size, 1, u64::from(header.version))?;
    add_varint_field_len(&mut size, 2, u64::from(header.num_posting_lists))?;
    add_varint_field_len(&mut size, 3, u64::from(header.num_documents))?;
    add_varint_field_len(&mut size, 4, u64::from(header.total_posting_lists))?;
    add_varint_field_len(&mut size, 5, u64::from(header.total_documents))?;
    add_varint_field_len(&mut size, 6, header.total_terms_in_collection)?;
    if header.average_document_length.to_bits() != 0 {
        checked_add(&mut size, key_len(7, WIRE_FIXED64)?)?;
        checked_add(&mut size, 8)?;
    }
    add_bytes_field_len(&mut size, 8, header.description.len())?;
    Ok(size)
}

fn encoded_posting_list_len(list: &CiffPostingList) -> Result<usize> {
    let mut size = 0_usize;
    add_bytes_field_len(&mut size, 1, list.term.len())?;
    add_varint_field_len(&mut size, 2, list.document_frequency)?;
    add_varint_field_len(&mut size, 3, list.collection_frequency)?;
    let mut previous = 0_u32;
    for posting in &list.postings {
        let gap = posting
            .document_id
            .checked_sub(previous)
            .ok_or_else(|| corrupt("posting ids are not increasing while sizing"))?;
        let nested = encoded_posting_len(gap, posting.term_frequency)?;
        add_bytes_field_len(&mut size, 4, nested)?;
        previous = posting.document_id;
    }
    Ok(size)
}

fn encoded_posting_len(document_gap: u32, term_frequency: u32) -> Result<usize> {
    let mut size = 0_usize;
    add_varint_field_len(&mut size, 1, u64::from(document_gap))?;
    add_varint_field_len(&mut size, 2, u64::from(term_frequency))?;
    Ok(size)
}

fn encoded_document_record_len(document: &CiffDocumentRecord) -> Result<usize> {
    let mut size = 0_usize;
    add_varint_field_len(&mut size, 1, u64::from(document.document_id))?;
    add_bytes_field_len(&mut size, 2, document.external_id.len())?;
    add_varint_field_len(&mut size, 3, u64::from(document.document_length))?;
    Ok(size)
}

fn add_varint_field_len(size: &mut usize, field: u32, value: u64) -> Result<()> {
    if value != 0 {
        checked_add(size, key_len(field, WIRE_VARINT)?)?;
        checked_add(size, varint_len(value))?;
    }
    Ok(())
}

fn add_bytes_field_len(size: &mut usize, field: u32, length: usize) -> Result<()> {
    if length != 0 {
        checked_add(size, key_len(field, WIRE_LENGTH_DELIMITED)?)?;
        checked_add(
            size,
            varint_len(
                u64::try_from(length)
                    .map_err(|_| Error::InvalidArgument("CIFF frame length exceeds u64".into()))?,
            ),
        )?;
        checked_add(size, length)?;
    }
    Ok(())
}

fn key_len(field: u32, wire_type: u8) -> Result<usize> {
    let key = u64::from(field)
        .checked_shl(3)
        .and_then(|value| value.checked_add(u64::from(wire_type)))
        .ok_or_else(|| Error::InvalidArgument("CIFF protobuf field key overflows".into()))?;
    Ok(varint_len(key))
}

fn varint_len(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

fn checked_add(size: &mut usize, additional: usize) -> Result<()> {
    *size = size
        .checked_add(additional)
        .ok_or_else(|| Error::InvalidArgument("CIFF encoded frame size overflows usize".into()))?;
    Ok(())
}

#[cfg(test)]
thread_local! {
    static WRITE_POSTING_LIST_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn contextual(error: Error, context: &str) -> Error {
    match error {
        Error::CorruptIndex(message) => Error::CorruptIndex(format!("{message} ({context})")),
        other => other,
    }
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::CorruptIndex(format!("CIFF: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_path() -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let number = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "indexsail-ciff-{}-{number}.ciff",
            std::process::id()
        ))
    }

    fn encoded_header(header: &CiffHeader) -> Vec<u8> {
        encode_header(header, encoded_header_len(header).unwrap())
    }

    fn encoded_posting_list(list: &CiffPostingList) -> Vec<u8> {
        encode_posting_list(list, encoded_posting_list_len(list).unwrap()).unwrap()
    }

    fn encoded_document(document: &CiffDocumentRecord) -> Vec<u8> {
        encode_document_record(document, encoded_document_record_len(document).unwrap())
    }

    // Independently transcribed from the official CIFF v1 .proto wire tags.
    // It deliberately includes a zero first d-gap and zero DocRecord id, which
    // proto3 omits from the wire and restores through scalar defaults.
    const OFFICIAL_WIRE_FIXTURE: &[u8] = &[
        0x1e, 0x08, 0x01, 0x10, 0x02, 0x18, 0x03, 0x20, 0x02, 0x28, 0x03, 0x30, 0x06, 0x39, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x42, 0x07, b'f', b'i', b'x', b't', b'u', b'r',
        b'e', 0x15, 0x0a, 0x05, b'a', b'p', b'p', b'l', b'e', 0x10, 0x02, 0x18, 0x03, 0x22, 0x02,
        0x10, 0x02, 0x22, 0x04, 0x08, 0x02, 0x10, 0x01, 0x12, 0x0a, 0x06, b'b', b'a', b'n', b'a',
        b'n', b'a', 0x10, 0x01, 0x18, 0x01, 0x22, 0x04, 0x08, 0x01, 0x10, 0x01, 0x05, 0x12, 0x01,
        b'A', 0x18, 0x02, 0x07, 0x08, 0x01, 0x12, 0x01, b'B', 0x18, 0x01, 0x07, 0x08, 0x02, 0x12,
        0x01, b'C', 0x18, 0x03,
    ];

    // Produced by the official PISA jsonl2ciff tool at CIFF commit
    // 0689e2a7536be4306c4eab5472e2bbe6a6696b58 from
    // {"id":"D1","vector":{"term":2.0}}. Its SHA-256 is
    // 35cacd057c276c444ab3a120161e45282eeb5eea91d799e4e5adcd1a98bfa5e2.
    // PISA stores a quantized learned impact in Posting.tf here, so cf (2)
    // legitimately exceeds total_terms_in_collection (1).
    const PISA_LEARNED_SPARSE_FIXTURE: &[u8] = &[
        0x30, 0x08, 0x01, 0x10, 0x01, 0x18, 0x01, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01, 0x39, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x42, 0x19, b'C', b'o', b'n', b'v', b'e', b'r',
        b't', b'e', b'd', b' ', b'f', b'r', b'o', b'm', b' ', b'J', b'S', b'O', b'N', b' ', b'l',
        b'i', b'n', b'e', b's', 0x0e, 0x0a, 0x04, b't', b'e', b'r', b'm', 0x10, 0x01, 0x18, 0x02,
        0x22, 0x02, 0x10, 0x02, 0x06, 0x12, 0x02, b'D', b'1', 0x18, 0x01,
    ];

    #[test]
    fn decodes_independent_official_wire_fixture() {
        let index = CiffIndex::read_from(
            &mut Cursor::new(OFFICIAL_WIRE_FIXTURE),
            CiffLimits::default(),
        )
        .unwrap();
        assert_eq!(index.header().version, crate::ciff::CIFF_FORMAT_VERSION);
        assert_eq!(
            index.header().average_document_length.to_bits(),
            2.0_f64.to_bits()
        );
        assert_eq!(
            index.posting_list("apple").unwrap().postings[1].document_id,
            2
        );
        assert_eq!(index.document(0).unwrap().external_id, "A");
    }

    #[test]
    fn canonical_writer_matches_independent_fixture_byte_for_byte() {
        let index = CiffIndex::read_from(
            &mut Cursor::new(OFFICIAL_WIRE_FIXTURE),
            CiffLimits::default(),
        )
        .unwrap();
        let mut written = Vec::new();
        index.write_to(&mut written, CiffLimits::default()).unwrap();
        assert_eq!(written, OFFICIAL_WIRE_FIXTURE);
    }

    #[test]
    fn accepts_pinned_pisa_learned_sparse_impact_fixture() {
        let index = CiffIndex::read_from(
            &mut Cursor::new(PISA_LEARNED_SPARSE_FIXTURE),
            CiffLimits::default(),
        )
        .unwrap();
        assert_eq!(index.header().total_terms_in_collection, 1);
        let list = index.posting_list("term").unwrap();
        assert_eq!(list.collection_frequency, 2);
        assert_eq!(list.postings[0].term_frequency, 2);
        assert_eq!(index.document(0).unwrap().document_length, 1);
        let mut canonical = Vec::new();
        index
            .write_to(&mut canonical, CiffLimits::default())
            .unwrap();
        assert_eq!(canonical, PISA_LEARNED_SPARSE_FIXTURE);
    }

    #[test]
    fn unknown_fields_are_skipped_for_forward_compatibility() {
        let mut bytes = OFFICIAL_WIRE_FIXTURE.to_vec();
        bytes[0] += 2;
        bytes.splice(31..31, [0x48, 0x01]);
        let index = CiffIndex::read_from(&mut Cursor::new(bytes), CiffLimits::default()).unwrap();
        assert_eq!(index.stats().contained_documents, 3);
    }

    #[test]
    fn rejects_truncation_overlong_lengths_and_trailing_messages() {
        for bytes in [vec![0x80], vec![0x02, 0x08], vec![0xff; 11]] {
            assert!(CiffIndex::read_from(&mut Cursor::new(bytes), CiffLimits::default()).is_err());
        }
        let mut bytes = OFFICIAL_WIRE_FIXTURE.to_vec();
        bytes.push(0);
        assert!(CiffIndex::read_from(&mut Cursor::new(bytes), CiffLimits::default()).is_err());
    }

    #[test]
    fn trailing_data_check_probes_exactly_one_byte() {
        struct OneByteAfterFixture {
            fixture: Cursor<Vec<u8>>,
            returned_trailing_byte: bool,
        }

        impl Read for OneByteAfterFixture {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.fixture.position() < self.fixture.get_ref().len() as u64 {
                    return self.fixture.read(buffer);
                }
                assert_eq!(buffer.len(), 1, "EOF detection must only probe one byte");
                assert!(
                    !self.returned_trailing_byte,
                    "reader must stop immediately after finding trailing data"
                );
                buffer[0] = 0x80;
                self.returned_trailing_byte = true;
                Ok(1)
            }
        }

        let mut reader = OneByteAfterFixture {
            fixture: Cursor::new(OFFICIAL_WIRE_FIXTURE.to_vec()),
            returned_trailing_byte: false,
        };
        let error = CiffIndex::read_from(&mut reader, CiffLimits::default()).unwrap_err();
        assert!(error.to_string().contains("unexpected message"));
    }

    #[test]
    fn rejects_declared_frame_and_collection_limits_before_bulk_allocation() {
        let limits = CiffLimits {
            max_frame_bytes: 29,
            ..CiffLimits::default()
        };
        assert!(CiffIndex::read_from(&mut Cursor::new(OFFICIAL_WIRE_FIXTURE), limits).is_err());

        let limits = CiffLimits {
            max_posting_lists: 1,
            ..CiffLimits::default()
        };
        assert!(CiffIndex::read_from(&mut Cursor::new(OFFICIAL_WIRE_FIXTURE), limits).is_err());

        let limits = CiffLimits {
            max_postings: 2,
            ..CiffLimits::default()
        };
        let error =
            CiffIndex::read_from(&mut Cursor::new(OFFICIAL_WIRE_FIXTURE), limits).unwrap_err();
        assert!(error.to_string().contains("postings exceed"));
    }

    #[test]
    fn empty_declared_list_fails_before_later_frames_are_requested() {
        let header = CiffHeader {
            version: 1,
            num_posting_lists: 2,
            num_documents: 0,
            total_posting_lists: 2,
            total_documents: 1,
            total_terms_in_collection: 1,
            average_document_length: 1.0,
            description: "zero-frame adversary".into(),
        };
        let mut bytes = Vec::new();
        write_delimited(&mut bytes, &encoded_header(&header)).unwrap();
        bytes.push(0); // A syntactically framed proto3-default PostingsList.
        let error =
            CiffIndex::read_from(&mut Cursor::new(bytes), CiffLimits::default()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("postings term must not be empty")
        );
    }

    #[test]
    fn streaming_reader_rejects_duplicate_terms_and_document_keys() {
        let list = CiffPostingList {
            term: "term".into(),
            document_frequency: 1,
            collection_frequency: 1,
            postings: vec![CiffPosting {
                document_id: 0,
                term_frequency: 1,
            }],
        };
        let mut duplicate_terms = Vec::new();
        write_delimited(
            &mut duplicate_terms,
            &encoded_header(&CiffHeader {
                version: 1,
                num_posting_lists: 2,
                num_documents: 0,
                total_posting_lists: 2,
                total_documents: 1,
                total_terms_in_collection: 1,
                average_document_length: 1.0,
                description: "duplicate-term adversary".into(),
            }),
        )
        .unwrap();
        for _ in 0..2 {
            write_delimited(&mut duplicate_terms, &encoded_posting_list(&list)).unwrap();
        }
        let error = CiffIndex::read_from(&mut Cursor::new(duplicate_terms), CiffLimits::default())
            .unwrap_err();
        assert!(error.to_string().contains("duplicate postings term"));

        for (second, expected) in [
            (
                CiffDocumentRecord {
                    document_id: 0,
                    external_id: "B".into(),
                    document_length: 0,
                },
                "duplicate document id",
            ),
            (
                CiffDocumentRecord {
                    document_id: 1,
                    external_id: "A".into(),
                    document_length: 0,
                },
                "duplicate collection document id",
            ),
        ] {
            let mut duplicate_documents = Vec::new();
            write_delimited(
                &mut duplicate_documents,
                &encoded_header(&CiffHeader {
                    version: 1,
                    num_posting_lists: 0,
                    num_documents: 2,
                    total_posting_lists: 0,
                    total_documents: 2,
                    total_terms_in_collection: 0,
                    average_document_length: 0.0,
                    description: "duplicate-document adversary".into(),
                }),
            )
            .unwrap();
            let first = CiffDocumentRecord {
                document_id: 0,
                external_id: "A".into(),
                document_length: 0,
            };
            write_delimited(&mut duplicate_documents, &encoded_document(&first)).unwrap();
            write_delimited(&mut duplicate_documents, &encoded_document(&second)).unwrap();
            let error =
                CiffIndex::read_from(&mut Cursor::new(duplicate_documents), CiffLimits::default())
                    .unwrap_err();
            assert!(error.to_string().contains(expected));
        }
    }

    #[test]
    fn rejects_duplicate_scalar_wrong_wire_and_invalid_utf8() {
        let mut duplicate = OFFICIAL_WIRE_FIXTURE.to_vec();
        duplicate[0] += 2;
        duplicate.splice(31..31, [0x08, 0x01]);
        assert!(CiffIndex::read_from(&mut Cursor::new(duplicate), CiffLimits::default()).is_err());

        let mut wrong_wire = OFFICIAL_WIRE_FIXTURE.to_vec();
        wrong_wire[1] = 0x0a;
        assert!(CiffIndex::read_from(&mut Cursor::new(wrong_wire), CiffLimits::default()).is_err());

        let mut invalid_utf8 = OFFICIAL_WIRE_FIXTURE.to_vec();
        invalid_utf8[35] = 0xff;
        assert!(
            CiffIndex::read_from(&mut Cursor::new(invalid_utf8), CiffLimits::default()).is_err()
        );
    }

    #[test]
    fn posting_prefix_sum_and_scalar_ranges_are_checked() {
        let mut list = Encoder::default();
        list.string(1, "term");
        list.varint(2, 3);
        list.varint(3, 3);
        for gap in [2_147_483_647_u64, 2_147_483_647_u64, 2] {
            let mut posting = Encoder::default();
            posting.varint(1, gap);
            posting.varint(2, 1);
            list.bytes(4, &posting.into_bytes());
        }
        let error = parse_posting_list(
            &list.into_bytes(),
            CiffLimits::default(),
            CiffLimits::default().max_postings,
        )
        .unwrap_err();
        assert!(error.to_string().contains("prefix sum"));

        let mut posting = Encoder::default();
        posting.varint(2, 2_147_483_648_u64);
        assert!(parse_posting(&posting.into_bytes()).is_err());
    }

    #[test]
    fn unsupported_version_is_distinct() {
        let mut bytes = OFFICIAL_WIRE_FIXTURE.to_vec();
        bytes[2] = 2;
        assert!(matches!(
            CiffIndex::read_from(&mut Cursor::new(bytes), CiffLimits::default()),
            Err(Error::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn save_policy_failure_does_not_truncate_an_existing_destination() {
        let index = CiffIndex::read_from(
            &mut Cursor::new(OFFICIAL_WIRE_FIXTURE),
            CiffLimits::default(),
        )
        .unwrap();
        let path = temp_path();
        std::fs::write(&path, b"sentinel").unwrap();
        let limits = CiffLimits {
            max_frame_bytes: 29,
            ..CiffLimits::default()
        };
        assert!(index.save_with_limits(&path, limits).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"sentinel");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn oversized_posting_frame_is_rejected_before_encoder_allocation() {
        let postings = (0..50_000)
            .map(|document_id| CiffPosting {
                document_id,
                term_frequency: 1,
            })
            .collect::<Vec<_>>();
        let list = CiffPostingList {
            term: "large".into(),
            document_frequency: postings.len() as u64,
            collection_frequency: postings.len() as u64,
            postings,
        };
        let documents = (0..50_000)
            .map(|document_id| CiffDocumentRecord {
                document_id,
                external_id: format!("D{document_id}"),
                document_length: 1,
            })
            .collect::<Vec<_>>();
        let header = CiffHeader {
            version: 1,
            num_posting_lists: 1,
            num_documents: 50_000,
            total_posting_lists: 1,
            total_documents: 50_000,
            total_terms_in_collection: 50_000,
            average_document_length: 1.0,
            description: "bounded writer".into(),
        };
        let index =
            CiffIndex::from_parts(header, vec![list], documents, CiffLimits::default()).unwrap();
        let path = temp_path();
        std::fs::write(&path, b"old index").unwrap();
        WRITE_POSTING_LIST_CALLS.set(0);
        let limits = CiffLimits {
            max_frame_bytes: 1_024,
            ..CiffLimits::default()
        };
        let error = index.save_with_limits(&path, limits).unwrap_err();
        assert!(error.to_string().contains("output frame"));
        assert_eq!(WRITE_POSTING_LIST_CALLS.get(), 0);
        assert_eq!(std::fs::read(&path).unwrap(), b"old index");
        std::fs::remove_file(path).unwrap();
    }
}
