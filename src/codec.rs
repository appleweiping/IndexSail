//! Variable-byte and Elias–Fano posting-list layouts.
//!
//! In the default v3 layout, strictly increasing document IDs and positions
//! become positive gaps. The opt-in v4 layout stores document IDs as a
//! monotone Elias–Fano sequence, with frequencies and positive position gaps
//! retaining base-128 variable bytes.

use crate::elias_fano;
use crate::error::{Error, Result};
use crate::index::Posting;
use crate::interpolative;

const MAX_POSITIONS_PER_POSTING: usize = 20_000_000;

/// Size comparison for the posting-list representation, excluding dictionary
/// strings and block-length prefixes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PostingCodecStats {
    pub posting_lists: usize,
    pub postings: usize,
    pub positions: usize,
    pub uncompressed_bytes: u64,
    pub encoded_bytes: u64,
}

impl PostingCodecStats {
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(self) -> f64 {
        if self.uncompressed_bytes == 0 {
            return 0.0;
        }
        self.encoded_bytes as f64 / self.uncompressed_bytes as f64
    }
}

pub(crate) fn encode_postings(postings: &[Posting]) -> Result<Vec<u8>> {
    for posting in postings {
        if posting.term_frequency == 0 || posting.term_frequency as usize != posting.positions.len()
        {
            return Err(Error::CorruptIndex(
                "term frequency does not match positions".into(),
            ));
        }
        validate_position_count(posting.positions.len())?;
    }
    let capacity = postings
        .iter()
        .try_fold(0_usize, |total, posting| {
            posting
                .positions
                .len()
                .checked_mul(5)
                .and_then(|position_bytes| position_bytes.checked_add(10))
                .and_then(|posting_bytes| total.checked_add(posting_bytes))
        })
        .ok_or_else(|| Error::InvalidArgument("posting block is too large".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| Error::InvalidArgument("posting block cannot be allocated safely".into()))?;
    let mut previous_doc: Option<u32> = None;

    for posting in postings {
        let document_gap = match previous_doc {
            None => posting.doc_id.checked_add(1),
            Some(previous) => posting.doc_id.checked_sub(previous),
        }
        .filter(|gap| *gap > 0)
        .ok_or_else(|| {
            Error::CorruptIndex("posting document ids are not strictly increasing".into())
        })?;
        encode_u32(document_gap, &mut output);
        encode_u32(posting.term_frequency, &mut output);

        let mut previous_position: Option<u32> = None;
        for &position in &posting.positions {
            let position_gap = match previous_position {
                None => position.checked_add(1),
                Some(previous) => position.checked_sub(previous),
            }
            .filter(|gap| *gap > 0)
            .ok_or_else(|| {
                Error::CorruptIndex("term positions are not strictly increasing".into())
            })?;
            encode_u32(position_gap, &mut output);
            previous_position = Some(position);
        }
        previous_doc = Some(posting.doc_id);
    }
    Ok(output)
}

pub(crate) fn decode_postings(bytes: &[u8], posting_count: usize) -> Result<Vec<Posting>> {
    if posting_count > bytes.len() / 3 {
        return Err(Error::CorruptIndex(
            "compressed posting count cannot fit in block".into(),
        ));
    }
    let mut cursor = 0;
    let mut previous_doc: Option<u32> = None;
    let mut postings = Vec::new();
    postings
        .try_reserve_exact(posting_count)
        .map_err(|_| Error::CorruptIndex("postings cannot be allocated safely".into()))?;
    for _ in 0..posting_count {
        let document_gap = decode_u32(bytes, &mut cursor)?;
        if document_gap == 0 {
            return Err(Error::CorruptIndex(
                "posting document gap must be positive".into(),
            ));
        }
        let doc_id = match previous_doc {
            None => document_gap.checked_sub(1),
            Some(previous) => previous.checked_add(document_gap),
        }
        .ok_or_else(|| Error::CorruptIndex("posting document id overflow".into()))?;

        let term_frequency = decode_u32(bytes, &mut cursor)?;
        if term_frequency == 0 {
            return Err(Error::CorruptIndex(
                "posting term frequency must be positive".into(),
            ));
        }
        let position_count = usize::try_from(term_frequency)
            .map_err(|_| Error::CorruptIndex("position count does not fit usize".into()))?;
        validate_position_count(position_count)?;
        if position_count > bytes.len().saturating_sub(cursor) {
            return Err(Error::CorruptIndex(
                "position count cannot fit in compressed posting block".into(),
            ));
        }
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(position_count)
            .map_err(|_| Error::CorruptIndex("positions cannot be allocated safely".into()))?;
        let mut previous_position: Option<u32> = None;
        for _ in 0..position_count {
            let position_gap = decode_u32(bytes, &mut cursor)?;
            if position_gap == 0 {
                return Err(Error::CorruptIndex(
                    "posting position gap must be positive".into(),
                ));
            }
            let position = match previous_position {
                None => position_gap.checked_sub(1),
                Some(previous) => previous.checked_add(position_gap),
            }
            .ok_or_else(|| Error::CorruptIndex("posting position overflow".into()))?;
            positions.push(position);
            previous_position = Some(position);
        }
        postings.push(Posting {
            doc_id,
            term_frequency,
            positions,
        });
        previous_doc = Some(doc_id);
    }
    if cursor != bytes.len() {
        return Err(Error::CorruptIndex(
            "trailing bytes in compressed posting block".into(),
        ));
    }
    Ok(postings)
}

/// Version 4 block: `u32` Elias–Fano byte length, canonical Elias–Fano
/// document IDs, then varbyte frequency and position gaps per document.
pub(crate) fn encode_elias_fano_postings(postings: &[Posting]) -> Result<Vec<u8>> {
    encode_monotone_postings(postings, elias_fano::encode, "Elias–Fano")
}

/// Version 5 block: `u32` interpolative byte length, canonical interpolative
/// document IDs, then unchanged varbyte frequency and position gaps.
pub(crate) fn encode_interpolative_postings(postings: &[Posting]) -> Result<Vec<u8>> {
    encode_monotone_postings(postings, interpolative::encode, "interpolative")
}

fn encode_monotone_postings(
    postings: &[Posting],
    encode_docs: fn(&[u32]) -> Result<Vec<u8>>,
    label: &str,
) -> Result<Vec<u8>> {
    let mut docs = Vec::new();
    docs.try_reserve_exact(postings.len())
        .map_err(|_| Error::InvalidArgument("posting IDs cannot be allocated safely".into()))?;
    for posting in postings {
        if posting.term_frequency == 0 || posting.term_frequency as usize != posting.positions.len()
        {
            return Err(Error::CorruptIndex(
                "term frequency does not match positions".into(),
            ));
        }
        validate_position_count(posting.positions.len())?;
        docs.push(posting.doc_id);
    }
    let doc_block = encode_docs(&docs)?;
    let encoded_docs_len = u32::try_from(doc_block.len())
        .map_err(|_| Error::InvalidArgument(format!("{label} document block exceeds u32")))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(4 + doc_block.len())
        .map_err(|_| Error::InvalidArgument("posting block cannot be allocated safely".into()))?;
    output.extend_from_slice(&encoded_docs_len.to_le_bytes());
    output.extend_from_slice(&doc_block);
    for posting in postings {
        encode_u32(posting.term_frequency, &mut output);
        let mut previous_position: Option<u32> = None;
        for &position in &posting.positions {
            let position_gap = match previous_position {
                None => position.checked_add(1),
                Some(previous) => position.checked_sub(previous),
            }
            .filter(|gap| *gap > 0)
            .ok_or_else(|| {
                Error::CorruptIndex("term positions are not strictly increasing".into())
            })?;
            encode_u32(position_gap, &mut output);
            previous_position = Some(position);
        }
    }
    Ok(output)
}

pub(crate) fn decode_elias_fano_postings(
    bytes: &[u8],
    posting_count: usize,
) -> Result<Vec<Posting>> {
    decode_monotone_postings(bytes, posting_count, elias_fano::decode, "Elias–Fano")
}

pub(crate) fn decode_interpolative_postings(
    bytes: &[u8],
    posting_count: usize,
) -> Result<Vec<Posting>> {
    decode_monotone_postings(bytes, posting_count, interpolative::decode, "interpolative")
}

fn decode_monotone_postings(
    bytes: &[u8],
    posting_count: usize,
    decode_docs: fn(&[u8], usize) -> Result<Vec<u32>>,
    label: &str,
) -> Result<Vec<Posting>> {
    let encoded_docs_len = u32::from_le_bytes(
        bytes
            .get(..4)
            .ok_or_else(|| Error::CorruptIndex(format!("truncated {label} posting header")))?
            .try_into()
            .map_err(|_| Error::CorruptIndex(format!("truncated {label} posting header")))?,
    ) as usize;
    let docs_end = 4_usize
        .checked_add(encoded_docs_len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| Error::CorruptIndex(format!("truncated {label} document block")))?;
    if posting_count > (bytes.len() - docs_end) / 2 {
        return Err(Error::CorruptIndex(format!(
            "postings cannot fit in {label} block"
        )));
    }
    let docs = decode_docs(&bytes[4..docs_end], posting_count)?;
    let mut postings = Vec::new();
    postings
        .try_reserve_exact(posting_count)
        .map_err(|_| Error::CorruptIndex("postings cannot be allocated safely".into()))?;
    let mut cursor = docs_end;
    for doc_id in docs {
        let term_frequency = decode_u32(bytes, &mut cursor)?;
        if term_frequency == 0 {
            return Err(Error::CorruptIndex(
                "posting term frequency must be positive".into(),
            ));
        }
        let position_count = usize::try_from(term_frequency)
            .map_err(|_| Error::CorruptIndex("position count does not fit usize".into()))?;
        validate_position_count(position_count)?;
        if position_count > bytes.len().saturating_sub(cursor) {
            return Err(Error::CorruptIndex(
                "position count cannot fit in compressed posting block".into(),
            ));
        }
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(position_count)
            .map_err(|_| Error::CorruptIndex("positions cannot be allocated safely".into()))?;
        let mut previous_position: Option<u32> = None;
        for _ in 0..position_count {
            let gap = decode_u32(bytes, &mut cursor)?;
            if gap == 0 {
                return Err(Error::CorruptIndex(
                    "posting position gap must be positive".into(),
                ));
            }
            let position = match previous_position {
                None => gap.checked_sub(1),
                Some(previous) => previous.checked_add(gap),
            }
            .ok_or_else(|| Error::CorruptIndex("posting position overflow".into()))?;
            positions.push(position);
            previous_position = Some(position);
        }
        postings.push(Posting {
            doc_id,
            term_frequency,
            positions,
        });
    }
    if cursor != bytes.len() {
        return Err(Error::CorruptIndex(format!(
            "trailing bytes in {label} posting block"
        )));
    }
    Ok(postings)
}

fn validate_position_count(position_count: usize) -> Result<()> {
    if position_count > MAX_POSITIONS_PER_POSTING {
        return Err(Error::CorruptIndex(format!(
            "posting position count exceeds {MAX_POSITIONS_PER_POSTING} item safety limit"
        )));
    }
    Ok(())
}

pub(crate) fn encoded_posting_bytes(postings: &[Posting]) -> Result<usize> {
    Ok(encode_postings(postings)?.len())
}

fn encode_u32(mut value: u32, output: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            return;
        }
    }
}

fn decode_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let mut value = 0_u32;
    for shift in (0..=28).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| Error::CorruptIndex("truncated variable-byte integer".into()))?;
        *cursor += 1;
        if shift == 28 && byte & 0xf0 != 0 {
            return Err(Error::CorruptIndex(
                "variable-byte integer exceeds u32".into(),
            ));
        }
        value |= u32::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Error::CorruptIndex(
        "unterminated variable-byte integer".into(),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Checksum {
    value: u64,
}

impl Checksum {
    pub(crate) const fn new() -> Self {
        Self {
            value: 0xcbf2_9ce4_8422_2325_u64,
        }
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.value ^= u64::from(byte);
            self.value = self.value.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub(crate) const fn finish(self) -> u64 {
        self.value
    }
}

pub(crate) fn checksum(bytes: &[u8]) -> u64 {
    let mut checksum = Checksum::new();
    checksum.update(bytes);
    checksum.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posting(doc_id: u32, positions: &[u32]) -> Posting {
        Posting {
            doc_id,
            term_frequency: u32::try_from(positions.len()).unwrap(),
            positions: positions.to_vec(),
        }
    }

    #[test]
    fn posting_codec_round_trips_large_gaps() {
        let original = vec![
            posting(0, &[0, 1, 127, 128]),
            posting(129, &[4, 65_535]),
            posting(u32::MAX, &[u32::MAX - 1]),
        ];
        let encoded = encode_postings(&original).unwrap();
        assert_eq!(decode_postings(&encoded, original.len()).unwrap(), original);
    }

    #[test]
    fn elias_fano_posting_codec_round_trips_and_rejects_boundary_corruption() {
        let original = vec![
            posting(0, &[0, 127]),
            posting(4, &[3, 128, 65_535]),
            posting(u32::MAX, &[u32::MAX - 1]),
        ];
        let block = encode_elias_fano_postings(&original).unwrap();
        assert_eq!(decode_elias_fano_postings(&block, 3).unwrap(), original);
        assert!(decode_elias_fano_postings(&block[..3], 3).is_err());
        assert!(decode_elias_fano_postings(&block[..block.len() - 1], 3).is_err());
        assert!(decode_elias_fano_postings(&block, 2).is_err());
        let mut trailing = block.clone();
        trailing.push(0);
        assert!(decode_elias_fano_postings(&trailing, 3).is_err());
        let mut bad_length = block.clone();
        bad_length[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_elias_fano_postings(&bad_length, 3).is_err());
        let mut bad_frequency = block.clone();
        let docs_len = u32::from_le_bytes(block[..4].try_into().unwrap()) as usize;
        bad_frequency[4 + docs_len] = 0;
        assert!(decode_elias_fano_postings(&bad_frequency, 3).is_err());
        assert!(encode_elias_fano_postings(&[posting(2, &[0]), posting(2, &[1])]).is_err());
        assert!(encode_elias_fano_postings(&[posting(2, &[1, 1])]).is_err());
    }

    #[test]
    fn elias_fano_dense_document_ids_reduce_posting_bytes() {
        let postings = (0..1_000)
            .map(|doc_id| posting(doc_id, &[0]))
            .collect::<Vec<_>>();
        let varbyte = encode_postings(&postings).unwrap();
        let elias_fano = encode_elias_fano_postings(&postings).unwrap();
        assert!(elias_fano.len() < varbyte.len());
        assert_eq!(
            decode_elias_fano_postings(&elias_fano, 1_000).unwrap(),
            postings
        );
    }

    #[test]
    fn interpolative_posting_codec_round_trips_and_rejects_boundary_corruption() {
        let original = vec![
            posting(0, &[0, 127]),
            posting(4, &[3, 128, 65_535]),
            posting(u32::MAX, &[u32::MAX - 1]),
        ];
        let block = encode_interpolative_postings(&original).unwrap();
        assert_eq!(decode_interpolative_postings(&block, 3).unwrap(), original);
        assert!(decode_interpolative_postings(&block[..3], 3).is_err());
        assert!(decode_interpolative_postings(&block[..block.len() - 1], 3).is_err());
        assert!(decode_interpolative_postings(&block, 2).is_err());
        let mut trailing = block.clone();
        trailing.push(0);
        assert!(decode_interpolative_postings(&trailing, 3).is_err());
        let mut bad_length = block.clone();
        bad_length[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_interpolative_postings(&bad_length, 3).is_err());
        let mut bad_frequency = block.clone();
        let docs_len = u32::from_le_bytes(block[..4].try_into().unwrap()) as usize;
        bad_frequency[4 + docs_len] = 0;
        assert!(decode_interpolative_postings(&bad_frequency, 3).is_err());
        assert!(encode_interpolative_postings(&[posting(2, &[0]), posting(2, &[1])]).is_err());
        assert!(encode_interpolative_postings(&[posting(2, &[1, 1])]).is_err());
    }

    #[test]
    fn interpolative_dense_document_ids_reduce_posting_bytes() {
        let postings = (0..1_000)
            .map(|doc_id| posting(doc_id, &[0]))
            .collect::<Vec<_>>();
        let varbyte = encode_postings(&postings).unwrap();
        let interpolative = encode_interpolative_postings(&postings).unwrap();
        assert!(interpolative.len() < varbyte.len());
        assert_eq!(
            decode_interpolative_postings(&interpolative, 1_000).unwrap(),
            postings
        );
    }

    #[test]
    fn interpolative_header_can_expand_a_sparse_posting_list() {
        let postings = [posting(0, &[0]), posting(u32::MAX, &[1])];
        let varbyte = encode_postings(&postings).unwrap();
        let interpolation = encode_interpolative_postings(&postings).unwrap();
        assert!(interpolation.len() > varbyte.len());
        assert_eq!(
            decode_interpolative_postings(&interpolation, 2).unwrap(),
            postings
        );
    }

    #[test]
    fn delta_varbyte_is_smaller_for_dense_postings() {
        let postings = (0..1_000)
            .map(|doc_id| posting(doc_id, &[0, 3, 7]))
            .collect::<Vec<_>>();
        let encoded = encode_postings(&postings).unwrap();
        let fixed_width = postings.len() * (3 + 3) * size_of::<u32>();
        assert!(encoded.len() < fixed_width / 3);
    }

    #[test]
    fn decoder_rejects_truncation_trailing_data_and_overflow() {
        let encoded = encode_postings(&[posting(5, &[2, 8])]).unwrap();
        assert!(decode_postings(&encoded[..encoded.len() - 1], 1).is_err());

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode_postings(&trailing, 1).is_err());

        assert!(decode_postings(&[0xff, 0xff, 0xff, 0xff, 0x10], 1).is_err());
        assert!(decode_postings(&[1, 1, 1], 2).is_err());
        assert!(decode_postings(&[1, 0xff, 0xff, 0xff, 0xff, 0x07], 1).is_err());
    }

    #[test]
    fn encoder_rejects_non_monotonic_input() {
        let unsorted = vec![posting(2, &[0]), posting(1, &[0])];
        assert!(encode_postings(&unsorted).is_err());
        assert!(encode_postings(&[posting(1, &[2, 2])]).is_err());
    }

    #[test]
    fn malformed_frequency_and_zero_gaps_are_rejected_at_the_codec_boundary() {
        assert!(
            encode_postings(&[Posting {
                doc_id: 0,
                term_frequency: 0,
                positions: Vec::new(),
            }])
            .unwrap_err()
            .to_string()
            .contains("term frequency does not match positions")
        );
        assert!(
            encode_postings(&[Posting {
                doc_id: 0,
                term_frequency: 2,
                positions: vec![0],
            }])
            .unwrap_err()
            .to_string()
            .contains("term frequency does not match positions")
        );
        for (block, reason) in [
            ([0, 1, 1], "document gap must be positive"),
            ([1, 0, 1], "term frequency must be positive"),
            ([1, 1, 0], "position gap must be positive"),
        ] {
            assert!(
                decode_postings(&block, 1)
                    .unwrap_err()
                    .to_string()
                    .contains(reason)
            );
        }
        assert_eq!(
            PostingCodecStats::default().ratio().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            PostingCodecStats {
                uncompressed_bytes: 8,
                encoded_bytes: 2,
                ..PostingCodecStats::default()
            }
            .ratio()
            .to_bits(),
            0.25_f64.to_bits()
        );
    }

    #[test]
    fn encoder_and_decoder_share_the_position_count_limit() {
        assert!(validate_position_count(MAX_POSITIONS_PER_POSTING).is_ok());
        assert!(validate_position_count(MAX_POSITIONS_PER_POSTING + 1).is_err());
    }

    #[test]
    fn checksum_is_stable_and_sensitive() {
        assert_eq!(checksum(b"IndexSail"), 0x7db0_e97a_206f_511e);
        assert_ne!(checksum(b"IndexSail"), checksum(b"indexsail"));
    }

    #[test]
    fn incremental_checksum_matches_one_shot_checksum() {
        let bytes = b"a deterministic payload split across arbitrary chunks";
        let mut incremental = Checksum::new();
        incremental.update(&bytes[..7]);
        incremental.update(&bytes[7..31]);
        incremental.update(&bytes[31..]);
        assert_eq!(incremental.finish(), checksum(bytes));
    }

    #[test]
    fn many_deterministic_posting_lists_round_trip() {
        let mut state = 17_u64;
        for list_number in 0_u32..128 {
            let mut postings = Vec::new();
            let mut doc_id = list_number;
            for _ in 0..=(list_number % 31) {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                doc_id = doc_id
                    .checked_add(u32::try_from(state % 10_000 + 1).unwrap())
                    .unwrap();
                let mut positions = Vec::new();
                let mut position = 0_u32;
                for _ in 0..=(state % 7) {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    position = position
                        .checked_add(u32::try_from(state % 1_000 + 1).unwrap())
                        .unwrap();
                    positions.push(position);
                }
                postings.push(posting(doc_id, &positions));
            }
            let encoded = encode_postings(&postings).unwrap();
            assert_eq!(decode_postings(&encoded, postings.len()).unwrap(), postings);
        }
    }
}
