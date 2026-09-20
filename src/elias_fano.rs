//! Canonical Elias–Fano encoding of a strictly increasing `u32` sequence.
//!
//! The wire representation is `last: u32`, packed low bits, then a unary high
//! bitmap, with bits numbered least-significant first in each byte.  The
//! sequence length is supplied by the enclosing posting-list record.  This
//! deliberately uses no sampling table: its iterator is linear and native
//! loading materializes postings before search.

use crate::error::{Error, Result};

const MAX_VALUES: usize = 20_000_000;
const MAX_BYTES: usize = 512 * 1024 * 1024;

fn shape(last: u32, count: usize) -> Result<(u32, usize, usize, usize)> {
    if count == 0
        || count > MAX_VALUES
        || u64::try_from(count).unwrap_or(u64::MAX) > u64::from(last) + 1
    {
        return Err(Error::CorruptIndex(
            "invalid Elias–Fano sequence length".into(),
        ));
    }
    let ratio = (u64::from(last) + 1)
        / u64::try_from(count).map_err(|_| {
            Error::CorruptIndex("Elias–Fano sequence length does not fit u64".into())
        })?;
    let lower_width = ratio.ilog2();
    let low_bits = u64::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(u64::from(lower_width)))
        .ok_or_else(|| Error::CorruptIndex("Elias–Fano low-bit length overflow".into()))?;
    let high_bits = (u64::from(last) >> lower_width)
        .checked_add(u64::try_from(count).unwrap_or(u64::MAX))
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| Error::CorruptIndex("Elias–Fano high-bit length overflow".into()))?;
    let low_bytes = usize::try_from(low_bits.div_ceil(8))
        .map_err(|_| Error::CorruptIndex("Elias–Fano low bits do not fit usize".into()))?;
    let high_bytes = usize::try_from(high_bits.div_ceil(8))
        .map_err(|_| Error::CorruptIndex("Elias–Fano high bits do not fit usize".into()))?;
    let total = 4_usize
        .checked_add(low_bytes)
        .and_then(|size| size.checked_add(high_bytes))
        .filter(|size| *size <= MAX_BYTES)
        .ok_or_else(|| {
            Error::CorruptIndex("Elias–Fano sequence exceeds byte safety limit".into())
        })?;
    Ok((
        lower_width,
        low_bytes,
        usize::try_from(high_bits)
            .map_err(|_| Error::CorruptIndex("Elias–Fano high bits do not fit usize".into()))?,
        total,
    ))
}

fn bit(bytes: &[u8], position: usize) -> bool {
    bytes[position / 8] & (1 << (position % 8)) != 0
}

fn set_bit(bytes: &mut [u8], position: usize) {
    bytes[position / 8] |= 1 << (position % 8);
}

fn read_low(bytes: &[u8], offset: usize, width: u32) -> u64 {
    let mut value = 0_u64;
    for digit in 0..width {
        if bit(bytes, offset + digit as usize) {
            value |= 1_u64 << digit;
        }
    }
    value
}

fn canonical_padding(bytes: &[u8], bit_count: usize) -> bool {
    (bit_count..bytes.len() * 8).all(|position| !bit(bytes, position))
}

pub(crate) fn encode(values: &[u32]) -> Result<Vec<u8>> {
    let &last = values
        .last()
        .ok_or_else(|| Error::CorruptIndex("empty Elias–Fano sequence".into()))?;
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::CorruptIndex(
            "Elias–Fano values must be strictly increasing".into(),
        ));
    }
    let (width, low_bytes, _high_bits, total) = shape(last, values.len())?;
    let mut result = Vec::new();
    result.try_reserve_exact(total).map_err(|_| {
        Error::InvalidArgument("Elias–Fano sequence cannot be allocated safely".into())
    })?;
    result.resize(total, 0);
    result[..4].copy_from_slice(&last.to_le_bytes());
    for (ordinal, &value) in values.iter().enumerate() {
        let lower = u64::from(value) & ((1_u64 << width) - 1);
        for digit in 0..width {
            if lower & (1_u64 << digit) != 0 {
                set_bit(
                    &mut result[4..4 + low_bytes],
                    ordinal * width as usize + digit as usize,
                );
            }
        }
        let high_position = usize::try_from(u64::from(value) >> width)
            .ok()
            .and_then(|high| high.checked_add(ordinal))
            .ok_or_else(|| Error::CorruptIndex("Elias–Fano high position overflow".into()))?;
        set_bit(&mut result[4 + low_bytes..], high_position);
    }
    Ok(result)
}

/// Sequential decoder. The constructor rejects malformed length/padding;
/// iteration validates the unary ranks, strict order, and declared maximum.
pub(crate) struct Iter<'a> {
    low: &'a [u8],
    high: &'a [u8],
    width: u32,
    count: usize,
    high_bits: usize,
    last: u32,
    ordinal: usize,
    next_high: usize,
    previous: Option<u32>,
}

impl<'a> Iter<'a> {
    pub(crate) fn new(bytes: &'a [u8], count: usize) -> Result<Self> {
        let last = u32::from_le_bytes(
            bytes
                .get(..4)
                .ok_or_else(|| Error::CorruptIndex("truncated Elias–Fano header".into()))?
                .try_into()
                .map_err(|_| Error::CorruptIndex("truncated Elias–Fano header".into()))?,
        );
        let (width, low_bytes, high_bits, total) = shape(last, count)?;
        if bytes.len() != total {
            return Err(Error::CorruptIndex(
                "Elias–Fano byte length mismatch".into(),
            ));
        }
        let low = &bytes[4..4 + low_bytes];
        let high = &bytes[4 + low_bytes..];
        if !canonical_padding(low, count * width as usize) || !canonical_padding(high, high_bits) {
            return Err(Error::CorruptIndex("nonzero Elias–Fano padding".into()));
        }
        Ok(Self {
            low,
            high,
            width,
            count,
            high_bits,
            last,
            ordinal: 0,
            next_high: 0,
            previous: None,
        })
    }
}

impl Iterator for Iter<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ordinal == self.count {
            return None;
        }
        while self.next_high < self.high_bits && !bit(self.high, self.next_high) {
            self.next_high += 1;
        }
        if self.next_high == self.high_bits {
            self.ordinal = self.count;
            return Some(Err(Error::CorruptIndex(
                "Elias–Fano bitmap has too few set bits".into(),
            )));
        }
        let upper = self.next_high - self.ordinal;
        let lower = read_low(self.low, self.ordinal * self.width as usize, self.width);
        let value = (u64::try_from(upper).unwrap_or(u64::MAX) << self.width) | lower;
        let value = u32::try_from(value)
            .map_err(|_| Error::CorruptIndex("Elias–Fano value exceeds u32".into()));
        self.next_high += 1;
        self.ordinal += 1;
        let checked = value.and_then(|value| {
            if self.previous.is_some_and(|previous| value <= previous) || value > self.last {
                return Err(Error::CorruptIndex(
                    "Elias–Fano values are not strictly increasing".into(),
                ));
            }
            if self.ordinal == self.count && value != self.last {
                return Err(Error::CorruptIndex(
                    "Elias–Fano maximum does not match final value".into(),
                ));
            }
            if self.ordinal == self.count
                && (self.next_high..self.high_bits).any(|position| bit(self.high, position))
            {
                return Err(Error::CorruptIndex(
                    "Elias–Fano bitmap has extra set bits".into(),
                ));
            }
            self.previous = Some(value);
            Ok(value)
        });
        Some(checked)
    }
}

pub(crate) fn decode(bytes: &[u8], count: usize) -> Result<Vec<u32>> {
    let mut output = Vec::new();
    let iter = Iter::new(bytes, count)?;
    output
        .try_reserve_exact(count)
        .map_err(|_| Error::CorruptIndex("Elias–Fano values cannot be allocated safely".into()))?;
    for value in iter {
        output.push(value?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_small_bit_oracles() {
        // [0, 2, 3], universe 4, n=3 => width 0, high bits at 0,3,5.
        assert_eq!(encode(&[0, 2, 3]).unwrap(), [3, 0, 0, 0, 0x29]);
        // [1, 3], universe 4, n=2 => width 1, lows 1,1; highs 0,2.
        assert_eq!(encode(&[1, 3]).unwrap(), [3, 0, 0, 0, 0x03, 0x05]);
        for sequence in [&[0, 2, 3][..], &[1, 3][..]] {
            assert_eq!(
                decode(&encode(sequence).unwrap(), sequence.len()).unwrap(),
                sequence
            );
        }
    }

    #[test]
    fn boundary_and_deterministic_sequences() {
        for sequence in [
            vec![0],
            vec![u32::MAX],
            vec![0, u32::MAX],
            (0..1024).collect(),
            (0..1024).map(|n| n * 4000).collect(),
        ] {
            let encoded = encode(&sequence).unwrap();
            assert_eq!(decode(&encoded, sequence.len()).unwrap(), sequence);
            assert_eq!(
                encode(&decode(&encoded, sequence.len()).unwrap()).unwrap(),
                encoded
            );
        }
        let mut state = 11_u64;
        for length in 1..64 {
            let mut sequence = Vec::new();
            let mut value = 0_u32;
            for _ in 0..length {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                value += u32::try_from(state % 1000 + 1).unwrap();
                sequence.push(value);
            }
            assert_eq!(
                decode(&encode(&sequence).unwrap(), length).unwrap(),
                sequence
            );
        }
    }

    #[test]
    fn rejects_invalid_shape_and_corrupt_bits() {
        assert!(encode(&[]).is_err());
        assert!(encode(&[1, 1]).is_err());
        assert!(encode(&[2, 1]).is_err());
        assert!(decode(&[0, 0, 0], 1).is_err());
        assert!(decode(&encode(&[0]).unwrap(), 2).is_err());
        let original = encode(&[1, 3]).unwrap();
        assert!(decode(&original[..original.len() - 1], 2).is_err());
        let mut trailing = original.clone();
        trailing.push(0);
        assert!(decode(&trailing, 2).is_err());
        let mut bad_padding = original.clone();
        *bad_padding.last_mut().unwrap() |= 0x80;
        assert!(decode(&bad_padding, 2).is_err());
        let mut missing = original.clone();
        *missing.last_mut().unwrap() = 0;
        assert!(decode(&missing, 2).is_err());
        let mut duplicate = original.clone();
        duplicate[4] = 0;
        assert!(decode(&duplicate, 2).is_err());
    }
}
