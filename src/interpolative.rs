//! Strict interpolative coding for monotone posting document IDs.
//!
//! The last ID is stored directly. The other IDs are recursively split at
//! their median and coded within the rank-adjusted interval left by their
//! already-known neighbors. Each median offset uses a truncated binary
//! codeword, written most-significant bit first into an LSB-first byte
//! stream. This is an `IndexSail` format, not PISA's byte layout.

use crate::error::{Error, Result};

const MAX_VALUES: usize = 20_000_000;
const MAX_BITS_PER_VALUE: usize = 32;

struct BitWriter {
    bytes: Vec<u8>,
    bits: u32,
    max_bits: u32,
}

impl BitWriter {
    const fn new(max_bits: u32) -> Self {
        Self {
            bytes: Vec::new(),
            bits: 0,
            max_bits,
        }
    }

    fn bit(&mut self, value: bool) -> Result<()> {
        if self.bits >= self.max_bits {
            return Err(Error::CorruptIndex(
                "interpolative bit budget exceeded".into(),
            ));
        }
        if self.bits % 8 == 0 {
            self.bytes.try_reserve(1).map_err(|_| {
                Error::InvalidArgument("interpolative bytes cannot be allocated safely".into())
            })?;
            self.bytes.push(0);
        }
        if value {
            let last = self.bytes.last_mut().ok_or_else(|| {
                Error::CorruptIndex("interpolative writer has no current byte".into())
            })?;
            *last |= 1 << (self.bits % 8);
        }
        self.bits += 1;
        Ok(())
    }

    fn fixed(&mut self, value: u64, width: u32) -> Result<()> {
        for shift in (0..width).rev() {
            self.bit(value & (1_u64 << shift) != 0)?;
        }
        Ok(())
    }

    fn truncated(&mut self, offset: u64, range: u64) -> Result<()> {
        if range == 0 || offset >= range {
            return Err(Error::CorruptIndex(
                "invalid interpolative rank interval".into(),
            ));
        }
        let width = range.ilog2();
        let cutoff = (1_u64 << (width + 1)) - range;
        if offset < cutoff {
            self.fixed(offset, width)
        } else {
            self.fixed(offset + cutoff, width + 1)
        }
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bits: u32,
    position: u32,
}

impl BitReader<'_> {
    fn bit(&mut self) -> Result<u64> {
        if self.position >= self.bits {
            return Err(Error::CorruptIndex(
                "truncated interpolative codeword".into(),
            ));
        }
        let value = u64::from((self.bytes[self.position as usize / 8] >> (self.position % 8)) & 1);
        self.position += 1;
        Ok(value)
    }

    fn fixed(&mut self, width: u32) -> Result<u64> {
        let mut value = 0_u64;
        for _ in 0..width {
            value = (value << 1) | self.bit()?;
        }
        Ok(value)
    }

    fn truncated(&mut self, range: u64) -> Result<u64> {
        if range == 0 {
            return Err(Error::CorruptIndex(
                "invalid interpolative rank interval".into(),
            ));
        }
        let width = range.ilog2();
        let cutoff = (1_u64 << (width + 1)) - range;
        let prefix = self.fixed(width)?;
        let offset = if prefix < cutoff {
            prefix
        } else {
            ((prefix << 1) | self.bit()?) - cutoff
        };
        if offset >= range {
            return Err(Error::CorruptIndex(
                "interpolative rank exceeds interval".into(),
            ));
        }
        Ok(offset)
    }
}

fn median_interval(length: usize, lower: i64, upper: i64) -> Result<(usize, i64, i64)> {
    let middle = length / 2;
    let right = length - middle - 1;
    let minimum = lower
        + i64::try_from(middle)
            .map_err(|_| Error::CorruptIndex("interpolative left rank does not fit i64".into()))?
        + 1;
    let maximum = upper
        - i64::try_from(right)
            .map_err(|_| Error::CorruptIndex("interpolative right rank does not fit i64".into()))?
        - 1;
    if minimum > maximum {
        return Err(Error::CorruptIndex(
            "interpolative interval is too small".into(),
        ));
    }
    Ok((middle, minimum, maximum))
}

fn encode_segment(writer: &mut BitWriter, values: &[u32], lower: i64, upper: i64) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let (middle, minimum, maximum) = median_interval(values.len(), lower, upper)?;
    let value = i64::from(values[middle]);
    if value < minimum || value > maximum {
        return Err(Error::CorruptIndex(
            "interpolative median outside interval".into(),
        ));
    }
    writer.truncated(
        u64::try_from(value - minimum)
            .map_err(|_| Error::CorruptIndex("interpolative offset does not fit u64".into()))?,
        u64::try_from(maximum - minimum + 1)
            .map_err(|_| Error::CorruptIndex("interpolative range does not fit u64".into()))?,
    )?;
    encode_segment(writer, &values[..middle], lower, value)?;
    encode_segment(writer, &values[middle + 1..], value, upper)
}

fn decode_segment(
    reader: &mut BitReader<'_>,
    out: &mut [u32],
    lower: i64,
    upper: i64,
) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let (middle, minimum, maximum) = median_interval(out.len(), lower, upper)?;
    let range = u64::try_from(maximum - minimum + 1)
        .map_err(|_| Error::CorruptIndex("interpolative range does not fit u64".into()))?;
    let value = minimum
        + i64::try_from(reader.truncated(range)?)
            .map_err(|_| Error::CorruptIndex("interpolative value does not fit i64".into()))?;
    out[middle] = u32::try_from(value)
        .map_err(|_| Error::CorruptIndex("interpolative value exceeds u32".into()))?;
    decode_segment(reader, &mut out[..middle], lower, value)?;
    decode_segment(reader, &mut out[middle + 1..], value, upper)
}

fn validate_shape(last: u32, count: usize) -> Result<u32> {
    if count == 0
        || count > MAX_VALUES
        || u64::try_from(count).unwrap_or(u64::MAX) > u64::from(last) + 1
    {
        return Err(Error::CorruptIndex(
            "invalid interpolative sequence length".into(),
        ));
    }
    let maximum = (count - 1)
        .checked_mul(MAX_BITS_PER_VALUE)
        .and_then(|bits| u32::try_from(bits).ok())
        .ok_or_else(|| Error::CorruptIndex("interpolative bit budget overflow".into()))?;
    Ok(maximum)
}

pub(crate) fn encode(values: &[u32]) -> Result<Vec<u8>> {
    let &last = values
        .last()
        .ok_or_else(|| Error::CorruptIndex("empty interpolative sequence".into()))?;
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::CorruptIndex(
            "interpolative IDs must be strictly increasing".into(),
        ));
    }
    let max_bits = validate_shape(last, values.len())?;
    let mut writer = BitWriter::new(max_bits);
    encode_segment(
        &mut writer,
        &values[..values.len() - 1],
        -1,
        i64::from(last),
    )?;
    let total = 8_usize
        .checked_add(writer.bytes.len())
        .ok_or_else(|| Error::CorruptIndex("interpolative byte length overflow".into()))?;
    let mut output = Vec::new();
    output.try_reserve_exact(total).map_err(|_| {
        Error::InvalidArgument("interpolative sequence cannot be allocated safely".into())
    })?;
    output.extend_from_slice(&last.to_le_bytes());
    output.extend_from_slice(&writer.bits.to_le_bytes());
    output.extend_from_slice(&writer.bytes);
    Ok(output)
}

pub(crate) fn decode(bytes: &[u8], count: usize) -> Result<Vec<u32>> {
    let header = bytes
        .get(..8)
        .ok_or_else(|| Error::CorruptIndex("truncated interpolative header".into()))?;
    let last = u32::from_le_bytes(
        header[..4]
            .try_into()
            .map_err(|_| Error::CorruptIndex("truncated interpolative maximum".into()))?,
    );
    let bits = u32::from_le_bytes(
        header[4..8]
            .try_into()
            .map_err(|_| Error::CorruptIndex("truncated interpolative bit count".into()))?,
    );
    let max_bits = validate_shape(last, count)?;
    if bits > max_bits {
        return Err(Error::CorruptIndex(
            "interpolative bit count exceeds safety limit".into(),
        ));
    }
    let payload_bytes = usize::try_from(bits.div_ceil(8))
        .map_err(|_| Error::CorruptIndex("interpolative payload does not fit usize".into()))?;
    if bytes.len() != 8 + payload_bytes {
        return Err(Error::CorruptIndex(
            "interpolative byte length mismatch".into(),
        ));
    }
    if bits % 8 != 0 && bytes.last().is_some_and(|byte| byte >> (bits % 8) != 0) {
        return Err(Error::CorruptIndex("nonzero interpolative padding".into()));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|_| {
        Error::CorruptIndex("interpolative values cannot be allocated safely".into())
    })?;
    output.resize(count, 0);
    output[count - 1] = last;
    let mut reader = BitReader {
        bytes: &bytes[8..],
        bits,
        position: 0,
    };
    decode_segment(&mut reader, &mut output[..count - 1], -1, i64::from(last))?;
    if reader.position != bits {
        return Err(Error::CorruptIndex(
            "trailing interpolative payload bits".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_short_bit_oracles() {
        // Strict interpolation stores the final value directly. For [0,2,3],
        // the median 2 has rank 1 in [1,2] => bit 1, then 0 in [0,1] => 0.
        assert_eq!(encode(&[0, 2, 3]).unwrap(), [3, 0, 0, 0, 2, 0, 0, 0, 0x01]);
        // In [0,2], truncated-binary values are 0, 10, 11; 1 => 10.
        assert_eq!(encode(&[1, 3]).unwrap(), [3, 0, 0, 0, 2, 0, 0, 0, 0x01]);
        // Offset 2 in the same three-value interval is the long word 11.
        assert_eq!(encode(&[2, 3]).unwrap(), [3, 0, 0, 0, 2, 0, 0, 0, 0x03]);
        // Dense IDs are determined by count and maximum; no payload bits.
        assert_eq!(encode(&[0, 1, 2]).unwrap(), [2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(encode(&[0]).unwrap(), [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn boundaries_and_deterministic_round_trips() {
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
        let mut state = 97_u64;
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
                decode(&encode(&sequence).unwrap(), sequence.len()).unwrap(),
                sequence
            );
        }
    }

    #[test]
    fn rejects_invalid_input_shape_and_corrupt_wire() {
        assert!(encode(&[]).is_err());
        assert!(encode(&[1, 1]).is_err());
        assert!(encode(&[2, 1]).is_err());
        assert!(decode(&[0, 0, 0], 1).is_err());
        assert!(decode(&[0; 8], 0).is_err());
        assert!(decode(&[0; 8], MAX_VALUES + 1).is_err());
        assert!(decode(&[0, 0, 0, 0, 0, 0, 0, 0], 2).is_err());
        let original = [3, 0, 0, 0, 2, 0, 0, 0, 0x01];
        assert_eq!(decode(&original, 3).unwrap(), [0, 2, 3]);
        assert!(decode(&original[..8], 3).is_err());
        let mut trailing = original.to_vec();
        trailing.push(0);
        assert!(decode(&trailing, 3).is_err());
        let mut bad_padding = original;
        bad_padding[8] |= 0x80;
        assert!(decode(&bad_padding, 3).is_err());
        let mut bad_bits = original;
        bad_bits[4] = 3;
        assert!(decode(&bad_bits, 3).is_err());
        let mut impossible_bits = original;
        impossible_bits[4..8].copy_from_slice(&65_u32.to_le_bytes());
        assert!(decode(&impossible_bits, 3).is_err());
        let mut truncated_codeword = original;
        truncated_codeword[4] = 1;
        assert!(decode(&truncated_codeword, 3).is_err());
        assert!(median_interval(3, -1, 1).is_err());
    }
}
