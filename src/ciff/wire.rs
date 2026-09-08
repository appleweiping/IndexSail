//! Small, bounded Protocol Buffers wire codec for the four CIFF v1 messages.
//!
//! CIFF is a stream of length-delimited protobuf messages. Pulling a general
//! protobuf runtime into the search hot path would add considerably more API
//! than this format needs, so this module implements the documented wire types
//! directly and keeps every allocation behind an explicit frame limit.

use std::io::{Read, Write};

use crate::error::{Error, Result};

pub(super) const WIRE_VARINT: u8 = 0;
pub(super) const WIRE_FIXED64: u8 = 1;
pub(super) const WIRE_LENGTH_DELIMITED: u8 = 2;
const WIRE_FIXED32: u8 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Field {
    pub number: u32,
    pub wire_type: u8,
}

#[derive(Debug)]
pub(super) struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub fn next_field(&mut self) -> Result<Option<Field>> {
        if self.position == self.bytes.len() {
            return Ok(None);
        }
        let key = self.read_varint()?;
        if key == 0 {
            return Err(corrupt("protobuf field key must not be zero"));
        }
        let number =
            u32::try_from(key >> 3).map_err(|_| corrupt("protobuf field number exceeds u32"))?;
        if number == 0 || number > 0x1fff_ffff {
            return Err(corrupt("protobuf field number is out of range"));
        }
        let wire_type = u8::try_from(key & 0x07).expect("three-bit wire type fits u8");
        if !matches!(
            wire_type,
            WIRE_VARINT | WIRE_FIXED64 | WIRE_LENGTH_DELIMITED | WIRE_FIXED32
        ) {
            return Err(corrupt(format!(
                "unsupported protobuf wire type {wire_type}"
            )));
        }
        Ok(Some(Field { number, wire_type }))
    }

    pub fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for shift in (0..=63).step_by(7) {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| corrupt("truncated protobuf varint"))?;
            self.position += 1;
            if shift == 63 && byte > 1 {
                return Err(corrupt("protobuf varint exceeds u64"));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(corrupt("protobuf varint exceeds ten bytes"))
    }

    pub fn read_i32_nonnegative(&mut self, label: &str) -> Result<u32> {
        let value = self.read_varint()?;
        u32::try_from(value)
            .ok()
            .filter(|value| i32::try_from(*value).is_ok())
            .ok_or_else(|| corrupt(format!("{label} must be a non-negative int32")))
    }

    pub fn read_i64_nonnegative(&mut self, label: &str) -> Result<u64> {
        let value = self.read_varint()?;
        i64::try_from(value)
            .is_ok()
            .then_some(value)
            .ok_or_else(|| corrupt(format!("{label} must be a non-negative int64")))
    }

    pub fn read_fixed64(&mut self) -> Result<u64> {
        let end = self
            .position
            .checked_add(8)
            .ok_or_else(|| corrupt("protobuf fixed64 offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| corrupt("truncated protobuf fixed64"))?;
        self.position = end;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("fixed64 slice has eight bytes"),
        ))
    }

    pub fn read_length_delimited(&mut self) -> Result<&'a [u8]> {
        let length = usize::try_from(self.read_varint()?)
            .map_err(|_| corrupt("protobuf length does not fit memory address space"))?;
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| corrupt("protobuf length offset overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| corrupt("truncated length-delimited protobuf field"))?;
        self.position = end;
        Ok(value)
    }

    pub fn read_string(&mut self, label: &str, max_bytes: usize) -> Result<String> {
        let bytes = self.read_length_delimited()?;
        if bytes.len() > max_bytes {
            return Err(corrupt(format!(
                "{label} exceeds the configured {max_bytes}-byte limit"
            )));
        }
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| corrupt(format!("{label} is not valid UTF-8")))
    }

    pub fn skip(&mut self, field: Field) -> Result<()> {
        match field.wire_type {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_FIXED64 => {
                self.read_fixed64()?;
            }
            WIRE_LENGTH_DELIMITED => {
                self.read_length_delimited()?;
            }
            WIRE_FIXED32 => {
                let end = self
                    .position
                    .checked_add(4)
                    .ok_or_else(|| corrupt("protobuf fixed32 offset overflow"))?;
                self.bytes
                    .get(self.position..end)
                    .ok_or_else(|| corrupt("truncated protobuf fixed32"))?;
                self.position = end;
            }
            _ => return Err(corrupt("unsupported protobuf wire type")),
        }
        Ok(())
    }
}

pub(super) fn require_wire(field: Field, expected: u8, label: &str) -> Result<()> {
    if field.wire_type != expected {
        return Err(corrupt(format!(
            "{label} uses protobuf wire type {}, expected {expected}",
            field.wire_type
        )));
    }
    Ok(())
}

pub(super) fn mark_once(seen: &mut u64, number: u32, label: &str) -> Result<()> {
    if number >= 64 {
        return Ok(());
    }
    let bit = 1_u64 << number;
    if *seen & bit != 0 {
        return Err(corrupt(format!("duplicate protobuf field '{label}'")));
    }
    *seen |= bit;
    Ok(())
}

pub(super) fn read_delimited(
    reader: &mut impl Read,
    max_frame_bytes: usize,
) -> Result<Option<Vec<u8>>> {
    let Some(length) = read_stream_varint(reader)? else {
        return Ok(None);
    };
    let length = usize::try_from(length)
        .map_err(|_| corrupt("CIFF frame length does not fit memory address space"))?;
    if length > max_frame_bytes {
        return Err(corrupt(format!(
            "CIFF frame has {length} bytes, exceeding the configured {max_frame_bytes}-byte limit"
        )));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| corrupt(format!("cannot allocate {length} bytes for a CIFF frame")))?;
    bytes.resize(length, 0_u8);
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => corrupt("truncated CIFF frame"),
            _ => Error::Io(error),
        })?;
    Ok(Some(bytes))
}

fn read_stream_varint(reader: &mut impl Read) -> Result<Option<u64>> {
    let mut value = 0_u64;
    for (index, shift) in (0..=63).step_by(7).enumerate() {
        let mut byte = [0_u8; 1];
        let read = reader.read(&mut byte)?;
        if read == 0 {
            return if index == 0 {
                Ok(None)
            } else {
                Err(corrupt("truncated CIFF frame-length varint"))
            };
        }
        if shift == 63 && byte[0] > 1 {
            return Err(corrupt("CIFF frame length exceeds u64"));
        }
        value |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(Some(value));
        }
    }
    Err(corrupt("CIFF frame length exceeds ten bytes"))
}

#[cfg(test)]
pub(super) fn write_delimited(writer: &mut impl Write, message: &[u8]) -> Result<()> {
    let length = u64::try_from(message.len())
        .map_err(|_| Error::InvalidArgument("CIFF frame length exceeds u64".into()))?;
    write_varint(writer, length)?;
    writer.write_all(message)?;
    Ok(())
}

pub(super) fn write_varint(writer: &mut impl Write, mut value: u64) -> Result<()> {
    let mut bytes = [0_u8; 10];
    let mut length = 0;
    loop {
        let low = u8::try_from(value & 0x7f).expect("seven bits fit u8");
        value >>= 7;
        bytes[length] = if value == 0 { low } else { low | 0x80 };
        length += 1;
        if value == 0 {
            break;
        }
    }
    writer.write_all(&bytes[..length])?;
    Ok(())
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(super) struct Encoder {
    bytes: Vec<u8>,
}

#[cfg(test)]
impl Encoder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn varint(&mut self, number: u32, value: u64) {
        if value == 0 {
            return;
        }
        self.key(number, WIRE_VARINT);
        encode_varint_vec(&mut self.bytes, value);
    }

    pub fn fixed64(&mut self, number: u32, value: u64) {
        if value == 0 {
            return;
        }
        self.key(number, WIRE_FIXED64);
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn string(&mut self, number: u32, value: &str) {
        if value.is_empty() {
            return;
        }
        self.bytes(number, value.as_bytes());
    }

    pub fn bytes(&mut self, number: u32, value: &[u8]) {
        if value.is_empty() {
            return;
        }
        self.key(number, WIRE_LENGTH_DELIMITED);
        encode_varint_vec(
            &mut self.bytes,
            u64::try_from(value.len()).expect("slice length fits u64 on supported targets"),
        );
        self.bytes.extend_from_slice(value);
    }

    fn key(&mut self, number: u32, wire_type: u8) {
        encode_varint_vec(
            &mut self.bytes,
            (u64::from(number) << 3) | u64::from(wire_type),
        );
    }
}

#[cfg(test)]
fn encode_varint_vec(bytes: &mut Vec<u8>, mut value: u64) {
    loop {
        let low = u8::try_from(value & 0x7f).expect("seven bits fit u8");
        value >>= 7;
        bytes.push(if value == 0 { low } else { low | 0x80 });
        if value == 0 {
            break;
        }
    }
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::CorruptIndex(format!("CIFF: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn varint_boundaries_round_trip() {
        for value in [
            0,
            1,
            127,
            128,
            16_383,
            16_384,
            u64::from(u32::MAX),
            u64::MAX,
        ] {
            let mut encoded = Vec::new();
            write_varint(&mut encoded, value).unwrap();
            let mut decoder = Decoder::new(&encoded);
            assert_eq!(decoder.read_varint().unwrap(), value);
            assert_eq!(decoder.next_field().unwrap(), None);
        }
    }

    #[test]
    fn rejects_overlong_and_overflowing_varints() {
        assert!(Decoder::new(&[0x80; 10]).read_varint().is_err());
        let mut overflow = [0x80; 10];
        overflow[9] = 2;
        assert!(Decoder::new(&overflow).read_varint().is_err());
    }

    #[test]
    fn delimited_reader_distinguishes_clean_eof_from_torn_length() {
        assert_eq!(read_delimited(&mut Cursor::new([]), 10).unwrap(), None);
        assert!(read_delimited(&mut Cursor::new([0x80]), 10).is_err());
        assert!(read_delimited(&mut Cursor::new([3, 1, 2]), 10).is_err());
    }

    #[test]
    fn frame_limit_is_checked_before_allocating_payload() {
        let mut source = Cursor::new([0x80, 0x01]);
        assert!(read_delimited(&mut source, 127).is_err());
    }

    #[test]
    fn unknown_supported_wire_types_can_be_skipped() {
        let bytes = [
            0x08, 0x01, 0x11, 0, 0, 0, 0, 0, 0, 0, 0, 0x1a, 1, 7, 0x25, 0, 0, 0, 0,
        ];
        let mut decoder = Decoder::new(&bytes);
        while let Some(field) = decoder.next_field().unwrap() {
            decoder.skip(field).unwrap();
        }
    }
}
