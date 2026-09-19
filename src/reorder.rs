//! Bounded, reproducible assignments of new internal document identifiers.
//!
//! The mapping is deliberately independent of PISA's on-disk formats and RNG.

use std::io::{Read, Write};

use crate::error::{Error, Result};

/// A tighter bound than the forward format: reordering materializes another
/// document vector and two identifier vectors before inversion.
pub const MAX_REORDER_DOCUMENTS: usize = 1_000_000;
/// Maximum persisted forward input admitted by the reordering CLI.
pub const MAX_REORDER_FORWARD_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_REORDER_OCCURRENCES: u64 = 20_000_000;
const MAX_MAPPING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FEATURE_BYTES: u64 = 64 * 1024 * 1024;

/// A checked bijection between old and new zero-based internal IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocIdMap {
    old_to_new: Vec<u32>,
    new_to_old: Vec<u32>,
}

impl DocIdMap {
    /// Validate an explicit old-ID → new-ID permutation.
    pub fn from_old_to_new(old_to_new: Vec<u32>) -> Result<Self> {
        let count = old_to_new.len();
        check_count(count)?;
        let mut new_to_old = Vec::new();
        new_to_old.try_reserve_exact(count).map_err(|_| {
            Error::InvalidArgument("could not allocate inverse document mapping".into())
        })?;
        new_to_old.resize(count, u32::MAX);
        for (old, &new) in old_to_new.iter().enumerate() {
            let position = usize::try_from(new).map_err(|_| {
                Error::InvalidArgument("new document ID does not fit this platform".into())
            })?;
            let slot = new_to_old.get_mut(position).ok_or_else(|| {
                Error::InvalidArgument(format!("new document ID {new} is outside 0..{count}"))
            })?;
            if *slot != u32::MAX {
                return Err(Error::InvalidArgument(format!(
                    "new document ID {new} is assigned more than once"
                )));
            }
            *slot = u32::try_from(old).expect("reordering bound fits u32");
        }
        Ok(Self {
            old_to_new,
            new_to_old,
        })
    }

    fn from_new_to_old(new_to_old: Vec<u32>) -> Result<Self> {
        let count = new_to_old.len();
        check_count(count)?;
        let mut old_to_new = vec![u32::MAX; count];
        for (new, &old) in new_to_old.iter().enumerate() {
            let position = usize::try_from(old).map_err(|_| {
                Error::InvalidArgument("old document ID does not fit this platform".into())
            })?;
            let slot = old_to_new.get_mut(position).ok_or_else(|| {
                Error::InvalidArgument(format!("old document ID {old} is outside 0..{count}"))
            })?;
            if *slot != u32::MAX {
                return Err(Error::InvalidArgument(format!(
                    "old document ID {old} is assigned more than once"
                )));
            }
            *slot = u32::try_from(new).expect("reordering bound fits u32");
        }
        Ok(Self {
            old_to_new,
            new_to_old,
        })
    }

    /// Fisher–Yates with `SplitMix64` and rejection sampling. Same seed and
    /// document count produce identical output on all supported platforms.
    pub fn random(count: usize, seed: u64) -> Result<Self> {
        check_count(count)?;
        let mut new_to_old = (0..count)
            .map(|old| u32::try_from(old).expect("reordering bound fits u32"))
            .collect::<Vec<_>>();
        let mut rng = SplitMix64(seed);
        for end in (1..count).rev() {
            let bound = u64::try_from(end + 1).expect("reordering bound fits u64");
            let threshold = bound.wrapping_neg() % bound;
            let chosen = loop {
                let draw = rng.next();
                if draw >= threshold {
                    break usize::try_from(draw % bound).expect("draw fits usize");
                }
            };
            new_to_old.swap(end, chosen);
        }
        Self::from_new_to_old(new_to_old)
    }

    /// Sort one UTF-8 feature line per old ID by feature bytes, breaking ties
    /// by old ID. A trailing newline is optional; blank features are valid.
    pub fn by_feature(count: usize, reader: impl Read) -> Result<Self> {
        check_count(count)?;
        let source = bounded_utf8(reader, MAX_FEATURE_BYTES, "feature")?;
        let features = source.lines().collect::<Vec<_>>();
        if features.len() != count {
            return Err(Error::InvalidArgument(format!(
                "feature file has {} lines; expected {count}",
                features.len()
            )));
        }
        let mut new_to_old = (0..count)
            .map(|old| u32::try_from(old).expect("reordering bound fits u32"))
            .collect::<Vec<_>>();
        new_to_old.sort_unstable_by(|&left, &right| {
            features[left as usize]
                .as_bytes()
                .cmp(features[right as usize].as_bytes())
                .then_with(|| left.cmp(&right))
        });
        Self::from_new_to_old(new_to_old)
    }

    /// Read PISA-style two-column `old new` text. This accepts any row order,
    /// but requires exactly one row for each old ID and a full bijection.
    pub fn from_mapping_reader(count: usize, reader: impl Read) -> Result<Self> {
        check_count(count)?;
        let source = bounded_utf8(reader, MAX_MAPPING_BYTES, "mapping")?;
        let mut old_to_new = vec![u32::MAX; count];
        let mut rows = 0_usize;
        for (line_index, line) in source.lines().enumerate() {
            let mut columns = line.split_whitespace();
            let old = parse_id(columns.next(), "old", line_index + 1)?;
            let new = parse_id(columns.next(), "new", line_index + 1)?;
            if columns.next().is_some() {
                return Err(Error::InvalidArgument(format!(
                    "mapping line {} must contain exactly two IDs",
                    line_index + 1
                )));
            }
            let slot = old_to_new.get_mut(old as usize).ok_or_else(|| {
                Error::InvalidArgument(format!("old document ID {old} is outside 0..{count}"))
            })?;
            if *slot != u32::MAX {
                return Err(Error::InvalidArgument(format!(
                    "old document ID {old} is assigned more than once"
                )));
            }
            *slot = new;
            rows += 1;
        }
        if rows != count {
            return Err(Error::InvalidArgument(format!(
                "mapping has {rows} rows; expected {count}"
            )));
        }
        Self::from_old_to_new(old_to_new)
    }

    pub fn len(&self) -> usize {
        self.old_to_new.len()
    }

    pub fn is_empty(&self) -> bool {
        self.old_to_new.is_empty()
    }

    pub fn old_to_new(&self) -> &[u32] {
        &self.old_to_new
    }

    pub fn new_to_old(&self) -> &[u32] {
        &self.new_to_old
    }

    /// Write two whitespace-delimited columns: old ID, new ID.
    pub fn write_old_to_new(&self, mut writer: impl Write) -> Result<()> {
        for (old, &new) in self.old_to_new.iter().enumerate() {
            writeln!(writer, "{old} {new}")?;
        }
        Ok(())
    }

    /// Write two whitespace-delimited columns: new ID, old ID.
    pub fn write_new_to_old(&self, mut writer: impl Write) -> Result<()> {
        for (new, &old) in self.new_to_old.iter().enumerate() {
            writeln!(writer, "{new} {old}")?;
        }
        Ok(())
    }
}

fn check_count(count: usize) -> Result<()> {
    if count > MAX_REORDER_DOCUMENTS {
        return Err(Error::InvalidArgument(format!(
            "reordering exceeds {MAX_REORDER_DOCUMENTS} document limit"
        )));
    }
    Ok(())
}

fn bounded_utf8(mut reader: impl Read, limit: u64, description: &str) -> Result<String> {
    let mut bytes = Vec::new();
    reader.by_ref().take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(Error::InvalidArgument(format!(
            "{description} file exceeds {limit} byte limit"
        )));
    }
    String::from_utf8(bytes)
        .map_err(|_| Error::InvalidArgument(format!("{description} file must be valid UTF-8")))
}

fn parse_id(value: Option<&str>, label: &str, line: usize) -> Result<u32> {
    value
        .ok_or_else(|| {
            Error::InvalidArgument(format!("mapping line {line} is missing {label} ID"))
        })?
        .parse::<u32>()
        .map_err(|_| Error::InvalidArgument(format!("mapping line {line} has invalid {label} ID")))
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_mapping_has_an_independent_inverse_oracle() {
        let map = DocIdMap::from_old_to_new(vec![2, 0, 3, 1]).unwrap();
        assert_eq!(map.old_to_new(), [2, 0, 3, 1]);
        assert_eq!(map.new_to_old(), [1, 3, 0, 2]);
        for old in 0..4 {
            assert_eq!(
                map.new_to_old()[map.old_to_new()[old] as usize],
                u32::try_from(old).unwrap()
            );
        }
        let mut written = Vec::new();
        map.write_old_to_new(&mut written).unwrap();
        assert_eq!(written, b"0 2\n1 0\n2 3\n3 1\n");
        written.clear();
        map.write_new_to_old(&mut written).unwrap();
        assert_eq!(written, b"0 1\n1 3\n2 0\n3 2\n");
        assert_eq!(
            DocIdMap::from_mapping_reader(4, b"3 1\n0 2\n2 3\n1 0\n".as_slice()).unwrap(),
            map
        );
    }

    #[test]
    fn random_seed_has_a_frozen_portable_assignment() {
        let map = DocIdMap::random(5, 7).unwrap();
        assert_eq!(map.new_to_old(), [4, 1, 3, 0, 2]);
        assert_eq!(map.old_to_new(), [3, 1, 4, 2, 0]);
        assert_eq!(DocIdMap::random(4, 42).unwrap().old_to_new(), [1, 3, 0, 2]);
        assert_eq!(DocIdMap::random(5, 7).unwrap(), map);
        assert_ne!(DocIdMap::random(5, 8).unwrap(), map);
        assert!(DocIdMap::random(0, 0).unwrap().is_empty());
    }

    #[test]
    fn feature_order_is_utf8_byte_order_then_original_id() {
        let map = DocIdMap::by_feature(5, "z\na\na\n\né\n".as_bytes()).unwrap();
        assert_eq!(map.new_to_old(), [3, 1, 2, 0, 4]);
        assert_eq!(map.old_to_new(), [3, 1, 2, 0, 4]);
        assert_eq!(
            DocIdMap::by_feature(1, b"\n".as_slice())
                .unwrap()
                .new_to_old(),
            [0]
        );
        assert!(DocIdMap::by_feature(2, b"one\n".as_slice()).is_err());
        assert!(DocIdMap::by_feature(1, b"\xff".as_slice()).is_err());
    }

    #[test]
    fn malformed_or_oversize_mappings_are_rejected() {
        for values in [vec![0, 0], vec![0, 2], vec![1]] {
            assert!(DocIdMap::from_old_to_new(values).is_err());
        }
        for source in [
            "0 1\n0 0\n",
            "0 0\n",
            "0 0 1\n1 1\n",
            "2 0\n1 1\n",
            "0 0\n1 0\n",
        ] {
            assert!(DocIdMap::from_mapping_reader(2, source.as_bytes()).is_err());
        }
        assert!(DocIdMap::random(MAX_REORDER_DOCUMENTS + 1, 0).is_err());
    }

    #[test]
    fn bounded_text_reader_probes_one_byte_past_its_limit() {
        assert_eq!(bounded_utf8(b"abcd".as_slice(), 4, "test").unwrap(), "abcd");
        assert!(bounded_utf8(b"abcde".as_slice(), 4, "test").is_err());
    }
}
