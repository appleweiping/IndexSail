//! Bounded, reproducible assignments of new internal document identifiers.
//!
//! The mapping is deliberately independent of PISA's on-disk formats and RNG.

use std::io::{Read, Write};

use crate::error::{Error, Result};
use crate::forward::ForwardIndex;

/// A tighter bound than the forward format: reordering materializes another
/// document vector and two identifier vectors before inversion.
pub const MAX_REORDER_DOCUMENTS: usize = 1_000_000;
/// Maximum persisted forward input admitted by the reordering CLI.
pub const MAX_REORDER_FORWARD_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_REORDER_OCCURRENCES: u64 = 20_000_000;
const MAX_MAPPING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FEATURE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_BISECTION_INCIDENCES: u64 = 2_000_000;
const MAX_BISECTION_DOCUMENTS: usize = 100_000;
const MAX_BISECTION_WORK: u64 = 200_000_000;
const MAX_BISECTION_DEPTH: usize = 20;
const MAX_BISECTION_ITERATIONS: usize = 20;

fn check_bisection_preflight(count: usize, options: BisectionOptions) -> Result<()> {
    check_count(count)?;
    if count > MAX_BISECTION_DOCUMENTS {
        return Err(Error::InvalidArgument(format!(
            "bisection exceeds {MAX_BISECTION_DOCUMENTS} document limit"
        )));
    }
    if options.depth == 0 || options.depth > MAX_BISECTION_DEPTH {
        return Err(Error::InvalidArgument(format!(
            "bisection depth must be in 1..={MAX_BISECTION_DEPTH}"
        )));
    }
    if options.iterations == 0 || options.iterations > MAX_BISECTION_ITERATIONS {
        return Err(Error::InvalidArgument(format!(
            "bisection iterations must be in 1..={MAX_BISECTION_ITERATIONS}"
        )));
    }
    Ok(())
}

fn check_bisection_work(
    count: usize,
    incidence_count: u64,
    options: BisectionOptions,
) -> Result<()> {
    if incidence_count > MAX_BISECTION_INCIDENCES {
        return Err(Error::InvalidArgument(format!(
            "bisection exceeds {MAX_BISECTION_INCIDENCES} token occurrence limit"
        )));
    }
    // Upper-bound comparison work for per-document term sorting and repeated
    // per-node ID/gain sorting, plus all term visits at each tree level.
    // Degree scratch is initialized once, then only touched terms are reset.
    let term_sort =
        incidence_count.checked_mul(u64::from((incidence_count.max(2) - 1).ilog2() + 1));
    let doc_sort = (count as u64).checked_mul(u64::from(((count.max(2) - 1) as u64).ilog2() + 1));
    let work = term_sort
        .zip(doc_sort)
        .and_then(|(terms, docs)| {
            docs.checked_add(incidence_count)
                .and_then(|per_level| per_level.checked_mul(options.depth as u64))
                .and_then(|all_levels| all_levels.checked_mul((options.iterations + 1) as u64))
                .and_then(|tree| terms.checked_add(tree))
        })
        .ok_or_else(|| Error::InvalidArgument("bisection work estimate overflow".into()))?;
    if work > MAX_BISECTION_WORK {
        return Err(Error::InvalidArgument(format!(
            "bisection exceeds {MAX_BISECTION_WORK} estimated work units; reduce depth or iterations"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct DegreeScratch {
    left: Vec<usize>,
    right: Vec<usize>,
    touched: Vec<u32>,
}

impl DegreeScratch {
    fn new(term_count: usize) -> Result<Self> {
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut touched = Vec::new();
        left.try_reserve_exact(term_count).map_err(|_| {
            Error::InvalidArgument("could not allocate bisection degree scratch".into())
        })?;
        right.try_reserve_exact(term_count).map_err(|_| {
            Error::InvalidArgument("could not allocate bisection degree scratch".into())
        })?;
        touched.try_reserve_exact(term_count).map_err(|_| {
            Error::InvalidArgument("could not allocate bisection touched-term scratch".into())
        })?;
        left.resize(term_count, 0);
        right.resize(term_count, 0);
        Ok(Self {
            left,
            right,
            touched,
        })
    }

    fn increment_left(&mut self, term: u32) {
        let index = term as usize;
        if self.left[index] == 0 && self.right[index] == 0 {
            self.touched.push(term);
        }
        self.left[index] += 1;
    }

    fn increment_right(&mut self, term: u32) {
        let index = term as usize;
        if self.left[index] == 0 && self.right[index] == 0 {
            self.touched.push(term);
        }
        self.right[index] += 1;
    }

    fn clear_touched(&mut self) {
        for term in self.touched.drain(..) {
            self.left[term as usize] = 0;
            self.right[term as usize] = 0;
        }
    }
}

/// Controls the deterministic, compression-proxy recursive bisection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BisectionOptions {
    pub depth: usize,
    pub iterations: usize,
}

impl BisectionOptions {
    /// A full-depth tree on small inputs, capped at eight levels and eight passes.
    pub fn for_documents(count: usize) -> Self {
        Self {
            depth: usize::min((count.max(2) - 1).ilog2() as usize + 1, 8),
            iterations: 8,
        }
    }
}

fn bit_proxy(degree: usize, partition_size: usize) -> f64 {
    if degree == 0 {
        return 0.0;
    }
    let degree = f64::from(u32::try_from(degree).expect("bisection degrees fit u32"));
    let partition_size = f64::from(u32::try_from(partition_size).expect("bisection size fits u32"));
    degree * (partition_size.log2() - (degree + 1.0).log2())
}

fn move_gain(from_degree: usize, to_degree: usize, from_size: usize, to_size: usize) -> f64 {
    bit_proxy(from_degree, from_size) + bit_proxy(to_degree, to_size)
        - bit_proxy(from_degree - 1, from_size)
        - bit_proxy(to_degree + 1, to_size)
}

fn pair_gain(
    left_terms: &[u32],
    right_terms: &[u32],
    left_degrees: &[usize],
    right_degrees: &[usize],
    left_size: usize,
    right_size: usize,
) -> f64 {
    let mut gain = 0.0;
    let mut left = 0;
    let mut right = 0;
    while left < left_terms.len() || right < right_terms.len() {
        match (left_terms.get(left), right_terms.get(right)) {
            (Some(&a), Some(&b)) if a == b => {
                left += 1;
                right += 1;
            }
            (Some(&a), Some(&b)) if a < b => {
                gain += move_gain(
                    left_degrees[a as usize],
                    right_degrees[a as usize],
                    left_size,
                    right_size,
                );
                left += 1;
            }
            (Some(_) | None, Some(&b)) => {
                gain += move_gain(
                    right_degrees[b as usize],
                    left_degrees[b as usize],
                    right_size,
                    left_size,
                );
                right += 1;
            }
            (Some(&a), None) => {
                gain += move_gain(
                    left_degrees[a as usize],
                    right_degrees[a as usize],
                    left_size,
                    right_size,
                );
                left += 1;
            }
            (None, None) => break,
        }
    }
    gain
}

fn process_bisection(
    order: &mut [u32],
    terms_by_doc: &[Vec<u32>],
    scratch: &mut DegreeScratch,
    gains: &mut [f64],
    iterations: usize,
) {
    let mid = order.len() / 2;
    if mid == 0 {
        return;
    }
    let (left, right) = order.split_at_mut(mid);
    let left_size = left.len();
    let right_size = right.len();
    for &doc in left.iter() {
        for &term in &terms_by_doc[doc as usize] {
            scratch.increment_left(term);
        }
    }
    for &doc in right.iter() {
        for &term in &terms_by_doc[doc as usize] {
            scratch.increment_right(term);
        }
    }
    for _ in 0..iterations {
        for &doc in left.iter() {
            gains[doc as usize] = terms_by_doc[doc as usize]
                .iter()
                .map(|&term| {
                    move_gain(
                        scratch.left[term as usize],
                        scratch.right[term as usize],
                        left_size,
                        right_size,
                    )
                })
                .sum();
        }
        for &doc in right.iter() {
            gains[doc as usize] = terms_by_doc[doc as usize]
                .iter()
                .map(|&term| {
                    move_gain(
                        scratch.right[term as usize],
                        scratch.left[term as usize],
                        right_size,
                        left_size,
                    )
                })
                .sum();
        }
        let by_gain = |&a: &u32, &b: &u32| {
            gains[b as usize]
                .total_cmp(&gains[a as usize])
                .then_with(|| a.cmp(&b))
        };
        left.sort_unstable_by(by_gain);
        right.sort_unstable_by(by_gain);
        let mut changed = false;
        for (left_doc, right_doc) in left.iter_mut().zip(right.iter_mut()) {
            let left_terms = &terms_by_doc[*left_doc as usize];
            let right_terms = &terms_by_doc[*right_doc as usize];
            let gain = pair_gain(
                left_terms,
                right_terms,
                &scratch.left,
                &scratch.right,
                left_size,
                right_size,
            );
            if gain > 1e-12 {
                for &term in left_terms {
                    scratch.left[term as usize] -= 1;
                    scratch.right[term as usize] += 1;
                }
                for &term in right_terms {
                    scratch.right[term as usize] -= 1;
                    scratch.left[term as usize] += 1;
                }
                std::mem::swap(left_doc, right_doc);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    scratch.clear_touched();
}

fn bisect_range(
    order: &mut [u32],
    terms_by_doc: &[Vec<u32>],
    scratch: &mut DegreeScratch,
    gains: &mut [f64],
    options: BisectionOptions,
) {
    if order.len() < 2 {
        return;
    }
    order.sort_unstable();
    process_bisection(order, terms_by_doc, scratch, gains, options.iterations);
    let (left, right) = order.split_at_mut(order.len() / 2);
    if options.depth > 1 && left.len() + right.len() > 2 {
        let child = BisectionOptions {
            depth: options.depth - 1,
            ..options
        };
        bisect_range(left, terms_by_doc, scratch, gains, child);
        bisect_range(right, terms_by_doc, scratch, gains, child);
    } else {
        left.sort_unstable();
        right.sort_unstable();
    }
}

/// A checked bijection between old and new zero-based internal IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocIdMap {
    old_to_new: Vec<u32>,
    new_to_old: Vec<u32>,
}

impl DocIdMap {
    /// Balanced recursive graph bisection over distinct field-qualified terms.
    /// Each accepted pair swap strictly reduces the degree-based bit proxy.
    pub fn recursive_graph_bisection(
        forward: &ForwardIndex,
        options: BisectionOptions,
    ) -> Result<Self> {
        let count = forward.documents().len();
        check_bisection_preflight(count, options)?;
        // The validated forward snapshot already records the exact token count.
        // Reject excessive graph-sort and tree work before materializing the
        // document-term incidence lists.
        check_bisection_work(count, forward.stats().occurrences, options)?;
        let mut incidence_count = 0_u64;
        let mut terms_by_doc = Vec::new();
        terms_by_doc
            .try_reserve_exact(count)
            .map_err(|_| Error::InvalidArgument("could not allocate bisection documents".into()))?;
        for document in forward.documents() {
            let mut terms = Vec::new();
            for field in document.document().fields().keys() {
                if let Some(sequence) = document.term_ids(field) {
                    incidence_count = incidence_count
                        .checked_add(sequence.len() as u64)
                        .ok_or_else(|| {
                            Error::InvalidArgument("bisection incidence count overflow".into())
                        })?;
                    if incidence_count > MAX_BISECTION_INCIDENCES {
                        return Err(Error::InvalidArgument(format!(
                            "bisection exceeds {MAX_BISECTION_INCIDENCES} token occurrence limit"
                        )));
                    }
                    terms.extend_from_slice(sequence);
                }
            }
            terms.sort_unstable();
            terms.dedup();
            terms_by_doc.push(terms);
        }
        check_bisection_work(count, incidence_count, options)?;
        let mut new_to_old = (0..count)
            .map(|id| u32::try_from(id).expect("bisection document bound fits u32"))
            .collect::<Vec<_>>();
        let mut gains = vec![0.0; count];
        let mut scratch = DegreeScratch::new(forward.terms().len())?;
        bisect_range(
            &mut new_to_old,
            &terms_by_doc,
            &mut scratch,
            &mut gains,
            options,
        );
        Self::from_new_to_old(new_to_old)
    }
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
    use crate::{Analyzer, Document};

    fn independent_proxy(order: &[u32], terms_by_doc: &[Vec<u32>], term_count: usize) -> f64 {
        let (left, right) = order.split_at(order.len() / 2);
        [left, right]
            .iter()
            .map(|side| {
                (0..term_count)
                    .map(|term| {
                        let degree = side
                            .iter()
                            .filter(|&&doc| {
                                terms_by_doc[doc as usize].contains(&u32::try_from(term).unwrap())
                            })
                            .count();
                        if degree == 0 {
                            0.0
                        } else {
                            f64::from(u32::try_from(degree).unwrap())
                                * (f64::from(u32::try_from(side.len()).unwrap())
                                    / f64::from(u32::try_from(degree + 1).unwrap()))
                                .log2()
                        }
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    #[test]
    fn pair_gain_matches_exhaustively_recomputed_objective() {
        for first_mask in 0..16_u32 {
            for second_mask in 0..16_u32 {
                let terms = (0..4)
                    .map(|doc| {
                        let mut value = Vec::new();
                        if first_mask & (1 << doc) != 0 {
                            value.push(0);
                        }
                        if second_mask & (1 << doc) != 0 {
                            value.push(1);
                        }
                        value
                    })
                    .collect::<Vec<_>>();
                let left_degree = (0..2)
                    .map(|term| terms[..2].iter().filter(|doc| doc.contains(&term)).count())
                    .collect::<Vec<_>>();
                let right_degree = (0..2)
                    .map(|term| terms[2..].iter().filter(|doc| doc.contains(&term)).count())
                    .collect::<Vec<_>>();
                let before = [0, 1, 2, 3];
                let after = [2, 1, 0, 3];
                let expected =
                    independent_proxy(&before, &terms, 2) - independent_proxy(&after, &terms, 2);
                let actual = pair_gain(&terms[0], &terms[2], &left_degree, &right_degree, 2, 2);
                assert!(
                    (expected - actual).abs() < 1e-10,
                    "masks {first_mask}, {second_mask}"
                );
                let mut order = before;
                let mut gains = [0.0; 4];
                let mut scratch = DegreeScratch::new(2).unwrap();
                bisect_range(
                    &mut order,
                    &terms,
                    &mut scratch,
                    &mut gains,
                    BisectionOptions {
                        depth: 1,
                        iterations: 8,
                    },
                );
                assert!(
                    independent_proxy(&order, &terms, 2)
                        <= independent_proxy(&before, &terms, 2) + 1e-10,
                    "masks {first_mask}, {second_mask}"
                );
                order.sort_unstable();
                assert_eq!(order, before);
            }
        }
    }

    #[test]
    fn seeded_small_graphs_preserve_bijections_and_never_worsen_root_proxy() {
        let mut rng = SplitMix64(0xA11C_E123);
        for case in 0..128 {
            let count = 2 + (rng.next() % 7) as usize;
            let term_count = 1 + (rng.next() % 6) as usize;
            let terms = (0..count)
                .map(|_| {
                    (0..term_count)
                        .filter(|_| rng.next() & 1 == 1)
                        .map(|term| u32::try_from(term).unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let original = (0..count)
                .map(|doc| u32::try_from(doc).unwrap())
                .collect::<Vec<_>>();
            let mut reordered = original.clone();
            let mut gains = vec![0.0; count];
            let mut scratch = DegreeScratch::new(term_count).unwrap();
            bisect_range(
                &mut reordered,
                &terms,
                &mut scratch,
                &mut gains,
                BisectionOptions {
                    depth: 1,
                    iterations: 8,
                },
            );
            assert!(
                independent_proxy(&reordered, &terms, term_count)
                    <= independent_proxy(&original, &terms, term_count) + 1e-10,
                "case {case}"
            );
            let mut sorted = reordered;
            sorted.sort_unstable();
            assert_eq!(sorted, original, "case {case}");
        }
    }

    #[test]
    fn multi_depth_partitions_have_independent_local_proxy_oracles() {
        let mut rng = SplitMix64(0xB15E_C710);
        for case in 0..128 {
            let count = 2 + (rng.next() % 7) as usize;
            let term_count = 1 + (rng.next() % 6) as usize;
            let terms = (0..count)
                .map(|_| {
                    (0..term_count)
                        .filter(|_| rng.next() & 1 == 1)
                        .map(|term| u32::try_from(term).unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut previous = (0..count)
                .map(|doc| u32::try_from(doc).unwrap())
                .collect::<Vec<_>>();
            let mut ranges = vec![(0, count)];
            for depth in 1..=3 {
                let mut current = (0..count)
                    .map(|doc| u32::try_from(doc).unwrap())
                    .collect::<Vec<_>>();
                let mut scratch = DegreeScratch::new(term_count).unwrap();
                let mut gains = vec![0.0; count];
                bisect_range(
                    &mut current,
                    &terms,
                    &mut scratch,
                    &mut gains,
                    BisectionOptions {
                        depth,
                        iterations: 8,
                    },
                );
                for &(start, end) in &ranges {
                    let before = &previous[start..end];
                    let after = &current[start..end];
                    let mut before_members = before.to_vec();
                    let mut after_members = after.to_vec();
                    before_members.sort_unstable();
                    after_members.sort_unstable();
                    assert_eq!(before_members, after_members, "case {case}, depth {depth}");
                    assert!(
                        independent_proxy(after, &terms, term_count)
                            <= independent_proxy(before, &terms, term_count) + 1e-10,
                        "case {case}, depth {depth}"
                    );
                }
                previous = current;
                ranges = ranges
                    .into_iter()
                    .flat_map(|(start, end)| {
                        if end - start < 2 {
                            vec![(start, end)]
                        } else {
                            let mid = start + (end - start) / 2;
                            vec![(start, mid), (mid, end)]
                        }
                    })
                    .collect();
            }
        }

        let terms = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![3],
            vec![2],
            vec![1],
            vec![0],
        ];
        let mut order = (0..8).collect::<Vec<u32>>();
        let mut scratch = DegreeScratch::new(4).unwrap();
        let mut gains = vec![0.0; 8];
        bisect_range(
            &mut order,
            &terms,
            &mut scratch,
            &mut gains,
            BisectionOptions {
                depth: 2,
                iterations: 8,
            },
        );
        assert_eq!(order, [3, 4, 2, 5, 1, 6, 0, 7]);
    }

    #[test]
    fn deep_bisection_reuses_degree_scratch_without_per_node_zeroing() {
        const COUNT: usize = 8_192;
        let terms = (0..COUNT)
            .map(|doc| vec![u32::try_from(doc).unwrap()])
            .collect::<Vec<_>>();
        let mut order = (0..COUNT)
            .map(|doc| u32::try_from(doc).unwrap())
            .collect::<Vec<_>>();
        let original = order.clone();
        let mut scratch = DegreeScratch::new(COUNT).unwrap();
        let left_ptr = scratch.left.as_ptr();
        let right_ptr = scratch.right.as_ptr();
        let mut gains = vec![0.0; COUNT];
        let options = BisectionOptions {
            depth: 20,
            iterations: 1,
        };
        assert!(check_bisection_work(COUNT, COUNT as u64, options).is_ok());
        bisect_range(&mut order, &terms, &mut scratch, &mut gains, options);
        assert_eq!(order, original);
        assert_eq!(scratch.left.as_ptr(), left_ptr);
        assert_eq!(scratch.right.as_ptr(), right_ptr);
        assert_eq!(scratch.touched, Vec::<u32>::new());
        assert!(scratch.left.iter().all(|&degree| degree == 0));
        assert!(scratch.right.iter().all(|&degree| degree == 0));
    }

    #[test]
    fn bisection_uses_field_qualified_presence_not_token_frequency() {
        let make_forward = |repeat: bool| {
            ForwardIndex::from_documents(
                Analyzer::default(),
                [
                    Document::from_fields(
                        "A",
                        [("body", if repeat { "red red red" } else { "red" })],
                    )
                    .unwrap(),
                    Document::from_fields("B", [("title", "red")]).unwrap(),
                    Document::from_fields("C", [("title", "red")]).unwrap(),
                    Document::from_fields("D", [("body", "red")]).unwrap(),
                ],
            )
            .unwrap()
        };
        let simple = make_forward(false);
        assert_eq!(simple.terms().len(), 2);
        let options = BisectionOptions {
            depth: 1,
            iterations: 8,
        };
        let map = DocIdMap::recursive_graph_bisection(&simple, options).unwrap();
        assert_eq!(map.new_to_old(), &[1, 2, 0, 3]);
        assert_eq!(
            map,
            DocIdMap::recursive_graph_bisection(&make_forward(true), options).unwrap()
        );
    }

    #[test]
    fn reordered_search_is_external_id_exact_across_source_orders() {
        use std::collections::BTreeMap;

        use crate::query::SearchQuery;
        use crate::search::SearchOptions;

        let source = [
            Document::from_fields("A", [("body", "red red")]).unwrap(),
            Document::from_fields("B", [("body", "blue")]).unwrap(),
            Document::from_fields("C", [("body", "blue red")]).unwrap(),
            Document::from_fields("D", [("body", "red")]).unwrap(),
        ];
        let options = BisectionOptions::for_documents(4);
        let mut observed = Vec::new();
        for documents in [
            source.clone(),
            [
                source[2].clone(),
                source[0].clone(),
                source[3].clone(),
                source[1].clone(),
            ],
        ] {
            let forward = ForwardIndex::from_documents(Analyzer::default(), documents).unwrap();
            let map = DocIdMap::recursive_graph_bisection(&forward, options).unwrap();
            let index = forward.reordered(&map).unwrap().invert().unwrap();
            let query = SearchQuery::from_text(index.analyzer(), "red blue", Some("body")).unwrap();
            let hits = index
                .search(
                    &query,
                    SearchOptions {
                        top_k: 4,
                        ..SearchOptions::default()
                    },
                )
                .unwrap()
                .hits;
            observed.push(
                hits.into_iter()
                    .map(|hit| (hit.external_id, hit.score.to_bits()))
                    .collect::<BTreeMap<_, _>>(),
            );
        }
        assert_eq!(observed[0], observed[1]);
        assert_eq!(observed[0].len(), 4);
    }

    #[test]
    fn bisection_rejects_document_incidence_and_work_boundaries() {
        let max = BisectionOptions {
            depth: 20,
            iterations: 20,
        };
        assert!(check_bisection_preflight(100_000, max).is_ok());
        assert!(check_bisection_preflight(100_001, max).is_err());
        assert!(check_bisection_work(1_000, 430_000, max).is_ok());
        assert!(check_bisection_work(1_000, 450_000, max).is_err());
        assert!(
            check_bisection_work(
                1,
                MAX_BISECTION_INCIDENCES + 1,
                BisectionOptions {
                    depth: 1,
                    iterations: 1
                }
            )
            .is_err()
        );
    }

    #[test]
    fn bisection_groups_crossed_terms_and_is_repeatable() {
        let forward = ForwardIndex::from_documents(
            Analyzer::default(),
            [
                Document::from_fields("A", [("body", "red red")]).unwrap(),
                Document::from_fields("B", [("body", "blue")]).unwrap(),
                Document::from_fields("C", [("body", "blue")]).unwrap(),
                Document::from_fields("D", [("body", "red")]).unwrap(),
            ],
        )
        .unwrap();
        let options = BisectionOptions {
            depth: 1,
            iterations: 8,
        };
        let map = DocIdMap::recursive_graph_bisection(&forward, options).unwrap();
        assert_eq!(
            map,
            DocIdMap::recursive_graph_bisection(&forward, options).unwrap()
        );
        assert_eq!(map.new_to_old(), &[1, 2, 0, 3]);
        assert_eq!(map.old_to_new(), &[2, 0, 1, 3]);
        let terms = vec![vec![0], vec![1], vec![1], vec![0]];
        assert!(
            independent_proxy(map.new_to_old(), &terms, 2)
                < independent_proxy(&[0, 1, 2, 3], &terms, 2)
        );
    }

    #[test]
    fn bisection_validates_parameters_and_handles_empty_input() {
        let empty =
            ForwardIndex::from_documents(Analyzer::default(), Vec::<Document>::new()).unwrap();
        let defaults = BisectionOptions::for_documents(0);
        assert!(
            DocIdMap::recursive_graph_bisection(&empty, defaults)
                .unwrap()
                .is_empty()
        );
        for options in [
            BisectionOptions {
                depth: 0,
                iterations: 1,
            },
            BisectionOptions {
                depth: 21,
                iterations: 1,
            },
            BisectionOptions {
                depth: 1,
                iterations: 0,
            },
            BisectionOptions {
                depth: 1,
                iterations: 21,
            },
        ] {
            assert!(DocIdMap::recursive_graph_bisection(&empty, options).is_err());
        }
    }

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
