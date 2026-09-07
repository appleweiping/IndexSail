use std::io::Write;
use std::time::{Duration, Instant};

use crate::analysis::Analyzer;
use crate::codec::PostingCodecStats;
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::IndexBuilder;
use crate::persistence::block_max_metadata_encoded_bytes;
use crate::query::{BooleanOperator, SearchQuery};
use crate::search::{PruningStrategy, SearchOptions, SearchStats};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkConfig {
    pub documents: usize,
    pub queries: usize,
    pub seed: u64,
    pub top_k: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            documents: 10_000,
            queries: 200,
            seed: 42,
            top_k: 10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    pub config: BenchmarkConfig,
    pub index_time: Duration,
    pub exhaustive_time: Duration,
    pub wand_time: Duration,
    pub block_max_time: Duration,
    pub exhaustive_stats: SearchStats,
    pub wand_stats: SearchStats,
    pub block_max_stats: SearchStats,
    pub checksum: u64,
    pub index_terms: usize,
    pub index_postings: usize,
    pub index_tokens: u64,
    pub posting_codec: PostingCodecStats,
    pub persisted_index_bytes: u64,
    pub block_max_metadata_bytes: u64,
    pub block_max_streams: usize,
    pub block_max_blocks: usize,
}

/// Run a deterministic synthetic benchmark and verify WAND against exhaustive search.
#[allow(clippy::too_many_lines)]
pub fn run(config: BenchmarkConfig) -> Result<BenchmarkReport> {
    if config.documents == 0 || config.queries == 0 || config.top_k == 0 {
        return Err(Error::InvalidArgument(
            "benchmark documents, queries, and top_k must be greater than zero".into(),
        ));
    }
    if config.documents > u32::MAX as usize {
        return Err(Error::InvalidArgument(
            "benchmark document count exceeds index limit".into(),
        ));
    }

    let analyzer = Analyzer::default();
    let mut random = Lcg::new(config.seed);
    let index_started = Instant::now();
    let mut builder = IndexBuilder::new(analyzer);
    for number in 0..config.documents {
        let mut title = String::new();
        let mut body = String::new();
        for _ in 0..6 {
            let word = random.word(256);
            push_word(&mut title, &word);
        }
        for _ in 0..72 {
            // A skewed vocabulary creates a realistic mix of common and rare terms.
            let bucket = random.next_u64() % 10;
            let vocabulary = if bucket < 7 { 64 } else { 512 };
            let word = random.word(vocabulary);
            push_word(&mut body, &word);
        }
        let category = format!("c{}", random.next_u64() % 8);
        builder.add_document(Document::from_fields(
            format!("doc-{number:08}"),
            [("title", title), ("body", body), ("category", category)],
        )?)?;
    }
    let index = builder.finish();
    let index_time = index_started.elapsed();

    let mut query_random = Lcg::new(config.seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut queries = Vec::with_capacity(config.queries);
    for _ in 0..config.queries {
        let text = format!(
            "{} {} {}",
            query_random.word(512),
            query_random.word(512),
            query_random.word(512)
        );
        queries.push(
            SearchQuery::from_text(analyzer, &text, Some("body"))?
                .with_operator(BooleanOperator::Or),
        );
    }

    let exhaustive_started = Instant::now();
    let mut exhaustive_results = Vec::with_capacity(queries.len());
    let mut exhaustive_stats = SearchStats::default();
    for query in &queries {
        let outcome = index.search(
            query,
            SearchOptions {
                top_k: config.top_k,
                pruning: PruningStrategy::Exhaustive,
                ..SearchOptions::default()
            },
        )?;
        add_stats(&mut exhaustive_stats, outcome.stats);
        exhaustive_results.push(outcome.hits);
    }
    let exhaustive_time = exhaustive_started.elapsed();

    let wand_started = Instant::now();
    let mut wand_stats = SearchStats::default();
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for (query, exhaustive) in queries.iter().zip(&exhaustive_results) {
        let outcome = index.search(
            query,
            SearchOptions {
                top_k: config.top_k,
                pruning: PruningStrategy::Wand,
                ..SearchOptions::default()
            },
        )?;
        add_stats(&mut wand_stats, outcome.stats);
        if outcome.hits.len() != exhaustive.len()
            || outcome.hits.iter().zip(exhaustive).any(|(left, right)| {
                left.doc_id != right.doc_id || left.score.to_bits() != right.score.to_bits()
            })
        {
            return Err(Error::CorruptIndex(
                "WAND benchmark results differ from exhaustive results".into(),
            ));
        }
        for hit in &outcome.hits {
            checksum ^= u64::from(hit.doc_id);
            checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
            checksum ^= hit.score.to_bits();
            checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let wand_time = wand_started.elapsed();

    // Block-max is checked against the same exhaustive results, and against the
    // same bit patterns rather than an epsilon: a pruning strategy that changed
    // a score in the last place would still be wrong about what it skipped.
    let block_max_started = Instant::now();
    let mut block_max_stats = SearchStats::default();
    for (query, exhaustive) in queries.iter().zip(&exhaustive_results) {
        let outcome = index.search(
            query,
            SearchOptions {
                top_k: config.top_k,
                pruning: PruningStrategy::BlockMaxWand,
                ..SearchOptions::default()
            },
        )?;
        add_stats(&mut block_max_stats, outcome.stats);
        if outcome.hits.len() != exhaustive.len()
            || outcome.hits.iter().zip(exhaustive).any(|(left, right)| {
                left.doc_id != right.doc_id || left.score.to_bits() != right.score.to_bits()
            })
        {
            return Err(Error::CorruptIndex(
                "block-max WAND benchmark results differ from exhaustive results".into(),
            ));
        }
    }
    let block_max_time = block_max_started.elapsed();

    let stats = index.stats();
    let posting_codec = index.posting_codec_stats()?;
    let mut persisted = Vec::new();
    index.write_to(&mut persisted)?;
    let persisted_index_bytes = u64::try_from(persisted.len())
        .map_err(|_| Error::InvalidArgument("persisted index size does not fit u64".into()))?;
    let block_max_metadata_bytes = block_max_metadata_encoded_bytes(&index)?;
    let block_max_streams = index.block_max.len();
    let block_max_blocks = index
        .block_max
        .values()
        .map(|metadata| metadata.default_bounds.len())
        .sum();

    Ok(BenchmarkReport {
        config,
        index_time,
        exhaustive_time,
        wand_time,
        block_max_time,
        exhaustive_stats,
        wand_stats,
        block_max_stats,
        checksum,
        index_terms: stats.terms,
        index_postings: stats.postings,
        index_tokens: stats.tokens,
        posting_codec,
        persisted_index_bytes,
        block_max_metadata_bytes,
        block_max_streams,
        block_max_blocks,
    })
}

/// Write a stable machine-readable benchmark report. Durations remain
/// machine-dependent; workload counters and checksum are deterministic.
pub fn write_json(report: &BenchmarkReport, mut writer: impl Write) -> Result<()> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema_version\": 3,")?;
    writeln!(writer, "  \"verified_exact\": true,")?;
    writeln!(writer, "  \"documents\": {},", report.config.documents)?;
    writeln!(writer, "  \"queries\": {},", report.config.queries)?;
    writeln!(writer, "  \"top_k\": {},", report.config.top_k)?;
    writeln!(writer, "  \"seed\": {},", report.config.seed)?;
    writeln!(writer, "  \"index_terms\": {},", report.index_terms)?;
    writeln!(writer, "  \"index_postings\": {},", report.index_postings)?;
    writeln!(writer, "  \"index_tokens\": {},", report.index_tokens)?;
    writeln!(
        writer,
        "  \"posting_codec\": {{\"raw_bytes\": {}, \"encoded_bytes\": {}, \"ratio\": {:.12}}},",
        report.posting_codec.uncompressed_bytes,
        report.posting_codec.encoded_bytes,
        report.posting_codec.ratio()
    )?;
    writeln!(
        writer,
        "  \"persistence\": {{\"format_version\": 3, \"serialized_bytes\": {}, \"base_index_bytes\": {}, \"block_max_metadata_bytes\": {}, \"block_max_streams\": {}, \"block_max_blocks\": {}}},",
        report.persisted_index_bytes,
        report
            .persisted_index_bytes
            .saturating_sub(report.block_max_metadata_bytes),
        report.block_max_metadata_bytes,
        report.block_max_streams,
        report.block_max_blocks
    )?;
    writeln!(
        writer,
        "  \"index_elapsed_micros\": {},",
        report.index_time.as_micros()
    )?;
    writeln!(
        writer,
        "  \"exhaustive_elapsed_micros\": {},",
        report.exhaustive_time.as_micros()
    )?;
    writeln!(
        writer,
        "  \"wand_elapsed_micros\": {},",
        report.wand_time.as_micros()
    )?;
    writeln!(
        writer,
        "  \"block_max_elapsed_micros\": {},",
        report.block_max_time.as_micros()
    )?;
    writeln!(
        writer,
        "  \"exhaustive\": {{\"evaluated\": {}, \"advanced\": {}, \"skipped\": {}}},",
        report.exhaustive_stats.evaluated_candidates,
        report.exhaustive_stats.postings_advanced,
        report.exhaustive_stats.postings_skipped
    )?;
    writeln!(
        writer,
        "  \"wand\": {{\"evaluated\": {}, \"advanced\": {}, \"skipped\": {}}},",
        report.wand_stats.evaluated_candidates,
        report.wand_stats.postings_advanced,
        report.wand_stats.postings_skipped
    )?;
    writeln!(
        writer,
        "  \"block_max_wand\": {{\"evaluated\": {}, \"advanced\": {}, \"skipped\": {}, \"precomputed_bounds_loaded\": {}, \"postings_covered_by_bounds\": {}, \"postings_scanned_for_bounds\": {}}},",
        report.block_max_stats.evaluated_candidates,
        report.block_max_stats.postings_advanced,
        report.block_max_stats.postings_skipped,
        report.block_max_stats.block_max_bounds_loaded,
        report.block_max_stats.block_max_postings_covered,
        report.block_max_stats.block_max_postings_scanned
    )?;
    writeln!(writer, "  \"checksum\": \"{:016x}\"", report.checksum)?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn add_stats(total: &mut SearchStats, current: SearchStats) {
    total.evaluated_candidates += current.evaluated_candidates;
    total.postings_advanced += current.postings_advanced;
    total.postings_skipped += current.postings_skipped;
    total.block_max_bounds_loaded += current.block_max_bounds_loaded;
    total.block_max_postings_covered += current.block_max_postings_covered;
    total.block_max_postings_scanned += current.block_max_postings_scanned;
}

fn push_word(output: &mut String, word: &str) {
    if !output.is_empty() {
        output.push(' ');
    }
    output.push_str(word);
}

#[derive(Clone, Copy, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xa076_1d64_78bd_642f,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn word(&mut self, vocabulary: u64) -> String {
        format!("t{:03}", self.next_u64() % vocabulary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_is_reproducible_by_checksum() {
        let config = BenchmarkConfig {
            documents: 120,
            queries: 12,
            seed: 7,
            top_k: 5,
        };
        let first = run(config).unwrap();
        let second = run(config).unwrap();
        assert_eq!(first.checksum, second.checksum);
        assert_eq!(first.index_terms, second.index_terms);
        assert_eq!(first.index_postings, second.index_postings);
    }

    #[test]
    fn benchmark_exercises_every_execution_path() {
        let report = run(BenchmarkConfig {
            documents: 100,
            queries: 8,
            seed: 19,
            top_k: 3,
        })
        .unwrap();
        assert!(report.exhaustive_stats.evaluated_candidates > 0);
        assert!(report.wand_stats.evaluated_candidates > 0);
        assert!(report.wand_stats.postings_advanced > 0);
        assert!(report.block_max_stats.evaluated_candidates > 0);
        assert!(report.block_max_stats.postings_advanced > 0);
        assert!(report.block_max_stats.block_max_bounds_loaded > 0);
        assert!(
            report.block_max_stats.block_max_postings_covered
                > report.block_max_stats.block_max_bounds_loaded
        );
        assert!(report.persisted_index_bytes > report.block_max_metadata_bytes);
        assert!(report.block_max_metadata_bytes > 8);
        assert!(report.block_max_streams > 0);
        assert!(report.block_max_blocks > 0);
        // `run` returns an error if any strategy disagrees with exhaustive, so
        // reaching here at all is the exactness check; this pins the direction.
        assert!(
            report.block_max_stats.evaluated_candidates
                <= report.exhaustive_stats.evaluated_candidates
        );
    }

    #[test]
    fn benchmark_json_records_storage_and_bound_loading_tradeoffs() {
        let report = run(BenchmarkConfig {
            documents: 100,
            queries: 8,
            seed: 23,
            top_k: 3,
        })
        .unwrap();
        let mut json = Vec::new();
        write_json(&report, &mut json).unwrap();
        let json = String::from_utf8(json).unwrap();
        assert!(json.contains("\"schema_version\": 3"));
        assert!(json.contains("\"block_max_metadata_bytes\""));
        assert!(json.contains("\"precomputed_bounds_loaded\""));
        assert!(json.contains("\"postings_scanned_for_bounds\": 0"));
    }

    #[test]
    fn benchmark_rejects_zero_dimensions() {
        assert!(
            run(BenchmarkConfig {
                documents: 0,
                ..BenchmarkConfig::default()
            })
            .is_err()
        );
        assert!(
            run(BenchmarkConfig {
                queries: 0,
                ..BenchmarkConfig::default()
            })
            .is_err()
        );
        assert!(
            run(BenchmarkConfig {
                top_k: 0,
                ..BenchmarkConfig::default()
            })
            .is_err()
        );
    }
}
