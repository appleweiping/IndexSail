//! Batch retrieval, exact executor verification, TREC run output, and metrics.

use std::collections::BTreeSet;
use std::io::Write;
use std::time::{Duration, Instant};

use crate::analysis::Analyzer;
use crate::error::{Error, Result};
use crate::index::InvertedIndex;
use crate::query::{BooleanOperator, SearchQuery};
use crate::search::{
    Bm25Params, PruningStrategy, ScoringModel, SearchHit, SearchOptions, SearchOutcome, SearchStats,
};
use crate::shard::ShardedIndex;
use crate::trec::{Qrels, Topic};

/// Common batch-evaluation surface for monolithic and sharded collections.
pub trait RetrievalBackend {
    fn analyzer(&self) -> Analyzer;
    fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome>;
}

impl RetrievalBackend for InvertedIndex {
    fn analyzer(&self) -> Analyzer {
        InvertedIndex::analyzer(self)
    }

    fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
        InvertedIndex::search(self, query, options)
    }
}

impl RetrievalBackend for ShardedIndex {
    fn analyzer(&self) -> Analyzer {
        ShardedIndex::analyzer(self)
    }

    fn search(&self, query: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
        ShardedIndex::search(self, query, options)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchConfig {
    pub top_k: usize,
    pub pruning: PruningStrategy,
    pub verify_exact: bool,
    pub field: Option<String>,
    pub operator: BooleanOperator,
    pub bm25: Bm25Params,
    pub scoring: ScoringModel,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            top_k: 1_000,
            pruning: PruningStrategy::Wand,
            verify_exact: false,
            field: None,
            operator: BooleanOperator::Or,
            bm25: Bm25Params::default(),
            scoring: ScoringModel::Bm25,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QueryMetrics {
    pub relevant_total: usize,
    pub relevant_retrieved: usize,
    pub average_precision: f64,
    pub reciprocal_rank: f64,
    pub ndcg: f64,
    pub recall: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryReport {
    pub topic_id: String,
    pub query_text: String,
    pub elapsed: Duration,
    pub stats: SearchStats,
    pub hits: Vec<SearchHit>,
    pub metrics: Option<QueryMetrics>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AggregateMetrics {
    pub map: f64,
    pub mrr: f64,
    pub mean_ndcg: f64,
    pub mean_recall: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchReport {
    pub config: BatchConfig,
    pub queries: Vec<QueryReport>,
    pub total_search_time: Duration,
    pub total_verification_time: Duration,
    pub total_stats: SearchStats,
    pub aggregate: Option<AggregateMetrics>,
}

/// Execute topics in input order, optionally evaluate qrels and cross-check
/// every result against the other executor.
pub fn evaluate_batch<B: RetrievalBackend + ?Sized>(
    index: &B,
    topics: &[Topic],
    qrels: Option<&Qrels>,
    config: BatchConfig,
) -> Result<BatchReport> {
    if topics.is_empty() {
        return Err(Error::InvalidArgument(
            "batch requires at least one topic".into(),
        ));
    }
    if config.top_k == 0 {
        return Err(Error::InvalidArgument(
            "batch top_k must be greater than zero".into(),
        ));
    }
    config.bm25.validate()?;
    SearchOptions {
        top_k: config.top_k,
        pruning: config.pruning,
        explain: false,
        bm25: config.bm25,
        scoring: config.scoring,
    }
    .validate()?;
    if !matches!(
        config.scoring,
        ScoringModel::Bm25 | ScoringModel::QuantizedBm25 { .. }
    ) && config.verify_exact
    {
        return Err(Error::InvalidArgument(
            "DPH, PL2, and QLD cannot use --verify until another exact executor is available"
                .into(),
        ));
    }
    let mut topic_ids = BTreeSet::new();
    if let Some(topic) = topics.iter().find(|topic| {
        topic.id.trim().is_empty()
            || topic.id.chars().any(char::is_whitespace)
            || topic.text.trim().is_empty()
            || !topic_ids.insert(topic.id.as_str())
    }) {
        return Err(Error::InvalidArgument(format!(
            "invalid or duplicate batch topic '{}'",
            topic.id
        )));
    }

    let mut queries = Vec::with_capacity(topics.len());
    let mut total_search_time = Duration::ZERO;
    let mut total_verification_time = Duration::ZERO;
    let mut total_stats = SearchStats::default();
    for topic in topics {
        let query = SearchQuery::from_text(index.analyzer(), &topic.text, config.field.as_deref())?
            .with_operator(config.operator);
        let options = SearchOptions {
            top_k: config.top_k,
            pruning: config.pruning,
            explain: false,
            bm25: config.bm25,
            scoring: config.scoring,
        };
        let started = Instant::now();
        let outcome = index.search(&query, options)?;
        let elapsed = started.elapsed();
        total_search_time += elapsed;
        add_stats(&mut total_stats, outcome.stats)?;

        if config.verify_exact {
            let oracle_options = SearchOptions {
                // A pruning run is checked against the exhaustive scan, which
                // is the only strategy that cannot be wrong about what it
                // skipped, and the exhaustive run is checked against `WAND`.
                pruning: match config.pruning {
                    PruningStrategy::Exhaustive => PruningStrategy::Wand,
                    PruningStrategy::Wand
                    | PruningStrategy::BlockMaxWand
                    | PruningStrategy::MaxScore
                    | PruningStrategy::BlockMaxMaxScore => PruningStrategy::Exhaustive,
                },
                ..options
            };
            let verification_started = Instant::now();
            let oracle = index.search(&query, oracle_options)?;
            total_verification_time += verification_started.elapsed();
            verify_hits(&topic.id, &outcome.hits, &oracle.hits)?;
        }

        let metrics = qrels.map(|qrels| metrics_for(&topic.id, &outcome.hits, qrels, config.top_k));
        queries.push(QueryReport {
            topic_id: topic.id.clone(),
            query_text: topic.text.clone(),
            elapsed,
            stats: outcome.stats,
            hits: outcome.hits,
            metrics,
        });
    }

    let aggregate = qrels.map(|_| aggregate_metrics(&queries));
    Ok(BatchReport {
        config,
        queries,
        total_search_time,
        total_verification_time,
        total_stats,
        aggregate,
    })
}

/// Write the six-column TREC run format in stable topic/rank order.
pub fn write_trec_run(report: &BatchReport, tag: &str, mut writer: impl Write) -> Result<()> {
    validate_run(report, tag)?;
    for query in &report.queries {
        for (rank, hit) in query.hits.iter().enumerate() {
            writeln!(
                writer,
                "{} Q0 {} {} {:.12} {}",
                query.topic_id,
                hit.external_id,
                rank + 1,
                hit.score,
                tag
            )?;
        }
    }
    Ok(())
}

/// Write a hand-encoded, stable JSON report suitable for regression jobs.
#[allow(clippy::too_many_lines)]
pub fn write_json_report(report: &BatchReport, mut writer: impl Write) -> Result<()> {
    validate_finite_report(report)?;
    let strategy = match report.config.pruning {
        PruningStrategy::Exhaustive => "exhaustive",
        PruningStrategy::Wand => "wand",
        PruningStrategy::BlockMaxWand => "block-max-wand",
        PruningStrategy::MaxScore => "maxscore",
        PruningStrategy::BlockMaxMaxScore => "block-max-maxscore",
    };
    let operator = match report.config.operator {
        BooleanOperator::And => "and",
        BooleanOperator::Or => "or",
    };
    writeln!(writer, "{{")?;
    let block_max_maxscore = report.config.pruning == PruningStrategy::BlockMaxMaxScore;
    let schema_version = match (report.config.scoring, block_max_maxscore) {
        (ScoringModel::Bm25, false) => 1,
        (ScoringModel::Bm25, true) => 3,
        (_, false) => 2,
        (_, true) => 4,
    };
    writeln!(writer, "  \"schema_version\": {schema_version},")?;
    match report.config.scoring {
        ScoringModel::Bm25 => {}
        ScoringModel::QuantizedBm25 { bits, max_impact } => {
            writeln!(writer, "  \"scorer\": \"qbm25\",")?;
            writeln!(writer, "  \"quant_bits\": {bits},")?;
            writeln!(writer, "  \"quant_max\": {max_impact},")?;
        }
        ScoringModel::Dph => writeln!(writer, "  \"scorer\": \"dph\",")?,
        ScoringModel::Pl2 { c } => {
            writeln!(writer, "  \"scorer\": \"pl2\",")?;
            writeln!(writer, "  \"pl2_c\": {c},")?;
        }
        ScoringModel::Qld { mu } => {
            writeln!(writer, "  \"scorer\": \"qld\",")?;
            writeln!(writer, "  \"qld_mu\": {mu},")?;
        }
    }
    writeln!(writer, "  \"strategy\": \"{strategy}\",")?;
    writeln!(writer, "  \"operator\": \"{operator}\",")?;
    writeln!(writer, "  \"top_k\": {},", report.config.top_k)?;
    write!(writer, "  \"field\": ")?;
    write_json_optional_string(&mut writer, report.config.field.as_deref())?;
    writeln!(writer, ",")?;
    writeln!(
        writer,
        "  \"verified_exact\": {},",
        report.config.verify_exact
    )?;
    writeln!(writer, "  \"query_count\": {},", report.queries.len())?;
    writeln!(
        writer,
        "  \"total_search_micros\": {},",
        report.total_search_time.as_micros()
    )?;
    writeln!(
        writer,
        "  \"total_verification_micros\": {},",
        report.total_verification_time.as_micros()
    )?;
    writeln!(
        writer,
        "  \"evaluated_candidates\": {},",
        report.total_stats.evaluated_candidates
    )?;
    if block_max_maxscore {
        writeln!(
            writer,
            "  \"block_bound_rejections\": {},",
            report.total_stats.block_bound_rejections
        )?;
    }
    writeln!(
        writer,
        "  \"postings_advanced\": {},",
        report.total_stats.postings_advanced
    )?;
    writeln!(
        writer,
        "  \"postings_skipped\": {},",
        report.total_stats.postings_skipped
    )?;
    writeln!(
        writer,
        "  \"block_max_bounds_loaded\": {},",
        report.total_stats.block_max_bounds_loaded
    )?;
    writeln!(
        writer,
        "  \"block_max_postings_covered\": {},",
        report.total_stats.block_max_postings_covered
    )?;
    writeln!(
        writer,
        "  \"block_max_postings_scanned\": {},",
        report.total_stats.block_max_postings_scanned
    )?;
    write!(writer, "  \"aggregate\": ")?;
    if let Some(metrics) = report.aggregate {
        writeln!(
            writer,
            "{{\"map\": {:.12}, \"mrr\": {:.12}, \"mean_ndcg\": {:.12}, \"mean_recall\": {:.12}}},",
            metrics.map, metrics.mrr, metrics.mean_ndcg, metrics.mean_recall
        )?;
    } else {
        writeln!(writer, "null,")?;
    }
    writeln!(writer, "  \"queries\": [")?;
    for (index, query) in report.queries.iter().enumerate() {
        writeln!(writer, "    {{")?;
        write!(writer, "      \"topic_id\": ")?;
        write_json_string(&mut writer, &query.topic_id)?;
        writeln!(writer, ",")?;
        write!(writer, "      \"query\": ")?;
        write_json_string(&mut writer, &query.query_text)?;
        writeln!(writer, ",")?;
        writeln!(
            writer,
            "      \"elapsed_micros\": {},",
            query.elapsed.as_micros()
        )?;
        writeln!(
            writer,
            "      \"evaluated_candidates\": {},",
            query.stats.evaluated_candidates
        )?;
        if block_max_maxscore {
            writeln!(
                writer,
                "      \"block_bound_rejections\": {},",
                query.stats.block_bound_rejections
            )?;
        }
        writeln!(
            writer,
            "      \"postings_advanced\": {},",
            query.stats.postings_advanced
        )?;
        writeln!(
            writer,
            "      \"postings_skipped\": {},",
            query.stats.postings_skipped
        )?;
        writeln!(
            writer,
            "      \"block_max_bounds_loaded\": {},",
            query.stats.block_max_bounds_loaded
        )?;
        writeln!(
            writer,
            "      \"block_max_postings_covered\": {},",
            query.stats.block_max_postings_covered
        )?;
        writeln!(
            writer,
            "      \"block_max_postings_scanned\": {},",
            query.stats.block_max_postings_scanned
        )?;
        write!(writer, "      \"metrics\": ")?;
        if let Some(metrics) = query.metrics {
            writeln!(
                writer,
                "{{\"relevant_total\": {}, \"relevant_retrieved\": {}, \"average_precision\": {:.12}, \"reciprocal_rank\": {:.12}, \"ndcg\": {:.12}, \"recall\": {:.12}}},",
                metrics.relevant_total,
                metrics.relevant_retrieved,
                metrics.average_precision,
                metrics.reciprocal_rank,
                metrics.ndcg,
                metrics.recall
            )?;
        } else {
            writeln!(writer, "null,")?;
        }
        writeln!(writer, "      \"hits\": [")?;
        for (rank, hit) in query.hits.iter().enumerate() {
            write!(
                writer,
                "        {{\"rank\": {}, \"document_id\": ",
                rank + 1
            )?;
            write_json_string(&mut writer, &hit.external_id)?;
            write!(writer, ", \"score\": {:.12}}}", hit.score)?;
            writeln!(
                writer,
                "{}",
                if rank + 1 == query.hits.len() {
                    ""
                } else {
                    ","
                }
            )?;
        }
        writeln!(writer, "      ]")?;
        writeln!(
            writer,
            "    }}{}",
            if index + 1 == report.queries.len() {
                ""
            } else {
                ","
            }
        )?;
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn verify_hits(topic_id: &str, selected: &[SearchHit], oracle: &[SearchHit]) -> Result<()> {
    if selected.len() != oracle.len() {
        return Err(Error::CorruptIndex(format!(
            "executor mismatch for topic '{topic_id}': {} versus {} hits",
            selected.len(),
            oracle.len()
        )));
    }
    for (rank, (left, right)) in selected.iter().zip(oracle).enumerate() {
        if left.doc_id != right.doc_id || left.score.to_bits() != right.score.to_bits() {
            return Err(Error::CorruptIndex(format!(
                "executor mismatch for topic '{topic_id}' at rank {}",
                rank + 1
            )));
        }
    }
    Ok(())
}

fn validate_run(report: &BatchReport, tag: &str) -> Result<()> {
    if tag.is_empty()
        || tag.len() > 64
        || tag.chars().any(char::is_whitespace)
        || tag.chars().any(char::is_control)
    {
        return Err(Error::InvalidArgument(
            "run tag must be 1..=64 non-whitespace, non-control characters".into(),
        ));
    }
    let mut topic_ids = BTreeSet::new();
    for query in &report.queries {
        if !valid_run_token(&query.topic_id) || !topic_ids.insert(query.topic_id.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "topic id '{}' cannot be represented uniquely in a TREC run",
                query.topic_id
            )));
        }
        if query.hits.len() > report.config.top_k {
            return Err(Error::InvalidArgument(format!(
                "topic '{}' has more hits than the configured cutoff",
                query.topic_id
            )));
        }
        let mut document_ids = BTreeSet::new();
        for hit in &query.hits {
            if !valid_run_token(&hit.external_id)
                || !document_ids.insert(hit.external_id.as_str())
                || !hit.score.is_finite()
            {
                return Err(Error::InvalidArgument(format!(
                    "document id '{}' or score cannot be represented in a TREC run",
                    hit.external_id
                )));
            }
        }
    }
    Ok(())
}

fn valid_run_token(value: &str) -> bool {
    !value.is_empty()
        && !value.chars().any(char::is_whitespace)
        && !value.chars().any(char::is_control)
}

fn validate_finite_report(report: &BatchReport) -> Result<()> {
    report.config.bm25.validate()?;
    SearchOptions {
        top_k: report.config.top_k,
        pruning: report.config.pruning,
        explain: false,
        bm25: report.config.bm25,
        scoring: report.config.scoring,
    }
    .validate()?;
    if !matches!(
        report.config.scoring,
        ScoringModel::Bm25 | ScoringModel::QuantizedBm25 { .. }
    ) && report.config.verify_exact
    {
        return Err(Error::InvalidArgument(
            "DPH, PL2, and QLD cannot report exact cross-executor verification".into(),
        ));
    }
    let finite_metrics = |metrics: QueryMetrics| {
        metrics.average_precision.is_finite()
            && metrics.reciprocal_rank.is_finite()
            && metrics.ndcg.is_finite()
            && metrics.recall.is_finite()
    };
    if report.queries.iter().any(|query| {
        query.hits.iter().any(|hit| !hit.score.is_finite())
            || query
                .metrics
                .is_some_and(|metrics| !finite_metrics(metrics))
    }) || report.aggregate.is_some_and(|metrics| {
        !metrics.map.is_finite()
            || !metrics.mrr.is_finite()
            || !metrics.mean_ndcg.is_finite()
            || !metrics.mean_recall.is_finite()
    }) {
        return Err(Error::InvalidArgument(
            "JSON report contains a non-finite number".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn metrics_for(topic_id: &str, hits: &[SearchHit], qrels: &Qrels, cutoff: usize) -> QueryMetrics {
    let judgments = qrels.judgments_for(topic_id);
    let relevant_total = qrels.relevant_count(topic_id);
    let mut relevant_retrieved = 0_usize;
    let mut precision_sum = 0.0;
    let mut reciprocal_rank = 0.0;
    let mut dcg = 0.0;
    for (index, hit) in hits.iter().take(cutoff).enumerate() {
        let relevance = judgments
            .and_then(|values| values.get(&hit.external_id))
            .copied()
            .unwrap_or(0);
        if relevance > 0 {
            relevant_retrieved += 1;
            #[allow(clippy::cast_precision_loss)]
            let precision = relevant_retrieved as f64 / (index + 1) as f64;
            precision_sum += precision;
            if reciprocal_rank == 0.0 {
                reciprocal_rank = 1.0 / (index + 1) as f64;
            }
        }
        dcg += discounted_gain(relevance, index);
    }

    let mut ideal = judgments
        .into_iter()
        .flat_map(|values| values.values())
        .copied()
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    ideal.sort_unstable_by(|left, right| right.cmp(left));
    let ideal_dcg = ideal
        .into_iter()
        .take(cutoff)
        .enumerate()
        .map(|(index, relevance)| discounted_gain(relevance, index))
        .sum::<f64>();

    QueryMetrics {
        relevant_total,
        relevant_retrieved,
        average_precision: if relevant_total == 0 {
            0.0
        } else {
            precision_sum / relevant_total as f64
        },
        reciprocal_rank,
        ndcg: if ideal_dcg == 0.0 {
            0.0
        } else {
            dcg / ideal_dcg
        },
        recall: if relevant_total == 0 {
            0.0
        } else {
            relevant_retrieved as f64 / relevant_total as f64
        },
    }
}

#[allow(clippy::cast_precision_loss)]
fn discounted_gain(relevance: i32, zero_based_rank: usize) -> f64 {
    if relevance <= 0 {
        return 0.0;
    }
    let relevance = u32::try_from(relevance).expect("positive relevance fits u32");
    let gain = 2_u64.pow(relevance) - 1;
    gain as f64 / (zero_based_rank as f64 + 2.0).log2()
}

#[allow(clippy::cast_precision_loss)]
fn aggregate_metrics(queries: &[QueryReport]) -> AggregateMetrics {
    let count = queries.len() as f64;
    let totals = queries.iter().filter_map(|query| query.metrics).fold(
        AggregateMetrics::default(),
        |mut total, metrics| {
            total.map += metrics.average_precision;
            total.mrr += metrics.reciprocal_rank;
            total.mean_ndcg += metrics.ndcg;
            total.mean_recall += metrics.recall;
            total
        },
    );
    AggregateMetrics {
        map: totals.map / count,
        mrr: totals.mrr / count,
        mean_ndcg: totals.mean_ndcg / count,
        mean_recall: totals.mean_recall / count,
    }
}

fn add_stats(total: &mut SearchStats, current: SearchStats) -> Result<()> {
    total.evaluated_candidates = total
        .evaluated_candidates
        .checked_add(current.evaluated_candidates)
        .ok_or_else(|| Error::InvalidArgument("batch candidate count overflow".into()))?;
    total.block_bound_rejections = total
        .block_bound_rejections
        .checked_add(current.block_bound_rejections)
        .ok_or_else(|| Error::InvalidArgument("batch block rejection count overflow".into()))?;
    total.postings_advanced = total
        .postings_advanced
        .checked_add(current.postings_advanced)
        .ok_or_else(|| Error::InvalidArgument("batch posting count overflow".into()))?;
    total.postings_skipped = total
        .postings_skipped
        .checked_add(current.postings_skipped)
        .ok_or_else(|| Error::InvalidArgument("batch posting count overflow".into()))?;
    total.block_max_bounds_loaded = total
        .block_max_bounds_loaded
        .checked_add(current.block_max_bounds_loaded)
        .ok_or_else(|| Error::InvalidArgument("batch block-bound count overflow".into()))?;
    total.block_max_postings_covered = total
        .block_max_postings_covered
        .checked_add(current.block_max_postings_covered)
        .ok_or_else(|| Error::InvalidArgument("batch block-posting count overflow".into()))?;
    total.block_max_postings_scanned = total
        .block_max_postings_scanned
        .checked_add(current.block_max_postings_scanned)
        .ok_or_else(|| Error::InvalidArgument("batch block-posting scan count overflow".into()))?;
    Ok(())
}

fn write_json_optional_string(writer: &mut impl Write, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => write_json_string(writer, value),
        None => writer.write_all(b"null").map_err(Error::from),
    }
}

fn write_json_string(writer: &mut impl Write, value: &str) -> Result<()> {
    writer.write_all(b"\"")?;
    for character in value.chars() {
        match character {
            '"' => writer.write_all(b"\\\"")?,
            '\\' => writer.write_all(b"\\\\")?,
            '\n' => writer.write_all(b"\\n")?,
            '\r' => writer.write_all(b"\\r")?,
            '\t' => writer.write_all(b"\\t")?,
            value if value <= '\u{001f}' => write!(writer, "\\u{:04x}", value as u32)?,
            value => write!(writer, "{value}")?,
        }
    }
    writer.write_all(b"\"")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Mutex;

    use super::*;
    use crate::trec::parse_qrels;
    use crate::{Analyzer, Document, IndexBuilder};

    fn fixture() -> (InvertedIndex, Vec<Topic>, Qrels) {
        let mut builder = IndexBuilder::new(Analyzer::default());
        for (id, body) in [
            ("D1", "rust search ranking"),
            ("D2", "rust compiler"),
            ("D3", "search evaluation"),
            ("D4", "unrelated"),
        ] {
            builder
                .add_document(Document::from_fields(id, [("body", body)]).unwrap())
                .unwrap();
        }
        let topics = vec![
            Topic {
                id: "1".into(),
                text: "rust search".into(),
            },
            Topic {
                id: "2".into(),
                text: "evaluation".into(),
            },
        ];
        let qrels = parse_qrels(Cursor::new("1 0 D1 2\n1 0 D2 1\n2 0 D3 1\n")).unwrap();
        (builder.finish(), topics, qrels)
    }

    #[test]
    fn batch_verifies_wand_and_computes_metrics() {
        let (index, topics, qrels) = fixture();
        let report = evaluate_batch(
            &index,
            &topics,
            Some(&qrels),
            BatchConfig {
                top_k: 3,
                field: Some("body".into()),
                verify_exact: true,
                ..BatchConfig::default()
            },
        )
        .unwrap();
        assert_eq!(report.queries.len(), 2);
        assert_eq!(report.queries[0].hits[0].external_id, "D1");
        let aggregate = report.aggregate.unwrap();
        assert!((aggregate.map - 1.0).abs() < 1e-12);
        assert!((aggregate.mrr - 1.0).abs() < 1e-12);
        assert!((aggregate.mean_ndcg - 1.0).abs() < 1e-12);
        assert!((aggregate.mean_recall - 1.0).abs() < 1e-12);
    }

    #[test]
    fn custom_bm25_batch_totals_block_bound_scans() {
        let (index, topics, _) = fixture();
        let report = evaluate_batch(
            &index,
            &topics,
            None,
            BatchConfig {
                top_k: 3,
                pruning: PruningStrategy::BlockMaxWand,
                field: Some("body".into()),
                bm25: Bm25Params { k1: 2.0, b: 0.5 },
                ..BatchConfig::default()
            },
        )
        .unwrap();
        let query_total = report
            .queries
            .iter()
            .map(|query| query.stats.block_max_postings_scanned)
            .sum::<usize>();

        assert!(query_total > 0);
        assert_eq!(report.total_stats.block_max_postings_scanned, query_total);
        let mut json = Vec::new();
        write_json_report(&report, &mut json).unwrap();
        let json = String::from_utf8(json).unwrap();
        assert!(json.contains(&format!("\"block_max_postings_scanned\": {query_total},")));
        assert_eq!(
            json.matches("\"block_max_postings_scanned\"").count(),
            report.queries.len() + 1
        );
    }

    #[test]
    fn average_precision_uses_total_relevant_denominator_at_cutoff() {
        let (index, topics, qrels) = fixture();
        let report = evaluate_batch(
            &index,
            &topics[..1],
            Some(&qrels),
            BatchConfig {
                top_k: 1,
                field: Some("body".into()),
                ..BatchConfig::default()
            },
        )
        .unwrap();
        let metrics = report.queries[0].metrics.unwrap();
        assert_eq!(metrics.relevant_total, 2);
        assert_eq!(metrics.relevant_retrieved, 1);
        assert!((metrics.average_precision - 0.5).abs() < 1e-12);
        assert!((metrics.recall - 0.5).abs() < 1e-12);
    }

    #[test]
    fn trec_run_and_json_are_stable_and_escaped() {
        let (index, mut topics, qrels) = fixture();
        topics[0].text = "rust \"search\"".into();
        let report = evaluate_batch(
            &index,
            &topics[..1],
            Some(&qrels),
            BatchConfig {
                top_k: 2,
                field: Some("body".into()),
                verify_exact: true,
                ..BatchConfig::default()
            },
        )
        .unwrap();
        let mut run = Vec::new();
        write_trec_run(&report, "indexsail-v1", &mut run).unwrap();
        let run = String::from_utf8(run).unwrap();
        assert!(run.starts_with("1 Q0 D1 1 "));
        assert!(run.lines().all(|line| line.split_whitespace().count() == 6));

        let mut json = Vec::new();
        write_json_report(&report, &mut json).unwrap();
        let json = String::from_utf8(json).unwrap();
        assert!(json.contains("rust \\\"search\\\""));
        assert!(json.contains("\"verified_exact\": true"));
        for counter in [
            "block_max_bounds_loaded",
            "block_max_postings_covered",
            "block_max_postings_scanned",
        ] {
            assert_eq!(json.matches(&format!("\"{counter}\"")).count(), 2);
        }
        assert!(json.ends_with("}\n"));
    }

    #[test]
    fn validation_rejects_duplicate_topics_and_bad_run_tags() {
        let (index, topics, _) = fixture();
        let duplicate = vec![topics[0].clone(), topics[0].clone()];
        assert!(evaluate_batch(&index, &duplicate, None, BatchConfig::default()).is_err());
        let report = evaluate_batch(&index, &topics[..1], None, BatchConfig::default()).unwrap();
        assert!(write_trec_run(&report, "bad tag", Vec::new()).is_err());

        let mut impossible_verification = report.clone();
        impossible_verification.config.scoring = ScoringModel::Dph;
        impossible_verification.config.pruning = PruningStrategy::Exhaustive;
        impossible_verification.config.verify_exact = true;
        let mut bytes = Vec::new();
        let error = write_json_report(&impossible_verification, &mut bytes).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot report exact cross-executor verification")
        );
        assert_eq!(bytes, Vec::<u8>::new());

        let mut invalid = report.clone();
        invalid.queries[0].hits[0].external_id = "two words".into();
        assert!(write_trec_run(&invalid, "valid", Vec::new()).is_err());

        let mut non_finite = report;
        non_finite.queries[0].hits[0].score = f64::NAN;
        assert!(write_json_report(&non_finite, Vec::new()).is_err());
    }

    #[test]
    fn pl2_batch_validation_and_json_do_not_publish_invalid_parameters() {
        let (index, topics, _) = fixture();
        let config = BatchConfig {
            field: Some("body".into()),
            pruning: PruningStrategy::Exhaustive,
            scoring: ScoringModel::Pl2 { c: 2.5 },
            ..BatchConfig::default()
        };
        let report = evaluate_batch(&index, &topics, None, config.clone()).unwrap();
        let mut bytes = Vec::new();
        write_json_report(&report, &mut bytes).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["scorer"], "pl2");
        assert_eq!(json["pl2_c"], 2.5);
        assert!(
            report
                .queries
                .iter()
                .all(|query| query.hits.iter().all(|hit| hit.score.is_finite()))
        );

        for c in [0.0, f64::NAN, f64::INFINITY] {
            let mut bad = report.clone();
            bad.config.scoring = ScoringModel::Pl2 { c };
            let mut output = Vec::new();
            assert!(write_json_report(&bad, &mut output).is_err());
            assert!(output.is_empty());
            assert!(
                evaluate_batch(
                    &index,
                    &topics,
                    None,
                    BatchConfig {
                        scoring: ScoringModel::Pl2 { c },
                        ..config.clone()
                    }
                )
                .is_err()
            );
        }
        assert!(
            evaluate_batch(
                &index,
                &topics,
                None,
                BatchConfig {
                    verify_exact: true,
                    ..config
                }
            )
            .unwrap_err()
            .to_string()
            .contains("--verify")
        );
    }

    #[test]
    fn qld_batch_report_is_versioned_and_invalid_mu_publishes_nothing() {
        let (index, topics, _) = fixture();
        let config = BatchConfig {
            field: Some("body".into()),
            pruning: PruningStrategy::Exhaustive,
            scoring: ScoringModel::Qld { mu: 2.0 },
            ..BatchConfig::default()
        };
        let report = evaluate_batch(&index, &topics, None, config.clone()).unwrap();
        let mut bytes = Vec::new();
        write_json_report(&report, &mut bytes).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["schema_version"], 2);
        assert_eq!(json["scorer"], "qld");
        assert_eq!(json["qld_mu"], 2.0);
        for mu in [0.0, f64::NAN, f64::INFINITY] {
            let mut invalid = report.clone();
            invalid.config.scoring = ScoringModel::Qld { mu };
            let mut output = Vec::new();
            assert!(write_json_report(&invalid, &mut output).is_err());
            assert!(output.is_empty());
            assert!(
                evaluate_batch(
                    &index,
                    &topics,
                    None,
                    BatchConfig {
                        scoring: ScoringModel::Qld { mu },
                        ..config.clone()
                    }
                )
                .is_err()
            );
        }
        assert!(
            evaluate_batch(
                &index,
                &topics,
                None,
                BatchConfig {
                    scoring: ScoringModel::Qld { mu: 2.0 },
                    verify_exact: true,
                    ..config
                }
            )
            .is_err()
        );
    }

    #[test]
    fn graded_metrics_and_missing_judgments_match_hand_calculation() {
        let hits = [
            SearchHit {
                doc_id: 0,
                external_id: "N".into(),
                score: 3.0,
                explanation: None,
            },
            SearchHit {
                doc_id: 1,
                external_id: "A".into(),
                score: 2.0,
                explanation: None,
            },
            SearchHit {
                doc_id: 2,
                external_id: "B".into(),
                score: 1.0,
                explanation: None,
            },
        ];
        let qrels = parse_qrels(Cursor::new("q 0 A 2\nq 0 B 1\nq 0 C 3\nnone 0 Z 0\n")).unwrap();
        let metrics = metrics_for("q", &hits, &qrels, 3);
        assert_eq!(metrics.relevant_total, 3);
        assert_eq!(metrics.relevant_retrieved, 2);
        assert!((metrics.average_precision - (0.5 + 2.0 / 3.0) / 3.0).abs() < 1e-12);
        assert!((metrics.reciprocal_rank - 0.5).abs() < 1e-12);
        assert!((metrics.recall - 2.0 / 3.0).abs() < 1e-12);
        assert!(metrics.ndcg > 0.0 && metrics.ndcg < 1.0);

        assert_eq!(
            metrics_for("none", &hits, &qrels, 3),
            QueryMetrics::default()
        );
        assert_eq!(
            metrics_for("absent", &hits, &qrels, 3),
            QueryMetrics::default()
        );
    }

    #[test]
    fn batch_rejects_invalid_topics_and_configuration_before_search() {
        let (index, topics, _) = fixture();
        for invalid in [
            vec![],
            vec![Topic {
                id: " ".into(),
                text: "rust".into(),
            }],
            vec![Topic {
                id: "bad id".into(),
                text: "rust".into(),
            }],
            vec![Topic {
                id: "ok".into(),
                text: "  ".into(),
            }],
            vec![topics[0].clone(), topics[0].clone()],
        ] {
            assert!(evaluate_batch(&index, &invalid, None, BatchConfig::default()).is_err());
        }
        assert!(
            evaluate_batch(
                &index,
                &topics,
                None,
                BatchConfig {
                    top_k: 0,
                    ..BatchConfig::default()
                }
            )
            .is_err()
        );
        assert!(
            evaluate_batch(
                &index,
                &topics,
                None,
                BatchConfig {
                    bm25: Bm25Params {
                        k1: f64::NAN,
                        b: 0.75
                    },
                    ..BatchConfig::default()
                }
            )
            .is_err()
        );
    }

    struct ScriptedBackend {
        calls: Mutex<Vec<PruningStrategy>>,
        selected: Vec<SearchHit>,
        oracle: Vec<SearchHit>,
    }

    impl RetrievalBackend for ScriptedBackend {
        fn analyzer(&self) -> Analyzer {
            Analyzer::default()
        }

        fn search(&self, _: &SearchQuery, options: SearchOptions) -> Result<SearchOutcome> {
            self.calls.lock().unwrap().push(options.pruning);
            Ok(SearchOutcome {
                hits: if options.pruning == PruningStrategy::Wand {
                    self.oracle.clone()
                } else {
                    self.selected.clone()
                },
                stats: SearchStats::default(),
            })
        }
    }

    #[test]
    fn exact_verification_detects_different_lengths_ranks_and_scores() {
        let hit = SearchHit {
            doc_id: 0,
            external_id: "D".into(),
            score: 1.0,
            explanation: None,
        };
        let topic = [Topic {
            id: "q".into(),
            text: "rust".into(),
        }];
        for (selected, oracle, expected) in [
            (vec![], vec![hit.clone()], "versus 1 hits"),
            (
                vec![SearchHit {
                    doc_id: 1,
                    ..hit.clone()
                }],
                vec![hit.clone()],
                "rank 1",
            ),
            (
                vec![SearchHit {
                    score: f64::from_bits(1.0_f64.to_bits() + 1),
                    ..hit.clone()
                }],
                vec![hit.clone()],
                "rank 1",
            ),
        ] {
            let backend = ScriptedBackend {
                calls: Mutex::new(Vec::new()),
                selected,
                oracle,
            };
            let error = evaluate_batch(
                &backend,
                &topic,
                None,
                BatchConfig {
                    pruning: PruningStrategy::Exhaustive,
                    verify_exact: true,
                    ..BatchConfig::default()
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(
                *backend.calls.lock().unwrap(),
                [PruningStrategy::Exhaustive, PruningStrategy::Wand]
            );
        }
    }

    #[test]
    fn run_writer_validates_all_tokens_duplicates_cutoff_and_finite_scores() {
        let (index, topics, _) = fixture();
        let report = evaluate_batch(&index, &topics[..1], None, BatchConfig::default()).unwrap();
        for tag in [
            "",
            "two words",
            "line\nbreak",
            "binary\u{1}tag",
            &"x".repeat(65),
        ] {
            assert!(write_trec_run(&report, tag, Vec::new()).is_err());
        }
        let mut invalid = report.clone();
        invalid.queries.push(report.queries[0].clone());
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.queries[0].topic_id = "bad\tid".into();
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.queries[0].topic_id = "bad\u{1}id".into();
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.queries[0].topic_id.clear();
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.queries[0].hits[0].external_id = "bad\u{1}id".into();
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.queries[0].hits[0].external_id.clear();
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.config.top_k = 0;
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report.clone();
        let duplicate_hit = invalid.queries[0].hits[0].clone();
        invalid.queries[0].hits.push(duplicate_hit);
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
        let mut invalid = report;
        invalid.queries[0].hits[0].score = f64::INFINITY;
        assert!(write_trec_run(&invalid, "ok", Vec::new()).is_err());
    }

    #[test]
    fn batch_statistics_reject_overflow_in_each_counter() {
        for field in 0..6 {
            let mut total = SearchStats::default();
            let mut current = SearchStats::default();
            match field {
                0 => {
                    total.evaluated_candidates = usize::MAX;
                    current.evaluated_candidates = 1;
                }
                1 => {
                    total.postings_advanced = usize::MAX;
                    current.postings_advanced = 1;
                }
                2 => {
                    total.postings_skipped = usize::MAX;
                    current.postings_skipped = 1;
                }
                3 => {
                    total.block_max_bounds_loaded = usize::MAX;
                    current.block_max_bounds_loaded = 1;
                }
                4 => {
                    total.block_max_postings_covered = usize::MAX;
                    current.block_max_postings_covered = 1;
                }
                5 => {
                    total.block_max_postings_scanned = usize::MAX;
                    current.block_max_postings_scanned = 1;
                }
                _ => unreachable!(),
            }
            assert!(
                add_stats(&mut total, current)
                    .unwrap_err()
                    .to_string()
                    .contains("overflow")
            );
        }
    }

    #[test]
    fn json_report_escapes_all_controls_and_rejects_nonfinite_metrics() {
        let (index, topics, qrels) = fixture();
        let mut report =
            evaluate_batch(&index, &topics[..1], Some(&qrels), BatchConfig::default()).unwrap();
        report.queries[0].topic_id = "q\"\\\n\r\t\u{0001}".into();
        report.queries[0].query_text = "search\u{0002}".into();
        report.config.field = Some("bo\u{0003}dy".into());
        report.queries[0].hits[0].external_id = "doc\u{0004}".into();
        let mut bytes = Vec::new();
        write_json_report(&report, &mut bytes).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded["field"], "bo\u{0003}dy");
        assert_eq!(decoded["queries"][0]["topic_id"], "q\"\\\n\r\t\u{0001}");
        assert_eq!(decoded["queries"][0]["query"], "search\u{0002}");
        assert_eq!(
            decoded["queries"][0]["hits"][0]["document_id"],
            "doc\u{0004}"
        );

        let mut invalid = report.clone();
        invalid.queries[0].metrics.as_mut().unwrap().ndcg = f64::NAN;
        assert!(write_json_report(&invalid, Vec::new()).is_err());
        let mut invalid = report.clone();
        invalid.aggregate.as_mut().unwrap().mean_recall = f64::INFINITY;
        assert!(write_json_report(&invalid, Vec::new()).is_err());
        let mut invalid = report;
        invalid.config.bm25.b = f64::NEG_INFINITY;
        assert!(write_json_report(&invalid, Vec::new()).is_err());
    }

    #[test]
    fn every_nonfinite_metric_is_rejected_before_json_writes() {
        let (index, topics, qrels) = fixture();
        let report =
            evaluate_batch(&index, &topics[..1], Some(&qrels), BatchConfig::default()).unwrap();
        for field in 0..8 {
            let mut invalid = report.clone();
            if field < 4 {
                let metrics = invalid.queries[0].metrics.as_mut().unwrap();
                match field {
                    0 => metrics.average_precision = f64::NAN,
                    1 => metrics.reciprocal_rank = f64::INFINITY,
                    2 => metrics.ndcg = f64::NEG_INFINITY,
                    3 => metrics.recall = f64::NAN,
                    _ => unreachable!(),
                }
            } else {
                let metrics = invalid.aggregate.as_mut().unwrap();
                match field {
                    4 => metrics.map = f64::NAN,
                    5 => metrics.mrr = f64::INFINITY,
                    6 => metrics.mean_ndcg = f64::NEG_INFINITY,
                    7 => metrics.mean_recall = f64::NAN,
                    _ => unreachable!(),
                }
            }
            let mut writer = b"existing report".to_vec();
            assert!(
                write_json_report(&invalid, &mut writer)
                    .unwrap_err()
                    .to_string()
                    .contains("non-finite")
            );
            assert_eq!(writer, b"existing report");
        }
    }
}
