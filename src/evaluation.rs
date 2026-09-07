//! Batch retrieval, exact executor verification, TREC run output, and metrics.

use std::collections::BTreeSet;
use std::io::Write;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::index::InvertedIndex;
use crate::query::{BooleanOperator, SearchQuery};
use crate::search::{Bm25Params, PruningStrategy, SearchHit, SearchOptions, SearchStats};
use crate::trec::{Qrels, Topic};

#[derive(Clone, Debug, PartialEq)]
pub struct BatchConfig {
    pub top_k: usize,
    pub pruning: PruningStrategy,
    pub verify_exact: bool,
    pub field: Option<String>,
    pub operator: BooleanOperator,
    pub bm25: Bm25Params,
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
pub fn evaluate_batch(
    index: &InvertedIndex,
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
                    | PruningStrategy::MaxScore => PruningStrategy::Exhaustive,
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
    };
    let operator = match report.config.operator {
        BooleanOperator::And => "and",
        BooleanOperator::Or => "or",
    };
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"schema_version\": 1,")?;
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

        let mut invalid = report.clone();
        invalid.queries[0].hits[0].external_id = "two words".into();
        assert!(write_trec_run(&invalid, "valid", Vec::new()).is_err());

        let mut non_finite = report;
        non_finite.queries[0].hits[0].score = f64::NAN;
        assert!(write_json_report(&non_finite, Vec::new()).is_err());
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
}
