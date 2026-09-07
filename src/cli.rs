use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::analysis::{AnalysisMode, Analyzer};
use crate::benchmark::{BenchmarkConfig, run as run_benchmark, write_json as write_benchmark_json};
use crate::document::Document;
use crate::error::{Error, Result};
use crate::evaluation::{BatchConfig, evaluate_batch, write_json_report, write_trec_run};
use crate::index::{IndexBuilder, InternalDocId, InvertedIndex};
use crate::persistence::{block_max_metadata_encoded_bytes, persisted_format_version};
use crate::query::{BooleanOperator, FieldFilter, PhraseFilter, SearchQuery};
use crate::search::{Bm25Params, PruningStrategy, SearchOptions, SearchOutcome};
use crate::shard::{ShardedIndex, ShardedIndexBuilder, persisted_sharded_format_version};
use crate::trec::{index_trec_collection, index_trec_collection_sharded, load_qrels, load_topics};

pub const HELP: &str = "\
IndexSail — compact local BM25 search\n\
\n\
USAGE:\n\
  indexsail --version\n\
  indexsail index --input COLLECTION --output INDEX.idx [--format tsv|trec] [--ascii]\n\
  indexsail shard-index --input COLLECTION --output SHARDS.idx --shards N [--format tsv|trec] [--ascii]\n\
  indexsail search --index INDEX.idx --query TEXT [OPTIONS]\n\
  indexsail shard-search --index SHARDS.idx --query TEXT [OPTIONS]\n\
  indexsail batch --index INDEX.idx --topics TOPICS --run RUN.txt [OPTIONS]\n\
  indexsail shard-batch --index SHARDS.idx --topics TOPICS --run RUN.txt [OPTIONS]\n\
  indexsail inspect --index INDEX.idx [--field NAME --term TERM]\n\
  indexsail shard-inspect --index SHARDS.idx [--field NAME --term TERM]\n\
  indexsail benchmark [--documents N] [--queries N] [--seed N] [--top-k N] [--shards N] [--json FILE]\n\
\n\
SEARCH OPTIONS:\n\
  --field NAME             Restrict all query terms to one field\n\
  --operator and|or        Boolean term semantics (default: or)\n\
  --phrase TEXT            Require an adjacent phrase\n\
  --phrase-field NAME      Restrict the phrase to one field\n\
  --filter NAME=VALUE      Require an exact stored field value; repeatable\n\
  --top-k N                Number of hits (default: 10)\n\
  --strategy wand|block-max-wand|maxscore|full\n\
                            Exact WAND, block-max WAND, MaxScore, or exhaustive evaluation\n\
  --k1 NUMBER --b NUMBER   BM25 parameters\n\
  --explain                Print per-term BM25 contributions\n\
\n\
BATCH OPTIONS:\n\
  --topics FILE            TSV or classic TREC <top> topics\n\
  --qrels FILE             Optional four-column TREC qrels\n\
  --run FILE               Required six-column TREC run output\n\
  --report FILE            Optional machine-readable JSON report\n\
  --tag NAME               Run tag (default: indexsail)\n\
  --verify                 Compare the selected strategy against exhaustive exactly\n\
  --field/--operator/...   Same ranking controls as search\n";

pub fn execute<I, S>(arguments: I, mut output: impl Write) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut arguments = arguments.into_iter().map(Into::into);
    let Some(command) = arguments.next() else {
        output.write_all(HELP.as_bytes())?;
        return Ok(());
    };
    let remaining = arguments.collect::<Vec<_>>();
    match command.as_str() {
        "help" | "--help" | "-h" => output.write_all(HELP.as_bytes()).map_err(Error::from),
        "version" | "--version" | "-V" => {
            writeln!(output, "indexsail {}", env!("CARGO_PKG_VERSION")).map_err(Error::from)
        }
        "index" => command_index(&remaining, &mut output),
        "shard-index" => command_shard_index(&remaining, &mut output),
        "search" => command_search(&remaining, &mut output),
        "shard-search" => command_shard_search(&remaining, &mut output),
        "batch" => command_batch(&remaining, &mut output),
        "shard-batch" => command_shard_batch(&remaining, &mut output),
        "inspect" => command_inspect(&remaining, &mut output),
        "shard-inspect" => command_shard_inspect(&remaining, &mut output),
        "benchmark" => command_benchmark(&remaining, &mut output),
        unknown => Err(Error::InvalidArgument(format!(
            "unknown command '{unknown}'; run 'indexsail help'"
        ))),
    }
}

fn command_index(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--ascii"],
        &["--input", "--output", "--format"],
    )?;
    let input = parsed.required_one("--input")?;
    let destination = parsed.required_one("--output")?;
    if paths_conflict(input, destination)? {
        return Err(Error::InvalidArgument(
            "index input and output paths must be distinct".into(),
        ));
    }
    let mode = if parsed.flag("--ascii") {
        AnalysisMode::Ascii
    } else {
        AnalysisMode::Unicode
    };
    let index = index_collection(
        input,
        Analyzer::new(mode),
        parsed.optional_one("--format")?.unwrap_or("tsv"),
    )?;
    index.save(destination)?;
    let stats = index.stats();
    writeln!(
        output,
        "indexed documents={} fields={} terms={} postings={} tokens={} output={}",
        stats.documents, stats.fields, stats.terms, stats.postings, stats.tokens, destination
    )?;
    Ok(())
}

fn command_shard_index(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--ascii"],
        &["--input", "--output", "--format", "--shards"],
    )?;
    let input = parsed.required_one("--input")?;
    let destination = parsed.required_one("--output")?;
    if paths_conflict(input, destination)? {
        return Err(Error::InvalidArgument(
            "index input and output paths must be distinct".into(),
        ));
    }
    let mode = if parsed.flag("--ascii") {
        AnalysisMode::Ascii
    } else {
        AnalysisMode::Unicode
    };
    let shard_count = parse_optional(parsed.optional_one("--shards")?, 0_usize, "shards")?;
    let index = index_sharded_collection(
        input,
        Analyzer::new(mode),
        parsed.optional_one("--format")?.unwrap_or("tsv"),
        shard_count,
    )?;
    index.save(destination)?;
    let stats = index.stats();
    writeln!(
        output,
        "indexed shards={} documents={} fields={} terms={} postings={} tokens={} output={}",
        index.shard_count(),
        stats.documents,
        stats.fields,
        stats.terms,
        stats.postings,
        stats.tokens,
        destination
    )?;
    Ok(())
}

fn command_search(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = parse_search_arguments(arguments)?;
    let index_path = parsed.required_one("--index")?;
    let index = InvertedIndex::load(index_path)?;
    let (query, options) = build_search_request(&parsed, index.analyzer())?;
    let pruning = options.pruning;
    let outcome = index.search(&query, options)?;
    write_search_output(&outcome, pruning, |doc_id| index.document(doc_id), output)
}

fn command_shard_search(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = parse_search_arguments(arguments)?;
    let index_path = parsed.required_one("--index")?;
    let index = ShardedIndex::load(index_path)?;
    let (query, options) = build_search_request(&parsed, index.analyzer())?;
    let pruning = options.pruning;
    let outcome = index.search(&query, options)?;
    write_search_output(&outcome, pruning, |doc_id| index.document(doc_id), output)?;
    writeln!(
        output,
        "collection=sharded shards={} global_documents={}",
        index.shard_count(),
        index.document_count()
    )?;
    Ok(())
}

fn parse_search_arguments(arguments: &[String]) -> Result<ParsedOptions> {
    ParsedOptions::parse(
        arguments,
        &["--explain"],
        &[
            "--index",
            "--query",
            "--field",
            "--operator",
            "--phrase",
            "--phrase-field",
            "--filter",
            "--top-k",
            "--strategy",
            "--k1",
            "--b",
        ],
    )
}

fn build_search_request(
    parsed: &ParsedOptions,
    analyzer: Analyzer,
) -> Result<(SearchQuery, SearchOptions)> {
    let query_text = parsed.required_one("--query")?;
    let field = parsed.optional_one("--field")?;
    let mut query = SearchQuery::from_text(analyzer, query_text, field)?;
    query = query.with_operator(match parsed.optional_one("--operator")?.unwrap_or("or") {
        "and" => BooleanOperator::And,
        "or" => BooleanOperator::Or,
        value => {
            return Err(Error::InvalidArgument(format!(
                "unknown operator '{value}', expected and or or"
            )));
        }
    });

    if let Some(phrase) = parsed.optional_one("--phrase")? {
        let phrase_field = parsed.optional_one("--phrase-field")?.map(str::to_owned);
        query = query.with_phrase(PhraseFilter::from_text(analyzer, phrase, phrase_field)?);
    } else if parsed.optional_one("--phrase-field")?.is_some() {
        return Err(Error::InvalidArgument(
            "--phrase-field requires --phrase".into(),
        ));
    }
    for filter in parsed.many("--filter") {
        let (name, value) = filter.split_once('=').ok_or_else(|| {
            Error::InvalidArgument(format!("filter '{filter}' must use NAME=VALUE syntax"))
        })?;
        query = query.with_filter(FieldFilter::exact(name, value)?);
    }

    let top_k = parse_optional(parsed.optional_one("--top-k")?, 10_usize, "top-k")?;
    let k1 = parse_optional(parsed.optional_one("--k1")?, 1.2_f64, "k1")?;
    let b = parse_optional(parsed.optional_one("--b")?, 0.75_f64, "b")?;
    let pruning = match parsed.optional_one("--strategy")?.unwrap_or("wand") {
        "wand" => PruningStrategy::Wand,
        "block-max-wand" | "bmw" => PruningStrategy::BlockMaxWand,
        "maxscore" | "max-score" => PruningStrategy::MaxScore,
        "full" | "exhaustive" => PruningStrategy::Exhaustive,
        value => {
            return Err(Error::InvalidArgument(format!(
                "unknown strategy '{value}', expected wand, block-max-wand, maxscore, or full"
            )));
        }
    };
    Ok((
        query,
        SearchOptions {
            top_k,
            pruning,
            explain: parsed.flag("--explain"),
            bm25: Bm25Params { k1, b },
        },
    ))
}

fn write_search_output<'a>(
    outcome: &SearchOutcome,
    pruning: PruningStrategy,
    document: impl Fn(InternalDocId) -> Option<&'a Document>,
    output: &mut impl Write,
) -> Result<()> {
    writeln!(output, "rank\tscore\tid\ttitle")?;
    for (rank, hit) in outcome.hits.iter().enumerate() {
        let document = document(hit.doc_id).expect("hit document exists");
        let title = document
            .field("title")
            .unwrap_or("")
            .replace(['\t', '\n', '\r'], " ");
        writeln!(
            output,
            "{}\t{:.6}\t{}\t{}",
            rank + 1,
            hit.score,
            hit.external_id,
            title
        )?;
        if let Some(explanation) = &hit.explanation {
            for term in &explanation.terms {
                writeln!(
                    output,
                    "  term={} field={} tf={} df={} len={} avg_len={:.3} idf={:.6} boost={:.3} score={:.6}",
                    term.term,
                    term.field,
                    term.term_frequency,
                    term.document_frequency,
                    term.document_length,
                    term.average_document_length,
                    term.inverse_document_frequency,
                    term.boost,
                    term.score
                )?;
            }
        }
    }
    writeln!(
        output,
        "strategy={pruning:?} evaluated={} advanced={} skipped={} block_max_bounds_loaded={} block_max_postings_covered={} block_max_postings_scanned={}",
        outcome.stats.evaluated_candidates,
        outcome.stats.postings_advanced,
        outcome.stats.postings_skipped,
        outcome.stats.block_max_bounds_loaded,
        outcome.stats.block_max_postings_covered,
        outcome.stats.block_max_postings_scanned
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn command_batch(arguments: &[String], output: &mut impl Write) -> Result<()> {
    command_batch_impl(arguments, output, false)
}

#[allow(clippy::too_many_lines)]
fn command_shard_batch(arguments: &[String], output: &mut impl Write) -> Result<()> {
    command_batch_impl(arguments, output, true)
}

#[allow(clippy::too_many_lines)]
fn command_batch_impl(arguments: &[String], output: &mut impl Write, sharded: bool) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--verify"],
        &[
            "--index",
            "--topics",
            "--qrels",
            "--run",
            "--report",
            "--tag",
            "--field",
            "--operator",
            "--top-k",
            "--strategy",
            "--k1",
            "--b",
        ],
    )?;
    let index_path = parsed.required_one("--index")?;
    let topics_path = parsed.required_one("--topics")?;
    let run_path = parsed.required_one("--run")?;
    let report_path = parsed.optional_one("--report")?;
    let mut paths = vec![(index_path, false), (topics_path, false), (run_path, true)];
    if let Some(path) = parsed.optional_one("--qrels")? {
        paths.push((path, false));
    }
    if let Some(path) = report_path {
        paths.push((path, true));
    }
    let mut conflict = false;
    for left in 0..paths.len() {
        for right in left + 1..paths.len() {
            if (paths[left].1 || paths[right].1) && paths_conflict(paths[left].0, paths[right].0)? {
                conflict = true;
            }
        }
    }
    if conflict {
        return Err(Error::InvalidArgument(
            "batch input, run, and report paths must be distinct".into(),
        ));
    }

    let topics = load_topics(topics_path)?;
    let qrels = parsed
        .optional_one("--qrels")?
        .map(load_qrels)
        .transpose()?;
    let operator = match parsed.optional_one("--operator")?.unwrap_or("or") {
        "and" => BooleanOperator::And,
        "or" => BooleanOperator::Or,
        value => {
            return Err(Error::InvalidArgument(format!(
                "unknown operator '{value}', expected and or or"
            )));
        }
    };
    let pruning = match parsed.optional_one("--strategy")?.unwrap_or("wand") {
        "wand" => PruningStrategy::Wand,
        "block-max-wand" | "bmw" => PruningStrategy::BlockMaxWand,
        "maxscore" | "max-score" => PruningStrategy::MaxScore,
        "full" | "exhaustive" => PruningStrategy::Exhaustive,
        value => {
            return Err(Error::InvalidArgument(format!(
                "unknown strategy '{value}', expected wand, block-max-wand, maxscore, or full"
            )));
        }
    };
    let config = BatchConfig {
        top_k: parse_optional(parsed.optional_one("--top-k")?, 1_000_usize, "top-k")?,
        pruning,
        verify_exact: parsed.flag("--verify"),
        field: parsed.optional_one("--field")?.map(str::to_owned),
        operator,
        bm25: Bm25Params {
            k1: parse_optional(parsed.optional_one("--k1")?, 1.2_f64, "k1")?,
            b: parse_optional(parsed.optional_one("--b")?, 0.75_f64, "b")?,
        },
    };
    let (report, shard_count) = if sharded {
        let index = ShardedIndex::load(index_path)?;
        let shard_count = index.shard_count();
        (
            evaluate_batch(&index, &topics, qrels.as_ref(), config)?,
            Some(shard_count),
        )
    } else {
        let index = InvertedIndex::load(index_path)?;
        (
            evaluate_batch(&index, &topics, qrels.as_ref(), config)?,
            None,
        )
    };

    let run_file = File::create(run_path)?;
    let mut run_writer = BufWriter::new(run_file);
    write_trec_run(
        &report,
        parsed.optional_one("--tag")?.unwrap_or("indexsail"),
        &mut run_writer,
    )?;
    run_writer.flush()?;

    if let Some(path) = report_path {
        let report_file = File::create(path)?;
        let mut report_writer = BufWriter::new(report_file);
        write_json_report(&report, &mut report_writer)?;
        report_writer.flush()?;
    }

    let hit_count = report
        .queries
        .iter()
        .map(|query| query.hits.len())
        .sum::<usize>();
    writeln!(
        output,
        "batch topics={} hits={} strategy={:?} verified={} elapsed_ms={:.3} evaluated={} advanced={} skipped={} block_max_bounds_loaded={} block_max_postings_covered={} block_max_postings_scanned={} run={}",
        report.queries.len(),
        hit_count,
        report.config.pruning,
        report.config.verify_exact,
        report.total_search_time.as_secs_f64() * 1_000.0,
        report.total_stats.evaluated_candidates,
        report.total_stats.postings_advanced,
        report.total_stats.postings_skipped,
        report.total_stats.block_max_bounds_loaded,
        report.total_stats.block_max_postings_covered,
        report.total_stats.block_max_postings_scanned,
        run_path
    )?;
    if let Some(metrics) = report.aggregate {
        writeln!(
            output,
            "metrics map={:.6} mrr={:.6} ndcg={:.6} recall={:.6}",
            metrics.map, metrics.mrr, metrics.mean_ndcg, metrics.mean_recall
        )?;
    }
    if let Some(path) = report_path {
        writeln!(output, "report={path}")?;
    }
    if let Some(shard_count) = shard_count {
        writeln!(output, "collection=sharded shards={shard_count}")?;
    }
    Ok(())
}

fn command_inspect(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--index", "--field", "--term"])?;
    let path = parsed.required_one("--index")?;
    let index = InvertedIndex::load(path)?;
    let format_version = persisted_format_version(path)?;
    let stats = index.stats();
    let codec = index.posting_codec_stats()?;
    let file_bytes = std::fs::metadata(path)?.len();
    let block_max_bytes = block_max_metadata_encoded_bytes(&index)?;
    let block_max_blocks = index
        .block_max
        .values()
        .map(|metadata| metadata.default_bounds.len())
        .sum::<usize>();
    writeln!(
        output,
        "format=IndexSail-v{} analyzer={:?} file_bytes={}",
        format_version,
        index.analyzer().mode(),
        file_bytes
    )?;
    writeln!(
        output,
        "documents={} fields={} terms={} postings={} tokens={}",
        stats.documents, stats.fields, stats.terms, stats.postings, stats.tokens
    )?;
    writeln!(
        output,
        "posting_codec=delta-varbyte positions={} raw_bytes={} encoded_bytes={} ratio={:.4}",
        codec.positions,
        codec.uncompressed_bytes,
        codec.encoded_bytes,
        codec.ratio()
    )?;
    writeln!(
        output,
        "block_max_metadata streams={} blocks={} v3_serialized_bytes={}",
        index.block_max.len(),
        block_max_blocks,
        block_max_bytes
    )?;
    for field in index.fields() {
        writeln!(
            output,
            "field={} avg_length={:.3}",
            field,
            index.average_field_length(field)
        )?;
    }

    match (
        parsed.optional_one("--field")?,
        parsed.optional_one("--term")?,
    ) {
        (None, None) => {}
        (Some(_), None) => {
            return Err(Error::InvalidArgument("--field requires --term".into()));
        }
        (field, Some(raw_term)) => {
            let normalized = index.analyzer().normalize_single(raw_term).ok_or_else(|| {
                Error::InvalidArgument("--term must analyze to exactly one token".into())
            })?;
            let fields: Vec<&str> = match field {
                Some(field) => vec![field],
                None => index.fields().into_iter().collect(),
            };
            for field in fields {
                let postings = index.postings(field, &normalized).unwrap_or(&[]);
                writeln!(
                    output,
                    "term={} field={} df={}",
                    normalized,
                    field,
                    postings.len()
                )?;
                for posting in postings {
                    let external_id = index
                        .document(posting.doc_id)
                        .expect("posting document exists")
                        .external_id();
                    writeln!(
                        output,
                        "  doc={} internal={} tf={} positions={:?}",
                        external_id, posting.doc_id, posting.term_frequency, posting.positions
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn command_shard_inspect(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--index", "--field", "--term"])?;
    let path = parsed.required_one("--index")?;
    let index = ShardedIndex::load(path)?;
    let format_version = persisted_sharded_format_version(path)?;
    let stats = index.stats();
    let embedded_versions = index
        .embedded_format_versions()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let embedded_format = if embedded_versions.len() == 1 {
        format!(
            "IndexSail-v{}",
            embedded_versions.iter().next().expect("one version")
        )
    } else {
        format!(
            "mixed({})",
            embedded_versions
                .iter()
                .map(|version| format!("IndexSail-v{version}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    writeln!(
        output,
        "format=IndexSail-sharded-v{} embedded_format={} analyzer={:?} file_bytes={} shards={}",
        format_version,
        embedded_format,
        index.analyzer().mode(),
        std::fs::metadata(path)?.len(),
        index.shard_count()
    )?;
    writeln!(
        output,
        "documents={} fields={} terms={} postings={} tokens={}",
        stats.documents, stats.fields, stats.terms, stats.postings, stats.tokens
    )?;
    for shard in index.physical_shard_stats() {
        writeln!(
            output,
            "shard={} documents={} fields={} terms={} postings={} tokens={}",
            shard.shard_id,
            shard.documents,
            shard.fields,
            shard.terms,
            shard.postings,
            shard.tokens
        )?;
    }
    for field in index.fields() {
        writeln!(
            output,
            "field={} global_avg_length={:.3}",
            field,
            index.average_field_length(field)
        )?;
    }

    match (
        parsed.optional_one("--field")?,
        parsed.optional_one("--term")?,
    ) {
        (None, None) => {}
        (Some(_), None) => {
            return Err(Error::InvalidArgument("--field requires --term".into()));
        }
        (field, Some(raw_term)) => {
            let normalized = index.analyzer().normalize_single(raw_term).ok_or_else(|| {
                Error::InvalidArgument("--term must analyze to exactly one token".into())
            })?;
            let fields = match field {
                Some(field) => vec![field],
                None => index.fields().into_iter().collect(),
            };
            for field in fields {
                writeln!(
                    output,
                    "term={} field={} global_df={}",
                    normalized,
                    field,
                    index.document_frequency(field, &normalized)
                )?;
                for shard_id in 0..index.shard_count() {
                    let frequency = index
                        .shard(shard_id)
                        .expect("shard id is in range")
                        .document_frequency(field, &normalized);
                    writeln!(output, "  shard={shard_id} df={frequency}")?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn command_benchmark(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &[],
        &[
            "--documents",
            "--queries",
            "--seed",
            "--top-k",
            "--shards",
            "--json",
        ],
    )?;
    let config = BenchmarkConfig {
        documents: parse_optional(
            parsed.optional_one("--documents")?,
            BenchmarkConfig::default().documents,
            "documents",
        )?,
        queries: parse_optional(
            parsed.optional_one("--queries")?,
            BenchmarkConfig::default().queries,
            "queries",
        )?,
        seed: parse_optional(
            parsed.optional_one("--seed")?,
            BenchmarkConfig::default().seed,
            "seed",
        )?,
        top_k: parse_optional(
            parsed.optional_one("--top-k")?,
            BenchmarkConfig::default().top_k,
            "top-k",
        )?,
        shards: parse_optional(
            parsed.optional_one("--shards")?,
            BenchmarkConfig::default().shards,
            "shards",
        )?,
    };
    let report = run_benchmark(config)?;
    if let Some(path) = parsed.optional_one("--json")? {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_benchmark_json(&report, &mut writer)?;
        writer.flush()?;
    }
    writeln!(
        output,
        "config documents={} queries={} top_k={} shards={} seed={}",
        config.documents, config.queries, config.top_k, config.shards, config.seed
    )?;
    writeln!(
        output,
        "index terms={} postings={} elapsed_ms={:.3}",
        report.index_terms,
        report.index_postings,
        report.index_time.as_secs_f64() * 1000.0
    )?;
    writeln!(
        output,
        "exhaustive elapsed_ms={:.3} evaluated={}",
        report.exhaustive_time.as_secs_f64() * 1000.0,
        report.exhaustive_stats.evaluated_candidates
    )?;
    writeln!(
        output,
        "wand elapsed_ms={:.3} evaluated={} advanced={} skipped={}",
        report.wand_time.as_secs_f64() * 1000.0,
        report.wand_stats.evaluated_candidates,
        report.wand_stats.postings_advanced,
        report.wand_stats.postings_skipped
    )?;
    writeln!(
        output,
        "block-max-wand elapsed_ms={:.3} evaluated={} advanced={} skipped={} precomputed_bounds_loaded={} postings_covered_by_bounds={} postings_scanned_for_bounds={}",
        report.block_max_time.as_secs_f64() * 1000.0,
        report.block_max_stats.evaluated_candidates,
        report.block_max_stats.postings_advanced,
        report.block_max_stats.postings_skipped,
        report.block_max_stats.block_max_bounds_loaded,
        report.block_max_stats.block_max_postings_covered,
        report.block_max_stats.block_max_postings_scanned
    )?;
    writeln!(
        output,
        "maxscore elapsed_ms={:.3} evaluated={} advanced={} skipped={}",
        report.maxscore_time.as_secs_f64() * 1000.0,
        report.maxscore_stats.evaluated_candidates,
        report.maxscore_stats.postings_advanced,
        report.maxscore_stats.postings_skipped
    )?;
    writeln!(
        output,
        "sharded-block-max-wand shards={} build_elapsed_ms={:.3} search_elapsed_ms={:.3} evaluated={} advanced={} skipped={} postings_scanned_for_bounds={} serialized_bytes={}",
        report.config.shards,
        report.sharded_build_time.as_secs_f64() * 1000.0,
        report.sharded_time.as_secs_f64() * 1000.0,
        report.sharded_stats.evaluated_candidates,
        report.sharded_stats.postings_advanced,
        report.sharded_stats.postings_skipped,
        report.sharded_stats.block_max_postings_scanned,
        report.sharded_index_bytes
    )?;
    writeln!(output, "verified=true checksum={:016x}", report.checksum)?;
    writeln!(
        output,
        "posting_codec raw_bytes={} encoded_bytes={} ratio={:.4}",
        report.posting_codec.uncompressed_bytes,
        report.posting_codec.encoded_bytes,
        report.posting_codec.ratio()
    )?;
    writeln!(
        output,
        "persistence format=3 serialized_bytes={} base_index_bytes={} block_max_metadata_bytes={} streams={} blocks={}",
        report.persisted_index_bytes,
        report
            .persisted_index_bytes
            .saturating_sub(report.block_max_metadata_bytes),
        report.block_max_metadata_bytes,
        report.block_max_streams,
        report.block_max_blocks
    )?;
    if let Some(path) = parsed.optional_one("--json")? {
        writeln!(output, "report={path}")?;
    }
    Ok(())
}

fn index_collection(
    path: impl AsRef<Path>,
    analyzer: Analyzer,
    format: &str,
) -> Result<InvertedIndex> {
    match format {
        "tsv" => index_tsv(path, analyzer),
        "trec" => index_trec_collection(path, analyzer),
        value => Err(Error::InvalidArgument(format!(
            "unknown collection format '{value}', expected tsv or trec"
        ))),
    }
}

fn index_sharded_collection(
    path: impl AsRef<Path>,
    analyzer: Analyzer,
    format: &str,
    shard_count: usize,
) -> Result<ShardedIndex> {
    match format {
        "tsv" => index_tsv_sharded(path, analyzer, shard_count),
        "trec" => index_trec_collection_sharded(path, analyzer, shard_count),
        value => Err(Error::InvalidArgument(format!(
            "unknown collection format '{value}', expected tsv or trec"
        ))),
    }
}

fn index_tsv(path: impl AsRef<Path>, analyzer: Analyzer) -> Result<InvertedIndex> {
    let mut builder = IndexBuilder::new(analyzer);
    visit_tsv(path, |document| builder.add_document(document).map(|_| ()))?;
    Ok(builder.finish())
}

fn index_tsv_sharded(
    path: impl AsRef<Path>,
    analyzer: Analyzer,
    shard_count: usize,
) -> Result<ShardedIndex> {
    let mut builder = ShardedIndexBuilder::new(analyzer, shard_count)?;
    visit_tsv(path, |document| builder.add_document(document).map(|_| ()))?;
    Ok(builder.finish())
}

fn visit_tsv(
    path: impl AsRef<Path>,
    mut add_document: impl FnMut(Document) -> Result<()>,
) -> Result<()> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .transpose()?
        .ok_or_else(|| Error::InvalidDocument("TSV input has no header".into()))?;
    let columns = header
        .trim_end_matches('\r')
        .split('\t')
        .collect::<Vec<_>>();
    if columns.len() < 2 || columns[0] != "id" {
        return Err(Error::InvalidDocument(
            "TSV header must start with id and contain at least one field".into(),
        ));
    }
    let mut seen_fields = BTreeSet::new();
    for field in &columns[1..] {
        if !seen_fields.insert(*field) {
            return Err(Error::InvalidDocument(format!(
                "duplicate TSV field '{field}'"
            )));
        }
        // Validate field names before processing an otherwise-empty corpus.
        Document::from_fields("header-validation", [(*field, "")])?;
    }
    for (line_index, line) in lines.enumerate() {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let values = line.split('\t').collect::<Vec<_>>();
        if values.len() != columns.len() {
            return Err(Error::InvalidDocument(format!(
                "TSV line {} has {} columns; expected {}",
                line_index + 2,
                values.len(),
                columns.len()
            )));
        }
        let fields = columns[1..]
            .iter()
            .zip(&values[1..])
            .map(|(name, value)| (*name, *value));
        add_document(Document::from_fields(values[0], fields)?)?;
    }
    Ok(())
}

fn parse_optional<T>(value: Option<&str>, default: T, label: &str) -> Result<T>
where
    T: std::str::FromStr,
{
    value.map_or(Ok(default), |value| {
        value.parse().map_err(|_| {
            Error::InvalidArgument(format!("{label} value '{value}' has the wrong type"))
        })
    })
}

fn paths_conflict(left: impl AsRef<Path>, right: impl AsRef<Path>) -> Result<bool> {
    let left = normalized_path(left.as_ref())?;
    let right = normalized_path(right.as_ref())?;
    #[cfg(windows)]
    {
        Ok(left
            .to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy()))
    }
    #[cfg(not(windows))]
    {
        Ok(left == right)
    }
}

fn normalized_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return std::fs::canonicalize(path).map_err(Error::from);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = std::fs::canonicalize(parent)?;
    let file_name = path.file_name().ok_or_else(|| {
        Error::InvalidArgument(format!("path '{}' has no file name", path.display()))
    })?;
    Ok(parent.join(file_name))
}

#[derive(Debug, Default)]
struct ParsedOptions {
    flags: BTreeSet<String>,
    values: BTreeMap<String, Vec<String>>,
}

impl ParsedOptions {
    fn parse(arguments: &[String], flags: &[&str], values: &[&str]) -> Result<Self> {
        let valid_flags = flags.iter().copied().collect::<BTreeSet<_>>();
        let valid_values = values.iter().copied().collect::<BTreeSet<_>>();
        let mut parsed = Self::default();
        let mut index = 0;
        while index < arguments.len() {
            let option = arguments[index].as_str();
            if valid_flags.contains(option) {
                if !parsed.flags.insert(option.to_owned()) {
                    return Err(Error::InvalidArgument(format!(
                        "option '{option}' was provided more than once"
                    )));
                }
                index += 1;
            } else if valid_values.contains(option) {
                let value = arguments.get(index + 1).ok_or_else(|| {
                    Error::InvalidArgument(format!("option '{option}' requires a value"))
                })?;
                parsed
                    .values
                    .entry(option.to_owned())
                    .or_default()
                    .push(value.clone());
                index += 2;
            } else {
                return Err(Error::InvalidArgument(format!(
                    "unknown option or positional argument '{option}'"
                )));
            }
        }
        Ok(parsed)
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    fn many(&self, name: &str) -> impl Iterator<Item = &str> {
        self.values
            .get(name)
            .into_iter()
            .flatten()
            .map(String::as_str)
    }

    fn optional_one(&self, name: &str) -> Result<Option<&str>> {
        let values = self.values.get(name);
        if values.is_some_and(|values| values.len() > 1) && name != "--filter" {
            return Err(Error::InvalidArgument(format!(
                "option '{name}' was provided more than once"
            )));
        }
        Ok(values.and_then(|values| values.first()).map(String::as_str))
    }

    fn required_one(&self, name: &str) -> Result<&str> {
        self.optional_one(name)?
            .ok_or_else(|| Error::InvalidArgument(format!("required option '{name}' is missing")))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn temp_path(extension: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let number = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "indexsail-cli-{}-{number}.{extension}",
            std::process::id()
        ))
    }

    #[test]
    fn no_command_prints_help() {
        let mut output = Vec::new();
        execute(Vec::<String>::new(), &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("indexsail search"));
        assert!(output.contains("wand|block-max-wand|maxscore|full"));
    }

    #[test]
    fn version_command_uses_package_version() {
        let mut output = Vec::new();
        execute(["--version"], &mut output).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            format!("indexsail {}\n", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn unknown_command_is_rejected() {
        assert!(execute(["unknown"], Vec::new()).is_err());
    }

    #[test]
    fn index_search_and_inspect_form_an_end_to_end_flow() {
        let corpus = temp_path("tsv");
        let index_path = temp_path("idx");
        std::fs::write(
            &corpus,
            "id\ttitle\tbody\tcategory\n1\tRust Search\tfast local engine\tguide\n2\tGrid\tpower model\treference\n",
        )
        .unwrap();

        let mut index_output = Vec::new();
        execute(
            [
                "index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                index_path.to_str().unwrap(),
            ],
            &mut index_output,
        )
        .unwrap();
        assert!(
            String::from_utf8(index_output)
                .unwrap()
                .contains("documents=2")
        );

        let mut search_output = Vec::new();
        execute(
            [
                "search",
                "--index",
                index_path.to_str().unwrap(),
                "--query",
                "local",
                "--explain",
            ],
            &mut search_output,
        )
        .unwrap();
        let search_output = String::from_utf8(search_output).unwrap();
        assert!(search_output.contains("1\t"));
        assert!(search_output.contains("Rust Search"));
        assert!(search_output.contains("term=local"));
        assert!(search_output.contains("block_max_bounds_loaded=0"));
        assert!(search_output.contains("block_max_postings_covered=0"));
        assert!(search_output.contains("block_max_postings_scanned=0"));

        let mut inspect_output = Vec::new();
        execute(
            [
                "inspect",
                "--index",
                index_path.to_str().unwrap(),
                "--field",
                "body",
                "--term",
                "local",
            ],
            &mut inspect_output,
        )
        .unwrap();
        assert!(String::from_utf8(inspect_output).unwrap().contains("df=1"));

        std::fs::remove_file(corpus).unwrap();
        std::fs::remove_file(index_path).unwrap();
    }

    #[test]
    fn malformed_tsv_reports_line_number() {
        let corpus = temp_path("tsv");
        let index_path = temp_path("idx");
        std::fs::write(&corpus, "id\ttitle\tbody\n1\tonly-two\n").unwrap();
        let error = execute(
            [
                "index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                index_path.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("line 2"));
        std::fs::remove_file(corpus).unwrap();
    }

    #[test]
    fn duplicate_non_repeatable_option_is_rejected() {
        let parsed = ParsedOptions::parse(
            &["--index".into(), "a".into(), "--index".into(), "b".into()],
            &[],
            &["--index"],
        )
        .unwrap();
        assert!(parsed.optional_one("--index").is_err());
    }

    #[test]
    fn repeated_filters_are_allowed() {
        let parsed = ParsedOptions::parse(
            &[
                "--filter".into(),
                "a=1".into(),
                "--filter".into(),
                "b=2".into(),
            ],
            &[],
            &["--filter"],
        )
        .unwrap();
        assert_eq!(parsed.many("--filter").collect::<Vec<_>>(), ["a=1", "b=2"]);
    }

    #[test]
    fn trec_collection_and_batch_evaluation_form_an_end_to_end_flow() {
        let collection = temp_path("trec");
        let index_path = temp_path("idx");
        let topics = temp_path("topics");
        let qrels = temp_path("qrels");
        let run = temp_path("run");
        let report = temp_path("json");
        std::fs::write(
            &collection,
            "<DOC>\n<DOCNO>D1</DOCNO>\n<TITLE>Rust</TITLE>\n<TEXT>local search ranking</TEXT>\n</DOC>\n<DOC>\n<DOCNO>D2</DOCNO>\n<TEXT>power grid model</TEXT>\n</DOC>\n",
        )
        .unwrap();
        std::fs::write(&topics, "1\tlocal search\n2\tgrid\n").unwrap();
        std::fs::write(&qrels, "1 0 D1 2\n2 0 D2 1\n").unwrap();

        execute(
            [
                "index",
                "--input",
                collection.to_str().unwrap(),
                "--output",
                index_path.to_str().unwrap(),
                "--format",
                "trec",
            ],
            Vec::new(),
        )
        .unwrap();
        let mut output = Vec::new();
        execute(
            [
                "batch",
                "--index",
                index_path.to_str().unwrap(),
                "--topics",
                topics.to_str().unwrap(),
                "--qrels",
                qrels.to_str().unwrap(),
                "--run",
                run.to_str().unwrap(),
                "--report",
                report.to_str().unwrap(),
                "--field",
                "body",
                "--top-k",
                "10",
                "--verify",
            ],
            &mut output,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("topics=2"));
        assert!(output.contains("verified=true"));
        assert!(output.contains("map=1.000000"));
        assert!(output.contains("block_max_bounds_loaded=0"));
        assert!(output.contains("block_max_postings_covered=0"));
        assert!(output.contains("block_max_postings_scanned=0"));
        assert!(std::fs::read_to_string(&run).unwrap().contains("1 Q0 D1 1"));
        let report_text = std::fs::read_to_string(&report).unwrap();
        assert!(report_text.contains("\"mean_ndcg\": 1.000000000000"));

        for path in [collection, index_path, topics, qrels, run, report] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn benchmark_can_write_machine_readable_report() {
        let report = temp_path("json");
        execute(
            [
                "benchmark",
                "--documents",
                "80",
                "--queries",
                "5",
                "--top-k",
                "3",
                "--seed",
                "9",
                "--json",
                report.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        let text = std::fs::read_to_string(&report).unwrap();
        assert!(text.contains("\"verified_exact\": true"));
        assert!(text.contains("\"checksum\""));
        std::fs::remove_file(report).unwrap();
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn sharded_cli_supports_index_search_inspect_and_batch() {
        let corpus = temp_path("shard.tsv");
        let index_path = temp_path("shards.idx");
        let topics = temp_path("shard.topics");
        let qrels = temp_path("shard.qrels");
        let run = temp_path("shard.run");
        let report = temp_path("shard.json");
        std::fs::write(
            &corpus,
            "id\ttitle\tbody\tcategory\nD1\tRust Search\tfast local search\tguide\nD2\tGrid\tpower model\treference\nD3\tSailing\tlocal retrieval engine\tguide\nD4\tOther\tsearch system\tnote\n",
        )
        .unwrap();
        std::fs::write(&topics, "1\tlocal search\n2\tpower model\n").unwrap();
        std::fs::write(&qrels, "1 0 D1 2\n1 0 D3 1\n2 0 D2 2\n").unwrap();

        let mut index_output = Vec::new();
        execute(
            [
                "shard-index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                index_path.to_str().unwrap(),
                "--shards",
                "3",
            ],
            &mut index_output,
        )
        .unwrap();
        let index_output = String::from_utf8(index_output).unwrap();
        assert!(index_output.contains("shards=3 documents=4"));

        let mut inspect_output = Vec::new();
        execute(
            [
                "shard-inspect",
                "--index",
                index_path.to_str().unwrap(),
                "--field",
                "body",
                "--term",
                "local",
            ],
            &mut inspect_output,
        )
        .unwrap();
        let inspect_output = String::from_utf8(inspect_output).unwrap();
        assert!(inspect_output.contains("format=IndexSail-sharded-v1"));
        assert!(inspect_output.contains("embedded_format=IndexSail-v3"));
        assert!(inspect_output.contains("term=local field=body global_df=2"));
        assert!(inspect_output.contains("shard=2 documents=1"));

        let mut search_output = Vec::new();
        execute(
            [
                "shard-search",
                "--index",
                index_path.to_str().unwrap(),
                "--query",
                "local search",
                "--field",
                "body",
                "--strategy",
                "block-max-wand",
                "--explain",
            ],
            &mut search_output,
        )
        .unwrap();
        let search_output = String::from_utf8(search_output).unwrap();
        assert_eq!(search_output.lines().next(), Some("rank\tscore\tid\ttitle"));
        assert!(search_output.contains("collection=sharded shards=3 global_documents=4"));
        assert!(search_output.contains("D1\tRust Search"));
        assert!(search_output.contains("df=2"));
        assert!(search_output.contains("block_max_bounds_loaded=0"));
        assert!(search_output.contains("block_max_postings_scanned="));

        let mut batch_output = Vec::new();
        execute(
            [
                "shard-batch",
                "--index",
                index_path.to_str().unwrap(),
                "--topics",
                topics.to_str().unwrap(),
                "--qrels",
                qrels.to_str().unwrap(),
                "--run",
                run.to_str().unwrap(),
                "--report",
                report.to_str().unwrap(),
                "--field",
                "body",
                "--strategy",
                "maxscore",
                "--verify",
            ],
            &mut batch_output,
        )
        .unwrap();
        let batch_output = String::from_utf8(batch_output).unwrap();
        assert!(batch_output.contains("verified=true"));
        assert!(batch_output.contains("collection=sharded shards=3"));
        assert!(std::fs::read_to_string(&run).unwrap().contains("1 Q0 D1 1"));
        assert!(
            std::fs::read_to_string(&report)
                .unwrap()
                .contains("\"verified_exact\": true")
        );

        for path in [corpus, index_path, topics, qrels, run, report] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn shard_index_requires_an_explicit_valid_shard_count() {
        let corpus = temp_path("invalid-shard.tsv");
        let index_path = temp_path("invalid-shards.idx");
        std::fs::write(&corpus, "id\tbody\n1\tone\n").unwrap();
        let error = execute(
            [
                "shard-index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                index_path.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("shard count"));
        std::fs::remove_file(corpus).unwrap();
    }

    #[test]
    fn destructive_path_aliases_are_rejected() {
        assert!(
            execute(
                ["index", "--input", "same", "--output", "./same"],
                Vec::new()
            )
            .is_err()
        );
    }
}
