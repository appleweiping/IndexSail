use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::analysis::{AnalysisMode, Analyzer};
use crate::atomic::{
    atomic_write, atomic_write_many, atomic_write_many_with, atomic_write_with,
    prevalidate_output_path,
};
use crate::benchmark::{BenchmarkConfig, run as run_benchmark, write_json as write_benchmark_json};
use crate::ciff::{CiffIndex, CiffSearchOptions};
use crate::collection::{
    CollectionFormat, CollectionLimits, index_collection, index_collection_sharded,
};
use crate::document::Document;
use crate::error::{Error, Result};
use crate::evaluation::{BatchConfig, evaluate_batch, write_json_report, write_trec_run};
use crate::forward::ForwardIndex;
use crate::index::{InternalDocId, InvertedIndex};
use crate::persistence::{block_max_metadata_encoded_bytes, persisted_format_version};
use crate::query::{BooleanOperator, FieldFilter, PhraseFilter, SearchQuery};
use crate::reorder::{BisectionOptions, DocIdMap, MAX_REORDER_FORWARD_BYTES};
use crate::search::{Bm25Params, PruningStrategy, SearchOptions, SearchOutcome};
use crate::shard::{ShardedIndex, persisted_sharded_format_version};
use crate::trec::{load_qrels, load_topics};

pub const HELP: &str = "\
IndexSail — compact local BM25 search\n\
\n\
USAGE:\n\
  indexsail --version\n\
  indexsail index --input COLLECTION --output INDEX.idx [--format tsv|trec|jsonl] [--ascii]\n\
  indexsail shard-index --input COLLECTION --output SHARDS.idx --shards N [--format tsv|trec|jsonl] [--ascii]\n\
  indexsail forward-build --input COLLECTION --output INDEX.fwd [--format tsv|trec|jsonl] [--ascii]\n\
  indexsail forward-invert --input INDEX.fwd --output INDEX.idx\n\
  indexsail reorder --input INDEX.fwd --forward-output NEW.fwd --index-output NEW.idx\n\
    --old-to-new OLD.map --new-to-old NEW.map (--random [--seed N] | --by-feature FILE | --from-mapping FILE | --bp [--depth N] [--iterations N])\n\
  indexsail forward-inspect --index INDEX.fwd [--document N] [--limit N]\n\
  indexsail lexicon --index INDEX.fwd [--id N | --field NAME --term TERM | --offset N --limit N]\n\
  indexsail search --index INDEX.idx --query TEXT [OPTIONS]\n\
  indexsail shard-search --index SHARDS.idx --query TEXT [OPTIONS]\n\
  indexsail batch --index INDEX.idx --topics TOPICS --run RUN.txt [OPTIONS]\n\
  indexsail shard-batch --index SHARDS.idx --topics TOPICS --run RUN.txt [OPTIONS]\n\
  indexsail inspect --index INDEX.idx [--field NAME --term TERM]\n\
  indexsail shard-inspect --index SHARDS.idx [--field NAME --term TERM]\n\
  indexsail ciff-export --index INDEX.idx --output INDEX.ciff [--description TEXT]\n\
  indexsail ciff-search --index INDEX.ciff --query TEXT [--ascii] [OPTIONS]\n\
  indexsail ciff-batch --index INDEX.ciff --topics TOPICS --run RUN.txt [OPTIONS]\n\
  indexsail ciff-inspect --index INDEX.ciff [--term TERM]\n\
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
  --field/--operator/...   Same ranking controls as search\n\
\n\
REORDER OPTIONS:\n\
  --random [--seed N]      Portable seeded shuffle (default seed: 0)\n\
  --by-feature FILE        One UTF-8 feature line per original document ID\n\
  --from-mapping FILE      Two columns: original ID, new ID\n\
  --bp [--depth N] [--iterations N]\n\
                            Deterministic recursive graph bisection\n\
  --old-to-new FILE        Write original-to-new two-column map\n\
  --new-to-old FILE        Write new-to-original two-column map\n\
\n\
CIFF OPTIONS:\n\
  --ascii                  Use ASCII-compatible query analysis instead of Unicode\n\
  --operator and|or        Boolean term semantics (default: or)\n\
  --top-k N                Number of hits (search: 10; batch: 1000)\n\
  --k1 NUMBER --b NUMBER   BM25 parameters; CIFF execution is exhaustive\n\
                            BM25 requires frequency-valued CIFF tf payloads\n";

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
        "forward-build" => command_forward_build(&remaining, &mut output),
        "forward-invert" => command_forward_invert(&remaining, &mut output),
        "reorder" => command_reorder(&remaining, &mut output),
        "forward-inspect" => command_forward_inspect(&remaining, &mut output),
        "lexicon" => command_lexicon(&remaining, &mut output),
        "search" => command_search(&remaining, &mut output),
        "shard-search" => command_shard_search(&remaining, &mut output),
        "batch" => command_batch(&remaining, &mut output),
        "shard-batch" => command_shard_batch(&remaining, &mut output),
        "inspect" => command_inspect(&remaining, &mut output),
        "shard-inspect" => command_shard_inspect(&remaining, &mut output),
        "ciff-export" => command_ciff_export(&remaining, &mut output),
        "ciff-search" => command_ciff_search(&remaining, &mut output),
        "ciff-batch" => command_ciff_batch(&remaining, &mut output),
        "ciff-inspect" => command_ciff_inspect(&remaining, &mut output),
        "benchmark" => command_benchmark(&remaining, &mut output),
        unknown => Err(Error::InvalidArgument(format!(
            "unknown command '{unknown}'; run 'indexsail help'"
        ))),
    }
}

fn command_ciff_export(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--index", "--output", "--description"])?;
    let source = parsed.required_one("--index")?;
    let destination = parsed.required_one("--output")?;
    if paths_conflict(source, destination)? {
        return Err(Error::InvalidArgument(
            "CIFF source and output paths must be distinct".into(),
        ));
    }
    let native = InvertedIndex::load(source)?;
    let description = parsed
        .optional_one("--description")?
        .unwrap_or("IndexSail export");
    let ciff = CiffIndex::from_native(&native, description)?;
    ciff.save(destination)?;
    let stats = ciff.stats();
    writeln!(
        output,
        "exported format=CIFF-v{} documents={} terms={} postings={} tokens={} bytes={} output={}",
        ciff.header().version,
        stats.contained_documents,
        stats.contained_posting_lists,
        stats.postings,
        stats.total_terms_in_collection,
        std::fs::metadata(destination)?.len(),
        destination
    )?;
    Ok(())
}

fn command_ciff_search(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--ascii"],
        &["--index", "--query", "--operator", "--top-k", "--k1", "--b"],
    )?;
    let path = parsed.required_one("--index")?;
    let index = CiffIndex::load(path)?;
    let analyzer = Analyzer::new(if parsed.flag("--ascii") {
        AnalysisMode::Ascii
    } else {
        AnalysisMode::Unicode
    });
    let options = CiffSearchOptions {
        top_k: parse_optional(parsed.optional_one("--top-k")?, 10_usize, "top-k")?,
        operator: parse_operator(parsed.optional_one("--operator")?)?,
        bm25: Bm25Params {
            k1: parse_optional(parsed.optional_one("--k1")?, 1.2_f64, "k1")?,
            b: parse_optional(parsed.optional_one("--b")?, 0.75_f64, "b")?,
        },
    };
    let outcome = index.search(analyzer, parsed.required_one("--query")?, options)?;
    writeln!(output, "rank\tscore\tid")?;
    for (rank, hit) in outcome.hits.iter().enumerate() {
        writeln!(
            output,
            "{}\t{:.6}\t{}",
            rank + 1,
            hit.score,
            hit.external_id
        )?;
    }
    writeln!(
        output,
        "format=CIFF-v{} strategy=exhaustive evaluated={} postings_visited={} total_documents={}",
        index.header().version,
        outcome.stats.evaluated_candidates,
        outcome.stats.postings_advanced,
        index.header().total_documents
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn command_ciff_batch(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--ascii"],
        &[
            "--index",
            "--topics",
            "--qrels",
            "--run",
            "--report",
            "--tag",
            "--operator",
            "--top-k",
            "--k1",
            "--b",
        ],
    )?;
    let index_path = parsed.required_one("--index")?;
    let topics_path = parsed.required_one("--topics")?;
    let run_path = parsed.required_one("--run")?;
    let report_path = parsed.optional_one("--report")?;
    ensure_distinct_batch_paths(
        index_path,
        topics_path,
        parsed.optional_one("--qrels")?,
        run_path,
        report_path,
    )?;

    let index = CiffIndex::load(index_path)?;
    let analyzer = Analyzer::new(if parsed.flag("--ascii") {
        AnalysisMode::Ascii
    } else {
        AnalysisMode::Unicode
    });
    let topics = load_topics(topics_path)?;
    let qrels = parsed
        .optional_one("--qrels")?
        .map(load_qrels)
        .transpose()?;
    let config = BatchConfig {
        top_k: parse_optional(parsed.optional_one("--top-k")?, 1_000_usize, "top-k")?,
        pruning: PruningStrategy::Exhaustive,
        verify_exact: false,
        field: None,
        operator: parse_operator(parsed.optional_one("--operator")?)?,
        bm25: Bm25Params {
            k1: parse_optional(parsed.optional_one("--k1")?, 1.2_f64, "k1")?,
            b: parse_optional(parsed.optional_one("--b")?, 0.75_f64, "b")?,
        },
    };
    let report = evaluate_batch(&index.retrieval(analyzer), &topics, qrels.as_ref(), config)?;
    let mut run_bytes = Vec::new();
    write_trec_run(
        &report,
        parsed.optional_one("--tag")?.unwrap_or("indexsail-ciff"),
        &mut run_bytes,
    )?;
    let report_bytes = if report_path.is_some() {
        let mut bytes = Vec::new();
        write_json_report(&report, &mut bytes)?;
        Some(bytes)
    } else {
        None
    };
    if let Some(path) = report_path {
        atomic_write_many(&[
            (Path::new(run_path), run_bytes.as_slice()),
            (
                Path::new(path),
                report_bytes
                    .as_deref()
                    .expect("a requested report was serialized before output opened"),
            ),
        ])?;
    } else {
        atomic_write(Path::new(run_path), &run_bytes)?;
    }
    let hits = report
        .queries
        .iter()
        .map(|query| query.hits.len())
        .sum::<usize>();
    writeln!(
        output,
        "batch format=CIFF-v{} topics={} hits={} strategy=Exhaustive elapsed_ms={:.3} evaluated={} postings_visited={} run={}",
        index.header().version,
        report.queries.len(),
        hits,
        report.total_search_time.as_secs_f64() * 1_000.0,
        report.total_stats.evaluated_candidates,
        report.total_stats.postings_advanced,
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
    Ok(())
}

fn command_ciff_inspect(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--index", "--term"])?;
    let path = parsed.required_one("--index")?;
    let index = CiffIndex::load(path)?;
    let stats = index.stats();
    writeln!(
        output,
        "format=CIFF-v{} file_bytes={} contained_documents={} total_documents={} contained_terms={} total_terms={} postings={} tokens={} avg_length={:.6}",
        index.header().version,
        std::fs::metadata(path)?.len(),
        stats.contained_documents,
        stats.total_documents,
        stats.contained_posting_lists,
        stats.total_posting_lists,
        stats.postings,
        stats.total_terms_in_collection,
        index.header().average_document_length
    )?;
    writeln!(
        output,
        "description={}",
        index
            .header()
            .description
            .chars()
            .map(|character| if character.is_control() {
                ' '
            } else {
                character
            })
            .collect::<String>()
    )?;
    if let Some(term) = parsed.optional_one("--term")? {
        if term.is_empty() || term.chars().any(char::is_control) {
            return Err(Error::InvalidArgument(
                "--term must be non-empty and contain no control characters".into(),
            ));
        }
        let list = index.posting_list(term);
        writeln!(
            output,
            "term={} df={} cf={}",
            term,
            list.map_or(0, |list| list.document_frequency),
            list.map_or(0, |list| list.collection_frequency)
        )?;
        if let Some(list) = list {
            for posting in &list.postings {
                let document = index.document(posting.document_id).ok_or_else(|| {
                    Error::CorruptIndex(format!(
                        "CIFF posting references missing document {}",
                        posting.document_id
                    ))
                })?;
                writeln!(
                    output,
                    "  doc={} internal={} tf={} length={}",
                    document.external_id,
                    document.document_id,
                    posting.term_frequency,
                    document.document_length
                )?;
            }
        }
    }
    Ok(())
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
        CollectionFormat::parse(parsed.optional_one("--format")?.unwrap_or("tsv"))?,
        CollectionLimits::default(),
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
    let index = index_collection_sharded(
        input,
        Analyzer::new(mode),
        CollectionFormat::parse(parsed.optional_one("--format")?.unwrap_or("tsv"))?,
        CollectionLimits::default(),
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

fn command_forward_build(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--ascii"],
        &["--input", "--output", "--format"],
    )?;
    let input = parsed.required_one("--input")?;
    let destination = parsed.required_one("--output")?;
    if paths_conflict(input, destination)? {
        return Err(Error::InvalidArgument(
            "forward input and output paths must be distinct".into(),
        ));
    }
    let mode = if parsed.flag("--ascii") {
        AnalysisMode::Ascii
    } else {
        AnalysisMode::Unicode
    };
    let format = CollectionFormat::parse(parsed.optional_one("--format")?.unwrap_or("tsv"))?;
    let forward = ForwardIndex::from_collection(
        input,
        Analyzer::new(mode),
        format,
        CollectionLimits::default(),
    )?;
    forward.save(destination)?;
    let stats = forward.stats();
    writeln!(
        output,
        "forward documents={} fields={} terms={} occurrences={} analyzer={:?} format={} output={}",
        stats.documents,
        stats.fields,
        stats.terms,
        stats.occurrences,
        forward.analyzer().mode(),
        format_name(format),
        destination
    )?;
    Ok(())
}

fn command_forward_invert(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--input", "--output"])?;
    let input = parsed.required_one("--input")?;
    let destination = parsed.required_one("--output")?;
    if paths_conflict(input, destination)? {
        return Err(Error::InvalidArgument(
            "forward source and inverted output paths must be distinct".into(),
        ));
    }
    let forward = ForwardIndex::load(input)?;
    let index = forward.invert()?;
    atomic_write_with(Path::new(destination), |writer| index.write_to(writer))?;
    let stats = index.stats();
    writeln!(
        output,
        "inverted documents={} fields={} terms={} postings={} tokens={} input={} output={}",
        stats.documents,
        stats.fields,
        stats.terms,
        stats.postings,
        stats.tokens,
        input,
        destination
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn command_reorder(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &["--random", "--bp"],
        &[
            "--input",
            "--forward-output",
            "--index-output",
            "--old-to-new",
            "--new-to-old",
            "--seed",
            "--by-feature",
            "--from-mapping",
            "--depth",
            "--iterations",
        ],
    )?;
    let source = Path::new(parsed.required_one("--input")?);
    let forward_output = Path::new(parsed.required_one("--forward-output")?);
    let index_output = Path::new(parsed.required_one("--index-output")?);
    let old_to_new_output = Path::new(parsed.required_one("--old-to-new")?);
    let new_to_old_output = Path::new(parsed.required_one("--new-to-old")?);
    let feature_source = parsed.optional_one("--by-feature")?.map(Path::new);
    let mapping_source = parsed.optional_one("--from-mapping")?.map(Path::new);
    let seed = parsed.optional_one("--seed")?;
    let depth = parsed.optional_one("--depth")?;
    let iterations = parsed.optional_one("--iterations")?;
    let methods = usize::from(parsed.flag("--random"))
        + usize::from(feature_source.is_some())
        + usize::from(mapping_source.is_some())
        + usize::from(parsed.flag("--bp"));
    if methods != 1 {
        return Err(Error::InvalidArgument(
            "choose exactly one of --random, --by-feature, --from-mapping, or --bp".into(),
        ));
    }
    if seed.is_some() && !parsed.flag("--random") {
        return Err(Error::InvalidArgument(
            "--seed is only valid with --random".into(),
        ));
    }
    if (depth.is_some() || iterations.is_some()) && !parsed.flag("--bp") {
        return Err(Error::InvalidArgument(
            "--depth and --iterations are only valid with --bp".into(),
        ));
    }

    validate_reorder_paths(
        [Some(source), feature_source, mapping_source],
        [
            forward_output,
            index_output,
            old_to_new_output,
            new_to_old_output,
        ],
    )?;

    let source_bytes = std::fs::metadata(source)?.len();
    if source_bytes > MAX_REORDER_FORWARD_BYTES + 28 {
        return Err(Error::InvalidArgument(format!(
            "reorder forward source exceeds {MAX_REORDER_FORWARD_BYTES} payload byte limit"
        )));
    }
    let forward = ForwardIndex::load(source)?;
    let count = forward.documents().len();
    let (method, mapping) = if parsed.flag("--random") {
        let seed = seed.unwrap_or("0").parse::<u64>().map_err(|_| {
            Error::InvalidArgument("--seed must be an unsigned 64-bit integer".into())
        })?;
        (
            format!("random seed={seed}"),
            DocIdMap::random(count, seed)?,
        )
    } else if let Some(path) = feature_source {
        (
            "by-feature".to_owned(),
            DocIdMap::by_feature(count, std::fs::File::open(path)?)?,
        )
    } else if parsed.flag("--bp") {
        let mut options = BisectionOptions::for_documents(count);
        options.depth = parse_optional(depth, options.depth, "depth")?;
        options.iterations = parse_optional(iterations, options.iterations, "iterations")?;
        (
            format!(
                "bp depth={} iterations={}",
                options.depth, options.iterations
            ),
            DocIdMap::recursive_graph_bisection(&forward, options)?,
        )
    } else {
        let path = mapping_source.expect("exactly one method was selected");
        (
            "from-mapping".to_owned(),
            DocIdMap::from_mapping_reader(count, std::fs::File::open(path)?)?,
        )
    };
    let reordered = forward.reordered(&mapping)?;
    let inverted = reordered.invert()?;
    let mut write_forward = |file: &mut std::fs::File| reordered.write_to(file);
    let mut write_index = |file: &mut std::fs::File| inverted.write_to(file);
    let mut write_old_to_new = |file: &mut std::fs::File| mapping.write_old_to_new(file);
    let mut write_new_to_old = |file: &mut std::fs::File| mapping.write_new_to_old(file);
    atomic_write_many_with(&mut [
        (forward_output, &mut write_forward),
        (index_output, &mut write_index),
        (old_to_new_output, &mut write_old_to_new),
        (new_to_old_output, &mut write_new_to_old),
    ])?;
    writeln!(
        output,
        "reordered documents={count} method={method} forward={} index={} old_to_new={} new_to_old={}",
        forward_output.display(),
        index_output.display(),
        old_to_new_output.display(),
        new_to_old_output.display(),
    )?;
    Ok(())
}

fn validate_reorder_paths(inputs: [Option<&Path>; 3], outputs: [&Path; 4]) -> Result<()> {
    for destination in outputs {
        prevalidate_output_path(destination)?;
        for input in inputs.into_iter().flatten() {
            if paths_conflict(input, destination)? {
                return Err(Error::InvalidArgument(
                    "reorder input and output paths must be distinct".into(),
                ));
            }
        }
    }
    for left in 0..outputs.len() {
        for right in left + 1..outputs.len() {
            if paths_conflict(outputs[left], outputs[right])? {
                return Err(Error::InvalidArgument(
                    "reorder output paths must be distinct".into(),
                ));
            }
        }
    }
    Ok(())
}

fn command_forward_inspect(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--index", "--document", "--limit"])?;
    let path = parsed.required_one("--index")?;
    let forward = ForwardIndex::load(path)?;
    let stats = forward.stats();
    writeln!(
        output,
        "forward format={} analyzer={:?} documents={} fields={} terms={} occurrences={}",
        crate::forward::FORWARD_FORMAT_VERSION,
        forward.analyzer().mode(),
        stats.documents,
        stats.fields,
        stats.terms,
        stats.occurrences
    )?;
    let Some(document_value) = parsed.optional_one("--document")? else {
        if parsed.optional_one("--limit")?.is_some() {
            return Err(Error::InvalidArgument(
                "--limit requires --document for forward-inspect".into(),
            ));
        }
        return Ok(());
    };
    let document_id = document_value.parse::<u32>().map_err(|_| {
        Error::InvalidArgument(format!(
            "document id '{document_value}' is not an unsigned integer"
        ))
    })?;
    let limit = parse_optional(parsed.optional_one("--limit")?, 20_usize, "limit")?;
    if limit == 0 || limit > 1_000 {
        return Err(Error::InvalidArgument(
            "forward preview limit must be between 1 and 1000".into(),
        ));
    }
    let document = usize::try_from(document_id)
        .ok()
        .and_then(|id| forward.documents().get(id))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("forward document {document_id} does not exist"))
        })?;
    writeln!(
        output,
        "document={} external_id={:?}",
        document_id,
        document.document().external_id()
    )?;
    for field in document.document().fields().keys() {
        let ids = document
            .term_ids(field)
            .expect("validated forward documents have one sequence per field");
        let preview = ids
            .iter()
            .take(limit)
            .map(|&id| {
                forward
                    .term(id)
                    .map(|term| term.term.as_str())
                    .ok_or_else(|| Error::CorruptIndex(format!("unknown forward term id {id}")))
            })
            .collect::<Result<Vec<_>>>()?;
        writeln!(
            output,
            "field={} occurrences={} preview={}",
            field,
            ids.len(),
            preview.join(" ")
        )?;
    }
    Ok(())
}

fn command_lexicon(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &[],
        &[
            "--index", "--id", "--field", "--term", "--offset", "--limit",
        ],
    )?;
    let forward = ForwardIndex::load(parsed.required_one("--index")?)?;
    let id = parsed.optional_one("--id")?;
    let field = parsed.optional_one("--field")?;
    let term = parsed.optional_one("--term")?;
    let offset = parsed.optional_one("--offset")?;
    let limit = parsed.optional_one("--limit")?;

    if let Some(id) = id {
        if field.is_some() || term.is_some() || offset.is_some() || limit.is_some() {
            return Err(Error::InvalidArgument(
                "--id cannot be combined with field, term, offset, or limit".into(),
            ));
        }
        let id = id.parse::<u32>().map_err(|_| {
            Error::InvalidArgument(format!("term id '{id}' is not an unsigned integer"))
        })?;
        let entry = forward
            .term(id)
            .ok_or_else(|| Error::InvalidArgument(format!("term id {id} does not exist")))?;
        writeln!(output, "{id}\t{}\t{}", entry.field, entry.term)?;
        return Ok(());
    }

    match (field, term) {
        (Some(field), Some(term)) => {
            if offset.is_some() || limit.is_some() {
                return Err(Error::InvalidArgument(
                    "field/term lookup cannot be combined with offset or limit".into(),
                ));
            }
            let normalized = forward.analyzer().normalize_single(term).ok_or_else(|| {
                Error::InvalidArgument(
                    "lexicon lookup term must normalize to exactly one token".into(),
                )
            })?;
            let id = forward.term_id(field, &normalized).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "term '{normalized}' does not exist in field '{field}'"
                ))
            })?;
            writeln!(output, "{id}\t{field}\t{normalized}")?;
            Ok(())
        }
        (None, None) => {
            let offset = parse_optional(offset, 0_usize, "offset")?;
            let limit = parse_optional(limit, 20_usize, "limit")?;
            if limit == 0 || limit > 1_000 {
                return Err(Error::InvalidArgument(
                    "lexicon page limit must be between 1 and 1000".into(),
                ));
            }
            for (id, entry) in forward.terms().iter().enumerate().skip(offset).take(limit) {
                writeln!(output, "{id}\t{}\t{}", entry.field, entry.term)?;
            }
            Ok(())
        }
        _ => Err(Error::InvalidArgument(
            "--field and --term must be supplied together".into(),
        )),
    }
}

const fn format_name(format: CollectionFormat) -> &'static str {
    match format {
        CollectionFormat::Tsv => "tsv",
        CollectionFormat::Trec => "trec",
        CollectionFormat::Jsonl => "jsonl",
    }
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
    ensure_distinct_batch_paths(
        index_path,
        topics_path,
        parsed.optional_one("--qrels")?,
        run_path,
        report_path,
    )?;

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

    let mut run_bytes = Vec::new();
    write_trec_run(
        &report,
        parsed.optional_one("--tag")?.unwrap_or("indexsail"),
        &mut run_bytes,
    )?;
    let report_bytes = if report_path.is_some() {
        let mut bytes = Vec::new();
        write_json_report(&report, &mut bytes)?;
        Some(bytes)
    } else {
        None
    };
    if let Some(path) = report_path {
        atomic_write_many(&[
            (Path::new(run_path), run_bytes.as_slice()),
            (
                Path::new(path),
                report_bytes
                    .as_deref()
                    .expect("a requested report was serialized before output opened"),
            ),
        ])?;
    } else {
        atomic_write(Path::new(run_path), &run_bytes)?;
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
    let json_path = parsed.optional_one("--json")?;
    if let Some(path) = json_path {
        // Validate before the potentially expensive benchmark. The atomic
        // writer repeats this check immediately before replacement.
        prevalidate_output_path(Path::new(path))?;
    }
    let report = run_benchmark(config)?;
    if let Some(path) = json_path {
        let mut bytes = Vec::new();
        write_benchmark_json(&report, &mut bytes)?;
        atomic_write(Path::new(path), &bytes)?;
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
    if let Some(path) = json_path {
        writeln!(output, "report={path}")?;
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

fn parse_operator(value: Option<&str>) -> Result<BooleanOperator> {
    match value.unwrap_or("or") {
        "and" => Ok(BooleanOperator::And),
        "or" => Ok(BooleanOperator::Or),
        value => Err(Error::InvalidArgument(format!(
            "unknown operator '{value}', expected and or or"
        ))),
    }
}

fn ensure_distinct_batch_paths(
    index: &str,
    topics: &str,
    qrels: Option<&str>,
    run: &str,
    report: Option<&str>,
) -> Result<()> {
    prevalidate_output_path(Path::new(run))?;
    if let Some(path) = report {
        prevalidate_output_path(Path::new(path))?;
    }
    let mut paths = vec![(index, false), (topics, false), (run, true)];
    if let Some(path) = qrels {
        paths.push((path, false));
    }
    if let Some(path) = report {
        paths.push((path, true));
    }
    for left in 0..paths.len() {
        for right in left + 1..paths.len() {
            if (paths[left].1 || paths[right].1) && paths_conflict(paths[left].0, paths[right].0)? {
                return Err(Error::InvalidArgument(
                    "batch input, run, and report paths must be distinct".into(),
                ));
            }
        }
    }
    Ok(())
}

fn paths_conflict(left: impl AsRef<Path>, right: impl AsRef<Path>) -> Result<bool> {
    let left = left.as_ref();
    let right = right.as_ref();
    if same_existing_file(left, right)? {
        return Ok(true);
    }
    let left = normalized_path(left)?;
    let right = normalized_path(right)?;
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

fn same_existing_file(left: &Path, right: &Path) -> Result<bool> {
    if !left.exists() || !right.exists() {
        return Ok(false);
    }
    same_file::is_same_file(left, right).map_err(Error::from)
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
    fn reorder_cli_rebuilds_both_indexes_and_writes_inverse_maps() {
        let collection = temp_path("tsv");
        let source = temp_path("fwd");
        let new_forward = temp_path("fwd");
        let new_index = temp_path("idx");
        let old_map = temp_path("map");
        let new_map = temp_path("map");
        let features = temp_path("features");
        std::fs::write(
            &collection,
            "id\tbody\nA\tblue sea\nB\tred wind\nC\tblue wind\n",
        )
        .unwrap();
        std::fs::write(&features, "z\na\nm\n").unwrap();
        execute(
            [
                "forward-build",
                "--input",
                collection.to_str().unwrap(),
                "--output",
                source.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        let command = [
            "reorder",
            "--input",
            source.to_str().unwrap(),
            "--forward-output",
            new_forward.to_str().unwrap(),
            "--index-output",
            new_index.to_str().unwrap(),
            "--old-to-new",
            old_map.to_str().unwrap(),
            "--new-to-old",
            new_map.to_str().unwrap(),
            "--by-feature",
            features.to_str().unwrap(),
        ];
        execute(command, Vec::new()).unwrap();
        assert_eq!(
            std::fs::read_to_string(&old_map).unwrap(),
            "0 2\n1 0\n2 1\n"
        );
        assert_eq!(
            std::fs::read_to_string(&new_map).unwrap(),
            "0 1\n1 2\n2 0\n"
        );
        let reordered = ForwardIndex::load(&new_forward).unwrap();
        assert_eq!(reordered.documents()[0].document().external_id(), "B");
        assert_eq!(reordered.documents()[1].document().external_id(), "C");
        assert_eq!(reordered.documents()[2].document().external_id(), "A");
        let index = InvertedIndex::load(&new_index).unwrap();
        assert_eq!(index.documents()[0].external_id(), "B");
        let mut search_output = Vec::new();
        execute(
            [
                "search",
                "--index",
                new_index.to_str().unwrap(),
                "--query",
                "blue",
            ],
            &mut search_output,
        )
        .unwrap();
        let search_output = String::from_utf8(search_output).unwrap();
        assert!(search_output.contains('A'));
        assert!(search_output.contains('C'));

        for path in [
            collection,
            source,
            new_forward,
            new_index,
            old_map,
            new_map,
            features,
        ] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn reorder_cli_custom_and_seeded_random_are_repeatable() {
        let source = temp_path("fwd");
        let new_forward = temp_path("fwd");
        let new_index = temp_path("idx");
        let old_map = temp_path("map");
        let new_map = temp_path("map");
        let custom = temp_path("map");
        ForwardIndex::from_documents(
            Analyzer::default(),
            [
                Document::from_fields("A", [("body", "blue")]).unwrap(),
                Document::from_fields("B", [("body", "red")]).unwrap(),
                Document::from_fields("C", [("body", "green")]).unwrap(),
            ],
        )
        .unwrap()
        .save(&source)
        .unwrap();
        let command = [
            "reorder",
            "--input",
            source.to_str().unwrap(),
            "--forward-output",
            new_forward.to_str().unwrap(),
            "--index-output",
            new_index.to_str().unwrap(),
            "--old-to-new",
            old_map.to_str().unwrap(),
            "--new-to-old",
            new_map.to_str().unwrap(),
        ];
        std::fs::write(&custom, b"2 1\n0 2\n1 0\n").unwrap();
        let mut mapped = command.to_vec();
        mapped.extend(["--from-mapping", custom.to_str().unwrap()]);
        execute(mapped, Vec::new()).unwrap();
        assert_eq!(
            std::fs::read_to_string(&old_map).unwrap(),
            "0 2\n1 0\n2 1\n"
        );
        let mut random = command.to_vec();
        random.extend(["--random", "--seed", "7"]);
        execute(random.clone(), Vec::new()).unwrap();
        let first = [
            std::fs::read(&new_forward).unwrap(),
            std::fs::read(&new_index).unwrap(),
            std::fs::read(&old_map).unwrap(),
            std::fs::read(&new_map).unwrap(),
        ];
        execute(random, Vec::new()).unwrap();
        assert_eq!(
            first,
            [
                std::fs::read(&new_forward).unwrap(),
                std::fs::read(&new_index).unwrap(),
                std::fs::read(&old_map).unwrap(),
                std::fs::read(&new_map).unwrap(),
            ]
        );
        let mut bp = command.to_vec();
        bp.extend(["--bp", "--depth", "2", "--iterations", "3"]);
        let mut output = Vec::new();
        execute(bp.clone(), &mut output).unwrap();
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("method=bp depth=2 iterations=3")
        );
        let first_bp = [
            std::fs::read(&new_forward).unwrap(),
            std::fs::read(&new_index).unwrap(),
            std::fs::read(&old_map).unwrap(),
            std::fs::read(&new_map).unwrap(),
        ];
        execute(bp, Vec::new()).unwrap();
        assert_eq!(
            first_bp,
            [
                std::fs::read(&new_forward).unwrap(),
                std::fs::read(&new_index).unwrap(),
                std::fs::read(&old_map).unwrap(),
                std::fs::read(&new_map).unwrap(),
            ]
        );
        for path in [source, new_forward, new_index, old_map, new_map, custom] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn reorder_rejects_aliases_and_bad_methods_before_changing_outputs() {
        let collection = temp_path("tsv");
        let source = temp_path("fwd");
        let new_forward = temp_path("fwd");
        let new_index = temp_path("idx");
        let old_map = temp_path("map");
        let new_map = temp_path("map");
        let custom = temp_path("map");
        std::fs::write(&collection, "id\tbody\nA\tblue\nB\tred\n").unwrap();
        execute(
            [
                "forward-build",
                "--input",
                collection.to_str().unwrap(),
                "--output",
                source.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        std::fs::write(&new_forward, b"old-forward").unwrap();
        std::fs::write(&new_index, b"old-index").unwrap();
        std::fs::write(&old_map, b"old-map").unwrap();
        std::fs::write(&new_map, b"old-inverse").unwrap();
        std::fs::write(&custom, b"0 0\n1 0\n").unwrap();
        let prefix = [
            "reorder",
            "--input",
            source.to_str().unwrap(),
            "--forward-output",
            new_forward.to_str().unwrap(),
            "--index-output",
            new_index.to_str().unwrap(),
            "--old-to-new",
            old_map.to_str().unwrap(),
            "--new-to-old",
            new_map.to_str().unwrap(),
        ];
        let mut invalid = prefix.to_vec();
        invalid.extend(["--from-mapping", custom.to_str().unwrap()]);
        assert!(execute(invalid, Vec::new()).is_err());
        assert_eq!(std::fs::read(&new_forward).unwrap(), b"old-forward");
        assert_eq!(std::fs::read(&new_index).unwrap(), b"old-index");
        assert_eq!(std::fs::read(&old_map).unwrap(), b"old-map");
        assert_eq!(std::fs::read(&new_map).unwrap(), b"old-inverse");

        let mut alias = prefix.to_vec();
        alias[4] = source.to_str().unwrap();
        alias.push("--random");
        assert!(
            execute(alias, Vec::new())
                .unwrap_err()
                .to_string()
                .contains("distinct")
        );
        let mut double_method = prefix.to_vec();
        double_method.extend(["--random", "--by-feature", custom.to_str().unwrap()]);
        assert!(execute(double_method, Vec::new()).is_err());
        let mut bad_seed = prefix.to_vec();
        bad_seed.extend(["--random", "--seed", "-1"]);
        assert!(execute(bad_seed, Vec::new()).is_err());
        let mut bad_depth = prefix.to_vec();
        bad_depth.extend(["--bp", "--depth", "0"]);
        assert!(execute(bad_depth, Vec::new()).is_err());
        let mut misplaced_depth = prefix.to_vec();
        misplaced_depth.extend(["--random", "--depth", "2"]);
        assert!(execute(misplaced_depth, Vec::new()).is_err());
        for path in [
            collection,
            source,
            new_forward,
            new_index,
            old_map,
            new_map,
            custom,
        ] {
            std::fs::remove_file(path).unwrap();
        }
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
    #[allow(clippy::too_many_lines)]
    fn ciff_export_search_inspect_and_trec_batch_form_an_end_to_end_flow() {
        let corpus = temp_path("ciff.tsv");
        let native = temp_path("ciff.idx");
        let ciff = temp_path("ciff");
        let topics = temp_path("ciff.topics");
        let qrels = temp_path("ciff.qrels");
        let run = temp_path("ciff.run");
        let report = temp_path("ciff.json");
        std::fs::write(
            &corpus,
            "id\ttitle\tbody\nD1\tRust Search\tfast local search engine\nD2\tGrid\tpower grid solver\nD3\tSailing\tlocal retrieval system\n",
        )
        .unwrap();
        std::fs::write(&topics, "1\tlocal search\n2\tpower grid\n").unwrap();
        std::fs::write(&qrels, "1 0 D1 2\n1 0 D3 1\n2 0 D2 2\n").unwrap();

        execute(
            [
                "index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                native.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        let mut export_output = Vec::new();
        execute(
            [
                "ciff-export",
                "--index",
                native.to_str().unwrap(),
                "--output",
                ciff.to_str().unwrap(),
                "--description",
                "CLI\nfixture\u{1b}",
            ],
            &mut export_output,
        )
        .unwrap();
        assert!(
            String::from_utf8(export_output)
                .unwrap()
                .starts_with("exported format=CIFF-v1 documents=3")
        );

        let mut inspect_output = Vec::new();
        execute(
            [
                "ciff-inspect",
                "--index",
                ciff.to_str().unwrap(),
                "--term",
                "local",
            ],
            &mut inspect_output,
        )
        .unwrap();
        let inspect_output = String::from_utf8(inspect_output).unwrap();
        assert!(inspect_output.contains("format=CIFF-v1"));
        assert!(inspect_output.contains("description=CLI fixture "));
        assert!(inspect_output.contains("term=local df=2 cf=2"));
        assert!(inspect_output.contains("doc=D1 internal=0 tf=1"));

        let mut search_output = Vec::new();
        execute(
            [
                "ciff-search",
                "--index",
                ciff.to_str().unwrap(),
                "--query",
                "local search",
                "--top-k",
                "2",
            ],
            &mut search_output,
        )
        .unwrap();
        let search_output = String::from_utf8(search_output).unwrap();
        assert_eq!(search_output.lines().next(), Some("rank\tscore\tid"));
        assert!(search_output.contains("D1"));
        assert!(search_output.contains("strategy=exhaustive"));

        let mut batch_output = Vec::new();
        execute(
            [
                "ciff-batch",
                "--index",
                ciff.to_str().unwrap(),
                "--topics",
                topics.to_str().unwrap(),
                "--qrels",
                qrels.to_str().unwrap(),
                "--run",
                run.to_str().unwrap(),
                "--report",
                report.to_str().unwrap(),
                "--top-k",
                "10",
            ],
            &mut batch_output,
        )
        .unwrap();
        let batch_output = String::from_utf8(batch_output).unwrap();
        assert!(batch_output.contains("batch format=CIFF-v1 topics=2"));
        assert!(batch_output.contains("map=1.000000"));
        assert!(std::fs::read_to_string(&run).unwrap().contains("1 Q0 D1 1"));
        assert!(
            std::fs::read_to_string(&report)
                .unwrap()
                .contains("\"strategy\": \"exhaustive\"")
        );

        for path in [corpus, native, ciff, topics, qrels, run, report] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn ciff_commands_reject_destructive_output_aliases_before_reading() {
        let error = execute(
            [
                "ciff-export",
                "--index",
                "same.ciff",
                "--output",
                "./same.ciff",
            ],
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("distinct"));

        let error = execute(
            [
                "ciff-batch",
                "--index",
                "same.ciff",
                "--topics",
                "topics.tsv",
                "--run",
                "./same.ciff",
            ],
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("distinct"));
    }

    #[test]
    fn ciff_export_rejects_a_hard_link_to_the_native_index_without_data_loss() {
        let corpus = temp_path("hardlink.tsv");
        let native = temp_path("hardlink.idx");
        let alias = temp_path("hardlink.ciff");
        std::fs::write(&corpus, "id\tbody\nD1\tlocal search\n").unwrap();
        execute(
            [
                "index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                native.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        let original = std::fs::read(&native).unwrap();
        std::fs::hard_link(&native, &alias).unwrap();
        let error = execute(
            [
                "ciff-export",
                "--index",
                native.to_str().unwrap(),
                "--output",
                alias.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("distinct"));
        assert_eq!(std::fs::read(&native).unwrap(), original);
        assert_eq!(std::fs::read(&alias).unwrap(), original);
        for path in [alias, native, corpus] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn batch_path_checks_detect_hard_links_for_every_input_and_output_role() {
        let index = temp_path("identity.idx");
        let topics = temp_path("identity.topics");
        let qrels = temp_path("identity.qrels");
        for (path, bytes) in [
            (&index, b"index".as_slice()),
            (&topics, b"topics".as_slice()),
            (&qrels, b"qrels".as_slice()),
        ] {
            std::fs::write(path, bytes).unwrap();
        }
        for input in [&index, &topics, &qrels] {
            let run = temp_path("identity.run");
            std::fs::hard_link(input, &run).unwrap();
            assert!(
                ensure_distinct_batch_paths(
                    index.to_str().unwrap(),
                    topics.to_str().unwrap(),
                    Some(qrels.to_str().unwrap()),
                    run.to_str().unwrap(),
                    None,
                )
                .is_err()
            );
            std::fs::remove_file(run).unwrap();

            let run = temp_path("identity.run");
            let report = temp_path("identity.json");
            std::fs::hard_link(input, &report).unwrap();
            assert!(
                ensure_distinct_batch_paths(
                    index.to_str().unwrap(),
                    topics.to_str().unwrap(),
                    Some(qrels.to_str().unwrap()),
                    run.to_str().unwrap(),
                    Some(report.to_str().unwrap()),
                )
                .is_err()
            );
            std::fs::remove_file(report).unwrap();
        }

        let run = temp_path("identity.run");
        let report = temp_path("identity.json");
        std::fs::write(&run, b"run").unwrap();
        std::fs::hard_link(&run, &report).unwrap();
        assert!(
            ensure_distinct_batch_paths(
                index.to_str().unwrap(),
                topics.to_str().unwrap(),
                Some(qrels.to_str().unwrap()),
                run.to_str().unwrap(),
                Some(report.to_str().unwrap()),
            )
            .is_err()
        );
        for path in [report, run, qrels, topics, index] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn batch_validation_failures_preserve_existing_native_and_ciff_outputs() {
        let corpus = temp_path("atomic.tsv");
        let native = temp_path("atomic.idx");
        let ciff = temp_path("atomic.ciff");
        let topics = temp_path("atomic.topics");
        let run = temp_path("atomic.run");
        let report = temp_path("atomic.json");
        std::fs::write(&corpus, "id\tbody\nD 1\tlocal search\n").unwrap();
        std::fs::write(&topics, "q1\tlocal\n").unwrap();
        execute(
            [
                "index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                native.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        execute(
            [
                "ciff-export",
                "--index",
                native.to_str().unwrap(),
                "--output",
                ciff.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();

        for command in ["batch", "ciff-batch"] {
            std::fs::write(&run, b"run sentinel").unwrap();
            std::fs::write(&report, b"report sentinel").unwrap();
            let selected_index = if command == "batch" { &native } else { &ciff };
            let error = execute(
                [
                    command,
                    "--index",
                    selected_index.to_str().unwrap(),
                    "--topics",
                    topics.to_str().unwrap(),
                    "--run",
                    run.to_str().unwrap(),
                    "--report",
                    report.to_str().unwrap(),
                    "--tag",
                    "invalid tag",
                ],
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("tag"));
            assert_eq!(std::fs::read(&run).unwrap(), b"run sentinel");
            assert_eq!(std::fs::read(&report).unwrap(), b"report sentinel");
        }

        for command in ["batch", "ciff-batch"] {
            std::fs::write(&run, b"run sentinel").unwrap();
            std::fs::write(&report, b"report sentinel").unwrap();
            let selected_index = if command == "batch" { &native } else { &ciff };
            let error = execute(
                [
                    command,
                    "--index",
                    selected_index.to_str().unwrap(),
                    "--topics",
                    topics.to_str().unwrap(),
                    "--run",
                    run.to_str().unwrap(),
                    "--report",
                    report.to_str().unwrap(),
                ],
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("document id"));
            assert_eq!(std::fs::read(&run).unwrap(), b"run sentinel");
            assert_eq!(std::fs::read(&report).unwrap(), b"report sentinel");
        }

        atomic_write(&run, b"replacement").unwrap();
        assert_eq!(std::fs::read(&run).unwrap(), b"replacement");
        for path in [report, run, topics, ciff, native, corpus] {
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
    fn benchmark_report_replacement_preserves_existing_hard_link_sibling() {
        let report = temp_path("benchmark-hardlink.json");
        let sibling = temp_path("benchmark-hardlink-backup.json");
        std::fs::write(&report, b"old benchmark").unwrap();
        std::fs::hard_link(&report, &sibling).unwrap();
        execute(
            [
                "benchmark",
                "--documents",
                "20",
                "--queries",
                "2",
                "--top-k",
                "2",
                "--seed",
                "7",
                "--json",
                report.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        assert!(
            std::fs::read_to_string(&report)
                .unwrap()
                .contains("\"checksum\"")
        );
        assert_eq!(std::fs::read(&sibling).unwrap(), b"old benchmark");
        std::fs::remove_file(report).unwrap();
        std::fs::remove_file(sibling).unwrap();
    }

    #[test]
    fn native_and_ciff_batch_commit_failure_restores_both_outputs() {
        let corpus = temp_path("transaction.tsv");
        let native = temp_path("transaction.idx");
        let ciff = temp_path("transaction.ciff");
        let topics = temp_path("transaction.topics");
        let run = temp_path("transaction.run");
        let report = temp_path("transaction.json");
        std::fs::write(&corpus, "id\tbody\nD1\tlocal search\n").unwrap();
        std::fs::write(&topics, "q1\tlocal\n").unwrap();
        execute(
            [
                "index",
                "--input",
                corpus.to_str().unwrap(),
                "--output",
                native.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        execute(
            [
                "ciff-export",
                "--index",
                native.to_str().unwrap(),
                "--output",
                ciff.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();

        for (command, selected_index) in [("batch", &native), ("ciff-batch", &ciff)] {
            std::fs::write(&run, b"old run").unwrap();
            std::fs::write(&report, b"old report").unwrap();
            crate::atomic::inject_error_after_install_for_test(1);
            let error = execute(
                [
                    command,
                    "--index",
                    selected_index.to_str().unwrap(),
                    "--topics",
                    topics.to_str().unwrap(),
                    "--run",
                    run.to_str().unwrap(),
                    "--report",
                    report.to_str().unwrap(),
                ],
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("injected output commit failure"));
            assert_eq!(std::fs::read(&run).unwrap(), b"old run");
            assert_eq!(std::fs::read(&report).unwrap(), b"old report");
        }

        for path in [report, run, topics, ciff, native, corpus] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn jsonl_forward_lexicon_invert_and_search_form_an_end_to_end_flow() {
        let collection = temp_path("forward.jsonl");
        let forward = temp_path("forward.fwd");
        let native = temp_path("forward.idx");
        std::fs::write(
            &collection,
            "{\"id\":\"D1\",\"fields\":{\"title\":\"Blue Sail\",\"body\":\"local blue search\"}}\n{\"id\":\"D2\",\"fields\":{\"title\":\"Grid\",\"body\":\"power model\"}}\n",
        )
        .unwrap();

        let mut build_output = Vec::new();
        execute(
            [
                "forward-build",
                "--input",
                collection.to_str().unwrap(),
                "--output",
                forward.to_str().unwrap(),
                "--format",
                "jsonl",
            ],
            &mut build_output,
        )
        .unwrap();
        let build_output = String::from_utf8(build_output).unwrap();
        assert!(build_output.contains("documents=2"));
        assert!(build_output.contains("format=jsonl"));

        let mut inspect_output = Vec::new();
        execute(
            [
                "forward-inspect",
                "--index",
                forward.to_str().unwrap(),
                "--document",
                "0",
                "--limit",
                "2",
            ],
            &mut inspect_output,
        )
        .unwrap();
        let inspect_output = String::from_utf8(inspect_output).unwrap();
        assert!(inspect_output.contains("format=1"));
        assert!(inspect_output.contains("external_id=\"D1\""));
        assert!(inspect_output.contains("field=body occurrences=3 preview=local blue"));

        let mut reverse_output = Vec::new();
        execute(
            [
                "lexicon",
                "--index",
                forward.to_str().unwrap(),
                "--field",
                "body",
                "--term",
                "BLUE",
            ],
            &mut reverse_output,
        )
        .unwrap();
        let reverse_output = String::from_utf8(reverse_output).unwrap();
        let term_id = reverse_output.split('\t').next().unwrap().to_owned();
        assert!(reverse_output.ends_with("\tbody\tblue\n"));

        let mut lookup_output = Vec::new();
        execute(
            [
                "lexicon",
                "--index",
                forward.to_str().unwrap(),
                "--id",
                &term_id,
            ],
            &mut lookup_output,
        )
        .unwrap();
        assert_eq!(String::from_utf8(lookup_output).unwrap(), reverse_output);

        execute(
            [
                "forward-invert",
                "--input",
                forward.to_str().unwrap(),
                "--output",
                native.to_str().unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        let mut search_output = Vec::new();
        execute(
            [
                "search",
                "--index",
                native.to_str().unwrap(),
                "--query",
                "blue",
                "--field",
                "body",
            ],
            &mut search_output,
        )
        .unwrap();
        assert!(
            String::from_utf8(search_output)
                .unwrap()
                .contains("D1\tBlue Sail")
        );

        for path in [native, forward, collection] {
            std::fs::remove_file(path).unwrap();
        }
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

    #[test]
    fn lexicon_rejects_ambiguous_selectors_and_invalid_page_bounds() {
        let forward_path = temp_path("lexicon-invalid.fwd");
        ForwardIndex::from_documents(
            Analyzer::default(),
            [Document::from_fields("D", [("body", "blue sea")]).unwrap()],
        )
        .unwrap()
        .save(&forward_path)
        .unwrap();
        let path = forward_path.to_str().unwrap();
        let cases: &[(&[&str], &str)] = &[
            (&["--id", "0", "--limit", "1"], "cannot be combined"),
            (&["--id", "not-a-number"], "not an unsigned integer"),
            (&["--id", "9999"], "does not exist"),
            (&["--field", "body"], "must be supplied together"),
            (&["--term", "blue"], "must be supplied together"),
            (
                &["--field", "body", "--term", "blue sea"],
                "exactly one token",
            ),
            (&["--field", "body", "--term", "absent"], "does not exist"),
            (
                &["--field", "body", "--term", "blue", "--offset", "0"],
                "cannot be combined",
            ),
            (&["--limit", "0"], "between 1 and 1000"),
            (&["--limit", "1001"], "between 1 and 1000"),
            (&["--offset", "negative"], "wrong type"),
        ];
        for (options, expected) in cases {
            let mut arguments = vec!["lexicon", "--index", path];
            arguments.extend_from_slice(options);
            let mut output = Vec::new();
            let error = execute(arguments, &mut output).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(output, Vec::<u8>::new());
        }
        std::fs::remove_file(forward_path).unwrap();
    }

    #[test]
    fn forward_inspect_rejects_unbound_limit_and_invalid_document_selector() {
        let forward_path = temp_path("inspect-invalid.fwd");
        ForwardIndex::from_documents(
            Analyzer::default(),
            [Document::from_fields("D", [("body", "blue sea")]).unwrap()],
        )
        .unwrap()
        .save(&forward_path)
        .unwrap();
        let path = forward_path.to_str().unwrap();
        for (options, expected) in [
            (vec!["--limit", "1"], "requires --document"),
            (vec!["--document", "-1"], "not an unsigned integer"),
            (vec!["--document", "1"], "does not exist"),
            (
                vec!["--document", "0", "--limit", "0"],
                "between 1 and 1000",
            ),
            (
                vec!["--document", "0", "--limit", "1001"],
                "between 1 and 1000",
            ),
            (vec!["--document", "0", "--limit", "invalid"], "wrong type"),
        ] {
            let mut arguments = vec!["forward-inspect", "--index", path];
            arguments.extend(options);
            let error = execute(arguments, Vec::new()).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
        std::fs::remove_file(forward_path).unwrap();
    }

    #[test]
    fn search_cli_combines_phrase_field_and_exact_filter_without_leaking_other_hits() {
        let index_path = temp_path("search-constraints.idx");
        let mut builder = crate::index::IndexBuilder::new(Analyzer::default());
        for (id, title, body, category) in [
            ("D1", "Blue\tSea", "blue sea", "guide"),
            ("D2", "Blue River", "blue river sea", "guide"),
            ("D3", "Sea Blue", "sea blue", "reference"),
        ] {
            builder
                .add_document(
                    Document::from_fields(
                        id,
                        [("title", title), ("body", body), ("category", category)],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        builder.finish().save(&index_path).unwrap();
        let path = index_path.to_str().unwrap();
        let mut output = Vec::new();
        execute(
            [
                "search",
                "--index",
                path,
                "--query",
                "blue sea",
                "--operator",
                "and",
                "--field",
                "body",
                "--phrase",
                "blue sea",
                "--phrase-field",
                "body",
                "--filter",
                "category=guide",
                "--strategy",
                "max-score",
                "--top-k",
                "3",
            ],
            &mut output,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("D1\tBlue Sea"));
        assert!(!output.contains("D2\t"));
        assert!(!output.contains("D3\t"));

        for (options, expected) in [
            (vec!["--phrase-field", "body"], "requires --phrase"),
            (vec!["--filter", "category"], "NAME=VALUE"),
            (vec!["--operator", "xor"], "unknown operator"),
            (vec!["--strategy", "unknown"], "unknown strategy"),
            (vec!["--top-k", "0"], "greater than zero"),
            (vec!["--k1", "0"], "BM25 k1"),
            (vec!["--b", "2"], "BM25 b"),
        ] {
            let mut arguments = vec!["search", "--index", path, "--query", "blue"];
            arguments.extend(options);
            let error = execute(arguments, Vec::new()).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
        std::fs::remove_file(index_path).unwrap();
    }
}
