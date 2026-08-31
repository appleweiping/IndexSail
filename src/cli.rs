use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::analysis::{AnalysisMode, Analyzer};
use crate::benchmark::{BenchmarkConfig, run as run_benchmark};
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::{IndexBuilder, InvertedIndex};
use crate::query::{BooleanOperator, FieldFilter, PhraseFilter, SearchQuery};
use crate::search::{Bm25Params, PruningStrategy, SearchOptions};

pub const HELP: &str = "\
IndexSail — compact local BM25 search\n\
\n\
USAGE:\n\
  indexsail index --input CORPUS.tsv --output INDEX.idx [--ascii]\n\
  indexsail search --index INDEX.idx --query TEXT [OPTIONS]\n\
  indexsail inspect --index INDEX.idx [--field NAME --term TERM]\n\
  indexsail benchmark [--documents N] [--queries N] [--seed N] [--top-k N]\n\
\n\
SEARCH OPTIONS:\n\
  --field NAME             Restrict all query terms to one field\n\
  --operator and|or        Boolean term semantics (default: or)\n\
  --phrase TEXT            Require an adjacent phrase\n\
  --phrase-field NAME      Restrict the phrase to one field\n\
  --filter NAME=VALUE      Require an exact stored field value; repeatable\n\
  --top-k N                Number of hits (default: 10)\n\
  --strategy wand|full     Safe WAND pruning or exhaustive evaluation\n\
  --k1 NUMBER --b NUMBER   BM25 parameters\n\
  --explain                Print per-term BM25 contributions\n";

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
        "index" => command_index(&remaining, &mut output),
        "search" => command_search(&remaining, &mut output),
        "inspect" => command_inspect(&remaining, &mut output),
        "benchmark" => command_benchmark(&remaining, &mut output),
        unknown => Err(Error::InvalidArgument(format!(
            "unknown command '{unknown}'; run 'indexsail help'"
        ))),
    }
}

fn command_index(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &["--ascii"], &["--input", "--output"])?;
    let input = parsed.required_one("--input")?;
    let destination = parsed.required_one("--output")?;
    let mode = if parsed.flag("--ascii") {
        AnalysisMode::Ascii
    } else {
        AnalysisMode::Unicode
    };
    let index = index_tsv(input, Analyzer::new(mode))?;
    index.save(destination)?;
    let stats = index.stats();
    writeln!(
        output,
        "indexed documents={} fields={} terms={} postings={} tokens={} output={}",
        stats.documents, stats.fields, stats.terms, stats.postings, stats.tokens, destination
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn command_search(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
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
    )?;
    let index_path = parsed.required_one("--index")?;
    let query_text = parsed.required_one("--query")?;
    let index = InvertedIndex::load(index_path)?;
    let field = parsed.optional_one("--field")?;
    let mut query = SearchQuery::from_text(index.analyzer(), query_text, field)?;
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
        query = query.with_phrase(PhraseFilter::from_text(
            index.analyzer(),
            phrase,
            phrase_field,
        )?);
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
        "full" | "exhaustive" => PruningStrategy::Exhaustive,
        value => {
            return Err(Error::InvalidArgument(format!(
                "unknown strategy '{value}', expected wand or full"
            )));
        }
    };
    let outcome = index.search(
        &query,
        SearchOptions {
            top_k,
            pruning,
            explain: parsed.flag("--explain"),
            bm25: Bm25Params { k1, b },
        },
    )?;

    writeln!(output, "rank\tscore\tid\ttitle")?;
    for (rank, hit) in outcome.hits.iter().enumerate() {
        let document = index.document(hit.doc_id).expect("hit document exists");
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
        "strategy={pruning:?} evaluated={} advanced={} skipped={}",
        outcome.stats.evaluated_candidates,
        outcome.stats.postings_advanced,
        outcome.stats.postings_skipped
    )?;
    Ok(())
}

fn command_inspect(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(arguments, &[], &["--index", "--field", "--term"])?;
    let path = parsed.required_one("--index")?;
    let index = InvertedIndex::load(path)?;
    let stats = index.stats();
    writeln!(
        output,
        "format=IndexSail-v1 analyzer={:?}",
        index.analyzer().mode()
    )?;
    writeln!(
        output,
        "documents={} fields={} terms={} postings={} tokens={}",
        stats.documents, stats.fields, stats.terms, stats.postings, stats.tokens
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

fn command_benchmark(arguments: &[String], output: &mut impl Write) -> Result<()> {
    let parsed = ParsedOptions::parse(
        arguments,
        &[],
        &["--documents", "--queries", "--seed", "--top-k"],
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
    };
    let report = run_benchmark(config)?;
    writeln!(
        output,
        "config documents={} queries={} top_k={} seed={}",
        config.documents, config.queries, config.top_k, config.seed
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
    writeln!(output, "verified=true checksum={:016x}", report.checksum)?;
    Ok(())
}

fn index_tsv(path: impl AsRef<Path>, analyzer: Analyzer) -> Result<InvertedIndex> {
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

    let mut builder = IndexBuilder::new(analyzer);
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
        builder.add_document(Document::from_fields(values[0], fields)?)?;
    }
    Ok(builder.finish())
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
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("indexsail search")
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
}
