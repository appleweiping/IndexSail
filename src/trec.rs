//! Local adapters for common TREC collection, topic, qrels, and run workflows.
//!
//! The adapters intentionally support a documented, deterministic subset of
//! the historical SGML formats instead of pretending to be a general SGML
//! parser. TSV topics are also accepted for generated experiments.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::analysis::Analyzer;
use crate::document::Document;
use crate::error::{Error, Result};
use crate::index::{IndexBuilder, InvertedIndex};

const MAX_TREC_RECORD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Topic {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Qrels {
    judgments: BTreeMap<String, BTreeMap<String, i32>>,
}

impl Qrels {
    pub fn relevance(&self, topic_id: &str, document_id: &str) -> Option<i32> {
        self.judgments
            .get(topic_id)
            .and_then(|documents| documents.get(document_id))
            .copied()
    }

    pub fn relevant_count(&self, topic_id: &str) -> usize {
        self.judgments.get(topic_id).map_or(0, |documents| {
            documents.values().filter(|&&value| value > 0).count()
        })
    }

    pub fn judgments_for(&self, topic_id: &str) -> Option<&BTreeMap<String, i32>> {
        self.judgments.get(topic_id)
    }

    pub fn topic_count(&self) -> usize {
        self.judgments.len()
    }
}

/// Load either `topic-id<TAB>query text` records or classic `<top>` records
/// with `<num> Number: ...` and `<title> ...` lines.
pub fn load_topics(path: impl AsRef<Path>) -> Result<Vec<Topic>> {
    let mut content = String::new();
    BufReader::new(File::open(path)?).read_to_string(&mut content)?;
    if content
        .lines()
        .any(|line| line.trim().eq_ignore_ascii_case("<top>"))
    {
        parse_sgml_topics(&content)
    } else {
        parse_tsv_topics(&content)
    }
}

/// Load standard whitespace-separated qrels: `topic iteration docid relevance`.
pub fn load_qrels(path: impl AsRef<Path>) -> Result<Qrels> {
    let reader = BufReader::new(File::open(path)?);
    parse_qrels(reader)
}

/// Stream a deterministic subset of TREC SGML collections into an index.
///
/// `<DOC>` and `</DOC>` markers must be on their own lines. Each record needs
/// one `<DOCNO>`. Repeated `<TEXT>`/`<BODY>` sections are concatenated; title
/// uses `<TITLE>` and `<HEADLINE>` sections. Unknown markup is stripped.
pub fn index_trec_collection(path: impl AsRef<Path>, analyzer: Analyzer) -> Result<InvertedIndex> {
    let reader = BufReader::new(File::open(path)?);
    index_trec_reader(reader, analyzer)
}

fn parse_tsv_topics(content: &str) -> Result<Vec<Topic>> {
    let mut topics = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let (id, text) = line.split_once('\t').ok_or_else(|| {
            Error::InvalidArgument(format!(
                "topic line {} must use topic-id<TAB>query text",
                line_index + 1
            ))
        })?;
        push_topic(&mut topics, &mut ids, id, text, line_index + 1)?;
    }
    if topics.is_empty() {
        return Err(Error::InvalidArgument(
            "topic file contains no topics".into(),
        ));
    }
    Ok(topics)
}

fn parse_sgml_topics(content: &str) -> Result<Vec<Topic>> {
    let mut topics = Vec::new();
    let mut ids = BTreeSet::new();
    let mut in_topic = false;
    let mut number: Option<String> = None;
    let mut title: Option<String> = None;
    let mut start_line = 0;
    for (line_index, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("<top>") {
            if in_topic {
                return Err(Error::InvalidArgument(format!(
                    "nested <top> at topic line {}",
                    line_index + 1
                )));
            }
            in_topic = true;
            number = None;
            title = None;
            start_line = line_index + 1;
        } else if trimmed.eq_ignore_ascii_case("</top>") {
            if !in_topic {
                return Err(Error::InvalidArgument(format!(
                    "unexpected </top> at topic line {}",
                    line_index + 1
                )));
            }
            let id = number.take().ok_or_else(|| {
                Error::InvalidArgument(format!("topic at line {start_line} has no <num>"))
            })?;
            let text = title.take().ok_or_else(|| {
                Error::InvalidArgument(format!("topic at line {start_line} has no <title>"))
            })?;
            push_topic(&mut topics, &mut ids, &id, &text, start_line)?;
            in_topic = false;
        } else if in_topic {
            if let Some(value) = strip_prefix_ascii_case(trimmed, "<num>") {
                let value = strip_prefix_ascii_case(value.trim(), "Number:")
                    .unwrap_or(value)
                    .trim();
                number = Some(strip_optional_closing_tag(value, "num").trim().to_owned());
            } else if let Some(value) = strip_prefix_ascii_case(trimmed, "<title>") {
                title = Some(strip_optional_closing_tag(value, "title").trim().to_owned());
            }
        }
    }
    if in_topic {
        return Err(Error::InvalidArgument(format!(
            "unterminated <top> beginning at topic line {start_line}"
        )));
    }
    if topics.is_empty() {
        return Err(Error::InvalidArgument(
            "topic file contains no topics".into(),
        ));
    }
    Ok(topics)
}

fn push_topic(
    topics: &mut Vec<Topic>,
    ids: &mut BTreeSet<String>,
    id: &str,
    text: &str,
    line: usize,
) -> Result<()> {
    let id = id.trim();
    let text = text.trim();
    if !valid_trec_token(id) {
        return Err(Error::InvalidArgument(format!(
            "topic id at line {line} must be non-empty and contain no whitespace or controls"
        )));
    }
    if text.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "topic '{id}' at line {line} has an empty query"
        )));
    }
    if !ids.insert(id.to_owned()) {
        return Err(Error::InvalidArgument(format!("duplicate topic id '{id}'")));
    }
    topics.push(Topic {
        id: id.to_owned(),
        text: text.to_owned(),
    });
    Ok(())
}

pub(crate) fn parse_qrels(reader: impl BufRead) -> Result<Qrels> {
    let mut judgments = BTreeMap::<String, BTreeMap<String, i32>>::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let columns = trimmed.split_whitespace().collect::<Vec<_>>();
        if columns.len() != 4 {
            return Err(Error::InvalidArgument(format!(
                "qrels line {} has {} columns; expected 4",
                line_index + 1,
                columns.len()
            )));
        }
        if !valid_trec_token(columns[0]) || !valid_trec_token(columns[2]) {
            return Err(Error::InvalidArgument(format!(
                "qrels topic/document id at line {} contains unsupported characters",
                line_index + 1
            )));
        }
        let relevance = columns[3].parse::<i32>().map_err(|_| {
            Error::InvalidArgument(format!(
                "qrels relevance '{}' at line {} is not an integer",
                columns[3],
                line_index + 1
            ))
        })?;
        if relevance > 31 {
            return Err(Error::InvalidArgument(format!(
                "qrels relevance {relevance} at line {} exceeds supported maximum 31",
                line_index + 1
            )));
        }
        let documents = judgments.entry(columns[0].to_owned()).or_default();
        if documents.insert(columns[2].to_owned(), relevance).is_some() {
            return Err(Error::InvalidArgument(format!(
                "duplicate qrels judgment for topic '{}' and document '{}'",
                columns[0], columns[2]
            )));
        }
    }
    if judgments.is_empty() {
        return Err(Error::InvalidArgument(
            "qrels file contains no judgments".into(),
        ));
    }
    Ok(Qrels { judgments })
}

fn index_trec_reader(reader: impl BufRead, analyzer: Analyzer) -> Result<InvertedIndex> {
    let mut builder = IndexBuilder::new(analyzer);
    let mut in_document = false;
    let mut record = String::new();
    let mut record_start = 0;
    let mut document_count = 0_usize;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("<DOC>") {
            if in_document {
                return Err(Error::InvalidDocument(format!(
                    "nested <DOC> at collection line {}",
                    line_index + 1
                )));
            }
            in_document = true;
            record.clear();
            record_start = line_index + 1;
        } else if trimmed.eq_ignore_ascii_case("</DOC>") {
            if !in_document {
                return Err(Error::InvalidDocument(format!(
                    "unexpected </DOC> at collection line {}",
                    line_index + 1
                )));
            }
            let document = parse_trec_document(&record, record_start)?;
            builder.add_document(document)?;
            document_count += 1;
            in_document = false;
        } else if in_document {
            let extra = line.len().saturating_add(1);
            if record.len().saturating_add(extra) > MAX_TREC_RECORD_BYTES {
                return Err(Error::InvalidDocument(format!(
                    "TREC record at line {record_start} exceeds {MAX_TREC_RECORD_BYTES} bytes"
                )));
            }
            record.push_str(&line);
            record.push('\n');
        } else if !trimmed.is_empty() {
            return Err(Error::InvalidDocument(format!(
                "content outside <DOC> at collection line {}",
                line_index + 1
            )));
        }
    }
    if in_document {
        return Err(Error::InvalidDocument(format!(
            "unterminated <DOC> beginning at collection line {record_start}"
        )));
    }
    if document_count == 0 {
        return Err(Error::InvalidDocument(
            "TREC collection contains no documents".into(),
        ));
    }
    Ok(builder.finish())
}

fn parse_trec_document(record: &str, line: usize) -> Result<Document> {
    let ids = extract_tag_values(record, "DOCNO");
    if ids.len() != 1 {
        return Err(Error::InvalidDocument(format!(
            "TREC record at line {line} must contain exactly one <DOCNO>"
        )));
    }
    let id = collapse_whitespace(&strip_markup(ids[0]));
    if !valid_trec_token(&id) {
        return Err(Error::InvalidDocument(format!(
            "TREC DOCNO at line {line} must contain one non-whitespace, non-control token"
        )));
    }
    let mut title_parts = extract_tag_values(record, "TITLE");
    title_parts.extend(extract_tag_values(record, "HEADLINE"));
    let mut body_parts = extract_tag_values(record, "TEXT");
    body_parts.extend(extract_tag_values(record, "BODY"));
    let title = normalize_sections(&title_parts);
    let body = normalize_sections(&body_parts);
    if title.is_empty() && body.is_empty() {
        return Err(Error::InvalidDocument(format!(
            "TREC document '{id}' has no TITLE, HEADLINE, TEXT, or BODY content"
        )));
    }
    Document::from_fields(id, [("title", title), ("body", body)])
}

fn extract_tag_values<'a>(input: &'a str, tag: &str) -> Vec<&'a str> {
    let uppercase = input.to_ascii_uppercase();
    let opening = format!("<{tag}>");
    let closing = format!("</{tag}>");
    let mut values = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = uppercase[cursor..].find(&opening) {
        let start = cursor + relative_start + opening.len();
        let Some(relative_end) = uppercase[start..].find(&closing) else {
            break;
        };
        let end = start + relative_end;
        values.push(&input[start..end]);
        cursor = end + closing.len();
    }
    values
}

fn normalize_sections(sections: &[&str]) -> String {
    sections
        .iter()
        .map(|section| collapse_whitespace(&strip_markup(section)))
        .filter(|section| !section.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_markup(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut inside_tag = false;
    for character in input.chars() {
        match character {
            '<' => inside_tag = true,
            '>' => inside_tag = false,
            _ if !inside_tag => output.push(character),
            _ => {}
        }
    }
    decode_named_entities(&output)
}

fn decode_named_entities(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut remaining = input;
    while !remaining.is_empty() {
        let matched = [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ]
        .into_iter()
        .find(|(entity, _)| remaining.starts_with(entity));
        if let Some((entity, value)) = matched {
            output.push(value);
            remaining = &remaining[entity.len()..];
        } else {
            let character = remaining
                .chars()
                .next()
                .expect("remaining input is nonempty");
            output.push(character);
            remaining = &remaining[character.len_utf8()..];
        }
    }
    output
}

fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_prefix_ascii_case<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    value
        .get(..prefix.len())
        .filter(|candidate| candidate.eq_ignore_ascii_case(prefix))?;
    value.get(prefix.len()..)
}

fn strip_optional_closing_tag<'a>(value: &'a str, tag: &str) -> &'a str {
    let uppercase = value.to_ascii_uppercase();
    let closing = format!("</{}>", tag.to_ascii_uppercase());
    uppercase
        .find(&closing)
        .and_then(|index| value.get(..index))
        .unwrap_or(value)
}

fn valid_trec_token(value: &str) -> bool {
    !value.is_empty()
        && !value.chars().any(char::is_whitespace)
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn parses_tsv_and_classic_sgml_topics() {
        let tsv =
            parse_tsv_topics("301\tInternational Organized Crime\n302\tPoliomyelitis\n").unwrap();
        assert_eq!(tsv[0].id, "301");
        assert_eq!(tsv[1].text, "Poliomyelitis");

        let sgml = parse_sgml_topics(
            "<top>\n<num> Number: 301</num>\n<title> International Organized Crime</title>\n<desc> ignored\n</top>\n",
        )
        .unwrap();
        assert_eq!(
            sgml,
            vec![Topic {
                id: "301".into(),
                text: "International Organized Crime".into()
            }]
        );
    }

    #[test]
    fn rejects_duplicate_or_malformed_topics() {
        assert!(parse_tsv_topics("1\tone\n1\ttwo\n").is_err());
        assert!(parse_tsv_topics("missing-tab\n").is_err());
        assert!(parse_sgml_topics("<top>\n<num> 1\n</top>\n").is_err());
    }

    #[test]
    fn parses_qrels_and_rejects_ambiguity() {
        let qrels =
            parse_qrels(Cursor::new("301 0 DOC1 2\n301 Q0 DOC2 0\n302 0 DOC3 -1\n")).unwrap();
        assert_eq!(qrels.relevance("301", "DOC1"), Some(2));
        assert_eq!(qrels.relevant_count("301"), 1);
        assert_eq!(qrels.topic_count(), 2);
        assert!(parse_qrels(Cursor::new("1 0 A 1\n1 0 A 0\n")).is_err());
        assert!(parse_qrels(Cursor::new("1 0 A 32\n")).is_err());
    }

    #[test]
    fn streams_trec_records_into_searchable_fields() {
        let collection = "\
<DOC>\n\
<DOCNO> DOC-1 </DOCNO>\n\
<HEADLINE>Local <B>search</B></HEADLINE>\n\
<TEXT>BM25 &amp; exact ranking.</TEXT>\n\
</DOC>\n\
<DOC>\n\
<DOCNO>DOC-2</DOCNO>\n\
<TITLE>Grid</TITLE>\n\
<BODY>power model</BODY>\n\
</DOC>\n";
        let index = index_trec_reader(Cursor::new(collection), Analyzer::default()).unwrap();
        assert_eq!(index.stats().documents, 2);
        assert_eq!(index.documents()[0].external_id(), "DOC-1");
        assert_eq!(index.documents()[0].field("title"), Some("Local search"));
        assert_eq!(
            index.documents()[0].field("body"),
            Some("BM25 & exact ranking.")
        );
        assert!(index.postings("body", "bm25").is_some());
    }

    #[test]
    fn collection_adapter_rejects_bad_boundaries_and_ids() {
        assert!(index_trec_reader(Cursor::new("outside\n"), Analyzer::default()).is_err());
        assert!(
            index_trec_reader(
                Cursor::new("<DOC>\n<DOCNO>two words</DOCNO>\n<TEXT>x</TEXT>\n</DOC>\n"),
                Analyzer::default()
            )
            .is_err()
        );
        assert!(
            index_trec_reader(
                Cursor::new("<DOC>\n<DOCNO>x</DOCNO>\n"),
                Analyzer::default()
            )
            .is_err()
        );
    }

    #[test]
    fn entity_decoding_is_single_pass() {
        assert_eq!(strip_markup("A &amp; B &lt; C"), "A & B < C");
        assert_eq!(strip_markup("&amp;lt;"), "&lt;");
    }
}
