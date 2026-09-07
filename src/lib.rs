//! `IndexSail` is a compact, deterministic local search engine.
//!
//! It provides a deterministic analyzer, a positional inverted index, BM25
//! ranking, Boolean queries, phrase and exact-field filters, exhaustive and
//! exact WAND top-k execution, explanations, TREC-style batch evaluation, and
//! a checksummed, delta/variable-byte version 3 format with persisted block
//! bounds and version 1/2 read compatibility.

pub mod analysis;
pub mod benchmark;
pub mod cli;
pub mod codec;
pub mod document;
pub mod error;
pub mod evaluation;
pub mod index;
pub mod persistence;
pub mod query;
pub mod search;
pub mod trec;

pub use analysis::{AnalysisMode, Analyzer, Token};
pub use codec::PostingCodecStats;
pub use document::Document;
pub use error::{Error, Result};
pub use evaluation::{
    AggregateMetrics, BatchConfig, BatchReport, QueryMetrics, QueryReport, evaluate_batch,
    write_json_report, write_trec_run,
};
pub use index::{IndexBuilder, IndexStats, InvertedIndex, Posting};
pub use query::{BooleanOperator, FieldFilter, PhraseFilter, QueryTerm, SearchQuery};
pub use search::{
    Bm25Params, Explanation, PruningStrategy, SearchHit, SearchOptions, SearchOutcome, SearchStats,
    TermContribution,
};
pub use trec::{Qrels, Topic, index_trec_collection, load_qrels, load_topics};
