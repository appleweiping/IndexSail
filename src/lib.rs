//! `IndexSail` is a compact, dependency-free local search engine.
//!
//! It provides a deterministic analyzer, a positional inverted index, BM25
//! ranking, Boolean queries, phrase and exact-field filters, exhaustive and
//! WAND-style top-k execution, explanations, and a versioned binary format.

pub mod analysis;
pub mod benchmark;
pub mod cli;
pub mod document;
pub mod error;
pub mod index;
pub mod persistence;
pub mod query;
pub mod search;

pub use analysis::{AnalysisMode, Analyzer, Token};
pub use document::Document;
pub use error::{Error, Result};
pub use index::{IndexBuilder, IndexStats, InvertedIndex, Posting};
pub use query::{BooleanOperator, FieldFilter, PhraseFilter, QueryTerm, SearchQuery};
pub use search::{
    Bm25Params, Explanation, PruningStrategy, SearchHit, SearchOptions, SearchOutcome, SearchStats,
    TermContribution,
};
