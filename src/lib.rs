//! `IndexSail` is a compact, deterministic local search engine.
//!
//! It provides a deterministic analyzer, a positional inverted index, BM25
//! ranking, Boolean queries, phrase and exact-field filters, exhaustive and
//! exact WAND top-k execution, explanations, TREC-style batch evaluation, and
//! a checksummed, delta/variable-byte version 3 format with persisted block
//! bounds and version 1/2 read compatibility. Collections can also be split
//! into deterministic physical shards while retaining exact global BM25
//! scores and stable cross-shard top-k merging. The CIFF v1 surface provides
//! bounded canonical index interchange plus direct exhaustive BM25 and TREC
//! evaluation over portable files.

pub mod analysis;
mod atomic;
pub mod benchmark;
pub mod ciff;
pub mod cli;
pub mod codec;
pub mod document;
pub mod error;
pub mod evaluation;
pub mod index;
pub mod persistence;
pub mod query;
pub mod search;
pub mod shard;
pub mod trec;

pub use analysis::{AnalysisMode, Analyzer, Token};
pub use ciff::{
    CIFF_FORMAT_VERSION, CiffDocumentRecord, CiffHeader, CiffIndex, CiffLimits, CiffPosting,
    CiffPostingList, CiffRetrieval, CiffSearchOptions, CiffStats,
};
pub use codec::PostingCodecStats;
pub use document::Document;
pub use error::{Error, Result};
pub use evaluation::{
    AggregateMetrics, BatchConfig, BatchReport, QueryMetrics, QueryReport, RetrievalBackend,
    evaluate_batch, write_json_report, write_trec_run,
};
pub use index::{IndexBuilder, IndexStats, InvertedIndex, Posting};
pub use query::{BooleanOperator, FieldFilter, PhraseFilter, QueryTerm, SearchQuery};
pub use search::{
    Bm25Params, Explanation, PruningStrategy, SearchHit, SearchOptions, SearchOutcome, SearchStats,
    TermContribution,
};
pub use shard::{
    PhysicalShardStats, SHARDED_PERSISTENCE_FORMAT_VERSION, ShardedIndex, ShardedIndexBuilder,
    persisted_sharded_format_version,
};
pub use trec::{
    Qrels, Topic, index_trec_collection, index_trec_collection_sharded, load_qrels, load_topics,
};
