//! Common Index File Format (CIFF) v1 interoperability.
//!
//! The implementation follows the OSIRRC CIFF v1 protobuf schema and its
//! delimited-message framing. It is intentionally dependency-light, applies
//! resource limits before collection-sized allocation, preserves the explicit
//! global statistics used for reproducible BM25 scoring of frequency indexes,
//! preserves learned-sparse impact payloads, and writes canonical deterministic
//! wire bytes.

mod framing;
mod model;
mod wire;

pub use model::{
    CIFF_FORMAT_VERSION, CiffDocumentRecord, CiffHeader, CiffIndex, CiffLimits, CiffPosting,
    CiffPostingList, CiffRetrieval, CiffSearchOptions, CiffStats,
};
