use std::fmt::{Display, Formatter};
use std::io;

/// Errors returned by `IndexSail`'s library and command-line interface.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    InvalidDocument(String),
    DuplicateDocumentId(String),
    InvalidQuery(String),
    InvalidArgument(String),
    CorruptIndex(String),
    UnsupportedVersion(u32),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::InvalidDocument(message) => write!(formatter, "invalid document: {message}"),
            Self::DuplicateDocumentId(id) => write!(formatter, "duplicate document id: {id}"),
            Self::InvalidQuery(message) => write!(formatter, "invalid query: {message}"),
            Self::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
            Self::CorruptIndex(message) => write!(formatter, "corrupt index: {message}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported index version: {version}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
