use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// A document with a stable external identifier and named text fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Document {
    external_id: String,
    fields: BTreeMap<String, String>,
}

impl Document {
    pub fn new(external_id: impl Into<String>) -> Result<Self> {
        let external_id = external_id.into();
        if external_id.trim().is_empty() {
            return Err(Error::InvalidDocument(
                "external id must not be empty".into(),
            ));
        }
        if external_id.contains(['\n', '\r', '\0']) {
            return Err(Error::InvalidDocument(
                "external id must not contain control separators".into(),
            ));
        }
        Ok(Self {
            external_id,
            fields: BTreeMap::new(),
        })
    }

    pub fn from_fields<I, K, V>(external_id: impl Into<String>, fields: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut document = Self::new(external_id)?;
        for (name, value) in fields {
            document.insert_field(name, value)?;
        }
        Ok(document)
    }

    pub fn insert_field(
        &mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Option<String>> {
        let name = name.into();
        validate_field_name(&name)?;
        Ok(self.fields.insert(name, value.into()))
    }

    pub fn external_id(&self) -> &str {
        &self.external_id
    }

    pub fn fields(&self) -> &BTreeMap<String, String> {
        &self.fields
    }

    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

pub(crate) fn validate_field_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidDocument(
            "field name must not be empty".into(),
        ));
    }
    if name.len() > 128 {
        return Err(Error::InvalidDocument(
            "field name exceeds 128 bytes".into(),
        ));
    }
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(Error::InvalidDocument(format!(
            "field name '{name}' contains unsupported characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_document_and_reads_fields() {
        let document = Document::from_fields(
            "doc-1",
            [("title", "Sailing Search"), ("body", "A compact engine")],
        )
        .unwrap();
        assert_eq!(document.external_id(), "doc-1");
        assert_eq!(document.field("title"), Some("Sailing Search"));
        assert_eq!(document.field("missing"), None);
    }

    #[test]
    fn rejects_blank_external_id() {
        assert!(matches!(
            Document::new("  "),
            Err(Error::InvalidDocument(_))
        ));
    }

    #[test]
    fn rejects_external_id_with_line_break() {
        assert!(Document::new("one\ntwo").is_err());
    }

    #[test]
    fn accepts_deliberately_limited_field_identifier_grammar() {
        for field in ["title", "title_en", "meta.category", "field-2"] {
            assert!(validate_field_name(field).is_ok());
        }
    }

    #[test]
    fn rejects_invalid_field_identifiers() {
        for field in ["", "two words", "标题", "a/b"] {
            assert!(validate_field_name(field).is_err(), "accepted {field:?}");
        }
    }

    #[test]
    fn inserting_same_field_replaces_value_explicitly() {
        let mut document = Document::new("doc").unwrap();
        assert_eq!(document.insert_field("title", "old").unwrap(), None);
        assert_eq!(
            document.insert_field("title", "new").unwrap(),
            Some("old".into())
        );
        assert_eq!(document.field("title"), Some("new"));
    }

    #[test]
    fn from_fields_is_deterministically_ordered() {
        let document = Document::from_fields("doc", [("z", "last"), ("a", "first")]).unwrap();
        assert_eq!(
            document
                .fields()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["a", "z"]
        );
    }
}
