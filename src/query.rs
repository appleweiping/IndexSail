use crate::analysis::Analyzer;
use crate::document::validate_field_name;
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BooleanOperator {
    And,
    #[default]
    Or,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryTerm {
    text: String,
    field: Option<String>,
    boost: f64,
}

impl QueryTerm {
    pub fn new(text: impl Into<String>, field: Option<String>, boost: f64) -> Result<Self> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(Error::InvalidQuery("query term must not be empty".into()));
        }
        if let Some(field) = &field {
            validate_query_field(field)?;
        }
        if !boost.is_finite() || boost <= 0.0 {
            return Err(Error::InvalidQuery(
                "term boost must be finite and greater than zero".into(),
            ));
        }
        Ok(Self { text, field, boost })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn field(&self) -> Option<&str> {
        self.field.as_deref()
    }

    pub const fn boost(&self) -> f64 {
        self.boost
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhraseFilter {
    terms: Vec<String>,
    field: Option<String>,
}

impl PhraseFilter {
    pub fn from_text(analyzer: Analyzer, text: &str, field: Option<String>) -> Result<Self> {
        if let Some(field) = &field {
            validate_query_field(field)?;
        }
        let terms: Vec<_> = analyzer
            .analyze(text)
            .into_iter()
            .map(|token| token.text)
            .collect();
        if terms.is_empty() {
            return Err(Error::InvalidQuery(
                "phrase must contain at least one searchable token".into(),
            ));
        }
        Ok(Self { terms, field })
    }

    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    pub fn field(&self) -> Option<&str> {
        self.field.as_deref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldFilter {
    field: String,
    value: String,
}

impl FieldFilter {
    pub fn exact(field: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let field = field.into();
        validate_query_field(&field)?;
        Ok(Self {
            field,
            value: value.into(),
        })
    }

    pub fn field(&self) -> &str {
        &self.field
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchQuery {
    terms: Vec<QueryTerm>,
    operator: BooleanOperator,
    phrases: Vec<PhraseFilter>,
    filters: Vec<FieldFilter>,
}

impl SearchQuery {
    pub fn from_text(analyzer: Analyzer, text: &str, field: Option<&str>) -> Result<Self> {
        if let Some(field) = &field {
            validate_query_field(field)?;
        }
        let terms: Vec<_> = analyzer
            .analyze(text)
            .into_iter()
            .map(|token| QueryTerm {
                text: token.text,
                field: field.map(str::to_owned),
                boost: 1.0,
            })
            .collect();
        if terms.is_empty() {
            return Err(Error::InvalidQuery(
                "query must contain at least one searchable token".into(),
            ));
        }
        Ok(Self {
            terms,
            operator: BooleanOperator::Or,
            phrases: Vec::new(),
            filters: Vec::new(),
        })
    }

    pub fn from_terms(terms: Vec<QueryTerm>) -> Result<Self> {
        if terms.is_empty() {
            return Err(Error::InvalidQuery(
                "query must contain at least one term".into(),
            ));
        }
        Ok(Self {
            terms,
            operator: BooleanOperator::Or,
            phrases: Vec::new(),
            filters: Vec::new(),
        })
    }

    #[must_use]
    pub const fn with_operator(mut self, operator: BooleanOperator) -> Self {
        self.operator = operator;
        self
    }

    #[must_use]
    pub fn with_phrase(mut self, phrase: PhraseFilter) -> Self {
        self.phrases.push(phrase);
        self
    }

    #[must_use]
    pub fn with_filter(mut self, filter: FieldFilter) -> Self {
        self.filters.push(filter);
        self
    }

    pub fn terms(&self) -> &[QueryTerm] {
        &self.terms
    }

    pub const fn operator(&self) -> BooleanOperator {
        self.operator
    }

    pub fn phrases(&self) -> &[PhraseFilter] {
        &self.phrases
    }

    pub fn filters(&self) -> &[FieldFilter] {
        &self.filters
    }
}

fn validate_query_field(field: &str) -> Result<()> {
    validate_field_name(field).map_err(|error| Error::InvalidQuery(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_query_is_analyzed_consistently() {
        let query = SearchQuery::from_text(Analyzer::default(), "Rust SEARCH", None).unwrap();
        assert_eq!(
            query
                .terms()
                .iter()
                .map(QueryTerm::text)
                .collect::<Vec<_>>(),
            ["rust", "search"]
        );
        assert_eq!(query.operator(), BooleanOperator::Or);
    }

    #[test]
    fn field_is_applied_to_every_analyzed_term() {
        let query =
            SearchQuery::from_text(Analyzer::default(), "local engine", Some("title")).unwrap();
        assert!(
            query
                .terms()
                .iter()
                .all(|term| term.field() == Some("title"))
        );
    }

    #[test]
    fn empty_query_is_rejected() {
        assert!(SearchQuery::from_text(Analyzer::default(), "--", None).is_err());
        assert!(SearchQuery::from_terms(Vec::new()).is_err());
    }

    #[test]
    fn invalid_boosts_are_rejected() {
        for boost in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(QueryTerm::new("term", None, boost).is_err());
        }
        assert!(QueryTerm::new("term", None, 2.5).is_ok());
    }

    #[test]
    fn phrase_uses_normalized_terms_and_optional_field() {
        let phrase =
            PhraseFilter::from_text(Analyzer::default(), "Exact PHRASE", Some("body".into()))
                .unwrap();
        assert_eq!(phrase.terms(), ["exact", "phrase"]);
        assert_eq!(phrase.field(), Some("body"));
    }

    #[test]
    fn punctuation_only_phrase_is_rejected() {
        assert!(PhraseFilter::from_text(Analyzer::default(), "...", None).is_err());
    }

    #[test]
    fn exact_filter_preserves_value_without_analysis() {
        let filter = FieldFilter::exact("category", "Power Systems").unwrap();
        assert_eq!(filter.field(), "category");
        assert_eq!(filter.value(), "Power Systems");
    }

    #[test]
    fn builder_methods_preserve_all_constraints() {
        let phrase = PhraseFilter::from_text(Analyzer::default(), "rust search", None).unwrap();
        let filter = FieldFilter::exact("kind", "guide").unwrap();
        let query = SearchQuery::from_text(Analyzer::default(), "rust", None)
            .unwrap()
            .with_operator(BooleanOperator::And)
            .with_phrase(phrase)
            .with_filter(filter);
        assert_eq!(query.operator(), BooleanOperator::And);
        assert_eq!(query.phrases().len(), 1);
        assert_eq!(query.filters().len(), 1);
    }

    #[test]
    fn invalid_field_name_is_rejected_at_query_boundary() {
        assert!(SearchQuery::from_text(Analyzer::default(), "term", Some("bad field")).is_err());
        assert!(FieldFilter::exact("bad/field", "x").is_err());
    }
}
