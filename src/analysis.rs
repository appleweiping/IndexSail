/// Tokenization behavior used at both indexing and query time.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AnalysisMode {
    /// Split on non-Unicode-alphanumeric characters and apply Unicode lowercase.
    #[default]
    Unicode,
    /// Keep only ASCII letters and digits, splitting on every other character.
    Ascii,
}

impl AnalysisMode {
    pub(crate) const fn wire_value(self) -> u8 {
        match self {
            Self::Unicode => 0,
            Self::Ascii => 1,
        }
    }

    pub(crate) fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unicode),
            1 => Some(Self::Ascii),
            _ => None,
        }
    }
}

/// A normalized token and its ordinal position in a field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token {
    pub text: String,
    pub position: u32,
}

/// A deliberately small and deterministic tokenizer.
///
/// Unicode mode uses the standard library's alphanumeric classification and
/// lowercase mapping. It does not perform NFKC/NFC normalization or stemming.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Analyzer {
    mode: AnalysisMode,
}

impl Analyzer {
    pub const fn new(mode: AnalysisMode) -> Self {
        Self { mode }
    }

    pub const fn mode(self) -> AnalysisMode {
        self.mode
    }

    pub fn analyze(self, input: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for character in input.chars() {
            let accepted = match self.mode {
                AnalysisMode::Unicode => character.is_alphanumeric(),
                AnalysisMode::Ascii => character.is_ascii_alphanumeric(),
            };
            if accepted {
                match self.mode {
                    AnalysisMode::Unicode => current.extend(character.to_lowercase()),
                    AnalysisMode::Ascii => current.push(character.to_ascii_lowercase()),
                }
            } else if !current.is_empty() {
                let position = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
                tokens.push(Token {
                    text: std::mem::take(&mut current),
                    position,
                });
            }
        }

        if !current.is_empty() {
            let position = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
            tokens.push(Token {
                text: current,
                position,
            });
        }
        tokens
    }

    pub fn normalize_single(self, input: &str) -> Option<String> {
        let mut tokens = self.analyze(input);
        (tokens.len() == 1).then(|| tokens.remove(0).text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_mode_lowercases_and_splits() {
        let tokens = Analyzer::default().analyze("Rust, SEARCH 2026!");
        assert_eq!(
            tokens,
            vec![
                Token {
                    text: "rust".into(),
                    position: 0
                },
                Token {
                    text: "search".into(),
                    position: 1
                },
                Token {
                    text: "2026".into(),
                    position: 2
                },
            ]
        );
    }

    #[test]
    fn unicode_mode_preserves_non_ascii_letters() {
        let words: Vec<_> = Analyzer::default()
            .analyze("电力 Système Модель")
            .into_iter()
            .map(|token| token.text)
            .collect();
        assert_eq!(words, ["电力", "système", "модель"]);
    }

    #[test]
    fn ascii_mode_splits_non_ascii_text() {
        let words: Vec<_> = Analyzer::new(AnalysisMode::Ascii)
            .analyze("café-RUST")
            .into_iter()
            .map(|token| token.text)
            .collect();
        assert_eq!(words, ["caf", "rust"]);
    }

    #[test]
    fn positions_are_token_ordinals() {
        let positions: Vec<_> = Analyzer::default()
            .analyze("one...two / three")
            .into_iter()
            .map(|token| token.position)
            .collect();
        assert_eq!(positions, [0, 1, 2]);
    }

    #[test]
    fn empty_and_punctuation_only_inputs_have_no_tokens() {
        assert!(Analyzer::default().analyze("").is_empty());
        assert!(Analyzer::default().analyze("—!?.").is_empty());
    }

    #[test]
    fn normalize_single_rejects_zero_or_multiple_tokens() {
        let analyzer = Analyzer::default();
        assert_eq!(analyzer.normalize_single("Rust"), Some("rust".into()));
        assert_eq!(analyzer.normalize_single("two words"), None);
        assert_eq!(analyzer.normalize_single("---"), None);
    }

    #[test]
    fn digits_remain_part_of_tokens() {
        let words: Vec<_> = Analyzer::default()
            .analyze("bm25 c3po")
            .into_iter()
            .map(|token| token.text)
            .collect();
        assert_eq!(words, ["bm25", "c3po"]);
    }

    #[test]
    fn wire_modes_round_trip_and_reject_unknown_values() {
        for mode in [AnalysisMode::Unicode, AnalysisMode::Ascii] {
            assert_eq!(AnalysisMode::from_wire(mode.wire_value()), Some(mode));
        }
        assert_eq!(AnalysisMode::from_wire(9), None);
    }
}
