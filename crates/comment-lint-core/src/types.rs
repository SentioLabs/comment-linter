//! Core enumerations for language identification and comment classification.

use serde::{Deserialize, Serialize};

/// Identifies a supported programming language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageId {
    Go,
    Python,
    TypeScript,
    JavaScript,
    Rust,
}

impl LanguageId {
    /// Returns the lowercase name of the language.
    pub fn name(&self) -> &'static str {
        match self {
            LanguageId::Go => "go",
            LanguageId::Python => "python",
            LanguageId::TypeScript => "typescript",
            LanguageId::JavaScript => "javascript",
            LanguageId::Rust => "rust",
        }
    }
}

/// Classifies the syntactic kind of a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommentKind {
    /// Single-line comment (e.g. `//`, `#`)
    Line,
    /// Multi-line block comment (e.g. `/* ... */`)
    Block,
    /// Documentation comment (e.g. `///`, `/** ... */`, `"""..."""`)
    Doc,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_id_name_returns_lowercase_string() {
        assert_eq!(LanguageId::Go.name(), "go");
        assert_eq!(LanguageId::Python.name(), "python");
        assert_eq!(LanguageId::TypeScript.name(), "typescript");
        assert_eq!(LanguageId::JavaScript.name(), "javascript");
        assert_eq!(LanguageId::Rust.name(), "rust");
    }

    #[test]
    fn language_id_serde_roundtrip() {
        for lang in [
            LanguageId::Go,
            LanguageId::Python,
            LanguageId::TypeScript,
            LanguageId::JavaScript,
            LanguageId::Rust,
        ] {
            let json = serde_json::to_string(&lang).unwrap();
            let deserialized: LanguageId = serde_json::from_str(&json).unwrap();
            assert_eq!(lang, deserialized);
        }
    }

    #[test]
    fn comment_kind_serde_roundtrip() {
        for kind in [CommentKind::Line, CommentKind::Block, CommentKind::Doc] {
            let json = serde_json::to_string(&kind).unwrap();
            let deserialized: CommentKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, deserialized);
        }
    }
}
