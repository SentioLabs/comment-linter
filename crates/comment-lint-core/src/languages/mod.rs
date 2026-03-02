//! Language trait and language detection utilities.

use std::path::Path;

use crate::extraction::comment::CommentContext;
use crate::types::LanguageId;

pub mod go;
pub mod javascript;
pub mod python;
pub mod rust_lang;
pub mod typescript;

/// Trait implemented by each supported language to provide tree-sitter
/// integration and comment extraction.
pub trait Language {
    /// The identifier for this language.
    fn id(&self) -> LanguageId;

    /// File extensions associated with this language (without leading dot).
    fn extensions(&self) -> &[&str];

    /// The tree-sitter [`Language`](tree_sitter::Language) grammar for parsing.
    fn tree_sitter_language(&self) -> tree_sitter::Language;

    /// Extract all comments from a parsed tree-sitter tree and source code.
    fn extract_comments(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        file_path: &Path,
    ) -> Vec<CommentContext>;
}

/// Return a boxed [`Language`] implementation for the given [`LanguageId`].
pub fn get_language(id: LanguageId) -> Box<dyn Language> {
    match id {
        LanguageId::Go => Box::new(go::GoLanguage),
        LanguageId::Python => Box::new(python::PythonLanguage),
        LanguageId::TypeScript => Box::new(typescript::TypeScriptLanguage),
        LanguageId::JavaScript => Box::new(javascript::JavaScriptLanguage),
        LanguageId::Rust => Box::new(rust_lang::RustLanguage),
    }
}

/// Detect the programming language from a file path's extension.
///
/// Returns `None` if the extension is not recognized.
pub fn detect_language(path: &Path) -> Option<LanguageId> {
    let ext = path.extension()?.to_str()?;
    match ext {
        "go" => Some(LanguageId::Go),
        "py" => Some(LanguageId::Python),
        "ts" | "tsx" => Some(LanguageId::TypeScript),
        "js" | "jsx" | "mjs" | "cjs" => Some(LanguageId::JavaScript),
        "rs" => Some(LanguageId::Rust),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_language_go() {
        assert_eq!(detect_language(Path::new("main.go")), Some(LanguageId::Go));
    }

    #[test]
    fn detect_language_python() {
        assert_eq!(
            detect_language(Path::new("script.py")),
            Some(LanguageId::Python)
        );
    }

    #[test]
    fn detect_language_typescript() {
        assert_eq!(
            detect_language(Path::new("app.ts")),
            Some(LanguageId::TypeScript)
        );
        assert_eq!(
            detect_language(Path::new("app.tsx")),
            Some(LanguageId::TypeScript)
        );
    }

    #[test]
    fn detect_language_javascript() {
        assert_eq!(
            detect_language(Path::new("index.js")),
            Some(LanguageId::JavaScript)
        );
        assert_eq!(
            detect_language(Path::new("index.jsx")),
            Some(LanguageId::JavaScript)
        );
    }

    #[test]
    fn detect_language_rust() {
        assert_eq!(
            detect_language(Path::new("lib.rs")),
            Some(LanguageId::Rust)
        );
    }

    #[test]
    fn detect_language_unknown_returns_none() {
        assert_eq!(detect_language(Path::new("data.csv")), None);
        assert_eq!(detect_language(Path::new("readme.md")), None);
    }
}
