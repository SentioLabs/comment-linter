//! The primary data structure representing an extracted comment and its context.

use std::path::PathBuf;

use crate::types::{CommentKind, LanguageId};
use serde::{Deserialize, Serialize};

/// Full context of a single comment extracted from source code.
///
/// Carries everything downstream analysis needs: the comment text, its
/// position, the language it was written in, and surrounding source context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentContext {
    /// Path to the source file containing this comment.
    pub file_path: PathBuf,
    /// 1-based line number where the comment starts.
    pub line: usize,
    /// 0-based column offset where the comment starts.
    pub column: usize,
    /// The raw text of the comment (without delimiters where possible).
    pub comment_text: String,
    /// Whether this is a line, block, or doc comment.
    pub comment_kind: CommentKind,
    /// The programming language of the source file.
    pub language: LanguageId,
    /// The tree-sitter node kind of the nearest adjacent declaration.
    pub adjacent_node_kind: String,
    /// A snippet of the source code surrounding the comment.
    pub surrounding_source: String,
    /// Identifiers found near the comment.
    pub nearby_identifiers: Vec<String>,
    /// Language keywords found near the comment.
    pub nearby_keywords: Vec<String>,
}
