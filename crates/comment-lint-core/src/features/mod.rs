//! Feature extraction and scored-comment types for comment quality analysis.

use serde::{Deserialize, Serialize};

pub mod cross_reference;
pub mod lexical;
pub mod semantic;
pub mod structural;

use crate::extraction::comment::CommentContext;

/// A fixed-size vector of features computed from a comment and its context.
///
/// Each field captures one signal that helps determine whether a comment is
/// superfluous.  All floating-point fields are in the range `[0.0, 1.0]`
/// unless otherwise noted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Jaccard similarity between comment tokens and adjacent code tokens.
    pub token_overlap_jaccard: f32,
    /// Ratio of comment tokens that are substrings of nearby identifiers.
    pub identifier_substring_ratio: f32,
    /// Number of tokens in the comment text.
    pub comment_token_count: usize,
    /// Whether this is a documentation comment (`///`, `/** */`, etc.).
    pub is_doc_comment: bool,
    /// Whether the comment appears immediately before a declaration.
    pub is_before_declaration: bool,
    /// Whether the comment is inline (on the same line as code).
    pub is_inline: bool,
    /// The tree-sitter node kind of the adjacent declaration.
    pub adjacent_node_kind: String,
    /// The nesting depth of the comment in the AST.
    pub nesting_depth: usize,
    /// Whether the comment contains a "why" indicator (e.g. "because", "workaround").
    pub has_why_indicator: bool,
    /// Whether the comment references an external resource (URL, ticket, etc.).
    pub has_external_ref: bool,
    /// Whether the comment uses an imperative verb followed by a noun.
    pub imperative_verb_noun: bool,
    /// Whether the comment is a section label (e.g. "// --- Helpers ---").
    pub is_section_label: bool,
    /// Whether the comment contains literal values (numbers, strings, etc.).
    pub contains_literal_values: bool,
    /// Whether the comment references other files.
    pub references_other_files: bool,
    /// Whether the comment references specific function or method names.
    pub references_specific_functions: bool,
    /// Whether the comment mirrors a data structure's field names.
    pub mirrors_data_structure: bool,
    /// Ratio of the comment's age to the surrounding code's age (if available).
    pub comment_code_age_ratio: Option<f32>,
}

/// A comment that has been scored for superfluousness.
#[derive(Debug, Clone)]
pub struct ScoredComment {
    /// The original extracted comment with its context.
    pub context: CommentContext,
    /// The computed feature vector.
    pub features: FeatureVector,
    /// The superfluousness score in `[0.0, 1.0]` (higher = more superfluous).
    pub score: f32,
    /// Confidence in the score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Human-readable reasons that contributed to the score.
    pub reasons: Vec<String>,
}
