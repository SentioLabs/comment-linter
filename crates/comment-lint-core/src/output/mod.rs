//! Output formatting for scored comments.
//!
//! Three formatters are provided:
//! - [`text::TextFormatter`] - Colored terminal output with star ratings.
//! - [`json::JsonFormatter`] - JSONL (one JSON object per line) for tooling.
//! - [`github::GithubFormatter`] - GitHub Actions workflow annotations.

pub mod github;
pub mod json;
pub mod text;

use crate::features::ScoredComment;
use std::io::Write;

/// Trait implemented by all output formatters.
pub trait OutputFormatter: Send + Sync {
    /// Write a single scored comment to the given writer.
    fn format_comment(
        &self,
        comment: &ScoredComment,
        writer: &mut dyn Write,
    ) -> std::io::Result<()>;

    /// Write a summary line after all comments have been emitted.
    fn format_summary(
        &self,
        total_comments: usize,
        superfluous_count: usize,
        file_count: usize,
        writer: &mut dyn Write,
    ) -> std::io::Result<()>;
}

/// Selects which output format to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Github,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::comment::CommentContext;
    use crate::features::{FeatureVector, ScoredComment};
    use crate::types::{CommentKind, LanguageId};
    use std::path::PathBuf;

    /// Helper to construct a minimal ScoredComment for test use.
    pub(crate) fn make_scored_comment(score: f32, confidence: f32) -> ScoredComment {
        ScoredComment {
            context: CommentContext {
                file_path: PathBuf::from("src/main.rs"),
                line: 42,
                column: 4,
                comment_text: "increment counter".to_string(),
                comment_kind: CommentKind::Line,
                language: LanguageId::Rust,
                adjacent_node_kind: "expression_statement".to_string(),
                surrounding_source: "counter += 1; // increment counter".to_string(),
                nearby_identifiers: vec!["counter".to_string()],
                nearby_keywords: vec!["let".to_string()],
            },
            features: FeatureVector {
                token_overlap_jaccard: 0.8,
                identifier_substring_ratio: 0.9,
                comment_token_count: 2,
                is_doc_comment: false,
                is_before_declaration: false,
                is_inline: true,
                adjacent_node_kind: "expression_statement".to_string(),
                nesting_depth: 1,
                has_why_indicator: false,
                has_external_ref: false,
                imperative_verb_noun: true,
                is_section_label: false,
                contains_literal_values: false,
                references_other_files: false,
                references_specific_functions: false,
                mirrors_data_structure: false,
                comment_code_age_ratio: None,
            },
            score,
            confidence,
            reasons: vec![
                "high token overlap".to_string(),
                "mirrors identifier".to_string(),
            ],
        }
    }

    #[test]
    fn output_format_enum_debug() {
        assert_eq!(format!("{:?}", OutputFormat::Text), "Text");
        assert_eq!(format!("{:?}", OutputFormat::Json), "Json");
        assert_eq!(format!("{:?}", OutputFormat::Github), "Github");
    }

    #[test]
    fn output_format_enum_clone_eq() {
        let fmt = OutputFormat::Text;
        let cloned = fmt;
        assert_eq!(fmt, cloned);
    }
}
