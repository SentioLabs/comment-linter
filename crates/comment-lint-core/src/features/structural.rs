//! Structural feature extraction: AST placement, doc-comment detection, inline detection.

use crate::extraction::comment::CommentContext;
use crate::types::CommentKind;

/// Tree-sitter node kinds that represent declarations across all supported languages.
///
/// - Go: function_declaration, method_declaration, type_declaration
/// - Rust: function_item, struct_item, enum_item, impl_item, trait_item, type_item
/// - Python: function_definition, class_definition, decorated_definition
/// - TS/JS: function_declaration, class_declaration, interface_declaration,
///          type_alias_declaration, method_definition, lexical_declaration
pub const DECLARATION_NODE_KINDS: &[&str] = &[
    // Go
    "function_declaration",
    "method_declaration",
    "type_declaration",
    // Rust
    "function_item",
    "struct_item",
    "enum_item",
    "impl_item",
    "trait_item",
    "type_item",
    // Python
    "function_definition",
    "class_definition",
    "decorated_definition",
    // TS/JS
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "method_definition",
    "lexical_declaration",
];

/// Returns `true` if the given tree-sitter node kind is a declaration.
pub fn is_before_declaration(adjacent_node_kind: &str) -> bool {
    DECLARATION_NODE_KINDS.contains(&adjacent_node_kind)
}

/// Returns `true` if the comment kind is a documentation comment.
pub fn is_doc_comment(kind: CommentKind) -> bool {
    matches!(kind, CommentKind::Doc)
}

/// Returns `true` if the comment is inline (starts at a column greater than 0).
pub fn is_inline(context: &CommentContext) -> bool {
    context.column > 0
}

/// Aggregated structural features extracted from a comment and its context.
#[derive(Debug, Clone)]
pub struct StructuralFeatures {
    /// Whether this is a documentation comment.
    pub is_doc_comment: bool,
    /// Whether the comment appears immediately before a declaration.
    pub is_before_declaration: bool,
    /// Whether the comment is inline (on the same line as code).
    pub is_inline: bool,
    /// The tree-sitter node kind of the adjacent declaration.
    pub adjacent_node_kind: String,
}

/// Extract all structural features from a comment context.
pub fn extract_structural_features(context: &CommentContext) -> StructuralFeatures {
    StructuralFeatures {
        is_doc_comment: is_doc_comment(context.comment_kind),
        is_before_declaration: is_before_declaration(&context.adjacent_node_kind),
        is_inline: is_inline(context),
        adjacent_node_kind: context.adjacent_node_kind.clone(),
    }
}

#[cfg(test)]
mod tests {
    use crate::extraction::comment::CommentContext;
    use crate::types::{CommentKind, LanguageId};
    use std::path::PathBuf;

    use super::*;

    /// Helper to build a minimal CommentContext for testing.
    fn make_context(column: usize, kind: CommentKind, adjacent_node_kind: &str) -> CommentContext {
        CommentContext {
            file_path: PathBuf::from("test.rs"),
            line: 1,
            column,
            comment_text: String::from("// test comment"),
            comment_kind: kind,
            language: LanguageId::Rust,
            adjacent_node_kind: adjacent_node_kind.to_string(),
            surrounding_source: String::new(),
            nearby_identifiers: vec![],
            nearby_keywords: vec![],
        }
    }

    // --- is_before_declaration tests ---

    #[test]
    fn is_before_declaration_returns_true_for_function_declaration() {
        assert!(is_before_declaration("function_declaration"));
    }

    #[test]
    fn is_before_declaration_returns_true_for_function_item() {
        assert!(is_before_declaration("function_item"));
    }

    #[test]
    fn is_before_declaration_returns_true_for_class_definition() {
        assert!(is_before_declaration("class_definition"));
    }

    #[test]
    fn is_before_declaration_returns_true_for_trait_item() {
        assert!(is_before_declaration("trait_item"));
    }

    #[test]
    fn is_before_declaration_returns_true_for_interface_declaration() {
        assert!(is_before_declaration("interface_declaration"));
    }

    #[test]
    fn is_before_declaration_returns_false_for_if_statement() {
        assert!(!is_before_declaration("if_statement"));
    }

    #[test]
    fn is_before_declaration_returns_false_for_empty_string() {
        assert!(!is_before_declaration(""));
    }

    // --- is_doc_comment tests ---

    #[test]
    fn is_doc_comment_returns_true_for_doc_kind() {
        assert!(is_doc_comment(CommentKind::Doc));
    }

    #[test]
    fn is_doc_comment_returns_false_for_line_kind() {
        assert!(!is_doc_comment(CommentKind::Line));
    }

    #[test]
    fn is_doc_comment_returns_false_for_block_kind() {
        assert!(!is_doc_comment(CommentKind::Block));
    }

    // --- is_inline tests ---

    #[test]
    fn is_inline_returns_false_when_column_is_zero() {
        let ctx = make_context(0, CommentKind::Line, "");
        assert!(!is_inline(&ctx));
    }

    #[test]
    fn is_inline_returns_true_when_column_is_positive() {
        let ctx = make_context(20, CommentKind::Line, "");
        assert!(is_inline(&ctx));
    }

    #[test]
    fn is_inline_returns_true_when_column_is_one() {
        let ctx = make_context(1, CommentKind::Line, "");
        assert!(is_inline(&ctx));
    }

    // --- extract_structural_features tests ---

    #[test]
    fn extract_structural_features_for_doc_comment_before_function() {
        let ctx = make_context(0, CommentKind::Doc, "function_item");
        let features = extract_structural_features(&ctx);

        assert!(features.is_doc_comment);
        assert!(features.is_before_declaration);
        assert!(!features.is_inline);
        assert_eq!(features.adjacent_node_kind, "function_item");
    }

    #[test]
    fn extract_structural_features_for_inline_line_comment() {
        let ctx = make_context(20, CommentKind::Line, "if_statement");
        let features = extract_structural_features(&ctx);

        assert!(!features.is_doc_comment);
        assert!(!features.is_before_declaration);
        assert!(features.is_inline);
        assert_eq!(features.adjacent_node_kind, "if_statement");
    }
}
