//! Rust language support for comment extraction using tree-sitter-rust.

use std::path::Path;

use crate::extraction::comment::CommentContext;
use crate::languages::Language;
use crate::types::{CommentKind, LanguageId};

/// Rust language implementation for comment extraction.
pub struct RustLanguage;

/// Rust declaration node kinds used to determine adjacent context.
const DECLARATION_KINDS: &[&str] = &[
    "function_item",
    "struct_item",
    "enum_item",
    "impl_item",
    "trait_item",
    "const_item",
    "static_item",
    "type_item",
    "mod_item",
    "use_declaration",
    "let_declaration",
];

/// Rust keywords to collect from surrounding context.
const RUST_KEYWORDS: &[&str] = &[
    "fn", "struct", "enum", "impl", "trait", "pub", "let", "mut", "const", "static", "mod",
    "use", "type", "where", "return", "match", "if", "else", "for", "while", "loop", "async",
    "await", "unsafe", "extern", "crate", "self", "super",
];

impl Language for RustLanguage {
    fn id(&self) -> LanguageId {
        LanguageId::Rust
    }

    fn extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn extract_comments(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        file_path: &Path,
    ) -> Vec<CommentContext> {
        let mut comments = Vec::new();
        let root = tree.root_node();
        collect_comments(root, source, file_path, &mut comments);
        comments
    }
}

/// Recursively walk the AST collecting comment nodes.
fn collect_comments(
    node: tree_sitter::Node,
    source: &str,
    file_path: &Path,
    comments: &mut Vec<CommentContext>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind == "line_comment" || kind == "block_comment" {
            if let Some(ctx) = build_comment_context(child, source, file_path) {
                comments.push(ctx);
            }
        }
        // Recurse into children to find comments inside function bodies, etc.
        collect_comments(child, source, file_path, comments);
    }
}

/// Build a `CommentContext` from a tree-sitter comment node.
fn build_comment_context(
    node: tree_sitter::Node,
    source: &str,
    file_path: &Path,
) -> Option<CommentContext> {
    let text = node.utf8_text(source.as_bytes()).ok()?;
    let (comment_text, comment_kind) = classify_and_strip(text, node.kind());
    let start = node.start_position();
    let line = start.row + 1; // 1-based
    let column = start.column; // 0-based
    let adjacent_node_kind = find_adjacent_declaration(node);
    let surrounding_source = extract_surrounding_source(source, start.row);
    let nearby_identifiers = extract_nearby_identifiers(node, source);
    let nearby_keywords = extract_nearby_keywords(node, source);

    Some(CommentContext {
        file_path: file_path.to_path_buf(),
        line,
        column,
        comment_text,
        comment_kind,
        language: LanguageId::Rust,
        adjacent_node_kind,
        surrounding_source,
        nearby_identifiers,
        nearby_keywords,
    })
}

/// Classify comment kind and strip the comment markers.
fn classify_and_strip(text: &str, node_kind: &str) -> (String, CommentKind) {
    match node_kind {
        "line_comment" => {
            if let Some(rest) = text.strip_prefix("///") {
                // Outer doc comment
                (rest.trim_start_matches(' ').to_string(), CommentKind::Doc)
            } else if let Some(rest) = text.strip_prefix("//!") {
                // Inner doc comment
                (rest.trim_start_matches(' ').to_string(), CommentKind::Doc)
            } else if let Some(rest) = text.strip_prefix("//") {
                // Regular line comment
                (rest.trim_start_matches(' ').to_string(), CommentKind::Line)
            } else {
                (text.to_string(), CommentKind::Line)
            }
        }
        "block_comment" => {
            if text.starts_with("/**") {
                // Block doc comment
                let inner = text
                    .strip_prefix("/**")
                    .unwrap_or(text)
                    .strip_suffix("*/")
                    .unwrap_or(text);
                (clean_block_comment(inner), CommentKind::Doc)
            } else {
                // Regular block comment
                let inner = text
                    .strip_prefix("/*")
                    .unwrap_or(text)
                    .strip_suffix("*/")
                    .unwrap_or(text);
                (clean_block_comment(inner), CommentKind::Block)
            }
        }
        _ => (text.to_string(), CommentKind::Line),
    }
}

/// Clean up block comment text by removing leading asterisks and extra whitespace.
fn clean_block_comment(inner: &str) -> String {
    let lines: Vec<&str> = inner.lines().collect();
    if lines.len() <= 1 {
        return inner.trim().to_string();
    }
    let cleaned: Vec<String> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("* ") {
                rest.to_string()
            } else if trimmed == "*" {
                String::new()
            } else {
                trimmed.to_string()
            }
        })
        .collect();
    cleaned.join("\n").trim().to_string()
}

/// Find the nearest adjacent declaration node (next sibling that is a declaration).
fn find_adjacent_declaration(node: tree_sitter::Node) -> String {
    // Look at the next sibling
    let mut sibling = node.next_named_sibling();
    while let Some(sib) = sibling {
        let kind = sib.kind();
        if DECLARATION_KINDS.contains(&kind) {
            return kind.to_string();
        }
        // If we hit a non-comment, non-declaration, stop looking
        if kind != "line_comment" && kind != "block_comment" {
            break;
        }
        sibling = sib.next_named_sibling();
    }
    String::new()
}

/// Extract surrounding source lines (up to 3 lines before and after).
fn extract_surrounding_source(source: &str, comment_row: usize) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let start = comment_row.saturating_sub(3);
    let end = (comment_row + 4).min(lines.len());
    lines[start..end].join("\n")
}

/// Extract identifiers from nearby declarations (siblings and children).
fn extract_nearby_identifiers(
    node: tree_sitter::Node,
    source: &str,
) -> Vec<String> {
    let mut identifiers = Vec::new();
    // Look at the next sibling declaration
    let mut sibling = node.next_named_sibling();
    while let Some(sib) = sibling {
        let kind = sib.kind();
        if DECLARATION_KINDS.contains(&kind) {
            collect_identifiers_from_node(sib, source, &mut identifiers);
            break;
        }
        if kind != "line_comment" && kind != "block_comment" {
            break;
        }
        sibling = sib.next_named_sibling();
    }
    // Also look at previous sibling
    if let Some(prev) = node.prev_named_sibling() {
        if DECLARATION_KINDS.contains(&prev.kind()) {
            collect_identifiers_from_node(prev, source, &mut identifiers);
        }
    }
    // Look at parent if it is a declaration
    if let Some(parent) = node.parent() {
        if DECLARATION_KINDS.contains(&parent.kind()) {
            collect_identifiers_from_node(parent, source, &mut identifiers);
        }
    }
    identifiers.sort();
    identifiers.dedup();
    identifiers
}

/// Recursively collect identifier nodes from a tree-sitter node.
fn collect_identifiers_from_node(
    node: tree_sitter::Node,
    source: &str,
    identifiers: &mut Vec<String>,
) {
    let kind = node.kind();
    if kind == "identifier" || kind == "type_identifier" {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            if !text.is_empty() {
                identifiers.push(text.to_string());
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers_from_node(child, source, identifiers);
    }
}

/// Extract language keywords from nearby source context.
fn extract_nearby_keywords(
    node: tree_sitter::Node,
    source: &str,
) -> Vec<String> {
    let mut keywords = Vec::new();
    let start_row = node.start_position().row;
    let lines: Vec<&str> = source.lines().collect();
    let context_start = start_row.saturating_sub(2);
    let context_end = (start_row + 3).min(lines.len());
    let context = lines[context_start..context_end].join(" ");
    for kw in RUST_KEYWORDS {
        if context.split_whitespace().any(|w| {
            // Match whole word or word with trailing punctuation
            w == *kw || w.starts_with(&format!("{kw}(")) || w.starts_with(&format!("{kw}<"))
        }) {
            keywords.push(kw.to_string());
        }
    }
    keywords.sort();
    keywords.dedup();
    keywords
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn parse_and_extract(source: &str) -> Vec<CommentContext> {
        let lang = RustLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang.tree_sitter_language()).unwrap();
        let tree = parser.parse(source, None).unwrap();
        lang.extract_comments(&tree, source, Path::new("test.rs"))
    }

    #[test]
    fn id_returns_rust() {
        let lang = RustLanguage;
        assert_eq!(lang.id(), LanguageId::Rust);
    }

    #[test]
    fn extensions_contains_rs() {
        let lang = RustLanguage;
        assert_eq!(lang.extensions(), &["rs"]);
    }

    #[test]
    fn tree_sitter_language_can_parse() {
        let lang = RustLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang.tree_sitter_language()).unwrap();
        let tree = parser.parse("fn main() {}", None);
        assert!(tree.is_some());
    }

    #[test]
    fn extract_line_comment() {
        let source = "// a regular comment\nfn main() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
        assert_eq!(comments[0].comment_text.trim(), "a regular comment");
        assert_eq!(comments[0].line, 1);
        assert_eq!(comments[0].column, 0);
        assert_eq!(comments[0].language, LanguageId::Rust);
        assert_eq!(comments[0].file_path, PathBuf::from("test.rs"));
    }

    #[test]
    fn extract_block_comment() {
        let source = "/* block comment */\nfn main() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Block);
        assert_eq!(comments[0].comment_text.trim(), "block comment");
    }

    #[test]
    fn extract_outer_doc_comment_triple_slash() {
        let source = "/// Doc comment for the function\nfn documented() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Doc);
        assert_eq!(
            comments[0].comment_text.trim(),
            "Doc comment for the function"
        );
    }

    #[test]
    fn extract_inner_doc_comment_bang() {
        let source = "//! Module-level doc comment\nfn main() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Doc);
        assert_eq!(
            comments[0].comment_text.trim(),
            "Module-level doc comment"
        );
    }

    #[test]
    fn extract_block_doc_comment() {
        let source = "/** Block doc comment */\nfn documented() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Doc);
        assert_eq!(comments[0].comment_text.trim(), "Block doc comment");
    }

    #[test]
    fn extract_nearby_identifiers_from_fn() {
        let source = "/// Adds two numbers\nfn add(a: i32, b: i32) -> i32 { a + b }\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert!(
            comments[0].nearby_identifiers.contains(&"add".to_string()),
            "Expected nearby_identifiers to contain 'add', got: {:?}",
            comments[0].nearby_identifiers
        );
    }

    #[test]
    fn extract_nearby_identifiers_from_struct() {
        let source = "/// A point in 2D space\nstruct Point {\n    x: f64,\n    y: f64,\n}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert!(
            comments[0]
                .nearby_identifiers
                .contains(&"Point".to_string()),
            "Expected nearby_identifiers to contain 'Point', got: {:?}",
            comments[0].nearby_identifiers
        );
    }

    #[test]
    fn extract_multiple_comments() {
        let source = r#"// first comment
fn foo() {}

// second comment
fn bar() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].comment_text.trim(), "first comment");
        assert_eq!(comments[1].comment_text.trim(), "second comment");
    }

    #[test]
    fn adjacent_node_kind_for_fn() {
        let source = "/// A function\nfn my_func() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].adjacent_node_kind, "function_item");
    }

    #[test]
    fn adjacent_node_kind_for_struct() {
        let source = "/// A struct\nstruct MyStruct {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].adjacent_node_kind, "struct_item");
    }
}
