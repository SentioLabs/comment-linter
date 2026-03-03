//! Go language support using tree-sitter-go grammar.

use std::path::{Path, PathBuf};

use crate::extraction::comment::CommentContext;
use crate::languages::Language;
use crate::types::{CommentKind, LanguageId};

/// Go language implementation for comment extraction.
pub struct GoLanguage;

/// Go keywords used for nearby keyword detection.
const GO_KEYWORDS: &[&str] = &[
    "break",
    "case",
    "chan",
    "const",
    "continue",
    "default",
    "defer",
    "else",
    "fallthrough",
    "for",
    "func",
    "go",
    "goto",
    "if",
    "import",
    "interface",
    "map",
    "package",
    "range",
    "return",
    "select",
    "struct",
    "switch",
    "type",
    "var",
];

/// Declaration node kinds that, when immediately following a comment, make it
/// a doc comment.
const DECLARATION_KINDS: &[&str] = &[
    "function_declaration",
    "method_declaration",
    "type_declaration",
    "const_declaration",
    "var_declaration",
];

impl Language for GoLanguage {
    fn id(&self) -> LanguageId {
        LanguageId::Go
    }

    fn extensions(&self) -> &[&str] {
        &["go"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_go::LANGUAGE.into()
    }

    fn extract_comments(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        file_path: &Path,
    ) -> Vec<CommentContext> {
        let source_lines: Vec<&str> = source.lines().collect();
        let mut comments = Vec::new();
        let mut cursor = tree.walk();

        collect_comments(&mut cursor, source, &source_lines, file_path, &mut comments);
        comments
    }
}

/// Recursively walk the tree-sitter AST collecting comment nodes.
fn collect_comments(
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    source_lines: &[&str],
    file_path: &Path,
    comments: &mut Vec<CommentContext>,
) {
    loop {
        let node = cursor.node();
        let kind = node.kind();

        // tree-sitter-go >=0.25 uses "comment" for both // and /* */ comments.
        // Older versions may use "block_comment" for /* */ comments.
        if kind == "comment" || kind == "block_comment" {
            if let Some(ctx) = build_comment_context(node, source, source_lines, file_path) {
                comments.push(ctx);
            }
        }

        // Recurse into children.
        if cursor.goto_first_child() {
            collect_comments(cursor, source, source_lines, file_path, comments);
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

/// Build a [`CommentContext`] from a single tree-sitter comment node.
fn build_comment_context(
    node: tree_sitter::Node,
    source: &str,
    source_lines: &[&str],
    file_path: &Path,
) -> Option<CommentContext> {
    let raw_text = node.utf8_text(source.as_bytes()).ok()?;
    let start = node.start_position();
    let line = start.row + 1; // 1-based
    let column = start.column; // 0-based

    // In tree-sitter-go >=0.25 both line and block comments use node kind
    // "comment". Distinguish by checking whether the text starts with "//".
    let is_line_comment = raw_text.starts_with("//");

    // Strip comment markers.
    let comment_text = strip_comment_markers(raw_text, is_line_comment);

    // Determine comment kind.
    let comment_kind = determine_comment_kind(node, source_lines, is_line_comment);

    // Get adjacent node kind (next named sibling).
    let adjacent_node_kind = find_adjacent_node_kind(node);

    // Extract surrounding source (5 lines above and below).
    let surrounding_source = extract_surrounding_source(source_lines, start.row);

    // Extract nearby identifiers.
    let nearby_identifiers = extract_nearby_identifiers(node, source);

    // Extract nearby keywords.
    let nearby_keywords = extract_nearby_keywords(node, source);

    Some(CommentContext {
        file_path: PathBuf::from(file_path),
        line,
        column,
        comment_text,
        comment_kind,
        language: LanguageId::Go,
        adjacent_node_kind,
        surrounding_source,
        nearby_identifiers,
        nearby_keywords,
    })
}

/// Strip `//` or `/* */` delimiters from comment text.
fn strip_comment_markers(raw: &str, is_line_comment: bool) -> String {
    if is_line_comment {
        let stripped = raw.strip_prefix("//").unwrap_or(raw);
        // Remove a single leading space after the marker if present.
        stripped.strip_prefix(' ').unwrap_or(stripped).to_string()
    } else {
        let stripped = raw
            .strip_prefix("/*")
            .unwrap_or(raw)
            .strip_suffix("*/")
            .unwrap_or(raw);
        stripped.trim().to_string()
    }
}

/// Determine whether a comment node is a doc comment, line comment, or block
/// comment.
///
/// A line comment (`//`) is classified as `Doc` when:
/// - Its next named sibling is a declaration (function, method, type, etc.)
/// - There is no blank line between the comment and the declaration.
fn determine_comment_kind(
    node: tree_sitter::Node,
    source_lines: &[&str],
    is_line_comment: bool,
) -> CommentKind {
    if !is_line_comment {
        return CommentKind::Block;
    }

    // Check if next named sibling is a declaration.
    if let Some(next) = node.next_named_sibling() {
        if DECLARATION_KINDS.contains(&next.kind()) {
            // Check there is no blank line between this comment and the declaration.
            let comment_end_row = node.end_position().row;
            let decl_start_row = next.start_position().row;

            let has_blank_line = (comment_end_row + 1..decl_start_row).any(|row| {
                source_lines
                    .get(row)
                    .is_some_and(|line| line.trim().is_empty())
            });

            if !has_blank_line {
                return CommentKind::Doc;
            }
        }
    }

    CommentKind::Line
}

/// Find the tree-sitter node kind of the nearest adjacent named sibling.
fn find_adjacent_node_kind(node: tree_sitter::Node) -> String {
    node.next_named_sibling()
        .map(|n| n.kind().to_string())
        .unwrap_or_default()
}

/// Extract surrounding source lines (up to 5 above and 5 below).
fn extract_surrounding_source(source_lines: &[&str], row: usize) -> String {
    let start = row.saturating_sub(5);
    let end = (row + 6).min(source_lines.len());
    source_lines[start..end].join("\n")
}

/// Walk nearby nodes to collect identifier names.
fn extract_nearby_identifiers(node: tree_sitter::Node, source: &str) -> Vec<String> {
    let mut identifiers = Vec::new();

    // Collect identifiers from the next named sibling and its children.
    if let Some(next) = node.next_named_sibling() {
        collect_identifiers_recursive(next, source, &mut identifiers);
    }

    // Also collect from previous named sibling.
    if let Some(prev) = node.prev_named_sibling() {
        collect_identifiers_recursive(prev, source, &mut identifiers);
    }

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    identifiers.retain(|id| seen.insert(id.clone()));
    identifiers
}

/// Recursively collect identifier and field_identifier node text.
fn collect_identifiers_recursive(
    node: tree_sitter::Node,
    source: &str,
    identifiers: &mut Vec<String>,
) {
    let kind = node.kind();
    if kind == "identifier" || kind == "field_identifier" {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            let s = text.to_string();
            if !s.is_empty() {
                identifiers.push(s);
            }
        }
    }

    // Recurse into children.
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_identifiers_recursive(cursor.node(), source, identifiers);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

/// Extract Go keywords from sibling nodes.
fn extract_nearby_keywords(node: tree_sitter::Node, source: &str) -> Vec<String> {
    let mut keywords = Vec::new();

    // Walk the next named sibling's subtree looking for keyword tokens.
    if let Some(next) = node.next_named_sibling() {
        collect_keywords_recursive(next, source, &mut keywords);
    }

    if let Some(prev) = node.prev_named_sibling() {
        collect_keywords_recursive(prev, source, &mut keywords);
    }

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    keywords.retain(|k| seen.insert(k.clone()));
    keywords
}

/// Recursively collect Go keyword tokens from a node subtree.
fn collect_keywords_recursive(node: tree_sitter::Node, source: &str, keywords: &mut Vec<String>) {
    // In tree-sitter-go, keywords appear as anonymous nodes with their text
    // matching Go keyword tokens (e.g. "func", "if", "return").
    if !node.is_named() {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            if GO_KEYWORDS.contains(&text) {
                keywords.push(text.to_string());
            }
        }
    }

    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_keywords_recursive(cursor.node(), source, keywords);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_extract(source: &str) -> Vec<CommentContext> {
        let lang = GoLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&lang.tree_sitter_language())
            .expect("failed to set Go language");
        let tree = parser.parse(source, None).expect("failed to parse");
        lang.extract_comments(&tree, source, Path::new("test.go"))
    }

    #[test]
    fn id_returns_go() {
        let lang = GoLanguage;
        assert_eq!(lang.id(), LanguageId::Go);
    }

    #[test]
    fn extensions_returns_go() {
        let lang = GoLanguage;
        assert_eq!(lang.extensions(), &["go"]);
    }

    #[test]
    fn extracts_line_comment() {
        let source = r#"package main

// This is a line comment
func main() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "This is a line comment");
        assert_eq!(c.comment_kind, CommentKind::Doc);
        assert_eq!(c.language, LanguageId::Go);
        assert_eq!(c.line, 3);
        assert_eq!(c.column, 0);
        assert_eq!(c.file_path, PathBuf::from("test.go"));
    }

    #[test]
    fn extracts_block_comment() {
        let source = r#"package main

/* This is a block comment */
func main() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "This is a block comment");
        assert_eq!(c.comment_kind, CommentKind::Block);
        assert_eq!(c.language, LanguageId::Go);
    }

    #[test]
    fn detects_doc_comment_before_exported_func() {
        let source = r#"package main

// Hello greets the caller.
func Hello() string {
    return "hello"
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        let c = &comments[0];
        assert_eq!(c.comment_kind, CommentKind::Doc);
        assert_eq!(c.comment_text, "Hello greets the caller.");
        assert_eq!(c.adjacent_node_kind, "function_declaration");
    }

    #[test]
    fn line_comment_not_before_declaration_is_line_kind() {
        let source = r#"package main

func main() {
    // inside a function body
    x := 1
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
        assert_eq!(comments[0].comment_text, "inside a function body");
    }

    #[test]
    fn extracts_nearby_identifiers_from_func_signature() {
        let source = r#"package main

// Add adds two numbers.
func Add(a int, b int) int {
    return a + b
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        let identifiers = &comments[0].nearby_identifiers;
        assert!(identifiers.contains(&"Add".to_string()));
        assert!(identifiers.contains(&"a".to_string()));
        assert!(identifiers.contains(&"b".to_string()));
    }

    #[test]
    fn extracts_surrounding_source() {
        let source = r#"package main

import "fmt"

// Greet prints a greeting.
func Greet(name string) {
    fmt.Println("Hello, " + name)
}

func main() {
    Greet("world")
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        let surrounding = &comments[0].surrounding_source;
        // The comment is on line 5. Surrounding should include lines around it.
        assert!(surrounding.contains("import \"fmt\""));
        assert!(surrounding.contains("// Greet prints a greeting."));
        assert!(surrounding.contains("func Greet(name string)"));
    }

    #[test]
    fn extracts_multiple_comments() {
        let source = r#"package main

// First comment
func First() {}

// Second comment
func Second() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].comment_text, "First comment");
        assert_eq!(comments[1].comment_text, "Second comment");
    }

    #[test]
    fn doc_comment_before_method_declaration() {
        let source = r#"package main

type Foo struct{}

// Bar is a method on Foo.
func (f Foo) Bar() {}
"#;
        let comments = parse_and_extract(source);
        let bar_comment = comments
            .iter()
            .find(|c| c.comment_text == "Bar is a method on Foo.")
            .unwrap();
        assert_eq!(bar_comment.comment_kind, CommentKind::Doc);
        assert_eq!(bar_comment.adjacent_node_kind, "method_declaration");
    }

    #[test]
    fn blank_line_between_comment_and_func_is_not_doc() {
        let source = r#"package main

// This is just a line comment

func Hello() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
    }

    #[test]
    fn extracts_nearby_keywords() {
        let source = r#"package main

// Process handles data.
func Process(data []byte) error {
    if len(data) == 0 {
        return nil
    }
    return nil
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        let keywords = &comments[0].nearby_keywords;
        assert!(keywords.contains(&"func".to_string()));
    }
}
