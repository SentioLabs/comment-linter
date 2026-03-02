//! JavaScript language support using tree-sitter-javascript grammar.

use std::path::Path;

use crate::extraction::comment::CommentContext;
use crate::languages::Language;
use crate::types::{CommentKind, LanguageId};

/// JavaScript language implementation for comment extraction.
pub struct JavaScriptLanguage;

/// JavaScript-specific declaration node kinds used for adjacent-node detection.
const JS_DECLARATION_KINDS: &[&str] = &[
    "function_declaration",
    "class_declaration",
    "method_definition",
    "variable_declaration",
    "lexical_declaration",
    "export_statement",
];

/// JavaScript keywords to detect near comments.
const JS_KEYWORDS: &[&str] = &[
    "function",
    "class",
    "const",
    "let",
    "var",
    "if",
    "else",
    "for",
    "while",
    "return",
    "import",
    "export",
    "async",
    "await",
    "new",
    "this",
    "switch",
    "case",
    "try",
    "catch",
    "throw",
    "typeof",
    "instanceof",
    "yield",
    "delete",
    "void",
    "in",
    "of",
    "do",
    "break",
    "continue",
    "default",
    "finally",
    "with",
    "debugger",
    "extends",
    "super",
    "static",
    "get",
    "set",
];

impl Language for JavaScriptLanguage {
    fn id(&self) -> LanguageId {
        LanguageId::JavaScript
    }

    fn extensions(&self) -> &[&str] {
        &["js", "jsx", "mjs", "cjs"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_javascript::LANGUAGE.into()
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

        collect_comments(&mut cursor, source, file_path, &source_lines, &mut comments);

        comments
    }
}

/// Recursively walk the tree-sitter AST and collect all comment nodes.
fn collect_comments(
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    file_path: &Path,
    source_lines: &[&str],
    comments: &mut Vec<CommentContext>,
) {
    loop {
        let node = cursor.node();

        if node.kind() == "comment" {
            if let Some(ctx) = build_comment_context(node, source, file_path, source_lines) {
                comments.push(ctx);
            }
        }

        // Recurse into children
        if cursor.goto_first_child() {
            collect_comments(cursor, source, file_path, source_lines, comments);
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

/// Build a `CommentContext` from a tree-sitter comment node.
fn build_comment_context(
    node: tree_sitter::Node,
    source: &str,
    file_path: &Path,
    source_lines: &[&str],
) -> Option<CommentContext> {
    let raw_text = node.utf8_text(source.as_bytes()).ok()?;
    let start = node.start_position();
    let line_1based = start.row + 1;
    let column_0based = start.column;

    let (comment_text, comment_kind) = classify_and_strip(raw_text);
    let adjacent_node_kind = find_adjacent_declaration(node);
    let surrounding_source = extract_surrounding_source(source_lines, start.row);
    let nearby_identifiers = extract_nearby_identifiers(node, source);
    let nearby_keywords = extract_nearby_keywords(node, source);

    Some(CommentContext {
        file_path: file_path.to_path_buf(),
        line: line_1based,
        column: column_0based,
        comment_text,
        comment_kind,
        language: LanguageId::JavaScript,
        adjacent_node_kind,
        surrounding_source,
        nearby_identifiers,
        nearby_keywords,
    })
}

/// Classify the comment kind and strip the comment markers.
///
/// - `/** ... */` is `Doc`
/// - `/* ... */` is `Block`
/// - `// ...` is `Line`
fn classify_and_strip(raw: &str) -> (String, CommentKind) {
    if raw.starts_with("/**") {
        // JSDoc comment
        let stripped = strip_block_markers(raw);
        (stripped, CommentKind::Doc)
    } else if raw.starts_with("/*") {
        // Block comment
        let stripped = strip_block_markers(raw);
        (stripped, CommentKind::Block)
    } else if raw.starts_with("//") {
        // Line comment - strip the `// ` or `//` prefix
        let text = raw.strip_prefix("//").unwrap_or(raw);
        let text = text.strip_prefix(' ').unwrap_or(text);
        (text.to_string(), CommentKind::Line)
    } else {
        (raw.to_string(), CommentKind::Line)
    }
}

/// Strip block comment delimiters (`/* ... */` or `/** ... */`) and clean up
/// the leading ` * ` prefixes common in JSDoc-style comments.
fn strip_block_markers(raw: &str) -> String {
    // Remove opening /** or /* and closing */
    let s = if raw.starts_with("/**") {
        &raw[3..]
    } else if raw.starts_with("/*") {
        &raw[2..]
    } else {
        raw
    };
    let s = if s.ends_with("*/") {
        &s[..s.len() - 2]
    } else {
        s
    };

    // Process each line: strip leading whitespace and optional ` * ` prefix
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() == 1 {
        // Single-line block comment: /* text */
        return s.trim().to_string();
    }

    let cleaned: Vec<String> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("* ") {
                trimmed[2..].to_string()
            } else if trimmed == "*" {
                String::new()
            } else {
                trimmed.to_string()
            }
        })
        .collect();

    // Remove leading/trailing empty lines
    let start = cleaned.iter().position(|l| !l.is_empty()).unwrap_or(0);
    let end = cleaned.iter().rposition(|l| !l.is_empty()).map(|i| i + 1).unwrap_or(0);
    cleaned[start..end].join("\n")
}

/// Find the adjacent declaration node kind (the next sibling that is a declaration).
fn find_adjacent_declaration(comment_node: tree_sitter::Node) -> String {
    // Look at the next sibling to see if it is a declaration
    let mut sibling = comment_node.next_named_sibling();
    while let Some(sib) = sibling {
        let kind = sib.kind();
        if JS_DECLARATION_KINDS.contains(&kind) {
            return kind.to_string();
        }
        // Skip over other comments (comment groups)
        if kind == "comment" {
            sibling = sib.next_named_sibling();
            continue;
        }
        break;
    }
    String::new()
}

/// Extract surrounding source lines (up to 3 lines before and after the comment).
fn extract_surrounding_source(source_lines: &[&str], comment_row: usize) -> String {
    let context_radius = 3;
    let start = comment_row.saturating_sub(context_radius);
    let end = (comment_row + context_radius + 1).min(source_lines.len());
    source_lines[start..end].join("\n")
}

/// Extract identifiers from the nearby declaration (next named sibling).
fn extract_nearby_identifiers(comment_node: tree_sitter::Node, source: &str) -> Vec<String> {
    let mut identifiers = Vec::new();

    // Look at the next named sibling (the declaration following the comment)
    if let Some(next) = comment_node.next_named_sibling() {
        collect_identifiers_recursive(next, source, &mut identifiers);
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    identifiers.retain(|id| seen.insert(id.clone()));
    identifiers
}

/// Recursively collect identifier and property_identifier nodes from a subtree.
fn collect_identifiers_recursive(
    node: tree_sitter::Node,
    source: &str,
    identifiers: &mut Vec<String>,
) {
    let kind = node.kind();
    if kind == "identifier" || kind == "property_identifier" {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            identifiers.push(text.to_string());
        }
    }

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

/// Extract keywords found near the comment by scanning the adjacent declaration.
fn extract_nearby_keywords(comment_node: tree_sitter::Node, source: &str) -> Vec<String> {
    let mut keywords = Vec::new();

    if let Some(next) = comment_node.next_named_sibling() {
        collect_keywords_recursive(next, source, &mut keywords);
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    keywords.retain(|k| seen.insert(k.clone()));
    keywords
}

/// Recursively collect JavaScript keyword tokens from a node subtree.
///
/// In tree-sitter grammars, keywords typically appear as anonymous (unnamed)
/// nodes whose text matches the keyword token.
fn collect_keywords_recursive(
    node: tree_sitter::Node,
    source: &str,
    keywords: &mut Vec<String>,
) {
    if !node.is_named() {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            if JS_KEYWORDS.contains(&text) {
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
    use std::path::PathBuf;

    fn parse_and_extract(source: &str) -> Vec<CommentContext> {
        let lang = JavaScriptLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&lang.tree_sitter_language())
            .expect("failed to set JavaScript language");
        let tree = parser.parse(source, None).expect("failed to parse");
        lang.extract_comments(&tree, source, Path::new("test.js"))
    }

    #[test]
    fn id_returns_javascript() {
        let lang = JavaScriptLanguage;
        assert_eq!(lang.id(), LanguageId::JavaScript);
    }

    #[test]
    fn extensions_returns_js_variants() {
        let lang = JavaScriptLanguage;
        assert_eq!(lang.extensions(), &["js", "jsx", "mjs", "cjs"]);
    }

    #[test]
    fn extracts_line_comment() {
        let source = r#"// This is a line comment
const x = 1;
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "This is a line comment");
        assert_eq!(c.comment_kind, CommentKind::Line);
        assert_eq!(c.language, LanguageId::JavaScript);
        assert_eq!(c.line, 1);
        assert_eq!(c.column, 0);
        assert_eq!(c.file_path, PathBuf::from("test.js"));
    }

    #[test]
    fn extracts_block_comment() {
        let source = r#"/* This is a block comment */
function hello() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "This is a block comment");
        assert_eq!(c.comment_kind, CommentKind::Block);
        assert_eq!(c.language, LanguageId::JavaScript);
    }

    #[test]
    fn detects_jsdoc_comment() {
        let source = r#"/**
 * Adds two numbers together.
 * @param {number} a - First number.
 * @param {number} b - Second number.
 * @returns {number} The sum.
 */
function add(a, b) {
    return a + b;
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_kind, CommentKind::Doc);
        assert!(c.comment_text.contains("Adds two numbers together."));
        assert!(c.comment_text.contains("@param"));
    }

    #[test]
    fn line_comment_before_function_declaration() {
        let source = r#"// greet says hello
function greet(name) {
    return "Hello, " + name;
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "greet says hello");
        assert_eq!(c.comment_kind, CommentKind::Line);
        assert_eq!(c.adjacent_node_kind, "function_declaration");
    }

    #[test]
    fn extracts_nearby_identifiers_from_function() {
        let source = r#"// Calculate the total
function calculateTotal(price, tax) {
    return price + tax;
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let identifiers = &comments[0].nearby_identifiers;
        assert!(identifiers.contains(&"calculateTotal".to_string()));
        assert!(identifiers.contains(&"price".to_string()));
        assert!(identifiers.contains(&"tax".to_string()));
    }

    #[test]
    fn extracts_nearby_identifiers_from_variable_declaration() {
        let source = r#"// The default configuration
const defaultConfig = { timeout: 3000 };
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let identifiers = &comments[0].nearby_identifiers;
        assert!(identifiers.contains(&"defaultConfig".to_string()));
    }

    #[test]
    fn extracts_nearby_identifiers_from_lexical_declaration() {
        let source = r#"// Maximum retries
let maxRetries = 5;
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let identifiers = &comments[0].nearby_identifiers;
        assert!(identifiers.contains(&"maxRetries".to_string()));
    }

    #[test]
    fn extracts_nearby_keywords() {
        let source = r#"// Process data
function process(data) {
    if (data.length === 0) {
        return null;
    }
    return data;
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let keywords = &comments[0].nearby_keywords;
        assert!(keywords.contains(&"function".to_string()));
    }

    #[test]
    fn extracts_surrounding_source() {
        let source = r#"import { foo } from 'bar';

// Process the input
function processInput(input) {
    return foo(input);
}

function other() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let surrounding = &comments[0].surrounding_source;
        assert!(surrounding.contains("import { foo }"));
        assert!(surrounding.contains("// Process the input"));
        assert!(surrounding.contains("function processInput(input)"));
    }

    #[test]
    fn extracts_multiple_comments() {
        let source = r#"// First comment
function first() {}

// Second comment
function second() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].comment_text, "First comment");
        assert_eq!(comments[1].comment_text, "Second comment");
    }

    #[test]
    fn jsdoc_before_class_declaration() {
        let source = r#"/**
 * Represents a user.
 */
class User {
    constructor(name) {
        this.name = name;
    }
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_kind, CommentKind::Doc);
        assert_eq!(c.adjacent_node_kind, "class_declaration");
    }

    #[test]
    fn comment_inside_function_body_is_line() {
        let source = r#"function main() {
    // inside a function body
    const x = 1;
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
        assert_eq!(comments[0].comment_text, "inside a function body");
    }

    #[test]
    fn block_comment_not_jsdoc() {
        let source = r#"/* just a block comment, not jsdoc */
const x = 1;
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Block);
    }

    #[test]
    fn jsdoc_before_variable_declaration() {
        let source = r#"/**
 * The default timeout.
 * @type {number}
 */
var DEFAULT_TIMEOUT = 3000;
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_kind, CommentKind::Doc);
        assert_eq!(c.adjacent_node_kind, "variable_declaration");
    }
}
