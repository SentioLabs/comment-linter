//! TypeScript language support using tree-sitter-typescript grammar.

use std::path::Path;

use crate::extraction::comment::CommentContext;
use crate::languages::Language;
use crate::types::{CommentKind, LanguageId};

/// TypeScript language implementation for comment extraction.
pub struct TypeScriptLanguage;

/// Node kinds that represent TypeScript declarations.
const TS_DECLARATION_KINDS: &[&str] = &[
    "function_declaration",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "method_definition",
    "lexical_declaration",
    "variable_declaration",
    "enum_declaration",
    "export_statement",
];

/// Node kinds that represent identifier nodes in TypeScript AST.
const TS_IDENTIFIER_KINDS: &[&str] = &[
    "identifier",
    "property_identifier",
    "type_identifier",
];

/// TypeScript language keywords to detect near comments.
const TS_KEYWORDS: &[&str] = &[
    "function",
    "class",
    "interface",
    "type",
    "const",
    "let",
    "var",
    "export",
    "import",
    "return",
    "if",
    "else",
    "for",
    "while",
    "async",
    "await",
    "enum",
    "extends",
    "implements",
    "new",
    "this",
    "super",
    "static",
    "abstract",
    "readonly",
    "private",
    "protected",
    "public",
];

impl Language for TypeScriptLanguage {
    fn id(&self) -> LanguageId {
        LanguageId::TypeScript
    }

    fn extensions(&self) -> &[&str] {
        &["ts", "tsx"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    }

    fn extract_comments(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        file_path: &Path,
    ) -> Vec<CommentContext> {
        let mut comments = Vec::new();
        let source_lines: Vec<&str> = source.lines().collect();
        let root = tree.root_node();

        collect_comments(&root, source, file_path, &source_lines, &mut comments);

        comments
    }
}

/// Recursively walk the AST and collect comment nodes.
fn collect_comments(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    source_lines: &[&str],
    comments: &mut Vec<CommentContext>,
) {
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        if child.kind() == "comment" {
            let text = child.utf8_text(source.as_bytes()).unwrap_or("");
            let (comment_text, comment_kind) = classify_and_strip_comment(text);

            let line = child.start_position().row + 1; // 1-based
            let column = child.start_position().column; // 0-based

            // Find adjacent (next sibling) declaration node
            let adjacent_node_kind = find_adjacent_declaration(&child);

            // Extract surrounding source (up to 3 lines before and after)
            let surrounding_source = extract_surrounding_source(source_lines, line);

            // Collect nearby identifiers and keywords from the adjacent/surrounding nodes
            let (nearby_identifiers, nearby_keywords) =
                extract_nearby_context(&child, source);

            comments.push(CommentContext {
                file_path: file_path.to_path_buf(),
                line,
                column,
                comment_text,
                comment_kind,
                language: LanguageId::TypeScript,
                adjacent_node_kind,
                surrounding_source,
                nearby_identifiers,
                nearby_keywords,
            });
        } else {
            collect_comments(&child, source, file_path, source_lines, comments);
        }
    }
}

/// Classify comment kind and strip delimiters to get clean text.
fn classify_and_strip_comment(text: &str) -> (String, CommentKind) {
    if text.starts_with("/**") {
        // JSDoc / Doc comment
        let stripped = strip_block_comment(text, "/**");
        (stripped, CommentKind::Doc)
    } else if text.starts_with("/*") {
        // Block comment
        let stripped = strip_block_comment(text, "/*");
        (stripped, CommentKind::Block)
    } else if text.starts_with("//") {
        // Line comment
        let stripped = text.trim_start_matches("//").trim().to_string();
        (stripped, CommentKind::Line)
    } else {
        (text.to_string(), CommentKind::Line)
    }
}

/// Strip block comment delimiters and clean up the text.
fn strip_block_comment(text: &str, open_delimiter: &str) -> String {
    let text = text.strip_prefix(open_delimiter).unwrap_or(text);
    let text = text.strip_suffix("*/").unwrap_or(text);

    // For multi-line block comments, strip leading " * " from each line
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() == 1 {
        return text.trim().to_string();
    }

    let cleaned: Vec<String> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("* ") {
                trimmed.strip_prefix("* ").unwrap_or(trimmed).to_string()
            } else if trimmed == "*" {
                String::new()
            } else {
                trimmed.to_string()
            }
        })
        .collect();

    cleaned
        .join("\n")
        .trim()
        .to_string()
}

/// Find the adjacent declaration node (the next named sibling after the comment).
fn find_adjacent_declaration(comment_node: &tree_sitter::Node) -> String {
    let mut sibling = comment_node.next_named_sibling();
    while let Some(node) = sibling {
        let kind = node.kind();
        if TS_DECLARATION_KINDS.contains(&kind) {
            return kind.to_string();
        }
        // If we hit a non-comment, non-declaration, stop looking
        if kind != "comment" {
            return kind.to_string();
        }
        sibling = node.next_named_sibling();
    }
    String::new()
}

/// Extract surrounding source lines (context window of +/- 3 lines).
fn extract_surrounding_source(source_lines: &[&str], comment_line: usize) -> String {
    let line_idx = comment_line.saturating_sub(1); // Convert 1-based to 0-based
    let start = line_idx.saturating_sub(3);
    let end = (line_idx + 4).min(source_lines.len());

    source_lines[start..end].join("\n")
}

/// Extract nearby identifiers and keywords from the context surrounding the comment.
fn extract_nearby_context(
    comment_node: &tree_sitter::Node,
    source: &str,
) -> (Vec<String>, Vec<String>) {
    let mut identifiers = Vec::new();
    let mut keywords = Vec::new();

    // Look at the next sibling (the adjacent declaration or statement)
    if let Some(next) = comment_node.next_named_sibling() {
        collect_identifiers_recursive(&next, source, &mut identifiers);
        collect_keywords_from_node(&next, source, &mut keywords);
    }

    // Also look at the parent node for context
    if let Some(parent) = comment_node.parent() {
        // Collect identifiers from the parent's children that are near the comment
        collect_keywords_from_node(&parent, source, &mut keywords);
    }

    // Deduplicate
    identifiers.sort();
    identifiers.dedup();
    keywords.sort();
    keywords.dedup();

    (identifiers, keywords)
}

/// Recursively collect identifier nodes from a subtree.
fn collect_identifiers_recursive(
    node: &tree_sitter::Node,
    source: &str,
    identifiers: &mut Vec<String>,
) {
    if TS_IDENTIFIER_KINDS.contains(&node.kind()) {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            let text = text.to_string();
            if !text.is_empty() && !identifiers.contains(&text) {
                identifiers.push(text);
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers_recursive(&child, source, identifiers);
    }
}

/// Collect keywords found in the text of a node and its children.
fn collect_keywords_from_node(
    node: &tree_sitter::Node,
    source: &str,
    keywords: &mut Vec<String>,
) {
    // Check direct children that are anonymous (keyword) nodes
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if !child.is_named() {
            if let Ok(text) = child.utf8_text(source.as_bytes()) {
                if TS_KEYWORDS.contains(&text) && !keywords.contains(&text.to_string()) {
                    keywords.push(text.to_string());
                }
            }
        }
        // Also check one level deeper for nested keyword tokens
        if child.is_named() && child.child_count() > 0 {
            let mut inner_cursor = child.walk();
            for grandchild in child.children(&mut inner_cursor) {
                if !grandchild.is_named() {
                    if let Ok(text) = grandchild.utf8_text(source.as_bytes()) {
                        if TS_KEYWORDS.contains(&text) && !keywords.contains(&text.to_string()) {
                            keywords.push(text.to_string());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn parse_and_extract(source: &str) -> Vec<CommentContext> {
        let lang = TypeScriptLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&lang.tree_sitter_language())
            .expect("Error loading TypeScript parser");
        let tree = parser.parse(source, None).unwrap();
        lang.extract_comments(&tree, source, Path::new("test.ts"))
    }

    #[test]
    fn id_returns_typescript() {
        let lang = TypeScriptLanguage;
        assert_eq!(lang.id(), LanguageId::TypeScript);
    }

    #[test]
    fn extensions_include_ts_and_tsx() {
        let lang = TypeScriptLanguage;
        let exts = lang.extensions();
        assert!(exts.contains(&"ts"));
        assert!(exts.contains(&"tsx"));
    }

    #[test]
    fn tree_sitter_language_can_parse() {
        let lang = TypeScriptLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang.tree_sitter_language()).unwrap();
        let tree = parser.parse("const x: number = 1;", None);
        assert!(tree.is_some());
    }

    #[test]
    fn extracts_line_comment() {
        let source = "// This is a line comment\nconst x = 1;\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "This is a line comment");
        assert_eq!(c.comment_kind, CommentKind::Line);
        assert_eq!(c.line, 1);
        assert_eq!(c.column, 0);
        assert_eq!(c.language, LanguageId::TypeScript);
        assert_eq!(c.file_path, PathBuf::from("test.ts"));
    }

    #[test]
    fn extracts_block_comment() {
        let source = "/* block comment */\nconst y = 2;\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "block comment");
        assert_eq!(c.comment_kind, CommentKind::Block);
        assert_eq!(c.language, LanguageId::TypeScript);
    }

    #[test]
    fn detects_jsdoc_as_doc_comment() {
        let source = "/** JSDoc comment */\nfunction greet() {}\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert_eq!(c.comment_text, "JSDoc comment");
        assert_eq!(c.comment_kind, CommentKind::Doc);
    }

    #[test]
    fn detects_multiline_jsdoc() {
        let source = r#"/**
 * Adds two numbers together.
 * @param a - First number.
 * @param b - Second number.
 * @returns The sum.
 */
function add(a: number, b: number): number {
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
    fn extracts_nearby_identifiers_from_function() {
        let source = r#"
// Greets the user
function greetUser(name: string): void {
    console.log(name);
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert!(
            c.nearby_identifiers.contains(&"greetUser".to_string()),
            "Expected 'greetUser' in nearby_identifiers: {:?}",
            c.nearby_identifiers
        );
    }

    #[test]
    fn extracts_nearby_identifiers_from_interface() {
        let source = r#"
// User profile interface
interface UserProfile {
    name: string;
    age: number;
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert!(
            c.nearby_identifiers.contains(&"UserProfile".to_string()),
            "Expected 'UserProfile' in nearby_identifiers: {:?}",
            c.nearby_identifiers
        );
    }

    #[test]
    fn extracts_nearby_identifiers_from_type_alias() {
        let source = r#"
// ID type alias
type UserId = string;
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert!(
            c.nearby_identifiers.contains(&"UserId".to_string()),
            "Expected 'UserId' in nearby_identifiers: {:?}",
            c.nearby_identifiers
        );
    }

    #[test]
    fn extracts_nearby_identifiers_from_class() {
        let source = r#"
/** Represents a user. */
class User {
    constructor(public name: string) {}
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let c = &comments[0];
        assert!(
            c.nearby_identifiers.contains(&"User".to_string()),
            "Expected 'User' in nearby_identifiers: {:?}",
            c.nearby_identifiers
        );
    }

    #[test]
    fn multiple_comments_extracted() {
        let source = r#"
// First comment
const a = 1;
/* Second comment */
const b = 2;
/** Third doc comment */
function foo() {}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 3);
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
        assert_eq!(comments[1].comment_kind, CommentKind::Block);
        assert_eq!(comments[2].comment_kind, CommentKind::Doc);
    }

    #[test]
    fn adjacent_node_kind_set_for_function() {
        let source = "/** Calculates sum */\nfunction sum(a: number, b: number): number { return a + b; }\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].adjacent_node_kind, "function_declaration");
    }

    #[test]
    fn adjacent_node_kind_set_for_interface() {
        let source = "// User interface\ninterface User { name: string; }\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].adjacent_node_kind, "interface_declaration");
    }

    #[test]
    fn adjacent_node_kind_set_for_type_alias() {
        let source = "// Type alias\ntype ID = string;\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].adjacent_node_kind, "type_alias_declaration");
    }

    #[test]
    fn surrounding_source_is_populated() {
        let source = "// a comment\nconst x = 1;\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert!(!comments[0].surrounding_source.is_empty());
        assert!(comments[0].surrounding_source.contains("// a comment"));
        assert!(comments[0].surrounding_source.contains("const x = 1;"));
    }

    #[test]
    fn extracts_nearby_keywords_from_function() {
        let source = r#"// Process data
function process(data: string[]): void {
    if (data.length === 0) {
        return;
    }
}
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let keywords = &comments[0].nearby_keywords;
        assert!(
            keywords.contains(&"function".to_string()),
            "Expected 'function' in nearby_keywords: {:?}",
            keywords
        );
    }

    #[test]
    fn block_comment_not_jsdoc() {
        let source = "/* just a block comment, not jsdoc */\nconst x = 1;\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_kind, CommentKind::Block);
    }

    #[test]
    fn jsdoc_before_class_declaration() {
        let source = r#"/**
 * Represents a user.
 */
class User {
    constructor(name: string) {
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
        let source = r#"function main(): void {
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
    fn extracts_nearby_identifiers_from_lexical_declaration() {
        let source = r#"// The default configuration
const defaultConfig = { timeout: 3000 };
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);

        let identifiers = &comments[0].nearby_identifiers;
        assert!(
            identifiers.contains(&"defaultConfig".to_string()),
            "Expected 'defaultConfig' in nearby_identifiers: {:?}",
            identifiers
        );
    }
}
