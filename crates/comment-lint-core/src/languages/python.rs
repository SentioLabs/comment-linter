//! Python language support via tree-sitter-python.

use std::path::{Path, PathBuf};

use crate::extraction::comment::CommentContext;
use crate::languages::Language;
use crate::types::{CommentKind, LanguageId};

/// Python language implementation for comment extraction.
pub struct PythonLanguage;

/// Python keywords used for nearby keyword extraction.
const PYTHON_KEYWORDS: &[&str] = &[
    "def", "class", "if", "elif", "else", "for", "while", "try", "except", "finally", "with",
    "return", "yield", "import", "from", "as", "pass", "break", "continue", "raise", "global",
    "nonlocal", "assert", "del", "lambda", "async", "await",
];

/// Python declaration node kinds for adjacent-node detection.
const DECLARATION_NODES: &[&str] = &[
    "function_definition",
    "class_definition",
    "decorated_definition",
];

/// Node kinds that hold identifiers.
const IDENTIFIER_NODES: &[&str] = &["identifier", "attribute"];

impl Language for PythonLanguage {
    fn id(&self) -> LanguageId {
        LanguageId::Python
    }

    fn extensions(&self) -> &[&str] {
        &["py"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn extract_comments(
        &self,
        tree: &tree_sitter::Tree,
        source: &str,
        file_path: &Path,
    ) -> Vec<CommentContext> {
        let mut comments = Vec::new();
        let source_bytes = source.as_bytes();
        let root = tree.root_node();

        Self::walk_node(root, source, source_bytes, file_path, &mut comments);

        comments
    }
}

impl PythonLanguage {
    /// Recursively walk the AST extracting comments and docstrings.
    fn walk_node(
        node: tree_sitter::Node,
        source: &str,
        source_bytes: &[u8],
        file_path: &Path,
        comments: &mut Vec<CommentContext>,
    ) {
        let kind = node.kind();

        if kind == "comment" {
            if let Some(ctx) = Self::extract_line_comment(node, source, source_bytes, file_path) {
                comments.push(ctx);
            }
        } else if kind == "expression_statement" {
            if let Some(ctx) = Self::extract_docstring(node, source, source_bytes, file_path) {
                comments.push(ctx);
            }
        }

        let child_count = node.child_count();
        for i in 0..child_count {
            if let Some(child) = node.child(i) {
                Self::walk_node(child, source, source_bytes, file_path, comments);
            }
        }
    }

    /// Extract a line comment (# ...) from a comment node.
    fn extract_line_comment(
        node: tree_sitter::Node,
        source: &str,
        source_bytes: &[u8],
        file_path: &Path,
    ) -> Option<CommentContext> {
        let text = node.utf8_text(source_bytes).ok()?;
        // Strip "# " or "#" prefix
        let comment_text = text
            .strip_prefix("# ")
            .or_else(|| text.strip_prefix("#"))
            .unwrap_or(text)
            .to_string();

        let line = node.start_position().row + 1; // 1-based
        let column = node.start_position().column; // 0-based
        let adjacent_node_kind = Self::find_adjacent_declaration(node);
        let surrounding_source = Self::get_surrounding_source(source, node.start_position().row);
        let (nearby_identifiers, nearby_keywords) =
            Self::collect_nearby_context(node, source_bytes);

        Some(CommentContext {
            file_path: PathBuf::from(file_path),
            line,
            column,
            comment_text,
            comment_kind: CommentKind::Line,
            language: LanguageId::Python,
            adjacent_node_kind,
            surrounding_source,
            nearby_identifiers,
            nearby_keywords,
        })
    }

    /// Determine if an expression_statement containing a string is a docstring.
    ///
    /// A docstring is a string literal that is the first statement in a
    /// function_definition body, class_definition body, or at module level.
    fn extract_docstring(
        node: tree_sitter::Node,
        source: &str,
        source_bytes: &[u8],
        file_path: &Path,
    ) -> Option<CommentContext> {
        // The node must be an expression_statement containing a string child.
        let string_node = node.child(0)?;
        if string_node.kind() != "string" {
            return None;
        }

        let string_text = string_node.utf8_text(source_bytes).ok()?;
        // Must be a triple-quoted string.
        if !(string_text.starts_with("\"\"\"") || string_text.starts_with("'''")) {
            return None;
        }

        // Check if this expression_statement is the first statement in its parent.
        if !Self::is_first_statement(node) {
            return None;
        }

        // Strip triple quotes and trim whitespace.
        let comment_text = Self::strip_triple_quotes(string_text);

        let line = node.start_position().row + 1;
        let column = node.start_position().column;
        let adjacent_node_kind = Self::find_adjacent_declaration(node);
        let surrounding_source = Self::get_surrounding_source(source, node.start_position().row);
        let (nearby_identifiers, nearby_keywords) =
            Self::collect_nearby_context(node, source_bytes);

        Some(CommentContext {
            file_path: PathBuf::from(file_path),
            line,
            column,
            comment_text,
            comment_kind: CommentKind::Doc,
            language: LanguageId::Python,
            adjacent_node_kind,
            surrounding_source,
            nearby_identifiers,
            nearby_keywords,
        })
    }

    /// Check if a node is the first statement in its parent block or module.
    fn is_first_statement(node: tree_sitter::Node) -> bool {
        let parent = match node.parent() {
            Some(p) => p,
            None => return false,
        };

        let parent_kind = parent.kind();

        // Module-level: parent is "module", this must be the first non-comment child.
        if parent_kind == "module" {
            return Self::is_first_non_comment_child(parent, node);
        }

        // Inside a block (function/class body): parent is "block"
        if parent_kind == "block" {
            if Self::is_first_non_comment_child(parent, node) {
                // Verify the block's parent is a function_definition, class_definition,
                // or decorated_definition.
                if let Some(grandparent) = parent.parent() {
                    let gp_kind = grandparent.kind();
                    return gp_kind == "function_definition"
                        || gp_kind == "class_definition"
                        || gp_kind == "decorated_definition";
                }
            }
        }

        false
    }

    /// Check if `target` is the first non-comment child of `parent`.
    fn is_first_non_comment_child(parent: tree_sitter::Node, target: tree_sitter::Node) -> bool {
        let child_count = parent.child_count();
        for i in 0..child_count {
            if let Some(child) = parent.child(i) {
                // Skip comment nodes when looking for first statement.
                if child.kind() == "comment" {
                    continue;
                }
                return child.id() == target.id();
            }
        }
        false
    }

    /// Strip triple quotes (""" or ''') from a docstring and trim whitespace.
    fn strip_triple_quotes(s: &str) -> String {
        let stripped = s
            .strip_prefix("\"\"\"")
            .or_else(|| s.strip_prefix("'''"))
            .unwrap_or(s);
        let stripped = stripped
            .strip_suffix("\"\"\"")
            .or_else(|| stripped.strip_suffix("'''"))
            .unwrap_or(stripped);
        stripped.trim().to_string()
    }

    /// Find the nearest adjacent declaration node (sibling or parent).
    fn find_adjacent_declaration(node: tree_sitter::Node) -> String {
        // Check next sibling
        if let Some(next) = node.next_named_sibling() {
            if DECLARATION_NODES.contains(&next.kind()) {
                return next.kind().to_string();
            }
        }

        // Check parent chain
        let mut current = node.parent();
        while let Some(parent) = current {
            if DECLARATION_NODES.contains(&parent.kind()) {
                return parent.kind().to_string();
            }
            // For nodes inside a block, check the block's parent
            if parent.kind() == "block" {
                if let Some(gp) = parent.parent() {
                    if DECLARATION_NODES.contains(&gp.kind()) {
                        return gp.kind().to_string();
                    }
                }
            }
            current = parent.parent();
        }

        String::new()
    }

    /// Get surrounding source lines (up to 2 lines before and after the comment).
    fn get_surrounding_source(source: &str, row: usize) -> String {
        let lines: Vec<&str> = source.lines().collect();
        let start = row.saturating_sub(2);
        let end = (row + 3).min(lines.len());
        lines[start..end].join("\n")
    }

    /// Collect nearby identifiers and keywords from sibling and parent nodes.
    fn collect_nearby_context(
        node: tree_sitter::Node,
        source_bytes: &[u8],
    ) -> (Vec<String>, Vec<String>) {
        let mut identifiers = Vec::new();
        let mut keywords = Vec::new();

        // Collect from parent context (e.g., function_definition, class_definition)
        let mut current = node.parent();
        while let Some(parent) = current {
            let parent_kind = parent.kind();
            if DECLARATION_NODES.contains(&parent_kind)
                || parent_kind == "module"
            {
                Self::collect_identifiers_from_node(parent, source_bytes, &mut identifiers);
                Self::collect_keywords_from_node(parent, source_bytes, &mut keywords);
                break;
            }
            if parent_kind == "block" {
                // Look at block's parent (the declaration)
                if let Some(gp) = parent.parent() {
                    if DECLARATION_NODES.contains(&gp.kind()) {
                        Self::collect_identifiers_from_node(gp, source_bytes, &mut identifiers);
                        Self::collect_keywords_from_node(gp, source_bytes, &mut keywords);
                        break;
                    }
                }
            }
            current = parent.parent();
        }

        // Also check next sibling for adjacent declarations
        if let Some(next) = node.next_named_sibling() {
            if DECLARATION_NODES.contains(&next.kind()) {
                Self::collect_identifiers_from_node(next, source_bytes, &mut identifiers);
                Self::collect_keywords_from_node(next, source_bytes, &mut keywords);
            }
        }

        // Deduplicate
        identifiers.sort();
        identifiers.dedup();
        keywords.sort();
        keywords.dedup();

        (identifiers, keywords)
    }

    /// Recursively collect identifier names from a node and its children.
    fn collect_identifiers_from_node(
        node: tree_sitter::Node,
        source_bytes: &[u8],
        identifiers: &mut Vec<String>,
    ) {
        if IDENTIFIER_NODES.contains(&node.kind()) {
            if let Ok(text) = node.utf8_text(source_bytes) {
                let text = text.to_string();
                if !identifiers.contains(&text) {
                    identifiers.push(text);
                }
            }
        }

        let child_count = node.child_count();
        for i in 0..child_count {
            if let Some(child) = node.child(i) {
                // Don't recurse into nested function/class bodies to avoid noise
                if child.kind() == "block" {
                    continue;
                }
                Self::collect_identifiers_from_node(child, source_bytes, identifiers);
            }
        }
    }

    /// Collect Python keywords from a node's text tokens.
    fn collect_keywords_from_node(
        node: tree_sitter::Node,
        source_bytes: &[u8],
        keywords: &mut Vec<String>,
    ) {
        if let Ok(text) = node.utf8_text(source_bytes) {
            if PYTHON_KEYWORDS.contains(&text) && !keywords.contains(&text.to_string()) {
                keywords.push(text.to_string());
            }
        }

        let child_count = node.child_count();
        for i in 0..child_count {
            if let Some(child) = node.child(i) {
                if child.kind() == "block" {
                    continue;
                }
                Self::collect_keywords_from_node(child, source_bytes, keywords);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn parse_and_extract(source: &str) -> Vec<CommentContext> {
        let lang = PythonLanguage;
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&lang.tree_sitter_language())
            .expect("failed to set language");
        let tree = parser.parse(source, None).expect("failed to parse");
        lang.extract_comments(&tree, source, Path::new("test.py"))
    }

    #[test]
    fn extracts_line_comment() {
        let source = "# This is a comment\nx = 1\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_text, "This is a comment");
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
        assert_eq!(comments[0].line, 1);
        assert_eq!(comments[0].column, 0);
        assert_eq!(comments[0].language, LanguageId::Python);
        assert_eq!(comments[0].file_path, PathBuf::from("test.py"));
    }

    #[test]
    fn extracts_inline_comment() {
        let source = "x = 1  # inline comment\n";
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_text, "inline comment");
        assert_eq!(comments[0].comment_kind, CommentKind::Line);
        assert_eq!(comments[0].line, 1);
        assert_eq!(comments[0].column, 7);
    }

    #[test]
    fn extracts_function_docstring() {
        let source = r#"def greet(name):
    """Say hello to the user."""
    print(f"Hello, {name}")
"#;
        let comments = parse_and_extract(source);
        let docs: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Doc)
            .collect();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].comment_text, "Say hello to the user.");
    }

    #[test]
    fn extracts_class_docstring() {
        let source = r#"class Greeter:
    """A class that greets."""
    pass
"#;
        let comments = parse_and_extract(source);
        let docs: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Doc)
            .collect();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].comment_text, "A class that greets.");
    }

    #[test]
    fn extracts_module_docstring() {
        let source = r#""""This is a module docstring."""

x = 1
"#;
        let comments = parse_and_extract(source);
        let docs: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Doc)
            .collect();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].comment_text, "This is a module docstring.");
    }

    #[test]
    fn non_first_string_is_not_docstring() {
        let source = r#"def foo():
    x = 1
    """This is NOT a docstring."""
    pass
"#;
        let comments = parse_and_extract(source);
        let docs: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Doc)
            .collect();
        assert_eq!(docs.len(), 0);
    }

    #[test]
    fn extracts_nearby_identifiers_from_def() {
        let source = r#"def calculate(price, tax):
    # Compute result
    return price * tax
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_text, "Compute result");
        // Should find identifiers from the function signature
        let ids = &comments[0].nearby_identifiers;
        assert!(ids.contains(&"calculate".to_string()));
        assert!(ids.contains(&"price".to_string()));
        assert!(ids.contains(&"tax".to_string()));
    }

    #[test]
    fn extracts_nearby_keywords() {
        let source = r#"def calculate(price, tax):
    # Compute result
    return price * tax
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        let kw = &comments[0].nearby_keywords;
        assert!(kw.contains(&"def".to_string()));
    }

    #[test]
    fn adjacent_node_kind_for_function_comment() {
        let source = r#"# Function comment
def hello():
    pass
"#;
        let comments = parse_and_extract(source);
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].adjacent_node_kind, "function_definition");
    }

    #[test]
    fn multiline_docstring() {
        let source = r#"def foo():
    """
    Multi-line
    docstring here.
    """
    pass
"#;
        let comments = parse_and_extract(source);
        let docs: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Doc)
            .collect();
        assert_eq!(docs.len(), 1);
        let text = &docs[0].comment_text;
        assert!(text.contains("Multi-line"));
        assert!(text.contains("docstring here."));
    }

    #[test]
    fn id_returns_python() {
        assert_eq!(PythonLanguage.id(), LanguageId::Python);
    }

    #[test]
    fn extensions_include_py() {
        let exts = PythonLanguage.extensions();
        assert!(exts.contains(&"py"));
    }

    #[test]
    fn decorated_function_docstring() {
        let source = r#"@staticmethod
def helper():
    """Helper docstring."""
    pass
"#;
        let comments = parse_and_extract(source);
        let docs: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Doc)
            .collect();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].comment_text, "Helper docstring.");
    }

    #[test]
    fn multiple_comments_extracted() {
        let source = r#"# First comment
x = 1
# Second comment
y = 2
"#;
        let comments = parse_and_extract(source);
        let lines: Vec<_> = comments
            .iter()
            .filter(|c| c.comment_kind == CommentKind::Line)
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].comment_text, "First comment");
        assert_eq!(lines[1].comment_text, "Second comment");
    }

    #[test]
    fn surrounding_source_is_populated() {
        let source = "# A comment\nx = 1\n";
        let comments = parse_and_extract(source);
        assert!(!comments[0].surrounding_source.is_empty());
    }
}
