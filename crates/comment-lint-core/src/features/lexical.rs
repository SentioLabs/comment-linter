//! Lexical feature extraction: tokenization, Jaccard similarity, and identifier
//! substring matching.

use std::collections::HashSet;

use crate::extraction::comment::CommentContext;

/// Common English stop words to filter from comment tokens.
pub const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "to", "for", "of", "in", "it", "this", "that", "and", "or",
    "but", "not", "with", "from", "by", "on", "at", "as", "be", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can",
];

/// Strip comment markers (// # /* */ /** */ /// //!) from the beginning and end
/// of comment text, then return the cleaned text.
fn strip_comment_markers(text: &str) -> &str {
    let trimmed = text.trim();

    // Order matters: longer prefixes first to avoid partial matches.
    let prefixes = ["///", "//!", "//", "/**", "/*", "#"];
    let suffixes = ["*/"];

    let mut result = trimmed;

    for prefix in &prefixes {
        if let Some(rest) = result.strip_prefix(prefix) {
            result = rest;
            break;
        }
    }

    for suffix in &suffixes {
        if let Some(rest) = result.strip_suffix(suffix) {
            result = rest;
            break;
        }
    }

    result.trim()
}

/// Strip comment markers and tokenize comment text into meaningful tokens.
///
/// 1. Strip comment markers (// # /* */ /** */ /// //!)
/// 2. Lowercase
/// 3. Split on whitespace and punctuation
/// 4. Remove stop words
/// 5. Remove single-char tokens
pub fn tokenize_comment(text: &str) -> Vec<String> {
    let stripped = strip_comment_markers(text);
    let lowered = stripped.to_lowercase();

    let stop_words: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    lowered
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .filter(|s| s.len() > 1)
        .filter(|s| !stop_words.contains(s.as_str()))
        .collect()
}

/// Split identifiers on camelCase (before uppercase) and snake_case (on
/// underscore) boundaries, lowercase each part, and skip single-char parts.
pub fn tokenize_identifiers(identifiers: &[String]) -> Vec<String> {
    let mut tokens = Vec::new();

    for ident in identifiers {
        // First split on underscores for snake_case
        for segment in ident.split('_') {
            // Then split camelCase within each segment
            let parts = split_camel_case(segment);
            for part in parts {
                let lowered = part.to_lowercase();
                if lowered.len() > 1 {
                    tokens.push(lowered);
                }
            }
        }
    }

    tokens
}

/// Split a string on camelCase boundaries.
/// For example, "getUserName" -> ["get", "User", "Name"]
fn split_camel_case(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();

    for ch in s.chars() {
        if ch.is_uppercase() && !current.is_empty() {
            parts.push(current);
            current = String::new();
        }
        current.push(ch);
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Compute Jaccard similarity between two token sets: |intersection| / |union|.
///
/// Returns 0.0 if both sets are empty.
pub fn jaccard_similarity(set_a: &[String], set_b: &[String]) -> f32 {
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }

    let a: HashSet<&str> = set_a.iter().map(|s| s.as_str()).collect();
    let b: HashSet<&str> = set_b.iter().map(|s| s.as_str()).collect();

    let intersection = a.intersection(&b).count();
    let union = a.union(&b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Fraction of comment tokens that appear as substrings in any identifier.
///
/// The substring check is case-insensitive. Returns 0.0 if either input is
/// empty.
pub fn identifier_substring_ratio(comment_tokens: &[String], identifiers: &[String]) -> f32 {
    if comment_tokens.is_empty() || identifiers.is_empty() {
        return 0.0;
    }

    let lowered_identifiers: Vec<String> = identifiers.iter().map(|s| s.to_lowercase()).collect();

    let matches = comment_tokens
        .iter()
        .filter(|token| {
            let lower_token = token.to_lowercase();
            lowered_identifiers
                .iter()
                .any(|ident| ident.contains(lower_token.as_str()))
        })
        .count();

    matches as f32 / comment_tokens.len() as f32
}

/// Extract lexical features from a comment context.
///
/// Returns `(jaccard_similarity, identifier_substring_ratio, token_count)`.
pub fn extract_lexical_features(context: &CommentContext) -> (f32, f32, usize) {
    let comment_tokens = tokenize_comment(&context.comment_text);
    let identifier_tokens = tokenize_identifiers(&context.nearby_identifiers);

    let token_count = comment_tokens.len();

    if token_count == 0 {
        return (0.0, 0.0, 0);
    }

    let jaccard = jaccard_similarity(&comment_tokens, &identifier_tokens);
    let substr_ratio = identifier_substring_ratio(&comment_tokens, &context.nearby_identifiers);

    (jaccard, substr_ratio, token_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::types::{CommentKind, LanguageId};

    fn make_context(comment_text: &str, identifiers: Vec<String>) -> CommentContext {
        CommentContext {
            file_path: PathBuf::from("test.rs"),
            line: 1,
            column: 0,
            comment_text: comment_text.to_string(),
            comment_kind: CommentKind::Line,
            language: LanguageId::Rust,
            adjacent_node_kind: String::new(),
            surrounding_source: String::new(),
            nearby_identifiers: identifiers,
            nearby_keywords: Vec::new(),
        }
    }

    // ---- tokenize_comment tests ----

    #[test]
    fn tokenize_comment_strips_line_comment_markers() {
        let tokens = tokenize_comment("// increment the counter");
        assert!(!tokens.contains(&"//".to_string()));
        assert!(tokens.contains(&"increment".to_string()));
        assert!(tokens.contains(&"counter".to_string()));
    }

    #[test]
    fn tokenize_comment_strips_hash_comment_markers() {
        let tokens = tokenize_comment("# increment the counter");
        assert!(!tokens.contains(&"#".to_string()));
        assert!(tokens.contains(&"increment".to_string()));
        assert!(tokens.contains(&"counter".to_string()));
    }

    #[test]
    fn tokenize_comment_strips_block_comment_markers() {
        let tokens = tokenize_comment("/* increment the counter */");
        assert!(!tokens.contains(&"/*".to_string()));
        assert!(!tokens.contains(&"*/".to_string()));
        assert!(tokens.contains(&"increment".to_string()));
        assert!(tokens.contains(&"counter".to_string()));
    }

    #[test]
    fn tokenize_comment_strips_doc_comment_markers() {
        let tokens = tokenize_comment("/// increment the counter");
        assert!(!tokens.contains(&"///".to_string()));
        assert!(tokens.contains(&"increment".to_string()));

        let tokens2 = tokenize_comment("/** increment the counter */");
        assert!(!tokens2.contains(&"/**".to_string()));
        assert!(tokens2.contains(&"increment".to_string()));

        let tokens3 = tokenize_comment("//! increment the counter");
        assert!(!tokens3.contains(&"//!".to_string()));
        assert!(tokens3.contains(&"increment".to_string()));
    }

    #[test]
    fn tokenize_comment_removes_stop_words() {
        let tokens = tokenize_comment("// this is the counter for a loop");
        // "this", "is", "the", "for", "a" are all stop words
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"for".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"counter".to_string()));
        assert!(tokens.contains(&"loop".to_string()));
    }

    #[test]
    fn tokenize_comment_lowercases_tokens() {
        let tokens = tokenize_comment("// Increment Counter");
        assert!(tokens.contains(&"increment".to_string()));
        assert!(tokens.contains(&"counter".to_string()));
        assert!(!tokens.contains(&"Increment".to_string()));
    }

    #[test]
    fn tokenize_comment_removes_single_char_tokens() {
        let tokens = tokenize_comment("// x is a variable");
        assert!(!tokens.contains(&"x".to_string()));
        // "is" and "a" are stop words too
        assert!(tokens.contains(&"variable".to_string()));
    }

    #[test]
    fn tokenize_comment_splits_on_punctuation() {
        let tokens = tokenize_comment("// increment,counter.value");
        assert!(tokens.contains(&"increment".to_string()));
        assert!(tokens.contains(&"counter".to_string()));
        assert!(tokens.contains(&"value".to_string()));
    }

    #[test]
    fn tokenize_comment_empty_input_returns_empty() {
        let tokens = tokenize_comment("");
        assert!(tokens.is_empty());
    }

    // ---- tokenize_identifiers tests ----

    #[test]
    fn tokenize_identifiers_splits_camel_case() {
        let ids = vec!["getUserName".to_string()];
        let tokens = tokenize_identifiers(&ids);
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"name".to_string()));
    }

    #[test]
    fn tokenize_identifiers_splits_snake_case() {
        let ids = vec!["get_user_name".to_string()];
        let tokens = tokenize_identifiers(&ids);
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"name".to_string()));
    }

    #[test]
    fn tokenize_identifiers_lowercases_all_parts() {
        let ids = vec!["GetUserName".to_string()];
        let tokens = tokenize_identifiers(&ids);
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"name".to_string()));
    }

    #[test]
    fn tokenize_identifiers_skips_single_char_parts() {
        let ids = vec!["a_big_value".to_string()];
        let tokens = tokenize_identifiers(&ids);
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"big".to_string()));
        assert!(tokens.contains(&"value".to_string()));
    }

    #[test]
    fn tokenize_identifiers_empty_input() {
        let tokens = tokenize_identifiers(&[]);
        assert!(tokens.is_empty());
    }

    // ---- jaccard_similarity tests ----

    #[test]
    fn jaccard_identical_sets() {
        let a = vec!["increment".to_string(), "counter".to_string()];
        let b = vec!["increment".to_string(), "counter".to_string()];
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = vec!["increment".to_string(), "counter".to_string()];
        let b = vec!["fetch".to_string(), "data".to_string()];
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec!["increment".to_string(), "counter".to_string(), "value".to_string()];
        let b = vec!["counter".to_string(), "value".to_string(), "total".to_string()];
        // intersection = {counter, value} = 2
        // union = {increment, counter, value, total} = 4
        // similarity = 2/4 = 0.5
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_empty_sets_returns_zero() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn jaccard_one_empty_set_returns_zero() {
        let a = vec!["increment".to_string()];
        let b: Vec<String> = vec![];
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    // ---- identifier_substring_ratio tests ----

    #[test]
    fn identifier_substring_ratio_full_match() {
        let comment_tokens = vec!["user".to_string(), "name".to_string()];
        let identifiers = vec!["getUserName".to_string()];
        let ratio = identifier_substring_ratio(&comment_tokens, &identifiers);
        assert!((ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn identifier_substring_ratio_no_match() {
        let comment_tokens = vec!["increment".to_string(), "counter".to_string()];
        let identifiers = vec!["fetchData".to_string()];
        let ratio = identifier_substring_ratio(&comment_tokens, &identifiers);
        assert!((ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn identifier_substring_ratio_partial_match() {
        let comment_tokens = vec!["user".to_string(), "count".to_string()];
        let identifiers = vec!["userName".to_string()];
        // "user" is a substring of "userName" but "count" is not
        let ratio = identifier_substring_ratio(&comment_tokens, &identifiers);
        assert!((ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn identifier_substring_ratio_empty_tokens_returns_zero() {
        let comment_tokens: Vec<String> = vec![];
        let identifiers = vec!["getUserName".to_string()];
        let ratio = identifier_substring_ratio(&comment_tokens, &identifiers);
        assert!((ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn identifier_substring_ratio_empty_identifiers_returns_zero() {
        let comment_tokens = vec!["user".to_string()];
        let identifiers: Vec<String> = vec![];
        let ratio = identifier_substring_ratio(&comment_tokens, &identifiers);
        assert!((ratio - 0.0).abs() < f32::EPSILON);
    }

    // ---- extract_lexical_features tests ----

    #[test]
    fn extract_lexical_features_returns_correct_token_count() {
        let ctx = make_context(
            "// increment the counter value",
            vec!["counter".to_string()],
        );
        let (_, _, token_count) = extract_lexical_features(&ctx);
        // "the" is a stop word; "increment", "counter", "value" remain
        assert_eq!(token_count, 3);
    }

    #[test]
    fn extract_lexical_features_computes_jaccard() {
        let ctx = make_context(
            "// increment counter",
            vec!["incrementCounter".to_string()],
        );
        let (jaccard, _, _) = extract_lexical_features(&ctx);
        // comment tokens: ["increment", "counter"]
        // identifier tokens: ["increment", "counter"]
        // jaccard = 2/2 = 1.0
        assert!((jaccard - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn extract_lexical_features_computes_substr_ratio() {
        let ctx = make_context(
            "// increment counter",
            vec!["incrementCounter".to_string()],
        );
        let (_, substr_ratio, _) = extract_lexical_features(&ctx);
        // Both "increment" and "counter" are substrings of "incrementCounter" (case-insensitive)
        assert!((substr_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn extract_lexical_features_empty_comment() {
        let ctx = make_context("//", vec!["something".to_string()]);
        let (jaccard, substr_ratio, token_count) = extract_lexical_features(&ctx);
        assert!((jaccard - 0.0).abs() < f32::EPSILON);
        assert!((substr_ratio - 0.0).abs() < f32::EPSILON);
        assert_eq!(token_count, 0);
    }
}
