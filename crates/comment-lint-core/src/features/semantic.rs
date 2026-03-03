//! Semantic feature extraction: why-indicators, external references,
//! imperative-verb-noun patterns, and section labels.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::lexical::{split_camel_case, STOP_WORDS};

// ── Compiled regexes ────────────────────────────────────────────────

static WHY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(because|since|workaround|hack|todo|fixme|note|nb|caveat|reason|due to|prevents|ensures|invariant|assumes|legacy|compat|backward|temporary|deprecated)\b")
        .expect("WHY_RE must compile")
});

static EXTERNAL_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(https?://\S+|#\d+|[A-Z]+-\d+|\bRFC\s*\d+|\bCVE-\d{4}-\d+)")
        .expect("EXTERNAL_REF_RE must compile")
});

/// Imperative verbs commonly found at the start of code comments.
const IMPERATIVE_VERBS: &[&str] = &[
    "get",
    "set",
    "check",
    "create",
    "initialize",
    "init",
    "fetch",
    "update",
    "delete",
    "remove",
    "add",
    "build",
    "parse",
    "validate",
    "compute",
    "calculate",
    "convert",
    "handle",
    "process",
    "return",
    "send",
    "load",
    "save",
    "store",
    "open",
    "close",
    "start",
    "stop",
    "run",
    "execute",
    "read",
    "write",
    "find",
    "search",
    "sort",
    "filter",
    "map",
    "reduce",
    "use",
    "call",
    "wait",
    "define",
    "configure",
    "apply",
    "ensure",
    "verify",
    "print",
    "log",
    "append",
    "insert",
    "merge",
    "copy",
    "format",
    "generate",
    "render",
    "connect",
    "disconnect",
    "register",
    "subscribe",
    "publish",
    "listen",
    "emit",
    "dispatch",
    "setup",
    "wrap",
    "mount",
];

/// Characters used as visual dividers in section labels.
const DIVIDER_CHARS: &[char] = &['-', '=', '*', '#'];

// ── Public functions ────────────────────────────────────────────────

/// Returns `true` if `text` contains a "why" indicator word or phrase.
pub fn has_why_indicator(text: &str) -> bool {
    WHY_RE.is_match(text)
}

/// Returns `true` if `text` contains an external reference (URL, ticket ID,
/// RFC, or CVE).
pub fn has_external_ref(text: &str) -> bool {
    EXTERNAL_REF_RE.is_match(text)
}

/// Returns `true` when the comment starts with an imperative verb AND
/// at least 33% of the remaining nouns (after filtering stop words and
/// single-char tokens) match nearby identifiers via case-insensitive
/// substring matching.
pub fn imperative_verb_noun(text: &str, nearby_identifiers: &[String]) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }

    let first = words[0].to_lowercase();
    if !IMPERATIVE_VERBS.contains(&first.as_str()) {
        return false;
    }

    let stop_words: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    let nouns: Vec<String> = words[1..]
        .iter()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1)
        .filter(|w| !stop_words.contains(w.as_str()))
        .collect();

    if nouns.is_empty() {
        return false;
    }

    let lowered_identifiers: Vec<String> = nearby_identifiers
        .iter()
        .map(|id| id.to_lowercase())
        .collect();

    let matching = nouns
        .iter()
        .filter(|noun| {
            lowered_identifiers
                .iter()
                .any(|ident| ident.contains(noun.as_str()))
        })
        .count();

    // At least 33% of nouns match (stop word removal already filters filler).
    matching * 3 >= nouns.len()
}

/// Returns `true` when the comment's verb+noun pattern matches a nearby
/// identifier's structure. For example, "delete user from database" matches
/// `DeleteUserByAuth0ID` because verb "delete" matches the first part and
/// noun "user" matches another part.
pub fn verb_noun_matches_identifier(text: &str, nearby_identifiers: &[String]) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }

    let verb = words[0].to_lowercase();
    if !IMPERATIVE_VERBS.contains(&verb.as_str()) {
        return false;
    }

    let stop_words: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    let nouns: Vec<String> = words[1..]
        .iter()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1)
        .filter(|w| !stop_words.contains(w.as_str()))
        .collect();

    if nouns.is_empty() {
        return false;
    }

    for ident in nearby_identifiers {
        // Split identifier by snake_case and camelCase into lowercased parts
        let parts: Vec<String> = ident
            .split('_')
            .flat_map(split_camel_case)
            .map(|p| p.to_lowercase())
            .filter(|p| p.len() > 1)
            .collect();

        if parts.is_empty() {
            continue;
        }

        // Verb must match the first part of the identifier
        if parts[0] != verb {
            continue;
        }

        // At least one noun must match any other part
        let other_parts = &parts[1..];
        let noun_matches = nouns
            .iter()
            .any(|noun| other_parts.iter().any(|part| part == noun));

        if noun_matches {
            return true;
        }
    }

    false
}

/// Returns `true` if `text` looks like a section label: at most 4 words after
/// stripping divider characters, and the first word is not an imperative verb.
pub fn is_section_label(text: &str) -> bool {
    let stripped: String = text
        .chars()
        .filter(|c| !DIVIDER_CHARS.contains(c))
        .collect();

    let words: Vec<&str> = stripped.split_whitespace().collect();

    if words.is_empty() || words.len() > 4 {
        return false;
    }

    let first = words[0].to_lowercase();
    !IMPERATIVE_VERBS.contains(&first.as_str())
}

// ── Aggregate struct ────────────────────────────────────────────────

/// All semantic features extracted from a single comment.
#[derive(Debug, Clone)]
pub struct SemanticFeatures {
    pub has_why_indicator: bool,
    pub has_external_ref: bool,
    pub imperative_verb_noun: bool,
    pub verb_noun_matches_identifier: bool,
    pub is_section_label: bool,
}

/// Extract all semantic features from `text` given `nearby_identifiers`.
pub fn extract_semantic_features(text: &str, nearby_identifiers: &[String]) -> SemanticFeatures {
    SemanticFeatures {
        has_why_indicator: has_why_indicator(text),
        has_external_ref: has_external_ref(text),
        imperative_verb_noun: imperative_verb_noun(text, nearby_identifiers),
        verb_noun_matches_identifier: verb_noun_matches_identifier(text, nearby_identifiers),
        is_section_label: is_section_label(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── has_why_indicator ───────────────────────────────────────────

    #[test]
    fn why_indicator_workaround() {
        assert!(has_why_indicator("workaround for bug"));
    }

    #[test]
    fn why_indicator_todo() {
        assert!(has_why_indicator("TODO fix"));
    }

    #[test]
    fn why_indicator_negative() {
        assert!(!has_why_indicator("Get columns"));
    }

    #[test]
    fn why_indicator_case_insensitive() {
        assert!(has_why_indicator("BECAUSE of a race condition"));
    }

    #[test]
    fn why_indicator_legacy() {
        assert!(has_why_indicator("legacy compatibility shim"));
    }

    // ── has_external_ref ────────────────────────────────────────────

    #[test]
    fn external_ref_url() {
        assert!(has_external_ref("see https://example.com/issue"));
    }

    #[test]
    fn external_ref_jira_ticket() {
        assert!(has_external_ref("JIRA-456"));
    }

    #[test]
    fn external_ref_github_issue() {
        assert!(has_external_ref("fixes #123"));
    }

    #[test]
    fn external_ref_rfc() {
        assert!(has_external_ref("per RFC 7231"));
    }

    #[test]
    fn external_ref_cve() {
        assert!(has_external_ref("mitigates CVE-2023-12345"));
    }

    #[test]
    fn external_ref_plain_text() {
        assert!(!has_external_ref("plain text comment"));
    }

    // ── imperative_verb_noun ────────────────────────────────────────

    #[test]
    fn imperative_verb_noun_with_matching_identifiers() {
        let identifiers = vec!["columns".to_string(), "rows".to_string()];
        assert!(imperative_verb_noun("Get columns", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_no_identifiers() {
        let identifiers: Vec<String> = vec![];
        assert!(!imperative_verb_noun("Get columns", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_non_imperative() {
        let identifiers = vec!["columns".to_string()];
        assert!(!imperative_verb_noun("The columns are wide", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_partial_overlap_passes_with_lower_threshold() {
        // "validate input data" -> nouns (after stop word filter): ["input", "data"]
        // identifiers has "input" -> 1/2 = 50% >= 33% -> true
        let identifiers = vec!["input".to_string()];
        assert!(imperative_verb_noun("validate input data", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_below_threshold() {
        // "validate input data records" -> nouns: ["input", "data", "records"]
        // identifiers has none -> 0/3 = 0% -> false
        let identifiers = vec!["something".to_string()];
        assert!(!imperative_verb_noun(
            "validate input data records",
            &identifiers
        ));
    }

    #[test]
    fn imperative_verb_noun_full_overlap() {
        let identifiers = vec!["input".to_string(), "data".to_string()];
        assert!(imperative_verb_noun("validate input data", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_filters_stop_words() {
        // "get the user object from the database" -> nouns after stop words: ["user", "object", "database"]
        // "user" is a substring of "GetUserByAuth0ID" (lowercased)
        let identifiers = vec!["GetUserByAuth0ID".to_string()];
        assert!(imperative_verb_noun(
            "get the user object from the database",
            &identifiers
        ));
    }

    #[test]
    fn imperative_verb_noun_substring_matching() {
        // "delete user" -> noun "user" is substring of "deleteUserByAuth0ID"
        let identifiers = vec!["deleteUserByAuth0ID".to_string()];
        assert!(imperative_verb_noun("delete user", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_filters_single_char_nouns() {
        // "get x" -> "x" is filtered (single-char) -> no nouns -> false
        let identifiers = vec!["x".to_string()];
        assert!(!imperative_verb_noun("get x", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_expanded_verbs() {
        // Test some of the newly added verbs
        let identifiers = vec!["config".to_string()];
        assert!(imperative_verb_noun("configure config", &identifiers));
        assert!(imperative_verb_noun("register config", &identifiers));
        assert!(imperative_verb_noun("emit config", &identifiers));
    }

    // ── verb_noun_matches_identifier ────────────────────────────────

    #[test]
    fn verb_noun_matches_identifier_basic() {
        // "delete user" → verb "delete", noun "user"
        // "DeleteUserByAuth0ID" → parts: ["delete", "user", "by", "auth0", "id"]
        // verb matches first part, "user" matches second → true
        let identifiers = vec!["DeleteUserByAuth0ID".to_string()];
        assert!(verb_noun_matches_identifier(
            "delete user from database",
            &identifiers
        ));
    }

    #[test]
    fn verb_noun_matches_identifier_snake_case() {
        let identifiers = vec!["get_user_name".to_string()];
        assert!(verb_noun_matches_identifier("get user name", &identifiers));
    }

    #[test]
    fn verb_noun_matches_identifier_verb_mismatch() {
        // Verb "fetch" doesn't match first part "delete"
        let identifiers = vec!["DeleteUser".to_string()];
        assert!(!verb_noun_matches_identifier("fetch user", &identifiers));
    }

    #[test]
    fn verb_noun_matches_identifier_no_noun_match() {
        // Verb "delete" matches, but noun "record" doesn't match any other part
        let identifiers = vec!["DeleteUser".to_string()];
        assert!(!verb_noun_matches_identifier("delete record", &identifiers));
    }

    #[test]
    fn verb_noun_matches_identifier_non_imperative() {
        let identifiers = vec!["TheUser".to_string()];
        assert!(!verb_noun_matches_identifier(
            "The user is logged in",
            &identifiers
        ));
    }

    #[test]
    fn verb_noun_matches_identifier_empty() {
        let identifiers: Vec<String> = vec![];
        assert!(!verb_noun_matches_identifier("delete user", &identifiers));
        assert!(!verb_noun_matches_identifier("", &identifiers));
    }

    #[test]
    fn verb_noun_matches_identifier_filters_stop_words() {
        // "get the user from the database" → after filtering: verb "get", nouns ["user", "database"]
        // "getUserFromDB" → parts: ["get", "user", "from", "db"]
        // verb matches first part, "user" matches second → true
        let identifiers = vec!["getUserFromDB".to_string()];
        assert!(verb_noun_matches_identifier(
            "get the user from the database",
            &identifiers
        ));
    }

    // ── is_section_label ────────────────────────────────────────────

    #[test]
    fn section_label_single_word() {
        assert!(is_section_label("Helpers"));
    }

    #[test]
    fn section_label_with_dividers() {
        assert!(is_section_label("--- Config ---"));
    }

    #[test]
    fn section_label_equals_dividers() {
        assert!(is_section_label("== Public API =="));
    }

    #[test]
    fn section_label_long_sentence() {
        assert!(!is_section_label(
            "This is a long sentence that describes something in detail"
        ));
    }

    #[test]
    fn section_label_imperative_verb_first() {
        assert!(!is_section_label("Get the data"));
    }

    // ── SemanticFeatures / extract ──────────────────────────────────

    #[test]
    fn extract_semantic_features_combines_all() {
        let identifiers = vec!["columns".to_string()];
        let features = extract_semantic_features("Get columns", &identifiers);
        assert!(!features.has_why_indicator);
        assert!(!features.has_external_ref);
        assert!(features.imperative_verb_noun);
        assert!(!features.is_section_label);
    }

    #[test]
    fn extract_semantic_features_why_and_ref() {
        let identifiers: Vec<String> = vec![];
        let features = extract_semantic_features(
            "workaround for upstream bug see https://example.com",
            &identifiers,
        );
        assert!(features.has_why_indicator);
        assert!(features.has_external_ref);
        assert!(!features.imperative_verb_noun);
        assert!(!features.is_section_label);
    }
}
