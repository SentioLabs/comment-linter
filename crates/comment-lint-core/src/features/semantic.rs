//! Semantic feature extraction: why-indicators, external references,
//! imperative-verb-noun patterns, and section labels.

use regex::Regex;
use std::sync::LazyLock;

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
    "get", "set", "check", "create", "initialize", "init", "fetch", "update",
    "delete", "remove", "add", "build", "parse", "validate", "compute",
    "calculate", "convert", "handle", "process", "return", "send", "load",
    "save", "store", "open", "close", "start", "stop", "run", "execute",
    "read", "write", "find", "search", "sort", "filter", "map", "reduce",
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
/// more than 50% of the remaining (non-verb) words overlap with
/// `nearby_identifiers`.
pub fn imperative_verb_noun(text: &str, nearby_identifiers: &[String]) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }

    let first = words[0].to_lowercase();
    if !IMPERATIVE_VERBS.contains(&first.as_str()) {
        return false;
    }

    let nouns: Vec<String> = words[1..]
        .iter()
        .map(|w| w.to_lowercase())
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
        .filter(|noun| lowered_identifiers.contains(noun))
        .count();

    // Strictly more than 50%.
    matching * 2 > nouns.len()
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
    pub is_section_label: bool,
}

/// Extract all semantic features from `text` given `nearby_identifiers`.
pub fn extract_semantic_features(
    text: &str,
    nearby_identifiers: &[String],
) -> SemanticFeatures {
    SemanticFeatures {
        has_why_indicator: has_why_indicator(text),
        has_external_ref: has_external_ref(text),
        imperative_verb_noun: imperative_verb_noun(text, nearby_identifiers),
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
    fn imperative_verb_noun_partial_overlap() {
        // "validate input data" -> nouns: ["input", "data"]
        // identifiers has only "input" -> 1/2 = 50% -> true (>50% means strictly more than half)
        // 50% is not >50%, so this should be false
        let identifiers = vec!["input".to_string()];
        assert!(!imperative_verb_noun("validate input data", &identifiers));
    }

    #[test]
    fn imperative_verb_noun_full_overlap() {
        let identifiers = vec!["input".to_string(), "data".to_string()];
        assert!(imperative_verb_noun("validate input data", &identifiers));
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
        assert!(!is_section_label("This is a long sentence that describes something in detail"));
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
