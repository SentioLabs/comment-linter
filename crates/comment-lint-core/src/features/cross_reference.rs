//! Cross-reference and staleness feature extraction.
//!
//! Detects literal values, file references, function references, and
//! data-structure mirroring in comment text — all signals that a comment
//! carries concrete, cross-referencing information (and therefore may become
//! stale).

use std::sync::LazyLock;

use regex::Regex;

// ---------------------------------------------------------------------------
// Compiled regexes (one-time init via LazyLock)
// ---------------------------------------------------------------------------

/// Matches numbers with 2+ digits or quoted strings.
static LITERAL_VALUES_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\d{2,}|"[^"]+"|'[^']+'"#).expect("LITERAL_VALUES_RE"));

/// Matches file-path patterns such as `foo/bar.go`, `../utils`, or
/// `see file.ext`.
static FILE_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:\.\./|[\w-]+/[\w./-]+|\b\w+\.\w{1,4}\b)").expect("FILE_REF_RE")
});

/// Matches backtick-wrapped identifiers like `processPayment`.
static BACKTICK_IDENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`[A-Za-z_]\w+`").expect("BACKTICK_IDENT_RE"));

/// Matches PascalCase identifiers with 2+ humps (e.g. `GetUserById`).
static PASCAL_CASE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+){1,}\b").expect("PASCAL_CASE_RE"));

/// Matches long snake_case identifiers with 3+ parts (e.g. `get_user_by_id`).
static SNAKE_CASE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-z]+(?:_[a-z]+){2,}\b").expect("SNAKE_CASE_RE"));

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Aggregated cross-reference features for a single comment.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossReferenceFeatures {
    pub contains_literal_values: bool,
    pub references_other_files: bool,
    pub references_specific_functions: bool,
    pub mirrors_data_structure: bool,
}

/// Returns `true` when the comment text contains literal values (numbers with
/// 2+ digits or quoted strings).
pub fn contains_literal_values(text: &str) -> bool {
    LITERAL_VALUES_RE.is_match(text)
}

/// Returns `true` when the comment text references other files via path-like
/// patterns.
pub fn references_other_files(text: &str) -> bool {
    FILE_REF_RE.is_match(text)
}

/// Returns `true` when the comment text references specific function or method
/// names (backtick identifiers, PascalCase with 2+ humps, or long snake_case
/// with 3+ parts).
pub fn references_specific_functions(text: &str) -> bool {
    BACKTICK_IDENT_RE.is_match(text)
        || PASCAL_CASE_RE.is_match(text)
        || SNAKE_CASE_RE.is_match(text)
}

/// Returns `true` when more than 60% of the given `identifiers` appear as
/// substrings in the comment `text`.
pub fn mirrors_data_structure(text: &str, identifiers: &[String]) -> bool {
    if identifiers.is_empty() {
        return false;
    }
    let lower = text.to_lowercase();
    let matches = identifiers
        .iter()
        .filter(|id| lower.contains(&id.to_lowercase()))
        .count();
    (matches as f64 / identifiers.len() as f64) > 0.6
}

/// Extracts all cross-reference features from `text`, using `identifiers` for
/// the data-structure mirroring check.
pub fn extract_cross_reference_features(
    text: &str,
    identifiers: &[String],
) -> CrossReferenceFeatures {
    CrossReferenceFeatures {
        contains_literal_values: contains_literal_values(text),
        references_other_files: references_other_files(text),
        references_specific_functions: references_specific_functions(text),
        mirrors_data_structure: mirrors_data_structure(text, identifiers),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- contains_literal_values -------------------------------------------

    #[test]
    fn literal_values_with_number() {
        assert!(contains_literal_values("timeout is 30 seconds"));
    }

    #[test]
    fn literal_values_with_quoted_string() {
        assert!(contains_literal_values(r#"default name is "admin""#));
    }

    #[test]
    fn literal_values_plain_text() {
        assert!(!contains_literal_values("handles retry"));
    }

    #[test]
    fn literal_values_single_digit_not_matched() {
        assert!(!contains_literal_values("step 1 of the process"));
    }

    // -- references_other_files --------------------------------------------

    #[test]
    fn file_ref_path_pattern() {
        assert!(references_other_files("see utils/helpers.go"));
    }

    #[test]
    fn file_ref_relative_path() {
        assert!(references_other_files("defined in ../utils"));
    }

    #[test]
    fn file_ref_see_file_ext() {
        assert!(references_other_files("see config.yaml for details"));
    }

    #[test]
    fn file_ref_plain_text() {
        assert!(!references_other_files("authentication"));
    }

    // -- references_specific_functions -------------------------------------

    #[test]
    fn func_ref_backtick_identifier() {
        assert!(references_specific_functions(
            "calls `processPayment` internally"
        ));
    }

    #[test]
    fn func_ref_pascal_case() {
        assert!(references_specific_functions("delegates to GetUserById"));
    }

    #[test]
    fn func_ref_long_snake_case() {
        assert!(references_specific_functions(
            "see get_user_by_id for details"
        ));
    }

    #[test]
    fn func_ref_plain_text() {
        assert!(!references_specific_functions("handles the retry logic"));
    }

    // -- mirrors_data_structure --------------------------------------------

    #[test]
    fn mirrors_matching_fields() {
        let identifiers: Vec<String> = vec!["name".into(), "age".into(), "email".into()];
        assert!(mirrors_data_structure(
            "contains name, age, and email fields",
            &identifiers,
        ));
    }

    #[test]
    fn mirrors_no_match() {
        let identifiers: Vec<String> = vec!["name".into(), "age".into(), "email".into()];
        assert!(!mirrors_data_structure("unrelated comment", &identifiers));
    }

    #[test]
    fn mirrors_empty_identifiers() {
        assert!(!mirrors_data_structure("some comment", &[]));
    }

    #[test]
    fn mirrors_partial_below_threshold() {
        let identifiers: Vec<String> = vec![
            "name".into(),
            "age".into(),
            "email".into(),
            "address".into(),
            "phone".into(),
        ];
        assert!(!mirrors_data_structure("the name field", &identifiers));
    }

    // -- extract_cross_reference_features ----------------------------------

    #[test]
    fn extract_features_all_true() {
        let identifiers: Vec<String> = vec!["name".into(), "age".into()];
        let features = extract_cross_reference_features(
            "see utils/helpers.go for `processPayment`, timeout is 30s, name and age fields",
            &identifiers,
        );
        assert!(features.contains_literal_values);
        assert!(features.references_other_files);
        assert!(features.references_specific_functions);
        assert!(features.mirrors_data_structure);
    }

    #[test]
    fn extract_features_all_false() {
        let features = extract_cross_reference_features("handles retry logic", &[]);
        assert!(!features.contains_literal_values);
        assert!(!features.references_other_files);
        assert!(!features.references_specific_functions);
        assert!(!features.mirrors_data_structure);
    }
}
