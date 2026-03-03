//! Integration tests for the config module.

use comment_lint_core::config::Config;
use std::io::Write;
use std::path::Path;

#[test]
fn test_default_threshold() {
    let cfg = Config::default();
    assert!(
        (cfg.general.threshold - 0.6).abs() < f64::EPSILON,
        "Expected default threshold 0.6, got {}",
        cfg.general.threshold
    );
}

#[test]
fn test_default_weights() {
    let cfg = Config::default();
    assert!(
        (cfg.weights.token_overlap_jaccard - 0.25).abs() < f64::EPSILON,
        "Expected token_overlap_jaccard 0.25, got {}",
        cfg.weights.token_overlap_jaccard
    );
    assert!(
        (cfg.weights.identifier_substring_ratio - 0.30).abs() < f64::EPSILON,
        "Expected identifier_substring_ratio 0.30, got {}",
        cfg.weights.identifier_substring_ratio
    );
    assert!(
        (cfg.weights.imperative_verb_noun - 0.20).abs() < f64::EPSILON,
        "Expected imperative_verb_noun 0.20, got {}",
        cfg.weights.imperative_verb_noun
    );
    assert!(
        (cfg.weights.verb_noun_matches_identifier - 0.25).abs() < f64::EPSILON,
        "Expected verb_noun_matches_identifier 0.25, got {}",
        cfg.weights.verb_noun_matches_identifier
    );
    assert!(
        (cfg.weights.is_section_label - 0.10).abs() < f64::EPSILON,
        "Expected is_section_label 0.10, got {}",
        cfg.weights.is_section_label
    );
    assert!(
        (cfg.weights.contains_literal_values - 0.05).abs() < f64::EPSILON,
        "Expected contains_literal_values 0.05, got {}",
        cfg.weights.contains_literal_values
    );
    assert!(
        (cfg.weights.references_other_files - 0.05).abs() < f64::EPSILON,
        "Expected references_other_files 0.05, got {}",
        cfg.weights.references_other_files
    );
    assert!(
        (cfg.weights.mirrors_data_structure - 0.05).abs() < f64::EPSILON,
        "Expected mirrors_data_structure 0.05, got {}",
        cfg.weights.mirrors_data_structure
    );
}

#[test]
fn test_default_negative_weights() {
    let cfg = Config::default();
    assert!(
        (cfg.negative.has_why_indicator - (-0.30)).abs() < f64::EPSILON,
        "Expected has_why_indicator -0.30, got {}",
        cfg.negative.has_why_indicator
    );
    assert!(
        (cfg.negative.has_external_ref - (-0.25)).abs() < f64::EPSILON,
        "Expected has_external_ref -0.25, got {}",
        cfg.negative.has_external_ref
    );
    assert!(
        (cfg.negative.is_doc_comment_on_public - (-0.20)).abs() < f64::EPSILON,
        "Expected is_doc_comment_on_public -0.20, got {}",
        cfg.negative.is_doc_comment_on_public
    );
}

#[test]
fn test_default_ignore_paths() {
    let cfg = Config::default();
    assert!(
        cfg.ignore.paths.contains(&"vendor/**".to_string()),
        "Expected vendor/** in default ignore paths"
    );
    assert!(
        cfg.ignore.paths.contains(&"node_modules/**".to_string()),
        "Expected node_modules/** in default ignore paths"
    );
    assert!(
        cfg.ignore.paths.contains(&"*.generated.*".to_string()),
        "Expected *.generated.* in default ignore paths"
    );
    assert!(
        cfg.ignore.paths.contains(&"target/**".to_string()),
        "Expected target/** in default ignore paths"
    );
}

#[test]
fn test_default_ignore_comment_patterns() {
    let cfg = Config::default();
    let patterns = &cfg.ignore.comment_patterns;
    assert!(patterns.contains(&"^//go:generate".to_string()));
    assert!(patterns.contains(&"^//nolint".to_string()));
    assert!(patterns.contains(&"^# type:".to_string()));
    assert!(patterns.contains(&"^# noqa".to_string()));
    assert!(patterns.contains(&"^# pylint:".to_string()));
    assert!(patterns.contains(&"^// eslint-".to_string()));
    assert!(patterns.contains(&"^// @ts-".to_string()));
    assert!(patterns.contains(&"^// Copyright".to_string()));
    assert!(patterns.contains(&"^# Copyright".to_string()));
}

#[test]
fn test_default_cache() {
    let cfg = Config::default();
    assert!(cfg.cache.enabled, "Expected cache enabled by default");
    assert_eq!(
        cfg.cache.directory, ".comment-lint-cache",
        "Expected default cache directory"
    );
}

#[test]
fn test_from_toml_str_full() {
    let toml_str = r#"
[general]
threshold = 0.8
min_confidence = 0.1
include_doc_comments = true

[weights]
token_overlap_jaccard = 0.30

[negative]
has_why_indicator = -0.50

[ignore]
paths = ["custom/**"]
comment_patterns = ["^# custom"]

[cache]
enabled = false
directory = "/tmp/cache"
"#;
    let cfg = Config::from_toml_str(toml_str).expect("valid TOML");
    assert!((cfg.general.threshold - 0.8).abs() < f64::EPSILON);
    assert!((cfg.general.min_confidence - 0.1).abs() < f64::EPSILON);
    assert!(cfg.general.include_doc_comments);
    assert!((cfg.weights.token_overlap_jaccard - 0.30).abs() < f64::EPSILON);
    assert!((cfg.negative.has_why_indicator - (-0.50)).abs() < f64::EPSILON);
    assert_eq!(cfg.ignore.paths, vec!["custom/**"]);
    assert!(!cfg.cache.enabled);
    assert_eq!(cfg.cache.directory, "/tmp/cache");
}

#[test]
fn test_empty_toml_gives_defaults() {
    let cfg = Config::from_toml_str("").expect("empty TOML is valid");
    assert!(
        (cfg.general.threshold - 0.6).abs() < f64::EPSILON,
        "Empty TOML should give default threshold 0.6, got {}",
        cfg.general.threshold
    );
    assert!(
        (cfg.weights.token_overlap_jaccard - 0.25).abs() < f64::EPSILON,
        "Empty TOML should give default token_overlap_jaccard"
    );
    assert!(cfg.cache.enabled, "Empty TOML should give cache enabled");
}

#[test]
fn test_partial_override() {
    let toml_str = r#"
[general]
threshold = 0.9
"#;
    let cfg = Config::from_toml_str(toml_str).expect("valid TOML");
    // Overridden value
    assert!(
        (cfg.general.threshold - 0.9).abs() < f64::EPSILON,
        "Threshold should be overridden to 0.9"
    );
    // Non-overridden values should keep defaults
    assert!(
        (cfg.weights.token_overlap_jaccard - 0.25).abs() < f64::EPSILON,
        "Non-overridden weights should keep defaults"
    );
    assert!(cfg.cache.enabled, "Non-overridden cache should keep default");
}

#[test]
fn test_from_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("test.toml");
    {
        let mut f = std::fs::File::create(&path).expect("create file");
        writeln!(f, "[general]\nthreshold = 0.75").expect("write");
    }
    let cfg = Config::from_file(&path).expect("load file");
    assert!(
        (cfg.general.threshold - 0.75).abs() < f64::EPSILON,
        "Expected threshold 0.75 from file"
    );
}

#[test]
fn test_from_file_not_found() {
    let result = Config::from_file(Path::new("/nonexistent/path.toml"));
    assert!(result.is_err(), "Should error on missing file");
}

#[test]
fn test_resolve_defaults_only() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cfg = Config::resolve(None, dir.path()).expect("resolve");
    assert!(
        (cfg.general.threshold - 0.6).abs() < f64::EPSILON,
        "Resolve with no files should give default threshold"
    );
}

#[test]
fn test_resolve_project_config_overrides() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cfg_path = dir.path().join("comment-lint.toml");
    {
        let mut f = std::fs::File::create(&cfg_path).expect("create file");
        writeln!(f, "[general]\nthreshold = 0.85").expect("write");
    }
    let cfg = Config::resolve(None, dir.path()).expect("resolve");
    assert!(
        (cfg.general.threshold - 0.85).abs() < f64::EPSILON,
        "Project comment-lint.toml should override default threshold"
    );
}

#[test]
fn test_resolve_explicit_overrides_project() {
    let dir = tempfile::tempdir().expect("temp dir");
    // Project config sets 0.85
    let project_cfg = dir.path().join("comment-lint.toml");
    {
        let mut f = std::fs::File::create(&project_cfg).expect("create");
        writeln!(f, "[general]\nthreshold = 0.85").expect("write");
    }
    // Explicit config sets 0.95
    let explicit_cfg = dir.path().join("explicit.toml");
    {
        let mut f = std::fs::File::create(&explicit_cfg).expect("create");
        writeln!(f, "[general]\nthreshold = 0.95").expect("write");
    }
    let cfg = Config::resolve(Some(&explicit_cfg), dir.path()).expect("resolve");
    assert!(
        (cfg.general.threshold - 0.95).abs() < f64::EPSILON,
        "Explicit path should override project config"
    );
}
