//! Integration tests for the full comment-lint pipeline across all 5 supported
//! languages.  Each language has two fixture files: one containing superfluous
//! comments (high token overlap with code identifiers) and one containing
//! valuable comments (why-indicators, external references, low overlap).

use std::path::PathBuf;

use comment_lint_core::config::Config;
use comment_lint_core::features::ScoredComment;
use comment_lint_core::pipeline::Pipeline;
use comment_lint_core::scoring::heuristic::HeuristicScorer;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Run the pipeline on a single fixture file with permissive settings so that
/// every comment is returned regardless of score.
fn run_fixture(fixture_path: &str) -> Vec<ScoredComment> {
    let mut config = Config::default();
    config.general.threshold = 0.0; // Do not filter -- return ALL scored comments
    config.general.min_confidence = 0.0;
    config.general.include_doc_comments = true; // Include doc comments
    config.ignore.paths = vec![]; // Do not ignore fixture paths
    config.ignore.comment_patterns = vec![]; // Do not ignore any comment patterns

    let scorer = HeuristicScorer::new(config.weights.clone(), config.negative.clone());
    let pipeline = Pipeline::new(config, Box::new(scorer));
    let result = pipeline.run(&[PathBuf::from(fixture_path)]);
    result.scored_comments
}

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------

#[test]
fn go_superfluous_comments_detected() {
    let results = run_fixture("tests/fixtures/go/superfluous.go");
    assert!(
        !results.is_empty(),
        "should find comments in Go superfluous fixture"
    );
    for sc in &results {
        assert!(
            sc.score >= 0.3,
            "Expected superfluous Go comment to score >= 0.3, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

#[test]
fn go_valuable_comments_not_flagged() {
    let results = run_fixture("tests/fixtures/go/valuable.go");
    assert!(
        !results.is_empty(),
        "should find comments in Go valuable fixture"
    );
    for sc in &results {
        assert!(
            sc.score < 0.6,
            "Expected valuable Go comment to score < 0.6, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

// ---------------------------------------------------------------------------
// Rust
// ---------------------------------------------------------------------------

#[test]
fn rust_superfluous_comments_detected() {
    let results = run_fixture("tests/fixtures/rust/superfluous.rs");
    assert!(
        !results.is_empty(),
        "should find comments in Rust superfluous fixture"
    );
    for sc in &results {
        assert!(
            sc.score >= 0.3,
            "Expected superfluous Rust comment to score >= 0.3, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

#[test]
fn rust_valuable_comments_not_flagged() {
    let results = run_fixture("tests/fixtures/rust/valuable.rs");
    assert!(
        !results.is_empty(),
        "should find comments in Rust valuable fixture"
    );
    for sc in &results {
        assert!(
            sc.score < 0.6,
            "Expected valuable Rust comment to score < 0.6, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

// ---------------------------------------------------------------------------
// Python
// ---------------------------------------------------------------------------

#[test]
fn python_superfluous_comments_detected() {
    let results = run_fixture("tests/fixtures/python/superfluous.py");
    assert!(
        !results.is_empty(),
        "should find comments in Python superfluous fixture"
    );
    for sc in &results {
        assert!(
            sc.score >= 0.3,
            "Expected superfluous Python comment to score >= 0.3, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

#[test]
fn python_valuable_comments_not_flagged() {
    let results = run_fixture("tests/fixtures/python/valuable.py");
    assert!(
        !results.is_empty(),
        "should find comments in Python valuable fixture"
    );
    for sc in &results {
        assert!(
            sc.score < 0.6,
            "Expected valuable Python comment to score < 0.6, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

// ---------------------------------------------------------------------------
// TypeScript
// ---------------------------------------------------------------------------

#[test]
fn typescript_superfluous_comments_detected() {
    let results = run_fixture("tests/fixtures/typescript/superfluous.ts");
    assert!(
        !results.is_empty(),
        "should find comments in TypeScript superfluous fixture"
    );
    for sc in &results {
        assert!(
            sc.score >= 0.3,
            "Expected superfluous TypeScript comment to score >= 0.3, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

#[test]
fn typescript_valuable_comments_not_flagged() {
    let results = run_fixture("tests/fixtures/typescript/valuable.ts");
    assert!(
        !results.is_empty(),
        "should find comments in TypeScript valuable fixture"
    );
    for sc in &results {
        assert!(
            sc.score < 0.6,
            "Expected valuable TypeScript comment to score < 0.6, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

// ---------------------------------------------------------------------------
// JavaScript
// ---------------------------------------------------------------------------

#[test]
fn javascript_superfluous_comments_detected() {
    let results = run_fixture("tests/fixtures/javascript/superfluous.js");
    assert!(
        !results.is_empty(),
        "should find comments in JavaScript superfluous fixture"
    );
    for sc in &results {
        assert!(
            sc.score >= 0.3,
            "Expected superfluous JavaScript comment to score >= 0.3, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}

#[test]
fn javascript_valuable_comments_not_flagged() {
    let results = run_fixture("tests/fixtures/javascript/valuable.js");
    assert!(
        !results.is_empty(),
        "should find comments in JavaScript valuable fixture"
    );
    for sc in &results {
        assert!(
            sc.score < 0.6,
            "Expected valuable JavaScript comment to score < 0.6, got {:.4} for '{}'",
            sc.score,
            sc.context.comment_text
        );
    }
}
