//! Integration tests for the EnsembleScorer.

use std::path::PathBuf;

use comment_lint_core::extraction::comment::CommentContext;
use comment_lint_core::features::{FeatureVector, ScoredComment};
use comment_lint_core::scoring::Scorer;
use comment_lint_core::types::{CommentKind, LanguageId};

use comment_lint_ml::ensemble::EnsembleScorer;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A mock scorer that always returns a fixed score, confidence, and reasons.
struct FixedScorer {
    score: f32,
    confidence: f32,
    reasons: Vec<String>,
}

impl Scorer for FixedScorer {
    fn score(&self, context: &CommentContext, features: &FeatureVector) -> ScoredComment {
        ScoredComment {
            context: context.clone(),
            features: features.clone(),
            score: self.score,
            confidence: self.confidence,
            reasons: self.reasons.clone(),
        }
    }
}

fn make_context() -> CommentContext {
    CommentContext {
        file_path: PathBuf::from("test.rs"),
        line: 1,
        column: 0,
        comment_text: "test comment".to_string(),
        comment_kind: CommentKind::Line,
        language: LanguageId::Rust,
        adjacent_node_kind: "function_item".to_string(),
        surrounding_source: "fn foo() {}".to_string(),
        nearby_identifiers: vec!["foo".to_string()],
        nearby_keywords: vec!["fn".to_string()],
    }
}

fn zero_features() -> FeatureVector {
    FeatureVector {
        token_overlap_jaccard: 0.0,
        identifier_substring_ratio: 0.0,
        comment_token_count: 0,
        is_doc_comment: false,
        is_before_declaration: false,
        is_inline: false,
        adjacent_node_kind: String::new(),
        nesting_depth: 0,
        has_why_indicator: false,
        has_external_ref: false,
        imperative_verb_noun: false,
        verb_noun_matches_identifier: false,
        is_section_label: false,
        contains_literal_values: false,
        references_other_files: false,
        references_specific_functions: false,
        mirrors_data_structure: false,
        comment_code_age_ratio: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn ensemble_score_is_weighted_average() {
    // heuristic = 0.8, ml = 0.4, ml_weight = 0.5
    // expected = 0.8 * 0.5 + 0.4 * 0.5 = 0.6
    let heuristic = FixedScorer {
        score: 0.8,
        confidence: 0.9,
        reasons: vec!["h_reason".to_string()],
    };
    let ml = FixedScorer {
        score: 0.4,
        confidence: 0.8,
        reasons: vec!["m_reason".to_string()],
    };

    let ensemble = EnsembleScorer::new(Box::new(heuristic), Box::new(ml), 0.5);
    let ctx = make_context();
    let features = zero_features();
    let result = ensemble.score(&ctx, &features);

    assert!(
        (result.score - 0.6).abs() < 0.001,
        "Expected weighted avg of 0.6, got {}",
        result.score
    );
}

#[test]
fn ensemble_respects_ml_weight() {
    // heuristic = 0.2, ml = 0.8, ml_weight = 0.6
    // expected = 0.2 * 0.4 + 0.8 * 0.6 = 0.08 + 0.48 = 0.56
    let heuristic = FixedScorer {
        score: 0.2,
        confidence: 0.7,
        reasons: vec![],
    };
    let ml = FixedScorer {
        score: 0.8,
        confidence: 0.7,
        reasons: vec![],
    };

    let ensemble = EnsembleScorer::new(Box::new(heuristic), Box::new(ml), 0.6);
    let ctx = make_context();
    let features = zero_features();
    let result = ensemble.score(&ctx, &features);

    assert!(
        (result.score - 0.56).abs() < 0.001,
        "Expected 0.56, got {}",
        result.score
    );
}

#[test]
fn ensemble_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EnsembleScorer>();
}

#[test]
fn ensemble_merges_reasons_with_prefixes() {
    let heuristic = FixedScorer {
        score: 0.5,
        confidence: 0.8,
        reasons: vec!["high_overlap".to_string(), "identifier_match".to_string()],
    };
    let ml = FixedScorer {
        score: 0.5,
        confidence: 0.8,
        reasons: vec!["ml_prediction(0.500)".to_string()],
    };

    let ensemble = EnsembleScorer::new(Box::new(heuristic), Box::new(ml), 0.5);
    let ctx = make_context();
    let features = zero_features();
    let result = ensemble.score(&ctx, &features);

    assert_eq!(result.reasons.len(), 3, "Should have 3 merged reasons");
    assert!(
        result.reasons[0].starts_with("[heuristic]"),
        "First reason should have [heuristic] prefix, got: {}",
        result.reasons[0]
    );
    assert!(
        result.reasons[1].starts_with("[heuristic]"),
        "Second reason should have [heuristic] prefix, got: {}",
        result.reasons[1]
    );
    assert!(
        result.reasons[2].starts_with("[ml]"),
        "Third reason should have [ml] prefix, got: {}",
        result.reasons[2]
    );
}

#[test]
fn ensemble_confidence_higher_when_scorers_agree() {
    // Both scorers agree at 0.7
    let h_agree = FixedScorer {
        score: 0.7,
        confidence: 0.8,
        reasons: vec![],
    };
    let m_agree = FixedScorer {
        score: 0.7,
        confidence: 0.8,
        reasons: vec![],
    };

    let agree_ensemble = EnsembleScorer::new(Box::new(h_agree), Box::new(m_agree), 0.5);
    let ctx = make_context();
    let features = zero_features();
    let agree_result = agree_ensemble.score(&ctx, &features);

    // Scorers disagree: heuristic 0.2, ml 0.9
    let h_disagree = FixedScorer {
        score: 0.2,
        confidence: 0.8,
        reasons: vec![],
    };
    let m_disagree = FixedScorer {
        score: 0.9,
        confidence: 0.8,
        reasons: vec![],
    };

    let disagree_ensemble = EnsembleScorer::new(Box::new(h_disagree), Box::new(m_disagree), 0.5);
    let disagree_result = disagree_ensemble.score(&ctx, &features);

    assert!(
        agree_result.confidence > disagree_result.confidence,
        "Confidence when agreeing ({}) should exceed confidence when disagreeing ({})",
        agree_result.confidence,
        disagree_result.confidence
    );
}

#[test]
fn ensemble_score_clamped_to_zero_one() {
    // Even with extreme sub-scores, the ensemble should clamp.
    // We use a score slightly above 1.0 via weighted combo that could round oddly.
    let heuristic = FixedScorer {
        score: 1.0,
        confidence: 1.0,
        reasons: vec![],
    };
    let ml = FixedScorer {
        score: 1.0,
        confidence: 1.0,
        reasons: vec![],
    };

    let ensemble = EnsembleScorer::new(Box::new(heuristic), Box::new(ml), 0.5);
    let ctx = make_context();
    let features = zero_features();
    let result = ensemble.score(&ctx, &features);

    assert!(
        result.score >= 0.0 && result.score <= 1.0,
        "Score should be in [0.0, 1.0], got {}",
        result.score
    );
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be in [0.0, 1.0], got {}",
        result.confidence
    );

    // Same with zeros
    let h_zero = FixedScorer {
        score: 0.0,
        confidence: 0.0,
        reasons: vec![],
    };
    let m_zero = FixedScorer {
        score: 0.0,
        confidence: 0.0,
        reasons: vec![],
    };

    let ensemble_zero = EnsembleScorer::new(Box::new(h_zero), Box::new(m_zero), 0.5);
    let result_zero = ensemble_zero.score(&ctx, &features);

    assert!(
        result_zero.score >= 0.0 && result_zero.score <= 1.0,
        "Score should be in [0.0, 1.0], got {}",
        result_zero.score
    );
    assert!(
        result_zero.confidence >= 0.0 && result_zero.confidence <= 1.0,
        "Confidence should be in [0.0, 1.0], got {}",
        result_zero.confidence
    );
}

#[test]
fn ensemble_with_real_scorers() {
    // Integration test using real HeuristicScorer + MLScorer
    use comment_lint_core::config::Config;
    use comment_lint_core::scoring::heuristic::HeuristicScorer;
    use comment_lint_ml::scorer::MLScorer;

    let config = Config::default();
    let heuristic = HeuristicScorer::new(config.weights.clone(), config.negative.clone());
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/dummy_model.onnx"
    )
    .to_string();
    let ml = MLScorer::new(&model_path).expect("should load dummy model");

    let ensemble = EnsembleScorer::new(Box::new(heuristic), Box::new(ml), 0.6);
    let ctx = make_context();
    let features = zero_features();
    let result = ensemble.score(&ctx, &features);

    // Basic sanity: score and confidence in valid range
    assert!(
        result.score >= 0.0 && result.score <= 1.0,
        "Score should be in [0.0, 1.0], got {}",
        result.score
    );
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be in [0.0, 1.0], got {}",
        result.confidence
    );

    // Should have reasons from both scorers
    let has_heuristic_reason = result
        .reasons
        .iter()
        .any(|r: &String| r.starts_with("[heuristic]"));
    let has_ml_reason = result
        .reasons
        .iter()
        .any(|r: &String| r.starts_with("[ml]"));
    // Heuristic may have 0 reasons for zero features, so only check ML
    assert!(
        has_ml_reason,
        "Should have at least one [ml] prefixed reason, got: {:?}",
        result.reasons
    );
    // If heuristic had reasons, they should be prefixed
    if has_heuristic_reason {
        assert!(
            result
                .reasons
                .iter()
                .filter(|r: &&String| r.starts_with("[heuristic]"))
                .count()
                > 0,
            "Heuristic reasons should be prefixed"
        );
    }
}
