//! ML-based comment scorer using ONNX Runtime inference.

use std::sync::Mutex;

use anyhow::Result;
use ort::session::Session;
use ort::value::TensorRef;

use comment_lint_core::extraction::comment::CommentContext;
use comment_lint_core::features::{FeatureVector, ScoredComment};
use comment_lint_core::scoring::Scorer;

use crate::tensor::{feature_vector_to_tensor, TENSOR_DIM};

/// Scores comments using a trained ONNX model.
///
/// The model takes a feature vector of dimension [`TENSOR_DIM`]
/// and outputs two probabilities: `[P(not_superfluous), P(superfluous)]`.
pub struct MLScorer {
    session: Mutex<Session>,
}

impl MLScorer {
    /// Create a new `MLScorer` by loading an ONNX model from `model_path`.
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl Scorer for MLScorer {
    fn score(&self, context: &CommentContext, features: &FeatureVector) -> ScoredComment {
        let tensor_data = feature_vector_to_tensor(features);

        // Create input tensor reference
        let tensor_ref = TensorRef::from_array_view(([1usize, TENSOR_DIM], &*tensor_data))
            .expect("failed to create tensor");

        // Run inference
        let mut session = self.session.lock().expect("session lock poisoned");
        let outputs = session
            .run(ort::inputs!["input" => tensor_ref])
            .expect("inference failed");

        // Extract probabilities: [P(not_superfluous), P(superfluous)]
        let (_shape, data) = outputs["probabilities"]
            .try_extract_tensor::<f32>()
            .expect("failed to extract output tensor");

        let score = data[1].clamp(0.0, 1.0); // P(superfluous)

        // Confidence from probability magnitude: how far from 0.5
        let confidence = ((score - 0.5).abs() * 2.0).clamp(0.0, 1.0);

        let reasons = vec![format!("ml_prediction({:.3})", score)];

        ScoredComment {
            context: context.clone(),
            features: features.clone(),
            score,
            confidence,
            reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comment_lint_core::extraction::comment::CommentContext;
    use comment_lint_core::features::FeatureVector;
    use comment_lint_core::types::{CommentKind, LanguageId};
    use std::path::PathBuf;

    fn dummy_model_path() -> String {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dummy_model.onnx").to_string()
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
            is_section_label: false,
            contains_literal_values: false,
            references_other_files: false,
            references_specific_functions: false,
            mirrors_data_structure: false,
            comment_code_age_ratio: None,
        }
    }

    #[test]
    fn test_ml_scorer_loads_model() {
        let scorer = MLScorer::new(&dummy_model_path());
        assert!(
            scorer.is_ok(),
            "Should load dummy model: {:?}",
            scorer.err()
        );
    }

    #[test]
    fn test_ml_scorer_invalid_model_path_errors() {
        let scorer = MLScorer::new("/nonexistent/model.onnx");
        assert!(scorer.is_err(), "Should error on nonexistent model");
    }

    #[test]
    fn test_ml_scorer_returns_valid_score_range() {
        let scorer = MLScorer::new(&dummy_model_path()).unwrap();
        let ctx = make_context();
        let features = zero_features();
        let result = scorer.score(&ctx, &features);
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "Score should be in [0,1], got {}",
            result.score
        );
    }

    #[test]
    fn test_ml_scorer_returns_valid_confidence_range() {
        let scorer = MLScorer::new(&dummy_model_path()).unwrap();
        let ctx = make_context();
        let features = zero_features();
        let result = scorer.score(&ctx, &features);
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence should be in [0,1], got {}",
            result.confidence
        );
    }

    #[test]
    fn test_ml_scorer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MLScorer>();
    }

    #[test]
    fn test_ml_scorer_produces_reasons() {
        let scorer = MLScorer::new(&dummy_model_path()).unwrap();
        let ctx = make_context();
        let features = zero_features();
        let result = scorer.score(&ctx, &features);
        assert!(!result.reasons.is_empty(), "Reasons should not be empty");
    }

    #[test]
    fn test_ml_scorer_high_overlap_scores_higher() {
        let scorer = MLScorer::new(&dummy_model_path()).unwrap();
        let ctx = make_context();

        let low = zero_features();
        let mut high = zero_features();
        high.token_overlap_jaccard = 0.95;
        high.identifier_substring_ratio = 0.9;
        high.imperative_verb_noun = true;

        let low_result = scorer.score(&ctx, &low);
        let high_result = scorer.score(&ctx, &high);

        assert!(
            high_result.score > low_result.score,
            "High overlap features ({}) should score higher than zero features ({})",
            high_result.score,
            low_result.score
        );
    }
}
