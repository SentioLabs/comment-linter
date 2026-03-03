//! Heuristic scorer: weighted sum of features with confidence scoring.

use crate::config::{NegativeWeights, WeightsConfig};
use crate::extraction::comment::CommentContext;
use crate::features::{FeatureVector, ScoredComment};
use crate::scoring::Scorer;

/// A scorer that computes a weighted sum of feature signals to produce
/// a superfluousness score, along with a confidence estimate and
/// human-readable reasons.
pub struct HeuristicScorer {
    weights: WeightsConfig,
    negative: NegativeWeights,
}

impl HeuristicScorer {
    /// Create a new `HeuristicScorer` with the given positive and negative weights.
    pub fn new(weights: WeightsConfig, negative: NegativeWeights) -> Self {
        Self { weights, negative }
    }

    /// Compute the superfluousness score as a weighted sum of features,
    /// clamped to `[0.0, 1.0]`.
    fn compute_score(&self, features: &FeatureVector) -> f32 {
        let mut score = 0.0_f64;

        // Positive weights (increase superfluity)
        score += features.token_overlap_jaccard as f64 * self.weights.token_overlap_jaccard;
        score +=
            features.identifier_substring_ratio as f64 * self.weights.identifier_substring_ratio;
        if features.imperative_verb_noun {
            score += self.weights.imperative_verb_noun;
        }
        if features.verb_noun_matches_identifier {
            score += self.weights.verb_noun_matches_identifier;
        }
        if features.is_section_label {
            score += self.weights.is_section_label;
        }
        if features.contains_literal_values {
            score += self.weights.contains_literal_values;
        }
        if features.references_other_files {
            score += self.weights.references_other_files;
        }
        if features.mirrors_data_structure {
            score += self.weights.mirrors_data_structure;
        }

        // Negative weights (reduce score for valuable signals)
        if features.has_why_indicator {
            score += self.negative.has_why_indicator; // negative value
        }
        if features.has_external_ref {
            score += self.negative.has_external_ref;
        }
        if features.is_doc_comment && features.is_before_declaration {
            score += self.negative.is_doc_comment_on_public;
        }

        (score as f32).clamp(0.0, 1.0)
    }

    /// Compute a confidence score based on agreement among feature signals.
    ///
    /// When features all point in the same direction (all superfluous or all
    /// valuable), confidence is high. When they disagree, confidence is lower.
    fn compute_confidence(&self, features: &FeatureVector) -> f32 {
        let mut superfluous_signals = 0;
        let mut valuable_signals = 0;

        if features.token_overlap_jaccard > 0.5 {
            superfluous_signals += 1;
        }
        if features.identifier_substring_ratio > 0.5 {
            superfluous_signals += 1;
        }
        if features.imperative_verb_noun {
            superfluous_signals += 1;
        }
        if features.verb_noun_matches_identifier {
            superfluous_signals += 1;
        }
        if features.is_section_label {
            superfluous_signals += 1;
        }

        if features.has_why_indicator {
            valuable_signals += 1;
        }
        if features.has_external_ref {
            valuable_signals += 1;
        }
        if features.is_doc_comment && features.is_before_declaration {
            valuable_signals += 1;
        }

        let total = superfluous_signals + valuable_signals;
        if total == 0 {
            return 0.5;
        }

        let dominant = superfluous_signals.max(valuable_signals);
        dominant as f32 / total as f32
    }

    /// Generate human-readable reason strings for significant feature signals.
    fn generate_reasons(&self, features: &FeatureVector) -> Vec<String> {
        let mut reasons = Vec::new();
        if features.token_overlap_jaccard > 0.5 {
            reasons.push(format!(
                "high_lexical_overlap ({:.2})",
                features.token_overlap_jaccard
            ));
        }
        if features.identifier_substring_ratio > 0.5 {
            reasons.push("identifier_match".to_string());
        }
        if features.imperative_verb_noun {
            reasons.push("imperative_verb_noun_pattern".to_string());
        }
        if features.verb_noun_matches_identifier {
            reasons.push("verb_noun_matches_identifier".to_string());
        }
        if features.is_section_label {
            reasons.push("section_label".to_string());
        }
        if features.contains_literal_values {
            reasons.push("contains_literals".to_string());
        }
        if features.references_other_files {
            reasons.push("references_files".to_string());
        }
        if features.mirrors_data_structure {
            reasons.push("mirrors_structure".to_string());
        }
        reasons
    }
}

impl Scorer for HeuristicScorer {
    fn score(&self, context: &CommentContext, features: &FeatureVector) -> ScoredComment {
        ScoredComment {
            context: context.clone(),
            features: features.clone(),
            score: self.compute_score(features),
            confidence: self.compute_confidence(features),
            reasons: self.generate_reasons(features),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::config::{Config, NegativeWeights, WeightsConfig};
    use crate::extraction::comment::CommentContext;
    use crate::features::FeatureVector;
    use crate::scoring::Scorer;
    use crate::types::{CommentKind, LanguageId};

    use super::HeuristicScorer;

    /// Helper to build a default CommentContext for tests.
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

    /// Helper to build a zeroed-out FeatureVector.
    fn make_features() -> FeatureVector {
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

    fn default_scorer() -> HeuristicScorer {
        let cfg = Config::default();
        HeuristicScorer::new(cfg.weights, cfg.negative)
    }

    #[test]
    fn all_superfluous_features_score_above_0_8() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        features.token_overlap_jaccard = 0.9;
        features.identifier_substring_ratio = 0.9;
        features.imperative_verb_noun = true;
        features.is_section_label = true;
        features.contains_literal_values = true;
        features.references_other_files = true;
        features.mirrors_data_structure = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.score > 0.8,
            "All-superfluous features should yield score > 0.8, got {}",
            result.score
        );
    }

    #[test]
    fn all_valuable_features_score_below_0_3() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        features.token_overlap_jaccard = 0.1;
        features.identifier_substring_ratio = 0.1;
        features.has_why_indicator = true;
        features.has_external_ref = true;
        features.is_doc_comment = true;
        features.is_before_declaration = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.score < 0.3,
            "All-valuable features should yield score < 0.3, got {}",
            result.score
        );
    }

    #[test]
    fn mixed_features_score_in_middle_range() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        // Strong positive signals
        features.token_overlap_jaccard = 0.9;
        features.identifier_substring_ratio = 0.8;
        features.imperative_verb_noun = true;
        features.is_section_label = true;
        features.contains_literal_values = true;
        // Plus a valuable signal pulling the score down
        features.has_why_indicator = true;

        let result = scorer.score(&context, &features);
        // Expected: 0.9*0.25 + 0.8*0.30 + 0.20 + 0.10 + 0.05 - 0.30 = 0.515
        assert!(
            result.score >= 0.3 && result.score <= 0.7,
            "Mixed features should yield score in [0.3, 0.7], got {}",
            result.score
        );
    }

    #[test]
    fn score_clamped_to_zero_one_with_extreme_weights() {
        // Extreme positive weights should clamp to 1.0
        let weights = WeightsConfig {
            token_overlap_jaccard: 5.0,
            identifier_substring_ratio: 5.0,
            imperative_verb_noun: 5.0,
            verb_noun_matches_identifier: 5.0,
            is_section_label: 5.0,
            contains_literal_values: 5.0,
            references_other_files: 5.0,
            mirrors_data_structure: 5.0,
        };
        let negative = NegativeWeights::default();
        let scorer = HeuristicScorer::new(weights, negative);
        let context = make_context();
        let mut features = make_features();
        features.token_overlap_jaccard = 1.0;
        features.imperative_verb_noun = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.score <= 1.0,
            "Score should be clamped to <= 1.0, got {}",
            result.score
        );
        assert!(
            result.score == 1.0,
            "Score should be clamped to exactly 1.0 with extreme weights, got {}",
            result.score
        );

        // Extreme negative weights should clamp to 0.0
        let weights = WeightsConfig::default();
        let negative = NegativeWeights {
            has_why_indicator: -5.0,
            has_external_ref: -5.0,
            is_doc_comment_on_public: -5.0,
        };
        let scorer = HeuristicScorer::new(weights, negative);
        let mut features = make_features();
        features.has_why_indicator = true;
        features.has_external_ref = true;
        features.is_doc_comment = true;
        features.is_before_declaration = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.score >= 0.0,
            "Score should be clamped to >= 0.0, got {}",
            result.score
        );
        assert!(
            result.score == 0.0,
            "Score should be clamped to exactly 0.0 with extreme negative weights, got {}",
            result.score
        );
    }

    #[test]
    fn confidence_high_when_features_all_agree_superfluous() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        // All superfluous signals, no valuable signals
        features.token_overlap_jaccard = 0.8;
        features.identifier_substring_ratio = 0.8;
        features.imperative_verb_noun = true;
        features.is_section_label = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.confidence >= 0.8,
            "Confidence should be >= 0.8 when all features agree (superfluous), got {}",
            result.confidence
        );
    }

    #[test]
    fn confidence_high_when_features_all_agree_valuable() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        // All valuable signals, no superfluous signals
        features.token_overlap_jaccard = 0.1;
        features.identifier_substring_ratio = 0.1;
        features.has_why_indicator = true;
        features.has_external_ref = true;
        features.is_doc_comment = true;
        features.is_before_declaration = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.confidence >= 0.8,
            "Confidence should be >= 0.8 when all features agree (valuable), got {}",
            result.confidence
        );
    }

    #[test]
    fn confidence_low_when_features_mixed() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        // Mix of superfluous and valuable signals
        features.token_overlap_jaccard = 0.8;
        features.identifier_substring_ratio = 0.8;
        features.imperative_verb_noun = true;
        features.has_why_indicator = true;
        features.has_external_ref = true;
        features.is_doc_comment = true;
        features.is_before_declaration = true;

        let result = scorer.score(&context, &features);
        assert!(
            result.confidence < 0.8,
            "Confidence should be < 0.8 when features are mixed, got {}",
            result.confidence
        );
    }

    #[test]
    fn reasons_generated_for_significant_features() {
        let scorer = default_scorer();
        let context = make_context();
        let mut features = make_features();
        features.token_overlap_jaccard = 0.8;
        features.identifier_substring_ratio = 0.8;
        features.imperative_verb_noun = true;
        features.is_section_label = true;
        features.contains_literal_values = true;
        features.references_other_files = true;
        features.mirrors_data_structure = true;

        let result = scorer.score(&context, &features);
        assert!(
            !result.reasons.is_empty(),
            "Reasons should not be empty for features with superfluous signals"
        );
        // Check specific reasons are present
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("high_lexical_overlap")),
            "Should contain 'high_lexical_overlap' reason, got: {:?}",
            result.reasons
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("identifier_match")),
            "Should contain 'identifier_match' reason, got: {:?}",
            result.reasons
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("imperative_verb_noun_pattern")),
            "Should contain 'imperative_verb_noun_pattern' reason, got: {:?}",
            result.reasons
        );
        assert!(
            result.reasons.iter().any(|r| r.contains("section_label")),
            "Should contain 'section_label' reason, got: {:?}",
            result.reasons
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("contains_literals")),
            "Should contain 'contains_literals' reason, got: {:?}",
            result.reasons
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("references_files")),
            "Should contain 'references_files' reason, got: {:?}",
            result.reasons
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("mirrors_structure")),
            "Should contain 'mirrors_structure' reason, got: {:?}",
            result.reasons
        );
    }

    #[test]
    fn default_weights_produce_reasonable_scores() {
        let scorer = default_scorer();
        let context = make_context();

        // Neutral features should produce low score
        let features = make_features();
        let result = scorer.score(&context, &features);
        assert!(
            result.score >= 0.0 && result.score <= 0.2,
            "Neutral features with default weights should produce score in [0.0, 0.2], got {}",
            result.score
        );

        // Moderate overlap should produce moderate score
        let mut features = make_features();
        features.token_overlap_jaccard = 0.6;
        features.identifier_substring_ratio = 0.4;
        let result = scorer.score(&context, &features);
        assert!(
            result.score >= 0.1 && result.score <= 0.5,
            "Moderate overlap with default weights should produce score in [0.1, 0.5], got {}",
            result.score
        );
    }
}
