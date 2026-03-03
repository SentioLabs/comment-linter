//! Ensemble scorer combining heuristic and ML scoring approaches.

use comment_lint_core::extraction::comment::CommentContext;
use comment_lint_core::features::{FeatureVector, ScoredComment};
use comment_lint_core::scoring::Scorer;

/// Combines a heuristic scorer and an ML scorer via a configurable weighted
/// average, producing a single unified score with merged reasons.
pub struct EnsembleScorer {
    heuristic: Box<dyn Scorer + Send + Sync>,
    ml: Box<dyn Scorer + Send + Sync>,
    ml_weight: f32,
}

impl EnsembleScorer {
    /// Create a new `EnsembleScorer`.
    ///
    /// `ml_weight` controls how much weight is given to the ML scorer
    /// (the heuristic scorer receives `1.0 - ml_weight`).
    pub fn new(
        heuristic: Box<dyn Scorer + Send + Sync>,
        ml: Box<dyn Scorer + Send + Sync>,
        ml_weight: f32,
    ) -> Self {
        Self {
            heuristic,
            ml,
            ml_weight,
        }
    }
}

impl Scorer for EnsembleScorer {
    fn score(&self, context: &CommentContext, features: &FeatureVector) -> ScoredComment {
        let h = self.heuristic.score(context, features);
        let m = self.ml.score(context, features);

        let score = (h.score * (1.0 - self.ml_weight) + m.score * self.ml_weight).clamp(0.0, 1.0);

        // Confidence: higher when both scorers agree.
        let agreement = 1.0 - (h.score - m.score).abs();
        let base_confidence = (h.confidence + m.confidence) / 2.0;
        let confidence = (base_confidence * 0.7 + agreement * 0.3).clamp(0.0, 1.0);

        let mut reasons = Vec::new();
        reasons.extend(h.reasons.iter().map(|r| format!("[heuristic] {}", r)));
        reasons.extend(m.reasons.iter().map(|r| format!("[ml] {}", r)));

        ScoredComment {
            context: context.clone(),
            features: features.clone(),
            score,
            confidence,
            reasons,
        }
    }
}
