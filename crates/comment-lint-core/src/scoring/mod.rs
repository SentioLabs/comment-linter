//! Scoring trait for comment superfluousness analysis.

use crate::extraction::comment::CommentContext;
use crate::features::{FeatureVector, ScoredComment};

/// Produces a superfluousness score for a comment given its context and
/// computed features.
///
/// Implementations may use heuristic rules, a trained model, or a
/// combination of both.
pub trait Scorer {
    /// Score a single comment, returning the full `ScoredComment`.
    fn score(&self, context: &CommentContext, features: &FeatureVector) -> ScoredComment;
}
