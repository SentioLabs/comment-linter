//! comment-lint-core: superfluous comment detection library
//!
//! This crate provides the core data types, traits, and utilities for
//! detecting superfluous comments in source code across multiple languages.

pub mod config;
pub mod extraction;
pub mod features;
pub mod languages;
pub mod output;
pub mod scoring;
pub mod types;

// Re-export key types at the crate root for convenience.
pub use extraction::comment::CommentContext;
pub use features::{FeatureVector, ScoredComment};
pub use languages::{detect_language, Language};
pub use scoring::Scorer;
pub use types::{CommentKind, LanguageId};
