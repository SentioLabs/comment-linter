//! Diff parser (implemented in T1).

use super::FileDelta;

/// Parse a unified diff string into a list of file deltas.
///
/// TODO(T1): Implement full unified diff parsing.
pub fn parse_unified_diff(_diff: &str) -> Result<Vec<FileDelta>, String> {
    Err("unified diff parser not yet implemented (T1)".to_string())
}
