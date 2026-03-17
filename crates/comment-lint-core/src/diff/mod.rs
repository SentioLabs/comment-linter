//! Diff parsing and filtering for CI/CD workflows.
//!
//! Parses unified diff from stdin and filters comments to only
//! those on added lines within changed hunks.

pub mod filter;
pub mod parser;

use std::collections::BTreeSet;
use std::path::PathBuf;

/// How the pipeline discovers files to lint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InputMode {
    /// Scan explicit file/directory paths (existing behavior).
    #[default]
    Files,
    /// Read a unified diff from stdin; lint only added lines.
    Diff,
}

/// A single file's changes extracted from a unified diff.
#[derive(Debug, Clone)]
pub struct FileDelta {
    /// Path to the file (new side, with `a/`/`b/` prefix stripped).
    pub path: PathBuf,
    /// 1-based line numbers of added (`+`) lines in the new file.
    pub added_lines: BTreeSet<usize>,
}
