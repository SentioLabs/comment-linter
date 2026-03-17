//! Diff filter: restricts linting to comments on added lines.

use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::FileDelta;

/// Filters comments to only those appearing on added lines in a diff.
#[derive(Debug)]
pub struct DiffFilter {
    changed_lines: HashMap<PathBuf, BTreeSet<usize>>,
}

impl DiffFilter {
    /// Create a new `DiffFilter` from a list of file deltas.
    pub fn new(deltas: Vec<FileDelta>) -> Self {
        let changed_lines = deltas
            .into_iter()
            .map(|d| (d.path, d.added_lines))
            .collect();
        Self { changed_lines }
    }

    /// Read a unified diff from stdin and build a `DiffFilter`.
    pub fn from_stdin() -> Result<Self, String> {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        Self::from_diff_text(&input)
    }

    /// Parse a unified diff string and build a `DiffFilter`.
    pub fn from_diff_text(diff_text: &str) -> Result<Self, String> {
        let deltas = super::parser::parse_unified_diff(diff_text)?;
        Ok(Self::new(deltas))
    }

    /// Check if a given file path and line number are within the added lines.
    pub fn includes(&self, path: &str, line: usize) -> bool {
        let p = Path::new(path);
        let normalized = p.strip_prefix("./").unwrap_or(p);
        self.changed_lines
            .get(normalized)
            .map_or(false, |lines| lines.contains(&line))
    }

    /// Iterate over all file paths in the diff.
    pub fn files(&self) -> impl Iterator<Item = &Path> {
        self.changed_lines.keys().map(|p| p.as_path())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use super::DiffFilter;
    use crate::diff::FileDelta;

    fn sample_deltas() -> Vec<FileDelta> {
        vec![
            FileDelta {
                path: PathBuf::from("src/foo.rs"),
                added_lines: BTreeSet::from([3, 5, 7]),
            },
            FileDelta {
                path: PathBuf::from("src/bar.rs"),
                added_lines: BTreeSet::from([1, 2, 10]),
            },
        ]
    }

    #[test]
    fn test_includes_added_line() {
        let filter = DiffFilter::new(sample_deltas());
        assert!(filter.includes("src/foo.rs", 5));
        assert!(filter.includes("src/bar.rs", 10));
    }

    #[test]
    fn test_excludes_non_added_line() {
        let filter = DiffFilter::new(sample_deltas());
        assert!(!filter.includes("src/foo.rs", 10));
        assert!(!filter.includes("src/bar.rs", 5));
    }

    #[test]
    fn test_excludes_unknown_file() {
        let filter = DiffFilter::new(sample_deltas());
        assert!(!filter.includes("src/unknown.rs", 5));
    }

    #[test]
    fn test_files_iterator() {
        let filter = DiffFilter::new(sample_deltas());
        let mut files: Vec<&str> = filter
            .files()
            .map(|p| p.to_str().unwrap())
            .collect();
        files.sort();
        assert_eq!(files, vec!["src/bar.rs", "src/foo.rs"]);
    }

    #[test]
    #[ignore]
    fn test_from_diff_text() {
        let diff = "\
diff --git a/src/foo.rs b/src/foo.rs
index 1234567..abcdefg 100644
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -1,3 +1,4 @@
 fn main() {
+    // added comment
     println!(\"hello\");
 }
";
        let filter = DiffFilter::from_diff_text(diff).unwrap();
        assert!(filter.includes("src/foo.rs", 2));
        assert!(!filter.includes("src/foo.rs", 1));
    }

    #[test]
    fn test_path_normalization() {
        let filter = DiffFilter::new(vec![FileDelta {
            path: PathBuf::from("src/foo.rs"),
            added_lines: BTreeSet::from([5]),
        }]);
        // Query with leading "./" should still match
        assert!(filter.includes("./src/foo.rs", 5));
        // And without leading "./" should also match
        assert!(filter.includes("src/foo.rs", 5));
    }
}
