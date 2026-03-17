//! Unified diff parser.
//!
//! Parses unified diff text into [`FileDelta`] structs using the `unidiff` crate.

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::FileDelta;

/// Parse a unified diff string into a list of [`FileDelta`]s.
///
/// Each `FileDelta` represents one file that had lines **added**.
/// Deleted files (`+++ /dev/null`) and binary files are skipped.
pub fn parse_unified_diff(diff_text: &str) -> Vec<FileDelta> {
    if diff_text.is_empty() {
        return Vec::new();
    }

    let mut patch = unidiff::PatchSet::new();
    if patch.parse(diff_text).is_err() {
        return Vec::new();
    }

    let mut deltas = Vec::new();

    for patched_file in patch {
        // Skip deleted files
        if patched_file.target_file == "/dev/null" {
            continue;
        }

        // Collect added line numbers from all hunks
        let mut added_lines = BTreeSet::new();
        for hunk in patched_file.hunks() {
            for line in hunk.lines() {
                if line.is_added() {
                    if let Some(line_no) = line.target_line_no {
                        added_lines.insert(line_no);
                    }
                }
            }
        }

        // Skip files with no added lines (e.g., binary files)
        if added_lines.is_empty() {
            continue;
        }

        // Strip b/ prefix from target path
        let path_str = patched_file.target_file.strip_prefix("b/")
            .unwrap_or(&patched_file.target_file);

        deltas.push(FileDelta {
            path: PathBuf::from(path_str),
            added_lines,
        });
    }

    deltas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_add_only_diff() {
        let diff = "\
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,5 @@
 fn main() {
+    println!(\"hello\");
+    println!(\"world\");
     // existing
 }
";
        let deltas = parse_unified_diff(diff);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].path, PathBuf::from("src/main.rs"));
        assert!(deltas[0].added_lines.contains(&2));
        assert!(deltas[0].added_lines.contains(&3));
        assert_eq!(deltas[0].added_lines.len(), 2);
    }

    #[test]
    fn test_parse_mixed_add_remove_modify() {
        let diff = "\
diff --git a/lib.rs b/lib.rs
--- a/lib.rs
+++ b/lib.rs
@@ -1,5 +1,5 @@
 fn foo() {
-    old_line();
+    new_line();
     context();
-    another_old();
+    another_new();
 }
";
        let deltas = parse_unified_diff(diff);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].path, PathBuf::from("lib.rs"));
        // Only added lines (the `+` lines), not removed or context
        assert!(deltas[0].added_lines.contains(&2));
        assert!(deltas[0].added_lines.contains(&4));
        assert_eq!(deltas[0].added_lines.len(), 2);
    }

    #[test]
    fn test_parse_multiple_files() {
        let diff = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,2 +1,3 @@
 line1
+added_a
 line2
diff --git a/b.rs b/b.rs
--- a/b.rs
+++ b/b.rs
@@ -1,2 +1,3 @@
 line1
+added_b
 line2
diff --git a/c.rs b/c.rs
--- a/c.rs
+++ b/c.rs
@@ -1,2 +1,3 @@
 line1
+added_c
 line2
";
        let deltas = parse_unified_diff(diff);
        assert_eq!(deltas.len(), 3);
        let paths: Vec<&PathBuf> = deltas.iter().map(|d| &d.path).collect();
        assert!(paths.contains(&&PathBuf::from("a.rs")));
        assert!(paths.contains(&&PathBuf::from("b.rs")));
        assert!(paths.contains(&&PathBuf::from("c.rs")));
    }

    #[test]
    fn test_parse_new_file() {
        let diff = "\
diff --git a/new_file.rs b/new_file.rs
--- /dev/null
+++ b/new_file.rs
@@ -0,0 +1,3 @@
+fn new() {
+    // brand new
+}
";
        let deltas = parse_unified_diff(diff);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].path, PathBuf::from("new_file.rs"));
        assert_eq!(deltas[0].added_lines.len(), 3);
        assert!(deltas[0].added_lines.contains(&1));
        assert!(deltas[0].added_lines.contains(&2));
        assert!(deltas[0].added_lines.contains(&3));
    }

    #[test]
    fn test_parse_deleted_file() {
        let diff = "\
diff --git a/deleted.rs b/deleted.rs
--- a/deleted.rs
+++ /dev/null
@@ -1,3 +0,0 @@
-fn old() {
-    // gone
-}
";
        let deltas = parse_unified_diff(diff);
        assert!(deltas.is_empty(), "Deleted files should be skipped");
    }

    #[test]
    fn test_parse_renamed_file() {
        let diff = "\
diff --git a/old_name.rs b/new_name.rs
similarity index 90%
rename from old_name.rs
rename to new_name.rs
--- a/old_name.rs
+++ b/new_name.rs
@@ -1,3 +1,4 @@
 fn renamed() {
+    // new line in renamed file
     // existing
 }
";
        let deltas = parse_unified_diff(diff);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].path, PathBuf::from("new_name.rs"));
    }

    #[test]
    fn test_parse_binary_file() {
        let diff = "\
diff --git a/image.png b/image.png
Binary files /dev/null and b/image.png differ
";
        let deltas = parse_unified_diff(diff);
        assert!(deltas.is_empty(), "Binary files should be skipped");
    }

    #[test]
    fn test_parse_empty_diff() {
        let deltas = parse_unified_diff("");
        assert!(deltas.is_empty());
    }
}
