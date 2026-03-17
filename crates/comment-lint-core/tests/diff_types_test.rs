//! Integration tests for diff module types.

use std::collections::BTreeSet;
use std::path::PathBuf;

use comment_lint_core::diff::{FileDelta, InputMode};

#[test]
fn input_mode_defaults_to_files() {
    let mode = InputMode::default();
    assert_eq!(mode, InputMode::Files);
}

#[test]
fn input_mode_clone_and_debug() {
    let mode = InputMode::Diff;
    let cloned = mode.clone();
    assert_eq!(cloned, InputMode::Diff);
    // Debug should not panic
    let debug_str = format!("{:?}", cloned);
    assert!(debug_str.contains("Diff"));
}

#[test]
fn input_mode_equality() {
    assert_eq!(InputMode::Files, InputMode::Files);
    assert_eq!(InputMode::Diff, InputMode::Diff);
    assert_ne!(InputMode::Files, InputMode::Diff);
}

#[test]
fn file_delta_construction() {
    let mut added = BTreeSet::new();
    added.insert(1);
    added.insert(5);
    added.insert(10);

    let delta = FileDelta {
        path: PathBuf::from("src/main.rs"),
        added_lines: added,
    };

    assert_eq!(delta.path, PathBuf::from("src/main.rs"));
    assert!(delta.added_lines.contains(&1));
    assert!(delta.added_lines.contains(&5));
    assert!(delta.added_lines.contains(&10));
    assert!(!delta.added_lines.contains(&2));
    assert_eq!(delta.added_lines.len(), 3);
}

#[test]
fn file_delta_clone_and_debug() {
    let delta = FileDelta {
        path: PathBuf::from("lib.rs"),
        added_lines: BTreeSet::from([3, 7]),
    };

    let cloned = delta.clone();
    assert_eq!(cloned.path, delta.path);
    assert_eq!(cloned.added_lines, delta.added_lines);

    // Debug should not panic
    let debug_str = format!("{:?}", cloned);
    assert!(debug_str.contains("lib.rs"));
}

#[test]
fn file_delta_empty_added_lines() {
    let delta = FileDelta {
        path: PathBuf::from("empty.rs"),
        added_lines: BTreeSet::new(),
    };

    assert!(delta.added_lines.is_empty());
}
