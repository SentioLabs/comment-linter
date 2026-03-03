use std::io::Write;
use std::process::Command;

use tempfile::TempDir;

/// Helper: write a file with given content into a temp dir and return its path.
fn write_temp_file(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dirs");
    }
    let mut f = std::fs::File::create(&path).expect("create file");
    f.write_all(content.as_bytes()).expect("write file");
    path
}

fn comment_lint_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_comment-lint"))
}

/// Run the binary with --export-features on a Go fixture file that has a superfluous comment.
/// Returns the parsed JSONL lines from stdout.
fn run_export(fixture_content: &str) -> Vec<serde_json::Value> {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_temp_file(&dir, "main.go", fixture_content);

    let output = comment_lint_cmd()
        .args(["--export-features", path.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.code().is_some(),
        "process should have an exit code. stderr: {}",
        stderr
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    stdout
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("failed to parse JSONL line: {}\nline: {}", e, l))
        })
        .collect()
}

const GO_FIXTURE: &str =
    "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";

// ---- Test: --export-features produces valid JSONL ----

#[test]
fn test_export_features_produces_valid_jsonl() {
    let records = run_export(GO_FIXTURE);
    assert!(
        !records.is_empty(),
        "export-features should produce at least one JSONL line"
    );
    for record in &records {
        assert!(
            record.is_object(),
            "each JSONL line should be a JSON object"
        );
    }
}

// ---- Test: --export-features output has required fields ----

#[test]
fn test_export_features_has_required_fields() {
    let records = run_export(GO_FIXTURE);
    assert!(!records.is_empty(), "should have at least one record");

    for record in &records {
        let obj = record.as_object().unwrap();
        assert!(obj.contains_key("file"), "should have 'file' field");
        assert!(obj.contains_key("line"), "should have 'line' field");
        assert!(obj.contains_key("column"), "should have 'column' field");
        assert!(obj.contains_key("language"), "should have 'language' field");
        assert!(
            obj.contains_key("comment_text"),
            "should have 'comment_text' field"
        );
        assert!(
            obj.contains_key("comment_kind"),
            "should have 'comment_kind' field"
        );
        assert!(
            obj.contains_key("heuristic_score"),
            "should have 'heuristic_score' field"
        );
        assert!(
            obj.contains_key("heuristic_confidence"),
            "should have 'heuristic_confidence' field"
        );
        assert!(obj.contains_key("features"), "should have 'features' field");
    }
}

// ---- Test: --export-features includes all feature fields ----

#[test]
fn test_export_features_includes_all_feature_fields() {
    let records = run_export(GO_FIXTURE);
    assert!(!records.is_empty(), "should have at least one record");

    for record in &records {
        let features = record["features"]
            .as_object()
            .expect("features should be an object");

        assert!(
            features.contains_key("token_overlap_jaccard"),
            "features should contain token_overlap_jaccard"
        );
        assert!(
            features.contains_key("identifier_substring_ratio"),
            "features should contain identifier_substring_ratio"
        );
        assert!(
            features.contains_key("comment_token_count"),
            "features should contain comment_token_count"
        );
        assert!(
            features.contains_key("is_doc_comment"),
            "features should contain is_doc_comment"
        );
        assert!(
            features.contains_key("is_before_declaration"),
            "features should contain is_before_declaration"
        );
        assert!(
            features.contains_key("is_inline"),
            "features should contain is_inline"
        );
        assert!(
            features.contains_key("has_why_indicator"),
            "features should contain has_why_indicator"
        );
        assert!(
            features.contains_key("has_external_ref"),
            "features should contain has_external_ref"
        );
        assert!(
            features.contains_key("imperative_verb_noun"),
            "features should contain imperative_verb_noun"
        );
        assert!(
            features.contains_key("is_section_label"),
            "features should contain is_section_label"
        );
    }
}

// ---- Test: --export-features does not include a summary line ----

#[test]
fn test_export_features_no_summary_line() {
    let records = run_export(GO_FIXTURE);
    for record in &records {
        let obj = record.as_object().unwrap();
        assert!(
            !obj.contains_key("type"),
            "export-features output should not contain summary lines with a 'type' field"
        );
    }
}

// ---- Test: --export-features overrides threshold to capture all comments ----

#[test]
fn test_export_features_captures_all_comments() {
    // Use a file with a comment that might be filtered by default threshold
    let go_src = "package main\n\n// main entry point\nfunc main() {\n}\n";
    let records = run_export(go_src);
    assert!(
        !records.is_empty(),
        "export-features should capture comments even with low scores (threshold forced to 0.0)"
    );
}

// ---- Test: --export-features flag appears in help ----

#[test]
fn test_export_features_flag_in_help() {
    let output = comment_lint_cmd()
        .arg("--help")
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--export-features"),
        "help output should mention --export-features flag, got: {}",
        stdout
    );
}
