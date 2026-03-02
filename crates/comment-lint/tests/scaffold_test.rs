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

// ---- Test: --help shows usage and exits 0 ----

#[test]
fn help_flag_shows_usage() {
    let output = comment_lint_cmd()
        .arg("--help")
        .output()
        .expect("failed to execute");

    assert!(
        output.status.success(),
        "help should exit with code 0, got {:?}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("comment-lint"),
        "help output should contain program name, got: {}",
        stdout
    );
    assert!(
        stdout.contains("--format"),
        "help output should mention --format flag, got: {}",
        stdout
    );
    assert!(
        stdout.contains("--threshold"),
        "help output should mention --threshold flag, got: {}",
        stdout
    );
}

// ---- Test: --version shows version info ----

#[test]
fn version_flag_shows_version() {
    let output = comment_lint_cmd()
        .arg("--version")
        .output()
        .expect("failed to execute");

    assert!(
        output.status.success(),
        "version should exit with code 0"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("comment-lint"),
        "version output should contain program name"
    );
}

// ---- Test: no arguments produces an error (paths required) ----

#[test]
fn no_args_exits_with_error() {
    let output = comment_lint_cmd()
        .output()
        .expect("failed to execute");

    assert!(
        !output.status.success(),
        "should exit with non-zero when no paths given"
    );
}

// ---- Test: processing a file with superfluous comments exits with code 1 ----

#[test]
fn superfluous_comment_exits_with_code_1() {
    let dir = tempfile::tempdir().expect("temp dir");
    // Use multi-line format so tree-sitter parses the comment correctly.
    // Include --include-doc-comments because Go comments before functions are doc comments.
    let go_src = "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg(path.to_str().unwrap())
        .arg("--threshold")
        .arg("0.0")
        .arg("--min-confidence")
        .arg("0.0")
        .arg("--include-doc-comments")
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 1,
        "should exit with code 1 when superfluous comments found, got {}. stdout: {}, stderr: {}",
        code,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---- Test: processing a clean file exits with code 0 ----

#[test]
fn clean_file_exits_with_code_0() {
    let dir = tempfile::tempdir().expect("temp dir");
    // A file with no comments should produce no findings
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 0,
        "should exit with code 0 when no superfluous comments found, got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---- Test: --format json produces valid JSON output ----

#[test]
fn json_format_produces_valid_json() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--format")
        .arg("json")
        .arg("--threshold")
        .arg("0.0")
        .arg("--min-confidence")
        .arg("0.0")
        .arg("--include-doc-comments")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Each line should be valid JSON
    for line in stdout.lines() {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        assert!(
            parsed.is_ok(),
            "each line of JSON output should be valid JSON, got: {}",
            line
        );
    }
    // Should have at least one comment line and one summary line
    assert!(
        stdout.lines().count() >= 2,
        "JSON output should have at least 2 lines (comment + summary), got: {}",
        stdout
    );
}

// ---- Test: --format github produces GitHub annotations ----

#[test]
fn github_format_produces_annotations() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--format")
        .arg("github")
        .arg("--threshold")
        .arg("0.0")
        .arg("--min-confidence")
        .arg("0.0")
        .arg("--include-doc-comments")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("::warning"),
        "GitHub format should produce ::warning annotations, got: {}",
        stdout
    );
}

// ---- Test: high threshold filters out results ----

#[test]
fn high_threshold_filters_results() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--threshold")
        .arg("0.99")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 0,
        "very high threshold should filter out all results, exit code should be 0, got {}",
        code
    );
}

// ---- Test: explicit config file is loaded ----

#[test]
fn explicit_config_file_overrides_threshold() {
    let dir = tempfile::tempdir().expect("temp dir");

    // Write a config that sets threshold very high
    let config_path = write_temp_file(
        &dir,
        "custom.toml",
        "[general]\nthreshold = 0.99\n",
    );

    let go_src = "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";
    let file_path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .arg(file_path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 0,
        "config with threshold=0.99 should filter all results, exit 0, got {}",
        code
    );
}

// ---- Test: CLI threshold overrides config threshold ----

#[test]
fn cli_threshold_overrides_config() {
    let dir = tempfile::tempdir().expect("temp dir");

    // Config sets low threshold
    let config_path = write_temp_file(
        &dir,
        "custom.toml",
        "[general]\nthreshold = 0.0\nmin_confidence = 0.0\n",
    );

    let go_src = "package main\n\n// increment counter\nfunc incrementCounter() {\n    counter++\n}\n";
    let file_path = write_temp_file(&dir, "main.go", go_src);

    // CLI overrides with high threshold
    let output = comment_lint_cmd()
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .arg("--threshold")
        .arg("0.99")
        .arg(file_path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 0,
        "CLI --threshold=0.99 should override config threshold=0.0, exit 0, got {}",
        code
    );
}

// ---- Test: invalid config path produces error exit code 2 ----

#[test]
fn invalid_config_path_exits_with_code_2() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let file_path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--config")
        .arg("/nonexistent/path/config.toml")
        .arg(file_path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code, 2,
        "invalid config path should exit with code 2, got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}
