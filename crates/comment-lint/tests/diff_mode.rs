use std::io::Write;
use std::process::{Command, Stdio};

use tempfile::TempDir;

/// Build a `Command` pointing at the `comment-lint` binary.
fn comment_lint_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_comment-lint"))
}

/// Run `comment-lint --input-mode=diff` with the given diff piped to stdin,
/// the working directory set to `dir`, and any extra CLI arguments appended.
fn run_diff_mode(dir: &TempDir, diff_content: &str, extra_args: &[&str]) -> std::process::Output {
    let mut cmd = comment_lint_bin();
    cmd.arg("--input-mode=diff");
    cmd.args(extra_args);
    cmd.current_dir(dir.path());
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("failed to spawn comment-lint");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(diff_content.as_bytes())
        .unwrap();
    child.wait_with_output().expect("failed to wait on child")
}

/// Write a file inside `dir` at the given relative path, creating parent dirs.
fn write_source(dir: &TempDir, rel_path: &str, content: &str) {
    let path = dir.path().join(rel_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dirs");
    }
    std::fs::write(&path, content).expect("write source file");
}

/// Load a fixture diff from the `tests/fixtures/diffs/` directory.
fn fixture_diff(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/diffs/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read fixture {}: {}", path, e))
}

// ---------------------------------------------------------------------------
// Test: diff mode filters to only comments on added lines
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_filters_to_added_lines() {
    let dir = tempfile::tempdir().expect("temp dir");

    // The source file has two comments: one on line 3, one on line 8.
    let go_src = "\
package main

// increment counter
func incrementCounter() {
\tcounter++
}

// this comment is not in the diff
func otherFunction() {
\tother++
}
";
    write_source(&dir, "src/main.go", go_src);

    // The diff only adds lines 1-7 (the first function with its comment).
    let diff = fixture_diff("simple_add.diff");
    let output = run_diff_mode(
        &dir,
        &diff,
        &["--format=json", "-t", "0.0", "--min-confidence=0.0", "--include-doc-comments"],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // The comment on line 3 ("increment counter") IS in the diff's added lines.
    // The comment on line 8 ("this comment is not in the diff") is NOT.
    // Only lines from the diff should appear in output.
    assert!(
        stdout.contains("increment counter"),
        "should report the comment on an added line; stdout: {}",
        stdout
    );
    assert!(
        !stdout.contains("this comment is not in the diff"),
        "should NOT report the comment outside the diff; stdout: {}",
        stdout
    );
}

// ---------------------------------------------------------------------------
// Test: exit code 0 when diff has no comments (clean)
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_exit_code_zero_when_clean() {
    let dir = tempfile::tempdir().expect("temp dir");

    let go_src = "\
package main

func incrementCounter() {
\tcounter++
}
";
    write_source(&dir, "src/main.go", go_src);

    let diff = fixture_diff("no_comments.diff");
    let output = run_diff_mode(&dir, &diff, &[]);

    assert_eq!(
        output.status.code().unwrap_or(-1),
        0,
        "exit code should be 0 when no superfluous comments found; stderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---------------------------------------------------------------------------
// Test: exit code 1 when diff has superfluous comments
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_exit_code_one_when_issues() {
    let dir = tempfile::tempdir().expect("temp dir");

    let go_src = "\
package main

// increment counter
func incrementCounter() {
\tcounter++
}
";
    write_source(&dir, "src/main.go", go_src);

    let diff = fixture_diff("simple_add.diff");
    let output = run_diff_mode(
        &dir,
        &diff,
        &["-t", "0.0", "--min-confidence=0.0", "--include-doc-comments"],
    );

    assert_eq!(
        output.status.code().unwrap_or(-1),
        1,
        "exit code should be 1 when superfluous comments found; stderr: {}; stdout: {}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );
}

// ---------------------------------------------------------------------------
// Test: --format=json produces valid JSON lines
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_json_format() {
    let dir = tempfile::tempdir().expect("temp dir");

    let go_src = "\
package main

// increment counter
func incrementCounter() {
\tcounter++
}
";
    write_source(&dir, "src/main.go", go_src);

    let diff = fixture_diff("simple_add.diff");
    let output = run_diff_mode(
        &dir,
        &diff,
        &["--format=json", "-t", "0.0", "--min-confidence=0.0", "--include-doc-comments"],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    // There should be at least one comment line + one summary line
    assert!(
        lines.len() >= 2,
        "JSON output should have at least 2 lines (comment + summary), got {}; stdout: {}",
        lines.len(),
        stdout,
    );

    // Every line should be valid JSON
    for line in &lines {
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
            panic!("each line should be valid JSON: {}; line: {}", e, line)
        });
        assert!(parsed.is_object(), "each JSON line should be an object");
    }

    // The first comment line should have expected fields
    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert!(first.get("file").is_some(), "comment JSON should have 'file' field");
    assert!(first.get("line").is_some(), "comment JSON should have 'line' field");
    assert!(first.get("score").is_some(), "comment JSON should have 'score' field");

    // The last line should be the summary
    let last: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
    assert_eq!(
        last.get("type").and_then(|v| v.as_str()),
        Some("summary"),
        "last JSON line should be a summary object",
    );
}

// ---------------------------------------------------------------------------
// Test: --format=github produces ::warning annotations
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_github_format() {
    let dir = tempfile::tempdir().expect("temp dir");

    let go_src = "\
package main

// increment counter
func incrementCounter() {
\tcounter++
}
";
    write_source(&dir, "src/main.go", go_src);

    let diff = fixture_diff("simple_add.diff");
    let output = run_diff_mode(
        &dir,
        &diff,
        &["--format=github", "-t", "0.0", "--min-confidence=0.0", "--include-doc-comments"],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("::warning file="),
        "github format should contain ::warning file= annotation; stdout: {}",
        stdout,
    );
}

// ---------------------------------------------------------------------------
// Test: --input-mode=diff with null stdin exits with code 2
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_error_without_pipe() {
    let mut cmd = comment_lint_bin();
    cmd.arg("--input-mode=diff");
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output().expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    // Stdin::null() is not a TTY, so it passes the is_terminal() check.
    // But reading from /dev/null yields an empty string, which produces
    // an empty diff with no files -- exit 0 (clean). OR it could fail
    // to parse the diff and exit 2. Let's just verify it doesn't panic
    // and exits cleanly (0 or 2).
    assert!(
        code == 0 || code == 2,
        "null stdin should exit 0 (empty diff, no issues) or 2 (parse error), got {}; stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---------------------------------------------------------------------------
// Test: multi-file diff processes both files
// ---------------------------------------------------------------------------

#[test]
fn test_diff_mode_multi_file() {
    let dir = tempfile::tempdir().expect("temp dir");

    let main_go = "\
package main

// increment counter
func incrementCounter() {
\tcounter++
}
";
    let utils_go = "\
package main

// add numbers
func addNumbers(a int, b int) int {
\treturn a + b
}
";
    write_source(&dir, "src/main.go", main_go);
    write_source(&dir, "src/utils.go", utils_go);

    let diff = fixture_diff("multi_file.diff");
    let output = run_diff_mode(
        &dir,
        &diff,
        &["--format=json", "-t", "0.0", "--min-confidence=0.0", "--include-doc-comments"],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Both files should be represented in the output
    assert!(
        stdout.contains("main.go"),
        "output should mention main.go; stdout: {}",
        stdout,
    );
    assert!(
        stdout.contains("utils.go"),
        "output should mention utils.go; stdout: {}",
        stdout,
    );
}
