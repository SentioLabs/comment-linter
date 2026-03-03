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

// ---- Test: --help mentions --scorer and --model-path flags ----

#[test]
fn help_shows_scorer_flag() {
    let output = comment_lint_cmd()
        .arg("--help")
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--scorer"),
        "help output should mention --scorer flag, got: {}",
        stdout
    );
}

#[test]
fn help_shows_model_path_flag() {
    let output = comment_lint_cmd()
        .arg("--help")
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--model-path"),
        "help output should mention --model-path flag, got: {}",
        stdout
    );
}

// ---- Test: default scorer is heuristic, runs normally ----

#[test]
fn default_scorer_is_heuristic() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        0,
        "default scorer (heuristic) should work normally, got exit code {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---- Test: explicit --scorer heuristic works ----

#[test]
fn explicit_heuristic_scorer_works() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("heuristic")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        0,
        "--scorer heuristic should work, got exit code {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---- Test: --scorer ml without ml feature compiled prints error ----
//
// When compiled without `--features ml`, requesting --scorer ml should
// produce an error message and exit with code 2.
//
// Note: This test only makes sense when compiled WITHOUT the ml feature,
// which is the default for `cargo test -p comment-lint`.

#[cfg(not(feature = "ml"))]
#[test]
fn scorer_ml_without_feature_exits_with_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("ml")
        .arg("--model-path")
        .arg("/some/model.onnx")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        2,
        "--scorer ml without ml feature should exit with code 2, got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ml") || stderr.contains("ML"),
        "stderr should mention ml feature requirement, got: {}",
        stderr,
    );
}

// ---- Test: --scorer with invalid value exits with error ----

#[test]
fn invalid_scorer_value_exits_with_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("nonexistent")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        2,
        "--scorer nonexistent should exit with code 2, got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

// ---- ML feature tests (only run when compiled with --features ml) ----

#[cfg(feature = "ml")]
#[test]
fn ml_scorer_with_valid_model_runs() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    // Use the dummy model from the comment-lint-ml test fixtures
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../comment-lint-ml/tests/fixtures/dummy_model.onnx"
    );

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("ml")
        .arg("--model-path")
        .arg(model_path)
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    // Exit 0 (no findings) or 1 (findings) are both acceptable
    assert!(
        code == 0 || code == 1,
        "--scorer ml with valid model should succeed (exit 0 or 1), got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[cfg(feature = "ml")]
#[test]
fn ml_scorer_with_invalid_model_path_exits_with_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("ml")
        .arg("--model-path")
        .arg("/nonexistent/model.onnx")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        2,
        "--scorer ml with invalid model path should exit with code 2, got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[cfg(feature = "ml")]
#[test]
fn ml_scorer_without_model_path_uses_config() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let file_path = write_temp_file(&dir, "main.go", go_src);

    // Use the dummy model from the comment-lint-ml test fixtures
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../comment-lint-ml/tests/fixtures/dummy_model.onnx"
    );

    // Write config with model_path in [ml] section
    let config_content = format!("[ml]\nmodel_path = \"{}\"\n", model_path);
    let config_path = write_temp_file(&dir, "comment-lint.toml", &config_content);

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("ml")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .arg(file_path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    // Exit 0 (no findings) or 1 (findings) are both acceptable
    assert!(
        code == 0 || code == 1,
        "--scorer ml with model_path from config should succeed, got exit code {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[cfg(feature = "ml")]
#[test]
fn ml_scorer_without_any_model_path_exits_with_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let go_src = "package main\n\nfunc main() {}\n";
    let path = write_temp_file(&dir, "main.go", go_src);

    let output = comment_lint_cmd()
        .arg("--scorer")
        .arg("ml")
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to execute");

    let code = output.status.code().unwrap_or(-1);
    assert_eq!(
        code,
        2,
        "--scorer ml without any model path should exit with code 2, got {}. stderr: {}",
        code,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("model") || stderr.contains("model-path"),
        "stderr should mention missing model path, got: {}",
        stderr,
    );
}
