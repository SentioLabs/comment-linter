use std::process::Command;

#[test]
fn binary_prints_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_comment-lint"))
        .output()
        .expect("failed to execute comment-lint binary");

    assert!(output.status.success(), "binary should exit successfully");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "comment-lint v0.1.0");
}

#[test]
fn core_crate_exists() {
    // Verify the core library crate is accessible as a dependency
    // This test simply ensures the workspace wiring is correct
    // by importing from the core crate via the binary crate
    let output = Command::new(env!("CARGO_BIN_EXE_comment-lint"))
        .output()
        .expect("failed to execute comment-lint binary");

    assert!(output.status.success());
}
