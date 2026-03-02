//! GitHub Actions annotation formatter.

use crate::features::ScoredComment;
use crate::output::OutputFormatter;
use std::io::Write;

/// Formatter that emits GitHub Actions workflow command annotations.
///
/// Each comment becomes a `::warning` annotation that GitHub renders inline
/// in pull request diffs.
pub struct GithubFormatter;

impl OutputFormatter for GithubFormatter {
    fn format_comment(
        &self,
        comment: &ScoredComment,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        let path = comment.context.file_path.display();
        let line = comment.context.line;
        let col = comment.context.column;
        let score = comment.score;
        let confidence = comment.confidence;
        let reasons = comment.reasons.join(", ");

        writeln!(
            writer,
            "::warning file={},line={},col={}::Superfluous comment (score: {:.2}, confidence: {:.2}): {}",
            path, line, col, score, confidence, reasons,
        )
    }

    fn format_summary(
        &self,
        _total_comments: usize,
        _superfluous_count: usize,
        _file_count: usize,
        _writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        // GitHub Actions does not use summary annotations -- intentionally a no-op.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::output::tests::make_scored_comment;
    use crate::output::OutputFormatter;

    #[test]
    fn github_formatter_produces_warning_annotation() {
        let formatter = super::GithubFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(
            output.starts_with("::warning "),
            "GitHub annotation should start with ::warning"
        );
    }

    #[test]
    fn github_formatter_contains_file_line_col() {
        let formatter = super::GithubFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("file=src/main.rs"), "should contain file parameter");
        assert!(output.contains("line=42"), "should contain line parameter");
        assert!(output.contains("col=4"), "should contain col parameter");
    }

    #[test]
    fn github_formatter_contains_score_and_confidence() {
        let formatter = super::GithubFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("score:"), "should contain score label");
        assert!(output.contains("confidence:"), "should contain confidence label");
    }

    #[test]
    fn github_formatter_contains_reasons() {
        let formatter = super::GithubFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(
            output.contains("high token overlap"),
            "should contain first reason"
        );
        assert!(
            output.contains("mirrors identifier"),
            "should contain second reason"
        );
    }

    #[test]
    fn github_formatter_summary_writes_nothing_or_comment() {
        let formatter = super::GithubFormatter;
        let mut buf = Vec::new();
        formatter.format_summary(100, 25, 10, &mut buf).unwrap();
        // GitHub Actions does not use summary — either empty or a comment is acceptable
        let output = String::from_utf8(buf).unwrap();
        // Just ensure it does not produce a ::warning or ::error for the summary
        assert!(
            !output.contains("::warning") && !output.contains("::error"),
            "summary should not produce an annotation"
        );
    }

    #[test]
    fn github_formatter_output_ends_with_newline() {
        let formatter = super::GithubFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.ends_with('\n'), "annotation should end with newline");
    }
}
