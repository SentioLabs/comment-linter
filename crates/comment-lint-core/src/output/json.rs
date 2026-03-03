//! JSONL output formatter for tooling integration.

use crate::features::ScoredComment;
use crate::output::OutputFormatter;
use serde_json::json;
use std::io::Write;
use std::time::Duration;

/// Formatter that emits one JSON object per line (JSONL) for machine consumption.
pub struct JsonFormatter;

impl OutputFormatter for JsonFormatter {
    fn format_comment(
        &self,
        comment: &ScoredComment,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        let value = json!({
            "file": comment.context.file_path.display().to_string(),
            "line": comment.context.line,
            "column": comment.context.column,
            "comment_text": comment.context.comment_text,
            "score": comment.score,
            "confidence": comment.confidence,
            "language": comment.context.language.name(),
            "reasons": comment.reasons,
        });
        let line = serde_json::to_string(&value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(writer, "{}", line)
    }

    fn format_summary(
        &self,
        total_comments: usize,
        superfluous_count: usize,
        file_count: usize,
        elapsed: Duration,
        cpu_time: Option<Duration>,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        let mut value = json!({
            "type": "summary",
            "total_comments": total_comments,
            "superfluous_count": superfluous_count,
            "file_count": file_count,
            "elapsed_ms": elapsed.as_millis(),
        });
        if let Some(cpu) = cpu_time {
            value["cpu_ms"] = serde_json::Value::from(cpu.as_millis() as u64);
        }
        let line = serde_json::to_string(&value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(writer, "{}", line)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use crate::output::tests::make_scored_comment;
    use crate::output::OutputFormatter;

    #[test]
    fn json_formatter_produces_valid_json() {
        let formatter = super::JsonFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert!(parsed.is_object(), "output should be a JSON object");
    }

    #[test]
    fn json_formatter_contains_all_required_fields() {
        let formatter = super::JsonFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let obj = parsed.as_object().unwrap();

        assert!(obj.contains_key("file"), "should have 'file' field");
        assert!(obj.contains_key("line"), "should have 'line' field");
        assert!(obj.contains_key("column"), "should have 'column' field");
        assert!(obj.contains_key("comment_text"), "should have 'comment_text' field");
        assert!(obj.contains_key("score"), "should have 'score' field");
        assert!(obj.contains_key("confidence"), "should have 'confidence' field");
        assert!(obj.contains_key("language"), "should have 'language' field");
        assert!(obj.contains_key("reasons"), "should have 'reasons' field");
    }

    #[test]
    fn json_formatter_field_values_are_correct() {
        let formatter = super::JsonFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();

        assert_eq!(parsed["file"], "src/main.rs");
        assert_eq!(parsed["line"], 42);
        assert_eq!(parsed["column"], 4);
        assert_eq!(parsed["comment_text"], "increment counter");
        assert_eq!(parsed["language"], "rust");
        assert_eq!(parsed["reasons"][0], "high token overlap");
        assert_eq!(parsed["reasons"][1], "mirrors identifier");
    }

    #[test]
    fn json_formatter_uses_compact_format() {
        let formatter = super::JsonFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        // Compact JSON should have no pretty-print indentation
        assert!(
            !output.contains("  "),
            "compact JSON should not contain indentation"
        );
    }

    #[test]
    fn json_formatter_summary_has_type_field() {
        let formatter = super::JsonFormatter;
        let mut buf = Vec::new();
        formatter.format_summary(100, 25, 10, Duration::from_millis(1234), None, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["type"], "summary");
        assert_eq!(parsed["total_comments"], 100);
        assert_eq!(parsed["superfluous_count"], 25);
        assert_eq!(parsed["file_count"], 10);
    }

    #[test]
    fn json_formatter_output_ends_with_newline() {
        let formatter = super::JsonFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.ends_with('\n'), "JSONL lines should end with newline");
    }
}
