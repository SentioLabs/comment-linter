//! JSONL output formatter for feature export (training data generation).

use crate::features::ScoredComment;
use crate::output::OutputFormatter;
use serde_json::json;
use std::io::Write;

/// Formatter that emits one JSON object per line with full feature vectors for ML training data.
pub struct FeaturesJsonlFormatter;

impl OutputFormatter for FeaturesJsonlFormatter {
    fn format_comment(
        &self,
        comment: &ScoredComment,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        let features_value = serde_json::to_value(&comment.features)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let value = json!({
            "file": comment.context.file_path.display().to_string(),
            "line": comment.context.line,
            "column": comment.context.column,
            "language": comment.context.language,
            "comment_text": comment.context.comment_text,
            "comment_kind": comment.context.comment_kind,
            "heuristic_score": comment.score as f64,
            "heuristic_confidence": comment.confidence as f64,
            "features": features_value,
        });

        let line = serde_json::to_string(&value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(writer, "{}", line)
    }

    fn format_summary(
        &self,
        _total_comments: usize,
        _superfluous_count: usize,
        _file_count: usize,
        _writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::output::tests::make_scored_comment;
    use crate::output::OutputFormatter;

    #[test]
    fn features_jsonl_produces_valid_json() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert!(parsed.is_object(), "output should be a JSON object");
    }

    #[test]
    fn features_jsonl_contains_required_fields() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let obj = parsed.as_object().unwrap();

        assert!(obj.contains_key("file"), "should have 'file' field");
        assert!(obj.contains_key("line"), "should have 'line' field");
        assert!(obj.contains_key("column"), "should have 'column' field");
        assert!(obj.contains_key("language"), "should have 'language' field");
        assert!(obj.contains_key("comment_text"), "should have 'comment_text' field");
        assert!(obj.contains_key("comment_kind"), "should have 'comment_kind' field");
        assert!(obj.contains_key("heuristic_score"), "should have 'heuristic_score' field");
        assert!(
            obj.contains_key("heuristic_confidence"),
            "should have 'heuristic_confidence' field"
        );
        assert!(obj.contains_key("features"), "should have 'features' field");
    }

    #[test]
    fn features_jsonl_field_values_are_correct() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();

        assert_eq!(parsed["file"], "src/main.rs");
        assert_eq!(parsed["line"], 42);
        assert_eq!(parsed["column"], 4);
        assert_eq!(parsed["language"], "rust");
        assert_eq!(parsed["comment_text"], "increment counter");
        assert_eq!(parsed["comment_kind"], "line");
    }

    #[test]
    fn features_jsonl_has_feature_vector_fields() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let features = parsed["features"].as_object().expect("features should be an object");

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
    }

    #[test]
    fn features_jsonl_feature_values_are_correct() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let features = &parsed["features"];

        // Check numeric feature values from make_scored_comment
        let jaccard = features["token_overlap_jaccard"].as_f64().unwrap();
        assert!(
            (jaccard - 0.8).abs() < 0.01,
            "token_overlap_jaccard should be ~0.8, got {}",
            jaccard
        );

        let ratio = features["identifier_substring_ratio"].as_f64().unwrap();
        assert!(
            (ratio - 0.9).abs() < 0.01,
            "identifier_substring_ratio should be ~0.9, got {}",
            ratio
        );

        assert_eq!(features["comment_token_count"], 2);
        assert_eq!(features["is_doc_comment"], false);
        assert_eq!(features["is_inline"], true);
    }

    #[test]
    fn features_jsonl_output_ends_with_newline() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.ends_with('\n'), "JSONL lines should end with newline");
    }

    #[test]
    fn features_jsonl_uses_compact_format() {
        let formatter = super::FeaturesJsonlFormatter;
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
    fn features_jsonl_summary_is_noop() {
        let formatter = super::FeaturesJsonlFormatter;
        let mut buf = Vec::new();
        formatter.format_summary(100, 25, 10, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.is_empty(), "summary should be a no-op, got: {}", output);
    }

    #[test]
    fn features_jsonl_heuristic_scores_are_f64() {
        let formatter = super::FeaturesJsonlFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert!(
            parsed["heuristic_score"].is_f64(),
            "heuristic_score should be a number"
        );
        assert!(
            parsed["heuristic_confidence"].is_f64(),
            "heuristic_confidence should be a number"
        );
    }
}
