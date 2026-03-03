//! Colored terminal output formatter with star ratings.

use crate::features::ScoredComment;
use crate::output::OutputFormatter;
use colored::Colorize;
use std::io::Write;
use std::time::Duration;

/// Terminal output formatter that displays scored comments with colored text
/// and Unicode star ratings.
pub struct TextFormatter;

/// Format a Duration as a human-readable string (e.g. "1.23s", "456ms").
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2}s")
    } else {
        format!("{:.0}ms", secs * 1000.0)
    }
}

/// Compute a star rating (1..=5) from a superfluousness score.
///
/// - 0.6..0.7  => 1 star
/// - 0.7..0.8  => 2 stars
/// - 0.8..0.85 => 3 stars
/// - 0.85..0.9 => 4 stars
/// - 0.9+      => 5 stars
fn star_rating(score: f32) -> usize {
    if score >= 0.9 {
        5
    } else if score >= 0.85 {
        4
    } else if score >= 0.8 {
        3
    } else if score >= 0.7 {
        2
    } else {
        1
    }
}

/// Render a star string: filled stars (\u{2605}) followed by empty stars (\u{2606}).
fn star_string(rating: usize) -> String {
    let filled = "\u{2605}".repeat(rating);
    let empty = "\u{2606}".repeat(5 - rating);
    format!("{}{}", filled, empty)
}

impl OutputFormatter for TextFormatter {
    fn format_comment(
        &self,
        comment: &ScoredComment,
        writer: &mut dyn Write,
    ) -> std::io::Result<()> {
        let path = comment.context.file_path.display();
        let line = comment.context.line;
        let col = comment.context.column;
        let score = comment.score;
        let stars = star_string(star_rating(score));
        let text = &comment.context.comment_text;
        let reasons = comment.reasons.join(", ");

        writeln!(
            writer,
            "{} {} {:.2} {} {} \"{}\" {} {}",
            format!("{}:{}:{}", path, line, col).bold(),
            "\u{2014}".dimmed(),
            score,
            stars.yellow(),
            "\u{2014}".dimmed(),
            text.cyan(),
            "\u{2014}".dimmed(),
            reasons,
        )
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
        let elapsed_str = format_duration(elapsed);
        let timing = match cpu_time {
            Some(cpu) => {
                let cpu_str = format_duration(cpu);
                let parallelism = if !elapsed.is_zero() {
                    cpu.as_secs_f64() / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                format!("{elapsed_str} wall, {cpu_str} cpu ({parallelism:.1}x)")
            }
            None => elapsed_str,
        };

        writeln!(
            writer,
            "\n{} {} comments scanned, {} superfluous, {} files in {}",
            "Summary:".bold(),
            total_comments,
            superfluous_count.to_string().red(),
            file_count,
            timing.dimmed(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::output::tests::make_scored_comment;
    use crate::output::OutputFormatter;
    use std::time::Duration;

    #[test]
    fn text_formatter_formats_comment_with_star_rating() {
        let formatter = super::TextFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        // Score 0.85 is in the 0.85..0.9 range => 4 stars
        assert!(output.contains("src/main.rs"), "should contain file path");
        assert!(output.contains("42"), "should contain line number");
        assert!(output.contains("4"), "should contain column number");
        assert!(
            output.contains("increment counter"),
            "should contain comment text"
        );
        assert!(
            output.contains("high token overlap"),
            "should contain reason"
        );
    }

    #[test]
    fn star_rating_one_star_for_score_0_65() {
        let formatter = super::TextFormatter;
        let comment = make_scored_comment(0.65, 0.8);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // score 0.65 is in 0.6..0.7 => 1 star
        assert!(
            output.contains("\u{2605}"),
            "should contain at least one filled star"
        );
    }

    #[test]
    fn star_rating_five_stars_for_score_0_95() {
        let formatter = super::TextFormatter;
        let comment = make_scored_comment(0.95, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // 5 filled stars
        let filled_count = output.matches('\u{2605}').count();
        assert_eq!(filled_count, 5, "score 0.95 should produce 5 filled stars");
    }

    #[test]
    fn star_rating_two_stars_for_score_0_75() {
        let formatter = super::TextFormatter;
        let comment = make_scored_comment(0.75, 0.8);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let filled_count = output.matches('\u{2605}').count();
        assert_eq!(filled_count, 2, "score 0.75 should produce 2 filled stars");
    }

    #[test]
    fn star_rating_four_stars_for_score_0_85() {
        let formatter = super::TextFormatter;
        let comment = make_scored_comment(0.85, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let filled_count = output.matches('\u{2605}').count();
        assert_eq!(filled_count, 4, "score 0.85 should produce 4 filled stars");
    }

    #[test]
    fn star_rating_three_stars_for_score_0_82() {
        let formatter = super::TextFormatter;
        let comment = make_scored_comment(0.82, 0.9);
        let mut buf = Vec::new();
        formatter.format_comment(&comment, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let filled_count = output.matches('\u{2605}').count();
        assert_eq!(filled_count, 3, "score 0.82 should produce 3 filled stars");
    }

    #[test]
    fn text_formatter_formats_summary() {
        let formatter = super::TextFormatter;
        let mut buf = Vec::new();
        formatter
            .format_summary(100, 25, 10, Duration::from_millis(1234), None, &mut buf)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("100"), "summary should show total comments");
        assert!(
            output.contains("25"),
            "summary should show superfluous count"
        );
        assert!(output.contains("10"), "summary should show file count");
    }
}
