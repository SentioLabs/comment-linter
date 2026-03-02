use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use comment_lint_core::config::Config;
use comment_lint_core::output::github::GithubFormatter;
use comment_lint_core::output::json::JsonFormatter;
use comment_lint_core::output::text::TextFormatter;
use comment_lint_core::output::OutputFormatter;
use comment_lint_core::pipeline::Pipeline;
use comment_lint_core::scoring::heuristic::HeuristicScorer;

#[derive(Parser, Debug)]
#[command(name = "comment-lint", version, about = "Detect superfluous code comments")]
struct Cli {
    /// Files or directories to scan
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Output format: text, json, github
    #[arg(short, long, default_value = "text")]
    format: String,

    /// Minimum superfluity score to report (0.0-1.0)
    #[arg(short = 't', long)]
    threshold: Option<f64>,

    /// Minimum confidence to report (0.0-1.0)
    #[arg(long)]
    min_confidence: Option<f64>,

    /// Also analyze doc comments
    #[arg(long)]
    include_doc_comments: bool,

    /// Path to config file (TOML)
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // 1. Load config with resolution chain
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut config = match Config::resolve(cli.config.as_deref(), &cwd) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load config: {e}");
            return ExitCode::from(2);
        }
    };

    // 2. Apply CLI overrides (only if explicitly set)
    if let Some(t) = cli.threshold {
        config.general.threshold = t;
    }
    if let Some(mc) = cli.min_confidence {
        config.general.min_confidence = mc;
    }
    if cli.include_doc_comments {
        config.general.include_doc_comments = true;
    }

    // 3. Create scorer (takes weights AND negative weights)
    let scorer = HeuristicScorer::new(config.weights.clone(), config.negative.clone());

    // 4. Create pipeline and run
    let pipeline = Pipeline::new(config, Box::new(scorer));
    let result = pipeline.run(&cli.paths);

    // 5. Select output formatter (unit structs, no constructor args)
    let formatter: Box<dyn OutputFormatter> = match cli.format.as_str() {
        "json" => Box::new(JsonFormatter),
        "github" => Box::new(GithubFormatter),
        _ => Box::new(TextFormatter),
    };

    // 6. Output results
    let mut stdout = std::io::stdout().lock();
    for comment in &result.scored_comments {
        if let Err(e) = formatter.format_comment(comment, &mut stdout) {
            eprintln!("error writing output: {e}");
            return ExitCode::from(2);
        }
    }

    // format_summary needs: total_comments, superfluous_count, file_count
    // PipelineResult only gives us scored_comments and files_processed/files_skipped
    let superfluous_count = result.scored_comments.len();
    if let Err(e) = formatter.format_summary(
        superfluous_count,
        superfluous_count,
        result.files_processed,
        &mut stdout,
    ) {
        eprintln!("error writing summary: {e}");
        return ExitCode::from(2);
    }

    // 7. Exit code: 0 = clean, 1 = issues found
    if result.scored_comments.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
