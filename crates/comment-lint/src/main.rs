use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use comment_lint_core::config::Config;
use comment_lint_core::output::features_jsonl::FeaturesJsonlFormatter;
use comment_lint_core::output::github::GithubFormatter;
use comment_lint_core::output::json::JsonFormatter;
use comment_lint_core::output::text::TextFormatter;
use comment_lint_core::output::OutputFormatter;
use comment_lint_core::pipeline::Pipeline;
use comment_lint_core::scoring::heuristic::HeuristicScorer;
use comment_lint_core::scoring::Scorer;

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

    /// Export all comment features as JSONL (for ML training data)
    #[arg(long)]
    export_features: bool,

    /// Scoring backend: "heuristic" (default), "ml", or "ensemble"
    #[arg(long, default_value = "heuristic")]
    scorer: String,

    /// Path to ONNX model file (required when --scorer ml, overrides config)
    #[arg(long)]
    model_path: Option<PathBuf>,
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

    // 3. Select output formatter; --export-features overrides config to capture everything
    let formatter: Box<dyn OutputFormatter> = if cli.export_features {
        config.general.threshold = 0.0;
        config.general.min_confidence = 0.0;
        config.general.include_doc_comments = true;
        Box::new(FeaturesJsonlFormatter)
    } else {
        match cli.format.as_str() {
            "json" => Box::new(JsonFormatter),
            "github" => Box::new(GithubFormatter),
            _ => Box::new(TextFormatter),
        }
    };

    // 4. Create scorer based on --scorer flag
    let scorer: Box<dyn Scorer + Send + Sync> = match cli.scorer.as_str() {
        "heuristic" => {
            Box::new(HeuristicScorer::new(config.weights.clone(), config.negative.clone()))
        }
        "ml" => {
            match create_ml_scorer(&cli, &config) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            }
        }
        "ensemble" => {
            match create_ensemble_scorer(&cli, &config) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            }
        }
        other => {
            eprintln!("error: unknown scorer '{other}', expected 'heuristic', 'ml', or 'ensemble'");
            return ExitCode::from(2);
        }
    };

    // 5. Create pipeline and run
    let pipeline = Pipeline::new(config, scorer);
    let result = pipeline.run(&cli.paths);

    // 6. Output results (renumbered after formatter selection moved earlier)
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

/// Create an ML scorer when the `ml` feature is enabled.
#[cfg(feature = "ml")]
fn create_ml_scorer(
    cli: &Cli,
    config: &Config,
) -> Result<Box<dyn Scorer + Send + Sync>, String> {
    use comment_lint_ml::scorer::MLScorer;

    // CLI --model-path takes priority, then config [ml].model_path
    let model_path = cli
        .model_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| config.ml.model_path.clone())
        .ok_or_else(|| {
            "ml scorer requires a model path via --model-path or [ml].model_path in config"
                .to_string()
        })?;

    let scorer = MLScorer::new(&model_path).map_err(|e| format!("failed to load ML model: {e}"))?;
    Ok(Box::new(scorer))
}

/// Create an ensemble scorer when the `ml` feature is enabled.
#[cfg(feature = "ml")]
fn create_ensemble_scorer(
    cli: &Cli,
    config: &Config,
) -> Result<Box<dyn Scorer + Send + Sync>, String> {
    use comment_lint_ml::ensemble::EnsembleScorer;
    use comment_lint_ml::scorer::MLScorer;

    // Resolve model path the same way as "ml"
    let model_path = cli
        .model_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| config.ml.model_path.clone())
        .ok_or_else(|| {
            "ensemble scorer requires a model path via --model-path or [ml].model_path in config"
                .to_string()
        })?;

    let heuristic = HeuristicScorer::new(config.weights.clone(), config.negative.clone());
    let ml = MLScorer::new(&model_path).map_err(|e| format!("failed to load ML model: {e}"))?;

    let ensemble = EnsembleScorer::new(Box::new(heuristic), Box::new(ml), 0.6);
    Ok(Box::new(ensemble))
}

/// Stub when the `ml` feature is not enabled.
#[cfg(not(feature = "ml"))]
fn create_ml_scorer(
    _cli: &Cli,
    _config: &Config,
) -> Result<Box<dyn Scorer + Send + Sync>, String> {
    Err("ml scorer is not available; rebuild with --features ml".to_string())
}

/// Stub when the `ml` feature is not enabled.
#[cfg(not(feature = "ml"))]
fn create_ensemble_scorer(
    _cli: &Cli,
    _config: &Config,
) -> Result<Box<dyn Scorer + Send + Sync>, String> {
    Err("ensemble scorer is not available; rebuild with --features ml".to_string())
}
