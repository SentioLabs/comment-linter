use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use comment_lint_core::config::Config;
use comment_lint_core::output::features_jsonl::FeaturesJsonlFormatter;
use comment_lint_core::output::github::GithubFormatter;
use comment_lint_core::output::json::JsonFormatter;
use comment_lint_core::output::text::TextFormatter;
use comment_lint_core::output::OutputFormatter;
use comment_lint_core::pipeline::Pipeline;
use comment_lint_core::scoring::heuristic::HeuristicScorer;
use comment_lint_core::scoring::Scorer;

#[derive(Clone, Debug, ValueEnum)]
enum InputMode {
    /// Scan explicit file/directory paths (default)
    Files,
    /// Read unified diff from stdin, lint only added lines
    Diff,
}

#[derive(Parser, Debug)]
#[command(
    name = "comment-lint",
    version,
    about = "Detect superfluous code comments"
)]
struct Cli {
    /// Files or directories to scan (required for --input-mode=files)
    #[arg()]
    paths: Vec<PathBuf>,

    /// Input mode: "files" (default) or "diff" (read unified diff from stdin)
    #[arg(long, value_enum, default_value_t = InputMode::Files)]
    input_mode: InputMode,

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

    /// Disable .gitignore filtering (scan all files regardless of .gitignore rules)
    #[arg(long)]
    no_gitignore: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Validate input mode
    match cli.input_mode {
        InputMode::Files => {
            if cli.paths.is_empty() {
                eprintln!("error: at least one path is required with --input-mode=files");
                return ExitCode::from(2);
            }
        }
        InputMode::Diff => {
            use std::io::IsTerminal;
            if std::io::stdin().is_terminal() {
                eprintln!("error: --input-mode=diff requires piped input (e.g., git diff | comment-lint --input-mode=diff)");
                return ExitCode::from(2);
            }
        }
    }

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
    if cli.no_gitignore {
        config.ignore.respect_gitignore = false;
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
        "heuristic" => Box::new(HeuristicScorer::new(
            config.weights.clone(),
            config.negative.clone(),
        )),
        "ml" => match create_ml_scorer(&cli, &config) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        "ensemble" => match create_ensemble_scorer(&cli, &config) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        other => {
            eprintln!("error: unknown scorer '{other}', expected 'heuristic', 'ml', or 'ensemble'");
            return ExitCode::from(2);
        }
    };

    // 5. Create pipeline and run
    let cpu_before = cpu_time();
    let wall_start = Instant::now();

    let pipeline = Pipeline::new(config, scorer);
    let result = match cli.input_mode {
        InputMode::Files => pipeline.run(&cli.paths),
        InputMode::Diff => {
            use comment_lint_core::diff::filter::DiffFilter;
            let filter = match DiffFilter::from_stdin() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("error: failed to parse diff from stdin: {e}");
                    return ExitCode::from(2);
                }
            };
            pipeline.with_diff_filter(filter).run(&[])
        }
    };

    let elapsed = wall_start.elapsed();
    let cpu_elapsed = cpu_before.and_then(|before| cpu_time().map(|after| after - before));

    // 6. Output results (renumbered after formatter selection moved earlier)
    let mut stdout = std::io::stdout().lock();
    for comment in &result.scored_comments {
        if let Err(e) = formatter.format_comment(comment, &mut stdout) {
            eprintln!("error writing output: {e}");
            return ExitCode::from(2);
        }
    }

    let superfluous_count = result.scored_comments.len();
    if let Err(e) = formatter.format_summary(
        result.total_comments_scanned,
        superfluous_count,
        result.files_processed,
        elapsed,
        cpu_elapsed,
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
fn create_ml_scorer(cli: &Cli, config: &Config) -> Result<Box<dyn Scorer + Send + Sync>, String> {
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
fn create_ml_scorer(_cli: &Cli, _config: &Config) -> Result<Box<dyn Scorer + Send + Sync>, String> {
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

/// Read total CPU time (user + system) for the current process.
///
/// Returns `None` on non-Linux platforms or if `/proc/self/stat` is unreadable.
fn cpu_time() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields: Vec<&str> = stat.split_whitespace().collect();
    // Fields 13 and 14 are utime and stime in clock ticks.
    if fields.len() < 15 {
        return None;
    }
    let utime: u64 = fields[13].parse().ok()?;
    let stime: u64 = fields[14].parse().ok()?;
    let ticks_per_sec = 100u64; // sysconf(_SC_CLK_TCK) is 100 on virtually all Linux
    let total_ticks = utime + stime;
    Some(Duration::from_nanos(
        total_ticks * 1_000_000_000 / ticks_per_sec,
    ))
}
